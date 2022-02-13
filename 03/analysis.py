import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import splu

DOUBLE_EPS = 1e-14
SIZING_EPS = 1e-6
MIN_EDGE_LENGTH = 1e-2
MAX_RADIUS = 0.5


def local_element_stiffness_matrix(L, A, E):
    K = np.ones((2, 2))
    K[0, 1] = -1.0
    K[1, 0] = -1.0
    L = max(L, MIN_EDGE_LENGTH)  # nasty trick to deal with collapsed edges
    K *= A * E / L
    return K


def element_rotation_matrix(start_node_coords, end_node_coords):
    L = np.linalg.norm(end_node_coords - start_node_coords)
    c_x = (end_node_coords[0] - start_node_coords[0]) / L
    c_y = (end_node_coords[1] - start_node_coords[1]) / L
    c_z = (end_node_coords[2] - start_node_coords[2]) / L
    R = np.array([[c_x, c_y, c_z, 0, 0, 0], [0, 0, 0, c_x, c_y, c_z]])
    return R


def global_element_stiffness_matrix(A, E, start_node_coords, end_node_coords):
    L = np.linalg.norm(end_node_coords - start_node_coords)
    local_stiffness_matrix = local_element_stiffness_matrix(L, A, E)
    rotation_matrix = element_rotation_matrix(start_node_coords, end_node_coords)
    global_Ke = (rotation_matrix.T.dot(local_stiffness_matrix)).dot(rotation_matrix)
    return global_Ke


def assemble_global_stiffness_matrix(
    nodes, edges, E, areas, edge_dof_map, dof_per_node=3
):
    n_dof = nodes.shape[0] * dof_per_node
    dof_per_element = dof_per_node * 2
    row = []
    col = []
    data = []
    for e, e_id, area in zip(edges, edge_dof_map, areas):
        start_node_coords = nodes[e[0]]
        end_node_coords = nodes[e[1]]
        Ke = global_element_stiffness_matrix(
            area, E, start_node_coords, end_node_coords
        )
        for i in range(dof_per_element):
            for j in range(dof_per_element):
                if abs(Ke[i, j]) > DOUBLE_EPS:
                    row.append(e_id[i])
                    col.append(e_id[j])
                    data.append(Ke[i, j])
    Ksp = csc_matrix((data, (row, col)), shape=(n_dof, n_dof), dtype=float)
    return Ksp


def node_id_to_dof_map(num_nodes, dof_per_node=3):
    return np.arange(num_nodes * dof_per_node).reshape(-1, dof_per_node)


def edge_id_to_dof_map(edges, node_dof_map):
    return np.hstack([node_dof_map[edges[:, 0]], node_dof_map[edges[:, 1]]])


def compute_permutation_matrix(
    num_nodes, support_nodes: dict, node_dof_map, dof_per_node=3
):
    # support nodes is dict {node_index: dof_fixities e.g. [0,0,0]}
    n_dofs = num_nodes * dof_per_node
    fixities_dofs = np.zeros(n_dofs)  # 0 means support is turned off
    for node in support_nodes.keys():
        dofs_indices = node_dof_map[node]
        fixities_dofs[dofs_indices] = support_nodes[node]
    # permutation map
    fixity_filter = fixities_dofs == 0
    n_fixed_dofs = np.int(np.sum(fixities_dofs))
    n_free_dofs = n_dofs - n_fixed_dofs
    dof_indices = np.arange(n_dofs)
    id_map = np.zeros_like(dof_indices)
    id_map[:n_free_dofs] = dof_indices[fixity_filter]
    id_map[n_free_dofs:] = dof_indices[~fixity_filter]
    # permutation matrix
    perm_data = []
    perm_row = []
    perm_col = []
    for i in range(n_dofs):
        perm_row.append(i)
        perm_col.append(id_map[i])
        perm_data.append(1)
    permutation_matrix = csc_matrix(
        (perm_data, (perm_row, perm_col)), shape=(n_dofs, n_dofs)
    )
    return permutation_matrix, n_free_dofs


def partition_matrix(matrix_to_partition, permutation_matrix, split_index):
    reordered = permutation_matrix * matrix_to_partition * permutation_matrix.T
    return (
        reordered[:split_index, :split_index],
        reordered[split_index:, split_index:],
    )


def partition_vector(vector_to_partition, permutation_matrix, split_index):
    reordered = permutation_matrix * vector_to_partition
    return (
        reordered[:split_index],
        reordered[split_index:],
    )


def solve(nodes, edges, loads, supports, E=210e9, areas=1, dof_per_node=3):
    """
    nodes: [[x_i,y_i,x_i]] node i
    edges: [[i,j]] edges from node i to node j
    loads: [[f_x_i, f_y_i, f_z_i]] load at node i
    supports: {node_index:[0/1,0/1,0/1]} dict of supports
    E: Young's modulus for all elements
    areas: [a_i] Cross-sectional areas for all elements

    Returns displacement vector for all nodes
    """
    num_nodes = nodes.shape[0]
    node_dof_map = node_id_to_dof_map(num_nodes)
    edge_dof_map = edge_id_to_dof_map(edges, node_dof_map)
    permutation_matrix, n_free_dofs = compute_permutation_matrix(
        num_nodes, supports, node_dof_map
    )
    K = assemble_global_stiffness_matrix(nodes, edges, E, areas, edge_dof_map)
    K_free, K_fixed = partition_matrix(K, permutation_matrix, n_free_dofs)
    F = loads.flatten()
    F_free, F_fixed = partition_vector(F, permutation_matrix, n_free_dofs)
    U_free = solve_stiffness_system(K_free, F_free)
    U = np.zeros(num_nodes * dof_per_node)
    U[:n_free_dofs] = U_free
    return (permutation_matrix.T * U).reshape((-1, dof_per_node))


def solve_stiffness_system(K_free, F_free):
    D_diag = 1 / np.sqrt(K_free.diagonal())
    D = diags(D_diag, format="csc")
    K_free_precond = (D.dot(K_free)).dot(
        D
    )  # precondition stiffness matrix to improve numerical stability
    K_LU = splu(K_free_precond, diag_pivot_thresh=0, options={"SymmetricMode": True})
    U_free = K_LU.solve(D.dot(F_free))
    return D_diag * U_free


def compute_element_lengths(nodes, edges):
    return np.sqrt(np.sum((nodes[edges[:, 0]] - nodes[edges[:, 1]]) ** 2, axis=1))


def compute_axial_forces(nodes, edges, disp, E=210e9, A=1):
    original_lengths = compute_element_lengths(nodes, edges)
    deformed_lengths = compute_element_lengths(nodes + disp, edges)
    delta = deformed_lengths - original_lengths
    return E * A * delta


def compute_self_weight(nodes, edges, areas, gamma):
    loads = np.zeros_like(nodes)
    element_lengths = compute_element_lengths(nodes, edges)
    for e, l, a in zip(edges, element_lengths, areas):
        half_load = a * l * gamma / 2
        loads[e[0], 2] -= half_load
        loads[e[1], 2] -= half_load
    return loads


def size_truss(
    nodes,
    edges,
    loads,
    supports,
    yield_stress,
    E,
    use_self_weight,
    gamma,
    wall_to_radius_ratio,
    buckling_safety=3,
    max_iterations=100,
    max_radius=MAX_RADIUS,
):
    n_elements = edges.shape[0]
    areas = np.ones(n_elements)
    # perform initial analysis
    disp = solve(nodes, edges, loads, supports, E, areas)
    axial_forces = compute_axial_forces(nodes, edges, disp, E, areas)
    lengths = compute_element_lengths(nodes, edges)
    sizes = [
        size_member(
            force, length, yield_stress, E, wall_to_radius_ratio, buckling_safety
        )
        for (force, length) in zip(axial_forces, lengths)
    ]
    new_areas = np.array([size[0] for size in sizes]) + 2 * DOUBLE_EPS
    new_radii = np.array([size[1] for size in sizes])
    max_area = np.pi * (
        max_radius ** 2 - (max_radius * (1 - wall_to_radius_ratio)) ** 2
    )
    # reanalyze and size until convergence
    iter_count = 0
    while (
        np.linalg.norm(new_areas - areas) > SIZING_EPS and iter_count < max_iterations
    ):
        areas = new_areas.copy()
        all_loads = (
            loads + compute_self_weight(nodes, edges, areas, gamma)
            if use_self_weight
            else loads
        )
        disp = solve(nodes, edges, all_loads, supports, E, areas)
        axial_forces = compute_axial_forces(nodes, edges, disp, E, areas)
        sizes = [
            size_member(
                force, length, yield_stress, E, wall_to_radius_ratio, buckling_safety
            )
            for (force, length) in zip(axial_forces, lengths)
        ]
        new_areas = np.minimum(max_area, np.array([size[0] for size in sizes]))
        new_radii = np.minimum(max_radius, np.array([size[1] for size in sizes]))
        iter_count += 1

    return new_areas, new_radii, axial_forces, disp


def size_member(
    force,
    length,
    yield_stress,
    E,
    wall_to_radius_ratio,
    buckling_safety=3,
    buckling_k=0.9,
    r_min=0.01,
):
    """
    returns section area and radius required to resist imposed force
    assumes hollow circular section
    """
    a_yield = np.abs(force) / yield_stress
    r_yield = np.sqrt(a_yield / np.pi / (1 - wall_to_radius_ratio ** 2))
    a_min = np.pi * r_min ** 2
    if buckling_safety and force < 0:
        euler_p_critical = -force * buckling_safety
        I_min = euler_p_critical / E / np.pi ** 2 * buckling_k * length ** 2
        r_buckling = np.sqrt(
            2 * np.sqrt(I_min / np.pi / (1 - wall_to_radius_ratio ** 4))
        )
        a_buckling = np.pi * r_buckling ** 2
        return max(a_yield, a_buckling, a_min), max(r_yield, r_buckling, r_min)
    return max(a_yield, a_min), max(r_yield, r_min)


if __name__ == "__main__":
    node_dof_map = node_id_to_dof_map(4)
    edges = [[0, 1], [1, 2], [2, 0], [3, 1], [3, 0], [2, 3]]
    print(edge_id_to_dof_map(np.array(edges), node_dof_map))
