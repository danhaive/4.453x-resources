import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import splu

DOUBLE_EPS = 1e-14
SIZING_EPS = 1e-6
MIN_EDGE_LENGTH = 1e-2
MAX_RADIUS = 0.5


def axial_stiffness_matrix(L, A, E):
    K = np.ones((2, 2))
    K[0, 1] = -1.0
    K[1, 0] = -1.0
    L = max(L, MIN_EDGE_LENGTH)  # nasty trick to deal with collapsed edges
    K *= A * E / L
    return K


def torsional_stiffness_matrix(L, J, G):
    """
    Parameters
    ----------
    L : [float]
        bar element length
    J : [float]
        torsional constant, unit: length unit^4
        In the case of circular, cylindrical shaft, it's equal to
        the polar moment of inertia of the cross section.
    G : [float]
        modulus of rigidity

    Return
    ------
    K_tor_x : 2x2 numpy array

    """
    return axial_stiffness_matrix(L, J, G)


def bending_stiffness_matrix(L, E, Iz, axis=2):
    """
    Parameters
    ----------
    L : [float]
        element length
    E : [float]
        Young's modulus
    Iz : [float]
        moment of inertia of the section about the z axis
            Iz = \int_A y^2 dA
    axis: int
        1 = local y axis , 2 = local z axis, default to 2

    Return
    ------
    K_bend_z : 4x4 numpy array
    """
    K = np.zeros([4, 4])
    sign = 1.0 if axis == 2 else -1.0
    K[0, :] = np.array([12.0, sign * 6 * L, -12.0, sign * 6 * L])
    K[1, :] = np.array([sign * 6 * L, 4 * (L ** 2), sign * -6 * L, 2 * (L ** 2)])
    K[2, :] = -K[0, :]
    K[3, :] = np.array([sign * 6 * L, 2 * (L ** 2), -sign * 6 * L, 4 * (L ** 2)])
    K *= E * Iz / L ** 3
    return K


def nu2G(nu, E):
    return E / (2 * (1 + nu))


def local_element_stiffness_matrix(L, A, Jx, Iy, Iz, E, nu):
    """complete 12x12 stiffness matrix for a bisymmetrical member.

        Since for small displacements the axial force effects, torsion,
        and bending about each axis are uncoupled, the influence coeff
        **relating** these effects are zero.

    Parameters
    ----------
    L : float
        element length
    A : float
        cross section area
    Jx : float
        torsional constant
        In the case of circular, cylindrical shaft, it's equal to
        the polar moment of inertia of the cross section.
    Iy : float
        moment of inertia w.r.t. y
    Iz : float
        moment of inertia w.r.t. z
    E : float
        Young's modulus
    nu : float
        Poisson ratio

    Returns
    -------
    K : 12x12 numpy array
    """
    G = nu2G(nu, E)
    # Fx1, Fx2 : u1, u2
    # 0, 6 : 0, 6
    axial_x_k = axial_stiffness_matrix(L, A, E)
    # Mx1, Mx2 : \theta_x1, \theta_x2
    # 3, 9 : 3, 9
    tor_x_k = torsional_stiffness_matrix(L, Jx, G)

    # Fy1, Mz1, Fy2, Mz2 : v1, \theta_z1, v2, \theta_z2
    # 1, 5, 7, 11 : 1, 5, 7, 11
    bend_z_k = bending_stiffness_matrix(L, E, Iz, axis=2)
    # Fz1, My1, Fz2, My2 : v1, \theta_z1, v2, \theta_z2
    # 2, 4, 8, 10 : 2, 4, 8, 10
    bend_y_k = bending_stiffness_matrix(L, E, Iz, axis=1)

    K = np.zeros([12, 12])
    K[np.ix_([0, 6], [0, 6])] += axial_x_k
    K[np.ix_([3, 9], [3, 9])] += tor_x_k
    K[np.ix_([1, 5, 7, 11], [1, 5, 7, 11])] += bend_z_k
    K[np.ix_([2, 4, 8, 10], [2, 4, 8, 10])] += bend_y_k
    return K


def element_rotation_matrix(start_node_coords, end_node_coords):
    L = np.linalg.norm(end_node_coords - start_node_coords)
    c_x = (end_node_coords[0] - start_node_coords[0]) / L
    c_y = (end_node_coords[1] - start_node_coords[1]) / L
    c_z = (end_node_coords[2] - start_node_coords[2]) / L
    R = np.array([[c_x, c_y, c_z, 0, 0, 0], [0, 0, 0, c_x, c_y, c_z]])
    return R


def local_to_global_transformation_matrix(
    start_node_coords, end_node_coords, rot_y2x=0.0
):
    L = np.linalg.norm(start_node_coords - end_node_coords)

    # by convention, the new x axis is along the element's direction
    # directional cosine of the new x axis in the global world frame
    c_x = (end_node_coords[0] - start_node_coords[0]) / L
    c_y = (end_node_coords[1] - start_node_coords[1]) / L
    R3 = np.zeros([3, 3])

    c_z = (end_node_coords[2] - start_node_coords[2]) / L
    # TODO rotaxis
    if abs(abs(c_z) - 1.0) < DOUBLE_EPS:
        # the element is parallel to global z axis
        # cross product is not defined, in this case
        # it's just a rotation about the global z axis
        # in x-y plane
        R3[0, 2] = -c_z
        R3[1, 1] = 1
        R3[2, 0] = c_z
    else:
        # local x_axis = element's vector
        new_x = np.array([c_x, c_y, c_z])
        # local y axis = cross product with global z axis
        new_y = -np.cross(new_x, [0, 0, 1.0])
        new_y /= np.linalg.norm(new_y)
        new_z = np.cross(new_x, new_y)
        R3[0, :] = new_x
        R3[1, :] = new_y
        R3[2, :] = new_z

    R = np.zeros((12, 12))
    for i in range(4):
        R[i * 3 : (i + 1) * 3, i * 3 : (i + 1) * 3] = R3
    return R


def global_element_stiffness_matrix(
    A, Jx, Iy, Iz, E, nu, start_node_coords, end_node_coords
):
    L = np.linalg.norm(end_node_coords - start_node_coords)
    local_stiffness_matrix = local_element_stiffness_matrix(L, A, Jx, Iy, Iz, E, nu)
    rotation_matrix = local_to_global_transformation_matrix(
        start_node_coords, end_node_coords
    )
    global_Ke = (rotation_matrix.T.dot(local_stiffness_matrix)).dot(rotation_matrix)
    return global_Ke


def assemble_global_stiffness_matrix(
    nodes, edges, E, nu, areas, jxs, iys, izs, edge_dof_map, dof_per_node=6
):
    n_dof = nodes.shape[0] * dof_per_node
    dof_per_element = dof_per_node * 2
    row = []
    col = []
    data = []
    for e, e_id, area, jx, iy, iz in zip(edges, edge_dof_map, areas, jxs, iys, izs):
        start_node_coords = nodes[e[0]]
        end_node_coords = nodes[e[1]]
        Ke = global_element_stiffness_matrix(
            area, jx, iy, iz, E, nu, start_node_coords, end_node_coords
        )
        for i in range(dof_per_element):
            for j in range(dof_per_element):
                if abs(Ke[i, j]) > DOUBLE_EPS:
                    row.append(e_id[i])
                    col.append(e_id[j])
                    data.append(Ke[i, j])
    Ksp = csc_matrix((data, (row, col)), shape=(n_dof, n_dof), dtype=float)
    return Ksp


def node_id_to_dof_map(num_nodes, dof_per_node=6):
    return np.arange(num_nodes * dof_per_node).reshape(-1, dof_per_node)


def edge_id_to_dof_map(edges, node_dof_map):
    return np.hstack([node_dof_map[edges[:, 0]], node_dof_map[edges[:, 1]]])


def compute_permutation_matrix(
    num_nodes, support_nodes: dict, node_dof_map, dof_per_node=6):
    # support nodes is dict {node_index: dof_fixities e.g. [0,0,0]}
    n_dofs = num_nodes * dof_per_node
    fixities_dofs = np.zeros(n_dofs)  # 0 means support is turned off
    for node in support_nodes.keys():
        dofs_indices = node_dof_map[node]
        fixities_dofs[dofs_indices] = support_nodes[node]
    # permutation map
    fixity_filter = fixities_dofs == 0
    n_fixed_dofs = np.sum(fixities_dofs).astype(int)
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


def solve(
    nodes,
    edges,
    loads,
    supports,
    E: float,
    nu: float,
    areas: np.ndarray,
    jxs: np.ndarray,
    iys: np.ndarray,
    izs: np.ndarray,
    dof_per_node=6,
):
    """
    nodes: [[x_i,y_i,x_i]] node i
    edges: [[i,j]] edges from node i to node j
    loads: [[f_x_i, f_y_i, f_z_i]] load at node i
    supports: {node_index:[0/1,0/1,0/1]} dict of supports
    E: Young's modulus for all elements
    mu: Poisson's ratio for all elements
    areas: [a_i] Cross-sectional areas for all elements
    jxs: [jx_i] torsion constants for all elements
    iys: [iy_i] moments of inertia w.r.t. y for all elements
    izs: [iz_i] moments of inertia w.r.t. z for all elements

    Returns displacement vector for all nodes
    """
    num_nodes = nodes.shape[0]
    node_dof_map = node_id_to_dof_map(num_nodes)
    edge_dof_map = edge_id_to_dof_map(edges, node_dof_map)
    permutation_matrix, n_free_dofs = compute_permutation_matrix(
        num_nodes, supports, node_dof_map
    )
    K = assemble_global_stiffness_matrix(
        nodes, edges, E, nu, areas, jxs, iys, izs, edge_dof_map
    )
    K_free, K_fixed = partition_matrix(K, permutation_matrix, n_free_dofs)
    F = loads.flatten()
    F_free, F_fixed = partition_vector(F, permutation_matrix, n_free_dofs)
    U_free = solve_stiffness_system(K_free, F_free)
    U = np.zeros(num_nodes * dof_per_node)
    U[:n_free_dofs] = U_free
    return (permutation_matrix.T * U).reshape((-1, dof_per_node))


def solve_circular_section(
    nodes,
    edges,
    loads,
    supports,
    E: float,
    nu: float,
    radii: np.ndarray,
    dof_per_node=6,
):

    areas = get_area_circle(radii)
    jxs = get_torsional_constant_circle(radii)
    izs = get_inertia_circle(radii)
    iys = get_inertia_circle(radii)

    return solve(
        nodes,
        edges,
        loads,
        supports,
        E,
        nu,
        areas,
        jxs,
        iys,
        izs,
        dof_per_node=dof_per_node,
    )


def solve_stiffness_system(K_free, F_free):
    D_diag = 1 / np.sqrt(K_free.diagonal())
    D = diags(D_diag, format="csc")
    K_free_precond = (D.dot(K_free)).dot(
        D
    )  # precondition stiffness matrix to improve numerical stability
    K_LU = splu(K_free_precond, diag_pivot_thresh=0, options={"SymmetricMode": True})
    U_free = K_LU.solve(D.dot(F_free))
    return D_diag * U_free


def get_area_circle(radius):
    return np.pi * radius ** 2


def get_inertia_circle(radius):
    return np.pi * radius ** 4 / 4


def get_torsional_constant_circle(radius):
    return np.pi * radius ** 4 / 2


if __name__ == "__main__":
    node_dof_map = node_id_to_dof_map(4)
    edges = [[0, 1], [1, 2], [2, 0], [3, 1], [3, 0], [2, 3]]
    print(edge_id_to_dof_map(np.array(edges), node_dof_map))
