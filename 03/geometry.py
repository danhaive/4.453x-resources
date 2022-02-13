import numpy as np
from geomdl import utilities, BSpline
from scipy.spatial import KDTree


def generate_3d_point_grid(width=10, n=3):
    cpts = np.zeros((n ** 2, 3))
    cpts[:, :2] = generate_2d_point_grid(m=n, n=n, start=-width / 2, end=width / 2)
    return cpts


def generate_2d_point_grid(m=3, n=3, start=0, end=1):
    u = np.linspace(start, end, m)
    v = np.linspace(start, end, n)
    x, y = np.meshgrid(u, v)
    return np.vstack([x.flatten(), y.flatten()]).T


def convert_to_packed_rgb(r, g, b):
    rgb = 65536 * r + 256 * g + b
    return rgb


def generated_connected_grid_on_surface(
    surface, n_subdivisions_u=5, n_subdivisions_v=5
):
    point_grid = generate_2d_point_grid(n_subdivisions_u, n_subdivisions_v)
    vertices = evaluate_surf(surface, point_grid)
    edges = create_grid_connectivity(n_subdivisions_u, n_subdivisions_v)
    return vertices, edges


def evaluate_surf(surface, parameters):
    return list(map(surface.evaluate_single, parameters))


def create_grid_connectivity(m, n):
    edges = []
    for i in range(m):
        edges += [[j, j + 1] for j in range(i * n, i * n + n - 1)]
    for i in range(n):
        edges += [[i + j * m, i + j * m + n] for j in range(m - 1)]
    return edges


def generate_custom_surface_var_central_cpoint(z, width, c=0):
    cpts = generate_3d_point_grid(width=width, n=3)
    cpts[4, 2] = z
    cpts[:, 2] += c
    surf = BSpline.Surface()
    surf.degree_u, surf.degree_v = 2, 2
    surf.set_ctrlpts(cpts.tolist(), 3, 3)
    surf.knotvector_u = utilities.generate_knot_vector(
        surf.degree_u, surf.ctrlpts_size_u
    )
    surf.knotvector_v = utilities.generate_knot_vector(
        surf.degree_v, surf.ctrlpts_size_v
    )
    return surf


def generate_symmetrical_custom_surface_with_6_var_cpts(z, width, c=0):
    assert len(z) == 6, "Incorrect number of control point heights provided."
    cpts = generate_3d_point_grid(width=width, n=3)
    cpts[:6, 2] = z
    cpts[6, 2] = z[0]
    cpts[7, 2] = z[1]
    cpts[8, 2] = z[2]
    cpts[:, 2] += c
    surf = BSpline.Surface()
    surf.degree_u, surf.degree_v = 2, 2
    surf.set_ctrlpts(cpts.tolist(), 3, 3)
    surf.knotvector_u = utilities.generate_knot_vector(
        surf.degree_u, surf.ctrlpts_size_u
    )
    surf.knotvector_v = utilities.generate_knot_vector(
        surf.degree_v, surf.ctrlpts_size_v
    )
    return surf


def generate_frame(surf1, surf2, m, n):
    n_vertices_per_surface = m * n
    point_grid = generate_2d_point_grid(m, n)  # common point-grid for both surfaces
    vertices_surf1 = evaluate_surf(surf1, point_grid)
    vertices_surf2 = evaluate_surf(surf2, point_grid)
    vertices = vertices_surf1 + vertices_surf2
    edges_surf1 = create_grid_connectivity(m, n)
    edges_surf2 = (
        np.array(create_grid_connectivity(m, n)) + n_vertices_per_surface
    ).tolist()
    edges = edges_surf1 + edges_surf2
    # verticals
    for i in range(n_vertices_per_surface):
        edges.append([i, i + n_vertices_per_surface])
    return vertices, edges


def generate_diagonals(surf1, surf2, m, n):
    n_vertices_per_surface = m * n
    point_grid_surf1 = generate_2d_point_grid(m, n)  # common point-grid for surf 1
    offset_surf = 0.5 / (m - 1)  # offset in parameter space for surf2
    point_grid_surf2 = generate_2d_point_grid(
        m - 1, n - 1, start=offset_surf, end=1 - offset_surf
    )  # common point-grid for surf 2

    vertices_surf1 = evaluate_surf(surf1, point_grid_surf1)
    vertices_surf2 = evaluate_surf(surf2, point_grid_surf2)
    n_v_surf1 = m * n
    n_v_surf2 = (m - 1) * (n - 1)

    vertices = vertices_surf1 + vertices_surf2
    edges = []
    for i in range(m - 1):
        for j in range(n - 1):
            anchor_index = i * (n - 1) + j
            surf2_v_index = anchor_index + n_v_surf1
            edges.append([surf2_v_index, anchor_index + i])  # bottom-left
            edges.append([surf2_v_index, anchor_index + i + 1])  # top-left
            edges.append([surf2_v_index, anchor_index + i + n])  # bottom-right
            edges.append([surf2_v_index, anchor_index + i + n + 1])  # top-right
    return vertices, edges


def generate_truss(surf1, surf2, m, n):
    n_vertices_per_surface = m * n
    point_grid_surf1 = generate_2d_point_grid(m, n)  # common point-grid for surf 1
    offset_surf = 0.5 / (m - 1)  # offset in parameter space for surf2
    point_grid_surf2 = generate_2d_point_grid(
        m - 1, n - 1, start=offset_surf, end=1 - offset_surf
    )  # common point-grid for surf 2

    vertices_surf1 = evaluate_surf(surf1, point_grid_surf1)
    vertices_surf2 = evaluate_surf(surf2, point_grid_surf2)
    n_v_surf1 = m * n
    n_v_surf2 = (m - 1) * (n - 1)

    vertices = vertices_surf1 + vertices_surf2
    edges = []
    # diagonals
    for i in range(m - 1):
        for j in range(n - 1):
            anchor_index = i * (n - 1) + j
            surf2_v_index = anchor_index + n_v_surf1
            edges.append([surf2_v_index, anchor_index + i])  # bottom-left
            edges.append([surf2_v_index, anchor_index + i + 1])  # top-left
            edges.append([surf2_v_index, anchor_index + i + n])  # bottom-right
            edges.append([surf2_v_index, anchor_index + i + n + 1])  # top-right
    # horizontals
    edges_surf1 = create_grid_connectivity(m, n)
    edges_surf2 = (
        np.array(create_grid_connectivity(m - 1, n - 1)) + n_v_surf1
    ).tolist()
    edges += edges_surf1 + edges_surf2
    return np.array(vertices), np.array(edges)


def generate_truss_with_columns(surf1, surf2, m, n, column_locations, n_connections):
    vertices, edges = generate_truss(surf1, surf2, m, n)
    n_vertices = vertices.shape[0]
    kd_tree = KDTree(vertices)
    _, vertex_indices = kd_tree.query(column_locations, k=n_connections)
    vertices = np.append(vertices, column_locations, axis=0)

    for i, v_indices in enumerate(vertex_indices):
        for v_index in v_indices:
            new_edges = np.array([[v_index, n_vertices + i]])
            edges = np.append(edges, new_edges, axis=0)
    return vertices, edges


def generate_space_truss_1(z_bottom, z_top, width, n_modules, column_locations):
    surf1 = generate_custom_surface_var_central_cpoint(z_bottom, width)
    surf2 = generate_custom_surface_var_central_cpoint(z_top, width, c=0.5)

    return generate_truss_with_columns(
        surf1, surf2, n_modules, n_modules, column_locations, 4
    )


def generate_space_truss_2(z1, z2, z3, z4, z5, z6, width, n_modules, column_locations):
    surf1 = generate_symmetrical_custom_surface_with_6_var_cpts([0] * 6, width, c=0.5)
    surf2 = generate_symmetrical_custom_surface_with_6_var_cpts(
        [z1, z2, z3, z4, z5, z6], width, c=0
    )

    return generate_truss_with_columns(
        surf1, surf2, n_modules, n_modules, column_locations, 4
    )
