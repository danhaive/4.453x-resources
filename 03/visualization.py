import asyncio
from trimesh.util import concatenate
from trimesh.creation import cylinder
import k3d
import numpy as np
from .analysis import MAX_RADIUS


class Timer:
    def __init__(self, timeout, callback):
        self._timeout = timeout
        self._callback = callback
        self._task = asyncio.ensure_future(self._job())

    async def _job(self):
        await asyncio.sleep(self._timeout)
        self._callback()

    def cancel(self):
        self._task.cancel()


def debounce(wait):
    """Decorator that will postpone a function's
    execution until after `wait` seconds
    have elapsed since the last time it was invoked."""

    def decorator(fn):
        timer = None

        def debounced(*args, **kwargs):
            nonlocal timer

            def call_it():
                fn(*args, **kwargs)

            if timer is not None:
                timer.cancel()
            timer = Timer(wait, call_it)

        return debounced

    return decorator


def create_truss_plot_object(vertices, edges, forces, radii):
    meshes = []
    vertex_attributes = []
    for e, r, f in zip(edges, radii, forces):
        segment_vertices = vertices[e]
        segment = np.round(vertices[e], 2)
        mesh = cylinder(radius=min(r, MAX_RADIUS), segment=segment)
        vertex_attributes += [f] * mesh.vertices.shape[0]
        meshes.append(mesh)
    vertex_attributes = np.array(vertex_attributes).flatten()
    merged_mesh = concatenate(meshes)
    # max_force = np.max(np.abs(forces))
    max_force = 1000000
    color_range = [-max_force /1000, +max_force / 1000]
    return k3d.mesh(
        merged_mesh.vertices.astype(np.float32),
        merged_mesh.faces.astype(np.int32),
        attribute=vertex_attributes / 1000,
        color_range=color_range,
        color_map=k3d.colormaps.matplotlib_color_maps.seismic_r,
    )


def reset_plot(plot):
    for plot_object in plot.objects:
        plot -= plot_object
        