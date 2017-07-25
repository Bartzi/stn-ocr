import numpy as np


def meshgrid(out_size):
    x, y = np.meshgrid(
        np.linspace(-1, 1, out_size.width),
        np.linspace(-1, 1, out_size.height),
        indexing='xy',
    )
    ones = np.ones(np.prod(x.shape))
    mesh = np.vstack((x.flatten(), y.flatten(), ones))

    return mesh


def get_sampling_grid(transform_params, output_size):
    grid = meshgrid(output_size)

    shape = transform_params.shape
    if len(shape) > 2:
        transform_params = transform_params.reshape(-1, shape[-1])

    sampling_grid = np.matmul(transform_params.reshape((len(transform_params), 2, 3)), grid)
    sampling_shape = sampling_grid.shape

    return np.reshape(sampling_grid, (shape[:-1] + sampling_shape[-2:]))
