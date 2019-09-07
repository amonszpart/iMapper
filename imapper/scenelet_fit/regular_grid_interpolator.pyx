# cython: language_level=3

import itertools

import cython
cimport numpy as np
import numpy as np
from scipy.interpolate.interpnd import _ndim_coords_from_arrays

from cpython cimport bool
# from libcpp cimport bool

# DTYPE = np.float64
# ctypedef np.float64_t DTYPE_t
# ctypedef np.float32_t DTYPEfloat32_t
# ctypedef np.int32_t DTYPEint32_t
from typedefs cimport DTYPE

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef np.ndarray[int, ndim=1] searchsorted(
   np.ndarray[DTYPEfloat32_t, ndim=1] grid_1d,
   np.ndarray[DTYPEfloat32_t, ndim=1] x_1d):
    cdef size_t nx = len(x_1d)
    cdef size_t n_grid = len(grid_1d)
    cdef np.ndarray[int, ndim=1] indices = np.zeros(nx, dtype=np.int32)
    cdef DTYPEint32_t i
    cdef DTYPEint32_t j
    for i in range(0, nx):
        j = 0
        while grid_1d[j] < x_1d[i] and j < n_grid:
            j += 1
        indices[i] = j - 1 # we need lower bound

    return indices

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef _find_indices(np.ndarray[np.float32_t, ndim=2] xi,
                   grid,
                   bool bounds_error):
    assert xi.dtype == np.float32, "Wrong type: %s" % xi.dtype

    cdef int n_dim = len(grid)
    cdef int n_points = xi.shape[1]

    # find relevant edges between which xi are situated
    cdef np.ndarray[int, ndim=2] indices = \
        np.zeros((n_dim, n_points), dtype=np.int32)

    # compute distance to lower edge in unity units
    cdef np.ndarray[DTYPEfloat32_t, ndim=2] norm_distances = \
        np.zeros((n_dim, n_points), dtype=np.float32)

    # check for out of bounds xi
    cdef np.ndarray[np.uint8_t, ndim=1] out_of_bounds = np.zeros(
       shape=(xi.shape[1]), dtype=np.uint8)

    # iterate through dimensions
    cdef int i_dim
    cdef np.ndarray[DTYPEfloat32_t, ndim=1] x
    cdef np.ndarray[DTYPEfloat32_t, ndim=1] _grid
    cdef np.ndarray[int, ndim=1] i_old
    cdef np.ndarray[int, ndim=1] i
    cdef np.int32_t ii
    cdef int _grid_size
    cdef int _grid_size_m2

    for i_dim in range(n_dim):
        x = xi[i_dim, :]
        _grid = grid[i_dim]
        _grid_size = _grid.size
        _grid_size_m2 = _grid_size - 2

        # i = searchsorted(_grid, x)

        i = np.searchsorted(_grid, x).astype(np.int32) - 1
        # assert np.allclose(i_old, i2), "No:\n%s\n%s" % (i_old, i2)
        # assert i_old.shape[0] == n_points, "No: %s" % repr(i_old)
        # i_old[i_old < 0] = 0
        # i_old[i_old > _grid.size - 2] = _grid.size - 2

        for ii in range(n_points):
            if i[ii] < 0:
                i[ii] = 0
            elif i[ii] > _grid_size_m2:
                i[ii] = _grid_size_m2
        # assert np.allclose(i_old, i2), "no"
        indices[i_dim, :] = i

        # TODO: cythonize
        norm_distances[i_dim, :] = (x - _grid[i]) / (_grid[i + 1] - _grid[i])
        if not bounds_error:
            out_of_bounds += (x < _grid[0]).astype(np.uint8)
            out_of_bounds += (x > _grid[_grid_size-1]).astype(np.uint8)
    # print("indices: %s" % repr(indices))
    return indices, norm_distances, out_of_bounds


cdef np.ndarray[DTYPE_t, ndim=1] _evaluate_linear_2d(
   np.ndarray[DTYPEint32_t, ndim=2] indices,
   np.ndarray[DTYPEfloat32_t, ndim=2] norm_distances,
   np.ndarray[np.uint8_t, ndim=1] out_of_bounds,
   np.ndarray[DTYPE_t, ndim=2] values_arg,
   DTYPEint32_t ndim):
    # slice for broadcasting over trailing dimensions in self.values
    # vslice = (slice(None),) + (None,)*(ndim - len(indices))

    # find relevant values
    # each i and i+1 represents a edge
    cdef size_t n_points = indices.shape[1]

    edges = itertools.product(*[[i, i + 1] for i in indices])
    cdef np.ndarray[DTYPE_t, ndim=1] values = np.zeros(n_points, dtype=DTYPE)
    for edge_indices in edges:
        weight = 1.
        for ei, i, yi in zip(edge_indices, indices, norm_distances):
            weight *= np.where(ei == i, 1 - yi, yi)
        values += np.asarray(values_arg[edge_indices]) * weight
    return values


cdef np.ndarray[DTYPE_t, ndim=1] _evaluate_linear_3d(
   np.ndarray[DTYPEint32_t, ndim=2] indices,
   np.ndarray[DTYPEfloat32_t, ndim=2] norm_distances,
   np.ndarray[np.uint8_t, ndim=1] out_of_bounds,
   np.ndarray[DTYPE_t, ndim=3] values_arg,
   DTYPEint32_t ndim):
    # slice for broadcasting over trailing dimensions in self.values
    # vslice = (slice(None),) + (None,)*(ndim - len(indices))

    # find relevant values
    # each i and i+1 represents a edge
    cdef size_t n_points = indices.shape[1]

    edges = itertools.product(*[[i, i + 1] for i in indices])
    cdef np.ndarray[DTYPE_t, ndim=1] values = np.zeros(n_points, dtype=DTYPE)
    for edge_indices in edges:
        weight = 1.
        for ei, i, yi in zip(edge_indices, indices, norm_distances):
            weight *= np.where(ei == i, 1 - yi, yi)
        values += np.asarray(values_arg[edge_indices]) * weight
    return values

cdef np.ndarray[DTYPEfloat32_t, ndim=2] _grid_to_array(
   tuple grid, size_t n_dims):
    # create a float-array for the grid as well
    grid_max_len = max(len(ps) for ps in grid)
    assert len(grid[0] <= grid_max_len)
    assert len(grid[1] <= grid_max_len)
    cdef np.ndarray[DTYPEfloat32_t, ndim=2] grid_array = np.zeros(
       shape=(len(grid), grid_max_len), dtype=np.float32)
    cdef int i_grid
    cdef int j_grid
    cdef int len_dim
    for i_grid in range(n_dims):
        len_dim = len(grid[i_grid])
        for j_grid in range(len_dim):
            grid_array[i_grid, j_grid] = grid[i_grid][j_grid]
    return grid_array

cdef class RegularGridInterpolator(object):
    """
    Interpolation on a regular grid in arbitrary dimensions

    The data must be defined on a regular grid; the grid spacing however may be
    uneven.  Linear and nearest-neighbour interpolation are supported. After
    setting up the interpolator object, the interpolation method (*linear* or
    *nearest*) may be chosen at each evaluation.

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions.

    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.

    method : str, optional
        The method of interpolation to perform. Supported are "linear" and
        "nearest". This parameter will become the default for the object's
        ``__call__`` method. Default is "linear".

    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.

    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.

    Methods
    -------
    __call__

    Notes
    -----
    Contrary to LinearNDInterpolator and NearestNDInterpolator, this class
    avoids expensive triangulation of the input data by taking advantage of the
    regular grid structure.

    .. versionadded:: 0.14

    Examples
    --------
    Evaluate a simple example function on the points of a 3D grid:

    ``data`` is now a 3D array with ``data[i,j,k] = f(x[i], y[j], z[k])``.
    Next, define an interpolating function from this data:

    >>> my_interpolating_function = RegularGridInterpolator((x, y, z), data)

    Evaluate the interpolating function at the two points
    ``(x,y,z) = (2.1, 6.2, 8.3)`` and ``(3.3, 5.2, 7.1)``:

    >>> pts = np.array([[2.1, 6.2, 8.3], [3.3, 5.2, 7.1]])
    >>> my_interpolating_function(pts)
    array([ 125.80469388,  146.30069388])

    which is indeed a close approximation to
    ``[f(2.1, 6.2, 8.3), f(3.3, 5.2, 7.1)]``.

    See also
    --------
    NearestNDInterpolator : Nearest neighbour interpolation on unstructured
                            data in N dimensions

    LinearNDInterpolator : Piecewise linear interpolant on unstructured data
                           in N dimensions

    References
    ----------
    .. [1] Python package *regulargrid* by Johannes Buchner, see
           https://pypi.python.org/pypi/regulargrid/
    .. [2] Trilinear interpolation. (2013, January 17). In Wikipedia, The Free
           Encyclopedia. Retrieved 27 Feb 2013 01:28.
           http://en.wikipedia.org/w/index.php?title=Trilinear_interpolation&oldid=533448871
    .. [3] Weiser, Alan, and Sergio E. Zarantonello. "A note on piecewise linear
           and multilinear table interpolation in many dimensions." MATH.
           COMPUT. 50.181 (1988): 189-196.
           http://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917826-0/S0025-5718-1988-0917826-0.pdf

    """
    # this class is based on code originally programmed by Johannes Buchner,
    # see https://github.com/JohannesBuchner/regulargrid

    # cdef object method
    # cdef bool bounds_error
    # cdef DTYPEfloat32_t fill_value
    # cdef object grid
    # cdef np.ndarray grid_array
    # cdef object values

    def __init__(self, points, values, method="linear", bounds_error=True,
                 fill_value=np.nan):
        cdef size_t n_dims = len(points)
        assert n_dims > 1 and n_dims <= 3, "No: %s" % repr(points.shape)

        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)
        self.method = method
        self.bounds_error = bounds_error

        if not hasattr(values, 'ndim'):
            # allow reasonable duck-typed values
            values = np.asarray(values)

        if len(points) > values.ndim:
            raise ValueError("There are %d point arrays, but values has %d "
                             "dimensions" % (len(points), values.ndim))

        if hasattr(values, 'dtype') and hasattr(values, 'astype'):
            if not np.issubdtype(values.dtype, np.inexact):
                values = values.astype(float)

        self.fill_value = fill_value
        if fill_value is not None:
            fill_value_dtype = np.asarray(fill_value).dtype
            if (hasattr(values, 'dtype') and not
            np.can_cast(fill_value_dtype, values.dtype,
                        casting='same_kind')):
                raise ValueError("fill_value must be either 'None' or "
                                 "of a type compatible with values")

        for i, p in enumerate(points):
            if not np.all(np.diff(p) > 0.):
                raise ValueError("The points in dimension %d must be strictly "
                                 "ascending" % i)
            if not np.asarray(p).ndim == 1:
                raise ValueError("The points in dimension %d must be "
                                 "1-dimensional" % i)
            if not values.shape[i] == len(p):
                raise ValueError("There are %d points and %d values in "
                                 "dimension %d" % (len(p), values.shape[i], i))
        self.grid = tuple([np.asarray(p) for p in points])
        self.grid_array = _grid_to_array(self.grid, n_dims)
        self.values = values


    def __call__(self, xi, method=None):
        """
        Interpolation at coordinates

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at

        method : str
            The method of interpolation to perform. Supported are "linear" and
            "nearest".

        """
        method = self.method if method is None else method
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)

        cdef int ndim = len(self.grid)
        xi = _ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             "%d, but this RegularGridInterpolator has "
                             "dimension %d" % (xi.shape[1], ndim))

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not np.logical_and(np.all(self.grid[i][0] <= p),
                                      np.all(p <= self.grid[i][-1])):
                    raise ValueError("One of the requested xi is out of bounds "
                                     "in dimension %d" % i)

        cdef np.ndarray[np.int32_t, ndim=2] indices
        cdef np.ndarray[np.uint8_t, ndim=1] out_of_bounds

        # indices, norm_distances, out_of_bounds = \
        #     _find_indices(xi.T, self.grid, self.bounds_error)
        indices, norm_distances, out_of_bounds = \
            self._py_find_indices(xi.T)
        # assert np.allclose(indices, indices2), \
        #     "No: %s %s" % (repr(indices), repr(indices2.shape))
        if method == "linear":
            result = self._py_evaluate_linear(indices,
                                              norm_distances,
                                              out_of_bounds)
            # if ndim == 2:
            #     result2 = _evaluate_linear_2d(indices,
            #                                   norm_distances,
            #                                   out_of_bounds,
            #                                   self.values,
            #                                   ndim)
            # elif ndim == 3:
            #     result2 = _evaluate_linear_3d(indices,
            #                                   norm_distances,
            #                                   out_of_bounds,
            #                                   self.values,
            #                                   ndim)
            # else:
            #     raise RuntimeError("Expected 2D or 3D, not %s" % ndim)

            # assert np.allclose(result, result2), "Don't match: %s %s" % (result, result2)
        elif method == "nearest":
            result = self._evaluate_nearest(indices,
                                            norm_distances,
                                            out_of_bounds)
        if not self.bounds_error and self.fill_value is not None:
            result[out_of_bounds] = self.fill_value
        # print("ndim: %s" % ndim)
        # print("xi_shape[:-1]: %s" % repr(xi_shape[:-1]))
        # print("shape: %s" % repr(self.values.shape[ndim:]))
        # print("shape2: %s" % repr(xi_shape[:-1] + self.values.shape[ndim:]))
        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:])
        # return result.reshape(xi_shape[:-1])

    def _py_evaluate_linear(self, indices, norm_distances, out_of_bounds):
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,)*(self.values.ndim - len(indices))

        # find relevant values
        # each i and i+1 represents a edge
        edges = itertools.product(*[[i, i + 1] for i in indices])
        values = 0.
        for edge_indices in edges:
            weight = 1.
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                weight *= np.where(ei == i, 1 - yi, yi)
            values += np.asarray(self.values[edge_indices]) * weight[vslice]
        return values

    def _evaluate_nearest(self, indices, norm_distances, out_of_bounds):
        idx_res = []
        for i, yi in zip(indices, norm_distances):
            idx_res.append(np.where(yi <= .5, i, i + 1))
        return self.values[idx_res]

    def _py_find_indices(self, xi):
        # find relevant edges between which xi are situated
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # check for out of bounds xi
        out_of_bounds = np.zeros((xi.shape[1]), dtype=np.uint8)
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
            i = np.searchsorted(grid, x) - 1
            i[i < 0] = 0
            i[i > grid.size - 2] = grid.size - 2
            indices.append(i)
            norm_distances.append((x - grid[i]) /
                                  (grid[i + 1] - grid[i]))
            if not self.bounds_error:
                out_of_bounds += (x < grid[0]).astype(np.uint8)
                out_of_bounds += (x > grid[-1]).astype(np.uint8)
        return np.array(indices, dtype=np.int32), \
               np.array(norm_distances, dtype=np.float32), \
               out_of_bounds

    @property
    def values(self):
        return self.values

    @property
    def grid(self):
        return self.grid

    @property
    def grid_array(self):
        return self.grid_array
