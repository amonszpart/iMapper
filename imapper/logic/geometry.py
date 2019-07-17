import numpy as np


def angle_axis(direction, angle, shape=(3, 3)):
    """
    Create a rotation matrix corresponding to the rotation around a general
    axis by a specified angle.

    R = dd^T + cos(a) (I - dd^T) + sin(a) skew(d)

    Parameters:

        angle : float a
        direction : array d
    """
    d = np.array(direction, dtype=np.float64)
    d /= np.linalg.norm(d)

    eye = np.eye(3, dtype=np.float64)
    ddt = np.outer(d, d)
    skew = np.array([[    0,  d[2],  -d[1]],
                     [-d[2],     0,  d[0]],
                     [d[1], -d[0],    0]], dtype=np.float64)

    mtx = ddt + np.cos(angle) * (eye - ddt) + np.sin(angle) * skew
    if shape[1] == 4:
        mtx = np.pad(mtx, pad_width=((0, 0), (0, 1)),
                     mode="constant", constant_values=0)
    if shape[0] == 4:
        mtx = np.pad(mtx, pad_width=((0, 1), (0, 0)),
                     mode="constant", constant_values=0)
        mtx[3, 3] = 1.
    assert mtx.shape == shape, "No: %s" % mtx.shape.__repr__()

    return mtx


def rotation_matrix(angle, direction, point=None, dtype=np.float32):
    """Return matrix to rotate about axis defined by point and direction.
    http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
    """
    assert direction.dtype == dtype, "Wrong: %s" % direction.dtype
    sina = dtype(np.math.sin(angle))
    cosa = dtype(np.math.cos(angle))
    direction = dtype(normalized(direction[:3]))
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    assert R.dtype == dtype, "Wrong: %s" % R.dtype
    R += np.outer(direction, direction) * (1.0 - cosa)
    assert R.dtype == dtype, "Wrong: %s" % R.dtype
    direction *= sina
    R += np.array(
        [
            [dtype(0.),     -direction[2],  direction[1]],
            [ direction[2], dtype(0.0),          -direction[0]],
            [-direction[1], direction[0],  dtype(0.0)]
        ]
    )
    assert R.dtype == dtype, "Wrong: %s" % R.dtype
    M = np.identity(4, dtype=dtype)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=dtype, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def normalized(a, axis=0):
    if (len(a.shape) == 2) and (np.any(np.array(a.shape) == 1)):
        return a / np.linalg.norm(a)
    else:
        norms = np.linalg.norm(a, axis=axis)
        return a / norms


def htransform(transform, a):
    assert transform.shape[0] >= 3 and transform.shape[1] == 4, \
        "Not a homog. transform with shape %s" % transform.shape.__repr__()
    tmp = np.matmul(transform[:3, :3], a)
    return tmp + np.expand_dims(transform[:3, 3], axis=1)


def homogeneous(a, axis=0, is_vector=False):
    """
    :param a: matrix-like 
    :param axis: which dimension to pad (0: rows, 1: cols)
    :param is_vector: pad with 0, not 1
    :return: 
    """
    assert False, "Deprecated, use htransform instead"

    assert axis < 2, "Only interpreted on the last two dimensions: %d" % axis
    # assert len(a.shape) <= 2, \
    #     "2D matrices assumed...%s" % (a.shape.__repr__())
    dim = len(a.shape)
    pad_width = [(0, 0) for i in range(dim)]
    pad_width[(dim - 2) + axis] = (0, 1)
    out = np.pad(
        a, pad_width=tuple(pad_width),
        mode='constant', constant_values=(0 if is_vector else 1))
    if dim == 2:
        shape = list(a.shape)
        shape[axis] += 1
        assert list(out.shape) == shape, \
            "Wrong shape from %s to %s, expected: %s" % \
            (a.shape.__repr__(), out.shape.__repr__(), shape.__repr__())
    elif dim == 3:
        shape = list(a.shape)
        shape[axis+1] += 1
        assert list(out.shape) == shape, \
            "Wrong shape from %s to %s, expected: %s" % \
            (a.shape.__repr__(), out.shape.__repr__(), shape.__repr__())

    return out


def hnormalized(a, axis=0):
    assert false, "Deprecated, use htransform instead"

    dim = len(a.shape)
    # assert len(a.shape) <= 2, \
    #     "2D matrices assumed...%s" % a.shape
    if dim <= 2:
        if axis == 0:  # rows
            out = a / a[-1, :]
            return out[:-1, :]
        elif axis == 1:
            out = a / a[:, -1]
            return out[:, :-1]
        else:
            raise RuntimeError("Axis %d not implemented" % axis)
    elif dim == 3:
        if axis == 0:  # rows
            print("a.shape: %s" % a.shape.__repr__())
            print("slices:\n%s\n%s" % (a[:, :-1, :], a[:, -2:-1, :]))
            out = a[:, :-1, :] / a[:, -2:-1, :]
            assert out.shape == (a.shape[0], a.shape[1] - 1, a.shape[2]), \
                "Wrong shape: %s vs %s" % (out.shape.__repr__(), a.shape.__repr__())
            return out
            # return out[:-1, :]
        elif axis == 1:
            out = a[:, :, :-1] / a[:, :, -2:-1]
            assert out.shape == (a.shape[0], a.shape[1], a.shape[2] - 1), \
                "Wrong shape: %i vs %i" % (out.shape, a.shape)
            return out
        else:
            raise RuntimeError("Axis %d not implemented" % axis)
    else:
        raise RuntimeError("Rank %d not implemented, shape: %s" % (dim, a.shape.__repr__()))


def rot_to_euler(R):
    """Returns XYZ angles from a 3x3 rotation matrix"""
    return (np.arctan2(R[2, 1], R[2, 2]),
            np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)),
            np.arctan2(R[1, 0], R[0, 0]))


# def rot_x(angle):
#     sn = np.sin(angle)
#     cn = np.cos(angle)
#     return np.matrix([[1., 0., 0.],
#                      [0., cn, -sn],
#                      [0., sn, cn]])

def rot_x(angle, shape=(4, 4)):
    """
    Generates a homogeneous rotation around the 3D X axis by the 
    angle in radians.
    :param angle: Angle of rotation in radians
    :param shape: Shape of output matrix, on of [(3, 3), (3, 4), (4, 4)]
    :return: A numpyr float32 ndarray with the shape as provided.
    """
    assert shape in [(3, 3), (3, 4), (4, 4)], \
        "Unexpected shape: %s" % shape
    sn = np.sin(angle)
    cn = np.cos(angle)
    return np.asarray(
        [[1., 0., 0., 0.],
         [0., cn, -sn, 0.],
         [0., sn, cn, 0.],
         [0., 0., 0., 1.]],
        dtype=np.float32)[:shape[0], :shape[1]]


# def rot_y(angle, shape=(3, 3)):
#     assert shape in [(3, 3), (3, 4), (4, 4)], \
#         "Unexpected shape: %s" % shape
#
#     sn = np.sin(angle)
#     cn = np.cos(angle)
#     out = np.matrix([[cn, 0., sn],
#                      [0., 1., 0.],
#                      [-sn, 0., cn]])
#     if shape[0] == 4 or shape[1] == 4:
#         out = np.pad(out,
#                      pad_width=((0, shape[0] == 4), (0, shape[1] == 4)),
#                      mode='constant', constant_values=0)
#     if shape[0] == 4:
#         out[3, 3] = 1.
#
#     return out

def rot_y(angle, shape=(4, 4)):
    """
    Generates a homogeneous rotation around the 3D -Y axis by the
    angle in radians.
    :param angle: Angle of rotation in radians
    :param shape: Shape of output matrix, on of [(3, 3), (3, 4), (4, 4)]
    :return: A numpy float32 ndarray with the shape as provided.
    """
    assert shape in [(3, 3), (3, 4), (4, 4)], \
        "Unexpected shape: %s" % shape
    sn = np.sin(angle)
    cn = np.cos(angle)
    return np.asarray(
        [[cn, 0., sn, 0.],
         [0., 1., 0., 0.],
         [-sn, 0., cn, 0.],
         [0., 0., 0., 1.]],
        dtype=np.float32)[:shape[0], :shape[1]]

def rot_y_y_up(angle, shape=(4, 4)):
    """
    Generates a homogeneous rotation around the 3D Y axis by the
    angle in radians.
    :param angle: Angle of rotation in radians
    :param shape: Shape of output matrix, on of [(3, 3), (3, 4), (4, 4)]
    :return: A numpy float32 ndarray with the shape as provided.
    """
    assert shape in [(3, 3), (3, 4), (4, 4)], \
        "Unexpected shape: %s" % shape
    sn = np.sin(angle)
    cn = np.cos(angle)
    return np.asarray(
       [[cn, 0., -sn, 0.],
        [0., 1., 0., 0.],
        [sn, 0., cn, 0.],
        [0., 0., 0., 1.]],
       dtype=np.float32)[:shape[0], :shape[1]]


def rot_z(angle, shape=(4, 4)):
    """
    Generates a homogeneous rotation around the 3D Z axis by the 
    angle in radians.
    :param angle: Angle of rotation in radians
    :param shape: Shape of output matrix, on of [(3, 3), (3, 4), (4, 4)]
    :return: A numpyr float32 ndarray with the shape as provided.
    """
    assert shape in [(3, 3), (3, 4), (4, 4)], \
        "Unexpected shape: %s" % shape
    sn = np.sin(angle)
    cn = np.cos(angle)
    return np.asarray(
        [[cn, -sn, 0., 0.],
         [sn, cn, 0., 0.],
         [0., 0., 1., 0.],
         [0., 0., 0., 1.]],
        dtype=np.float32)[:shape[0], :shape[1]]


def scale(scale, shape=(4, 4)):
    assert shape in [(3, 3), (3, 4), (4, 4)], \
        "Unexpected shape: %s" % shape
    if isinstance(scale, float) or isinstance(scale, np.float32):
        scale = [scale, scale, scale]
    elif isinstance(scale, list) and len(scale) == 1:
        scale = [scale[0], scale[0], scale[0]]
    elif isinstance(scale, tuple) and len(scale) == 1:
        scale = [scale[0], scale[0], scale[0]]

    return np.asarray(
        [[scale[0], 0., 0., 0.],
         [0., scale[1], 0., 0.],
         [0., 0., scale[2], 0.],
         [0., 0., 0., 1.]],
        dtype=np.float32)[:shape[0], :shape[1]]


def get_rotation(transform):
    """Gets rotation part of transform
    :param transform: A minimum 3x3 matrix
    """
    assert transform.shape[0] >= 3 \
           and transform.shape[1] >= 3, \
        "Not a transform? Shape: %s" % \
        transform.shape.__repr__()
    assert len(transform.shape) == 2, \
        "Assumed 2D matrices: %s" % \
        transform.shape.__repr__()

    return transform[:3, :3]


def angle_3d(v0, v1):
    cr = np.cross(v0, v1)
    return np.math.atan2(np.linalg.norm(cr),
                         np.matmul(v0, v1))


def project_vec(vec, tr_ground):
    """Project a vector to the x-z plane in tr_ground"""
    norm_up = normalized(tr_ground[:3, 1])
    return vec - np.dot(np.dot(vec, norm_up), norm_up)


def translation(vec, shape=(4, 4)):
    return np.array(
       [[1., 0., 0., vec[0]],
        [0., 1., 0., vec[1]],
        [0., 0., 1., vec[2]],
        [0., 0., 0., 1.]]
    ).astype(vec.dtype)[:shape[0], :shape[1]]