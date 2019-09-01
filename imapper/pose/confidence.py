import numpy as np


def get_conf_thresholded(conf, thresh_log_conf, dtype_np):
    """Normalizes a confidence score to (0..1).

    Args:
        conf (float):
            Unnormalized confidence.
        dtype_np (type):
            Desired return type.

    Returns:
        confidence (np.float32):
            Normalized joint confidence.
    """

    # 1. / (1. + np.exp(-5000. * conf + 5))
    # https://www.desmos.com/calculator/olqbvoffua
    # + 9.5: 0.0019 => 0.5
    # + 5  : 0.0010 => 0.5
    # + 6.5: 0.0013 => 0.5
    return np.where(
      conf < dtype_np(0.),
      dtype_np(0.),
      dtype_np(1.) /
      (dtype_np(1.) + np.exp(dtype_np(-5000.) * conf + dtype_np(9.5)))
    ).astype(dtype_np)


def get_confs(query_2d_full, frame_id, thresh_log_conf, mx_conf, dtype_np):
    """

    Args:
        query_2d_full (stealth.logic.skeleton.Skeleton):
            Skeleton with confidences.
        frame_id (int):
            Frame id.

    Returns:
        confs (List[float]):
            Confidences at frame_id.
    """

    confs = np.zeros(query_2d_full.poses.shape[-1],
                     dtype=dtype_np)
    is_normalized = query_2d_full.is_confidence_normalized()
    if query_2d_full.has_confidence(frame_id):
        for joint, conf in query_2d_full.confidence[frame_id].items():
            cnf = dtype_np(conf) \
                if is_normalized \
                else get_conf_thresholded(conf, thresh_log_conf, dtype_np)

            if mx_conf is not None and mx_conf < cnf:
                mx_conf = dtype_np(cnf)
            confs[joint] = dtype_np(cnf)
    if mx_conf is None:
        return confs
    else:
        assert isinstance(mx_conf, dtype_np)
        return confs, mx_conf
