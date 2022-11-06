import numpy as np


def project(X, output_range=(0, 1), absmax=None, input_is_positive_only=False):
    """Projects a tensor into a value range.
    Source: https://github.com/albermax/innvestigate/blob/master/src/innvestigate/utils/visualizations.py
    Projects the tensor values into the specified range.
    :param X: A tensor.
    :param output_range: The output value range.
    :param absmax: A tensor specifying the absmax used for normalizing.
      Default the absmax along the first axis.
    :param input_is_positive_only: Is the input value range only positive.
    :return: The tensor with the values project into output range.
    """

    if absmax is None:
        absmax = np.max(np.abs(X), axis=tuple(range(1, len(X.shape))))
    absmax = np.asarray(absmax)
    
    mask = absmax != 0
    absmax = np.expand_dims(absmax, -1)
    if mask.sum() > 0:
        X[mask] = X[mask] / absmax[mask]

    if input_is_positive_only is False:
        X = (X + 1) / 2  # [0, 1]
    X = X.clip(0, 1)

    X = output_range[0] + (X * (output_range[1] - output_range[0]))
    return X

def heatmap(X, cmap, reduce_op="sum", reduce_axis=-1, alpha_cmap=False, **kwargs):
    """Creates a heatmap/color map.
    Source: https://github.com/albermax/innvestigate/blob/master/src/innvestigate/utils/visualizations.py
    Create a heatmap or colormap out of the input tensor.
    :param X: A image tensor with 4 axes.
    :param cmap: The color map to use. Default 'seismic'.
    :param reduce_op: Operation to reduce the color axis.
      Either 'sum' or 'absmax'.
    :param reduce_axis: Axis to reduce.
    :param alpha_cmap: Should the alpha component of the cmap be included.
    :param kwargs: Arguments passed on to :func:`project`
    :return: The tensor as color-map.
    """

    tmp = X
    shape = tmp.shape

    if reduce_op == "sum":
        tmp = tmp.sum(axis=reduce_axis)
    elif reduce_op == "absmax":
        pos_max = tmp.max(axis=reduce_axis)
        neg_max = (-tmp).max(axis=reduce_axis)
        abs_neg_max = -neg_max
        tmp = np.select(
            [pos_max >= abs_neg_max, pos_max < abs_neg_max], [pos_max, neg_max]
        )
    else:
        raise NotImplementedError(f"reduce_op {reduce_op} not implemented for heatmap.")

    tmp = project(tmp, output_range=(0, 255), **kwargs).astype(np.int64)

    if alpha_cmap:
        tmp = cmap(tmp.flatten()).T
    else:
        tmp = cmap(tmp.flatten())[:, :3].T
    tmp = tmp.T

    shape = list(shape)
    shape[reduce_axis] = 3 + alpha_cmap
    return tmp.reshape(shape).astype(np.float32)


def restore_original_image_from_array(x, data_format='channels_last'):
    mean = [103.939, 116.779, 123.68]

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] += mean[0]
            x[1, :, :] += mean[1]
            x[2, :, :] += mean[2]
        else:
            x[:, 0, :, :] += mean[0]
            x[:, 1, :, :] += mean[1]
            x[:, 2, :, :] += mean[2]
    else:
        x[..., 0] += mean[0]
        x[..., 1] += mean[1]
        x[..., 2] += mean[2]

    if data_format == 'channels_first':
        # 'BGR'->'RGB'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'BGR'->'RGB'
        x = x[..., ::-1]

    return x