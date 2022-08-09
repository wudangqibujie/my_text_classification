import tensorflow as tf
import numpy as np


def assert_rank(tensor, expected_rank, name=None):
    if name is None:
        name = tensor.name
    expected_rank_dict = dict()
    if isinstance(expected_rank, int):
        expected_rank_dict[expected_rank] = True
    else:
        for ix in expected_rank:
            expected_rank_dict[ix] = True
    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
    if name is None:
        name = tensor.name
    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)
    shape = tensor.shape.as_list()
    non_static_indexes = []
    for ix, dim in enumerate(shape):
        if dim is None:
            non_static_indexes.append(ix)
    if not non_static_indexes:
        return shape
    dyn_shape = tf.shape(tensor)
    for ix in non_static_indexes:
        shape[ix] = dyn_shape[ix]
    return shape


def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))
    ))
    return x * cdf


def reshape_to_matrix(input_tensor):
    ndims = input_tensor.shape.ndims
    if ndims == 2:
        return input_tensor
    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    if len(orig_shape_list) == 2:
        return output_tensor
    output_shape = get_shape_list(output_tensor)
    orig_dims = orig_shape_list[0: -1]
    width = output_shape[-1]
    return tf.reshape(output_tensor, orig_dims + [width])


def get_activation(activation_string):
    if not isinstance(activation_string, str):
        return activation_string
    if not activation_string:
        return None
    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu6
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise
