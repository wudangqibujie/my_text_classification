import tensorflow as tf


def assert_rank(tensor, expected_rank, name=None):
    if name is None:
        name = tensor.name
    expected_rank_dict = dict()
    if isinstance(expected_rank, int):
        expected_rank_dict[expected_rank] = True
    else:
        for ix in expected_rank:
            expected_rank_dict[ix] = True
    actual_rank = tensor.shape.ndim
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
