# from tensorflow.python.ops import array_ops
from tensorflow.losses import cosine_distance
import tensorflow as tf



def triplet_loss(anchor_vector, positive_vector, negative_vector, metric='cosine_dist', margin=0.009):
    """Computes the triplet loss with semi-hard negative mining.
    The loss encourages the positive distances (between a pair of embeddings with
    the same labels) to be smaller than the minimum negative distance among
    which are at least greater than the positive distance plus the margin constant
    (called semi-hard negative) in the mini-batch. If no such negative exists,
    uses the largest negative distance instead.
    See: https://arxiv.org/abs/1503.03832.
    Args:
    labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
        multiclass integer labels.
    embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
        be l2 normalized.
    metric: 'cosine_dist' (default)
    margin: Float, margin term in the loss definition. default based on https://arxiv.org/pdf/1508.01585.pdf
    Returns:
    triplet_loss: tf.float32 scalar.
    """
    d_pos = cosine_distance(anchor_vector, positive_vector, axis=1)
    d_neg = cosine_distance(anchor_vector, negative_vector, axis=1)
    loss = tf.maximum(0., margin + d_pos - d_neg)
    loss = tf.reduce_mean(loss)
    return loss

#   # Build pairwise squared distance matrix.
#   pdist_matrix = cosine_distance(embeddings)
#   # Build pairwise binary adjacency matrix.
#   adjacency = math_ops.equal(labels, array_ops.transpose(labels))
#   # Invert so we can select negatives only.
#   adjacency_not = math_ops.logical_not(adjacency)

#   batch_size = array_ops.size(labels)

#   # Compute the mask.
#   pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
#   mask = math_ops.logical_and(
#       array_ops.tile(adjacency_not, [batch_size, 1]),
#       math_ops.greater(
#           pdist_matrix_tile, array_ops.reshape(
#               array_ops.transpose(pdist_matrix), [-1, 1])))
#   mask_final = array_ops.reshape(
#       math_ops.greater(
#           math_ops.reduce_sum(
#               math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True),
#           0.0), [batch_size, batch_size])
#   mask_final = array_ops.transpose(mask_final)

#   adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
#   mask = math_ops.cast(mask, dtype=dtypes.float32)

#   # negatives_outside: smallest D_an where D_an > D_ap.
#   negatives_outside = array_ops.reshape(
#       masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
#   negatives_outside = array_ops.transpose(negatives_outside)

#   # negatives_inside: largest D_an.
#   negatives_inside = array_ops.tile(
#       masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
#   semi_hard_negatives = array_ops.where(
#       mask_final, negatives_outside, negatives_inside)

#   loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

#   mask_positives = math_ops.cast(
#       adjacency, dtype=dtypes.float32) - array_ops.diag(
#           array_ops.ones([batch_size]))

#   # In lifted-struct, the authors multiply 0.5 for upper triangular
#   #   in semihard, they take all positive pairs except the diagonal.
#   num_positives = math_ops.reduce_sum(mask_positives)

#   triplet_loss = math_ops.truediv(
#       math_ops.reduce_sum(
#           math_ops.maximum(
#               math_ops.multiply(loss_mat, mask_positives), 0.0)),
#       num_positives,
#       name='triplet_semihard_loss')

#   return triplet_loss