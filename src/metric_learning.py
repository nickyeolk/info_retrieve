# from tensorflow.python.ops import array_ops
from tensorflow.losses import cosine_distance
import tensorflow as tf
# from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

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


def contrastive_loss(labels, embeddings_anchor, embeddings_positive,
                     margin=1.0):
  """Computes the contrastive loss.
  This loss encourages the embedding to be close to each other for
  the samples of the same label and the embedding to be far apart at least
  by the margin constant for the samples of different labels.
  See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

  Args:
      labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
      binary labels indicating positive vs negative pair.
      embeddings_anchor: 2-D float `Tensor` of embedding vectors for the anchor
      images. Embeddings should be l2 normalized.
      embeddings_positive: 2-D float `Tensor` of embedding vectors for the
      positive images. Embeddings should be l2 normalized.
      margin: margin term in the loss definition.

  Returns:
      contrastive_loss: tf.float32 scalar.
  """
  # Get per pair distances
  distances = cosine_distance(embeddings_anchor, embeddings_positive, axis=1)
  # Add contrastive loss for the siamese network.
  #   label here is {0,1} for neg, pos.
  return tf.reduce_mean(
      tf.cast(labels, dtypes.float32) * distances +
      (1. - tf.cast(labels, dtypes.float32)) *
      tf.maximum(margin - distances, 0.),
      name='contrastive_loss')