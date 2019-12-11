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
    cosine_distance = tf.keras.losses.CosineSimilarity(axis=1)
    d_pos = cosine_distance(anchor_vector, positive_vector)
    d_neg = cosine_distance(anchor_vector, negative_vector)
    loss = tf.maximum(0., margin + d_pos - d_neg)
    loss = tf.reduce_mean(loss)
    return loss

