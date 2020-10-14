import tensorflow as tf

def snr_metric(y_true,y_pred):
  sum_1 = tf.math.reduce_sum(y_true ** 2)
  sum_2 = tf.math.reduce_sum((y_true - y_pred) ** 2)
  numerator = tf.math.log(sum_1/sum_2)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return 10 * (numerator / denominator)