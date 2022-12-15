import tensorflow as tf


def classifier(x, n_classes=1, pre_task=None, activation=None):
    """Create classification task

    # Arguments
        x: input tensor.
        n_classes: number of classes to classify.
        pre_task: optional Keras tensor is used to flatten or
        pooling networks before connecting to the prediction layer.
        activation: activation name is used in output layer.

    # Outputs
        a tensor with output layers is used for classification

    # Raises
        None
    """
    if pre_task is not None:
        x = pre_task()(x)

    x = tf.keras.layers.Dense(n_classes)(x)

    if activation is not None:
        x = tf.keras.layers.Activation(activation)(x)
    return x
