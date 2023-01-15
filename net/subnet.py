import tensorflow as tf


def conv1d_bn_mp(x, filters, kernel_size, strides, padding, name=None):
    """Utility function to apply Conv1D, BatchNormalization, MaxPooling1D
    # Arguments
        x: input tensor.
        filters: filters in `Conv1D`.
        kernel_size: kernel_size in `Con1D`
        strides: a list,
            0: stride in `Con1D`
            1: stride in `MaxPooling1D`
        padding: a list,
            0: padding in `Con1D`
            1: padding in `MaxPooling1D`
    # Returns
        Output tensor:
            Conv1D,
            BatchNormalization,
            MaxPooling1D
    """

    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides[0],
        padding=padding[0],
        activation="relu",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(strides=strides[1], padding=padding[1])(x)
    return x
