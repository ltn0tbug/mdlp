import tensorflow as tf
from .subnet import conv1d_bn_mp


def FedIDSnet(x):
    """ Create FedIDSnet instance.

    # Arguments
        x: input tensor.

    # Returns
        Output a FedIDSnet tensor.

    # Raises
        None
    """

    # Save input
    shortcut = x

    # Define layer
    for _ in range(3):
        x = conv1d_bn_mp(
            x=x, filters=32, kernel_size=3, strides=[1, 1], padding=["same", "same"]
        )

    x = tf.keras.layers.Flatten()(x)

    y = tf.keras.layers.GRU(3)(shortcut)

    x = tf.keras.layers.Concatenate()([x, y])

    x = tf.keras.layers.Dense(64)(x)

    return x

def DeepFednet(x):
    """ Create DeepFednet instance.

    # Arguments
        x: input tensor.

    # Returns
        Output a DeepFednet tensor.

    # Raises
        None
    """

    # Save input
    shortcut = x

    # Define layer
    for _ in range(3):
        x = conv1d_bn_mp(
            x=x, filters=32, kernel_size=3, strides=[1, 1], padding=["same", "same"]
        )

    x = tf.keras.layers.Flatten()(x)

    y = tf.keras.layers.GRU(3, return_sequences=True)(shortcut)

    y = tf.keras.layers.GRU(3)(y)

    x = tf.keras.layers.Concatenate()([x, y])

    x = tf.keras.layers.Dense(64)(x)

    return x
