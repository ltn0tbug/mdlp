import tensorflow as tf
from mdlp.net import FedIDSnet, DeepFednet
from mdlp.model.utils import classifier

"""
    # Author
    -   Nguyen Huu Quyen

    # Reference paper
    -   None

    # Reference Implementation
    -   https://colab.research.google.com/drive/1JPxYbqHYcMwJklxm74YPrI3RTdNUOKjH#scrollTo=aosz0tsNXTTE
"""


def FedIDSmodel(input_tensor=None, input_shape=None, n_classes=2):
    """Create model with FedIDS network

    # Arguments
        input_tensor: input tensor.
        input_shape: input shape of the input tensor. If `input_tensor` is set, the `input_shape` will be ignored.
        n_classes: in case of `n_classes` = 1, the model will use the `sigmoid` activation for prediction layers (the loss function in this case should be like `binary_crossentropy`). And in the others case, the model will user the `softmax` activation for prediction layers (the loss function in this case should be like `categorical_crossentropy`)

    # Returns
        Output a FedIDSnet model.

    # Raises
        ValueError: Both `input_tensor` and `input_shape` are empty.
    """

    if input_tensor is not None:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            inputs = tf.keras.layers.Input(tensor=input_tensor)
        else:
            inputs = input_tensor
    elif input_shape is not None:
        inputs = tf.keras.layers.Input(shape=input_shape)
    else:
        raise ValueError("Both `input_tensor` and `input_shape` are empty.")

    x = FedIDSnet(inputs)
    if n_classes == 1:
        outputs = classifier(x, n_classes, activation="sigmoid")
    else:
        outputs = classifier(x, n_classes, activation="softmax")

    return tf.keras.Model(inputs, outputs)


def DeepFedmodel(input_tensor=None, input_shape=None, n_classes=2, drop_out=0.2):
    """Create model with DeepFed network with `dropout` and `softmax` one prediction layer

    # Arguments
        input_tensor: input tensor.
        input_shape: input shape of the input tensor. If `input_tensor` is set, the `input_shape` will be ignored.
        n_classes: the n_classes should be at least 2 (equal to 2 for anomaly prediction). the model will use the `softmax` activation for prediction layers (the loss function in this case should be like `categorical_crossentropy`).
        drop_out: drop the value of prediction layer before activation layer. 

    # Returns
        Output a DeepFednet model.

    # Raises
        ValueError: Both `input_tensor` and `input_shape` are empty.
        ValueError: `n_classes` must be equal to or greater than 2.
    """

    if input_tensor is not None:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            inputs = tf.keras.layers.Input(tensor=input_tensor)
        else:
            inputs = input_tensor
    elif input_shape is not None:
        inputs = tf.keras.layers.Input(shape=input_shape)
    else:
        raise ValueError("Both `input_tensor` and `input_shape` are empty.")
    
    if n_classes < 2:
        raise ValueError("`n_classes` must be equal to or greater than 2.")

    x = DeepFednet(inputs)
    x = tf.keras.layers.Dense(n_classes)(x)
    x = tf.keras.layers.Dropout(drop_out,input_shape=(n_classes,))(x)
    outputs = tf.keras.layers.Softmax()(x)
    #outputs = classifier(x, 1, activation="sigmoid")

    return tf.keras.Model(inputs, outputs)