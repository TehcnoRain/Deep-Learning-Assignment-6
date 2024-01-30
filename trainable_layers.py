from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def define_dense_model_with_hidden_layers(input_length, 
                                          activation_func_array=['sigmoid', 'sigmoid'],
                                          hidden_layers_sizes=[50, 20],
                                          output_function='softmax',
                                          output_length=10):
    """
    Define a dense model with multiple hidden layers.

    Parameters:
    input_length (int): The number of inputs for the first layer.
    activation_func_array (list): The activation functions for the hidden layers.
    hidden_layers_sizes (list): The number of neurons in each hidden layer.
    output_function (str): The activation function for the output layer.
    output_length (int): The number of outputs (neurons in the output layer).

    Returns:
    keras.Sequential: The defined model with multiple hidden layers.
    """
    model = keras.Sequential()

    # Add the first hidden layer with the input_shape
    model.add(keras.layers.Dense(hidden_layers_sizes[0], activation=activation_func_array[0], input_shape=(input_length,)))

    # Add additional hidden layers
    for size, activation in zip(hidden_layers_sizes[1:], activation_func_array[1:]):
        model.add(keras.layers.Dense(size, activation=activation))

    # Add the output layer
    model.add(keras.layers.Dense(output_length, activation=output_function))

    return model


def set_layers_to_trainable(model, trainable_layer_numbers):
    """
    Set specific layers of the model to trainable or non-trainable.

    Parameters:
    model (keras.Model): The model to modify.
    trainable_layer_numbers (list): The indices of the layers to be set as trainable.

    Returns:
    keras.Model: The modified model with specific layers set as trainable or non-trainable.
    """
    # Iterate through all layers of the model
    for layer_num, layer in enumerate(model.layers):
        if layer_num in trainable_layer_numbers:
            layer.trainable = True
        else:
            layer.trainable = False

    return model