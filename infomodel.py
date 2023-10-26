from keras.models import load_model

def describe_layer(layer):
    """Return a string description of the layer."""
    layer_type = layer.__class__.__name__
    config = layer.get_config()
    
    if layer_type == 'Conv2D':
        return f"model.add(Conv2D({config['filters']}, {config['kernel_size']}, activation='{config['activation']}', padding='{config['padding']}'))"
    elif layer_type == 'MaxPooling2D':
        return f"model.add(MaxPooling2D(pool_size={config['pool_size']}))"
    elif layer_type == 'Dense':
        return f"model.add(Dense({config['units']}, activation='{config['activation']}'))"
    elif layer_type == 'Dropout':
        return f"model.add(Dropout({config['rate']}))"
    elif layer_type == 'Flatten':
        return f"model.add(Flatten())"
    elif layer_type == 'InputLayer':
        return f"model.add(InputLayer(input_shape={config['batch_input_shape'][1:]}))"
    elif layer_type == 'ZeroPadding2D':
        return f"model.add(ZeroPadding2D(padding={config['padding']}))"
    elif layer_type == 'BatchNormalization':
        return f"model.add(BatchNormalization())"
    elif layer_type == 'ReLU':
        return f"model.add(ReLU())"
    elif layer_type == 'DepthwiseConv2D':
        return f"model.add(DepthwiseConv2D(kernel_size={config['kernel_size']}, strides={config['strides']}, padding='{config['padding']}'))"
    elif layer_type == 'Add':
        return f"model.add(Add())"
    # ... add more layer types as needed ...
    else:
        return f"model.add({layer_type}())  # Check this layer's parameters manually"


def get_model_description(layers):
    """Return a string description of the model architecture."""
    descriptions = []
    for layer in layers:
        if hasattr(layer, 'layers'):  # If it's a Sequential or Model layer
            descriptions.extend(get_model_description(layer.layers))
        else:
            descriptions.append(describe_layer(layer))
    return descriptions

# Load the model
model = load_model("keras_model.h5", compile=False)

# Get model description
descriptions = get_model_description(model.layers)
for desc in descriptions:
    print(desc)
