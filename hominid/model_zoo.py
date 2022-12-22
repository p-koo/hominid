import tensorflow.keras as keras
import layers
from tensorflow.keras.regularizers import l1, l2, l1_l2


def PairwiseConvAtt(
    input_shape, 
    output_shape, 
    num_filters=128,
    kernel_size=15,
    diag=l2(1e-6), 
    offdiag=l2(1e-3),
    conv_activation='relu', 
    conv_dropout=0.2,
    conv_batchnorm=None,
    max_pool=10, 
    mha_heads=4,
    mha_dims=96,
    mha_dropout=0.1,
    mha_layernorm=False,
    dense_units=[256, 256],
    dense_dropout=[0.4, 0.4],
    dense_batchnorm=True,
    dense_activation='relu',
    output_activation='linear',
    ):

    inputs = keras.layers.Input(shape=input_shape)
    
    # pairwise conv layer
    nn = layers.PairwiseConv1D(num_filters, 
                               kernel_size=kernel_size, 
                               padding='same', 
                               kernel_regularizer=PairwiseReg(diag=diag, offdiag=offdiag), 
                               use_bias=True)(inputs)
    if conv_batchnorm:
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(conv_activation, name='conv_activation')(nn)
    nn = keras.layers.MaxPooling1D(max_pool)(nn)
    nn = keras.layers.Dropout(conv_dropout)(nn)

    # multi-head attention layer
    if mha_layernorm:
        nn = keras.layers.LayerNormalization()(nn)
    nn, att = MultiHeadAttention(num_heads=mha_heads, dims=mha_dims)(nn, nn, nn)
    nn = keras.layers.Dropout(mha_dropout)(nn)

    # mlp layers
    nn = keras.layers.Flatten()(nn)
    for units, dropout in zip(dense_units, dense_dropout):
        nn = keras.layers.Dense(units)(nn)
        if dense_batch_norm:
            nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation(dense_activation)(nn)
        nn = keras.layers.Dropout(dropout)(nn)

    # output layer
    if output_activation='linear':
        outputs = keras.layers.Dense(output_shape, activation='linear')(nn)
    else:
        logits = keras.layers.Dense(output_shape, activation='linear', name='logits')(nn)
        outputs = keras.layers.Activation(output_activation)(logits)
    
    return inputs, outputs




def AdditiveConvAtt(
    input_shape, 
    output_shape, 
    num_filters=128,
    kernel_size=15,
    conv_activation='relu', 
    conv_dropout=0.2,
    conv_batchnorm=None,
    max_pool=10, 
    mha_heads=4,
    mha_dims=96,
    mha_dropout=0.1,
    mha_layernorm=False,
    dense_units=[256, 256],
    dense_dropout=[0.4, 0.4],
    dense_batchnorm=True,
    dense_activation='relu',
    output_activation='linear',
    ):

    inputs = keras.layers.Input(shape=input_shape)

    # additive conv layer
    nn = nn = keras.layers.Conv1D(filters=128, kernel_size=15, padding='same', use_bias=True)(inputs)
    if conv_batchnorm:
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(conv_activation, name='conv_activation')(nn)
    nn = keras.layers.MaxPooling1D(max_pool)(nn)
    nn = keras.layers.Dropout(conv_dropout)(nn)

    # multi-head attention layer
    if mha_layernorm:
        nn = keras.layers.LayerNormalization()(nn)
    nn, att = MultiHeadAttention(num_heads=mha_heads, dims=mha_dims)(nn, nn, nn)
    nn = keras.layers.Dropout(mha_dropout)(nn)

    # mlp layers
    nn = keras.layers.Flatten()(nn)
    for units, dropout in zip(dense_units, dense_dropout):
        nn = keras.layers.Dense(units)(nn)
        if dense_batch_norm:
            nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation(dense_activation)(nn)
        nn = keras.layers.Dropout(dropout)(nn)

    # output layer
    if output_activation='linear':
        outputs = keras.layers.Dense(output_shape, activation='linear')(nn)
    else:
        logits = keras.layers.Dense(output_shape, activation='linear', name='logits')(nn)
        outputs = keras.layers.Activation(output_activation)(logits)
    
    return inputs, outputs




def AdditiveConvAttBase(
    input_shape, 
    num_tasks, 
    filters=128,
    kernel_size=15,
    conv_batchnorm=False,
    conv_activation='exponential',
    conv_dropout=0.2,
    downsample_factor=3,
    mha_heads=4,
    mha_dims=128,
    mha_dropout=0.1,
    mha_layernorm=False,
    bottleneck=128,
    decode_filters=64,
    decode_kernel_size=7,
    decode_batchnorm=True,
    decode_activation='relu',
    decode_dropout=0.4,
    num_resid=4,
    task_filters=32,
    task_kernel_size=7,
    task_dropout=0.2,
    task_activation='softplus',

    ):

    inputs = keras.layers.Input(shape=input_shape)

    # zero-pad to ensure L can downsample exactly with 2^downsample
    max_pool = 2**downsample_factor
    remainder = tf.math.mod(input_shape[0], max_pool)
    inputs_padded = tf.keras.layers.ZeroPadding1D((0, max_pool-remainder.numpy()))(inputs)

    # convolutional layer
    nn = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=True)(inputs_padded) 
    if conv_batchnorm:
        nn = keras.layers.BatchNormalization()(nn)
    nn_connect = keras.layers.Activation(conv_activation, name='conv_activation')(nn) 
    nn = keras.layers.Dropout(conv_dropout)(nn_connect)
    nn = keras.layers.MaxPool1D(max_pool, padding='same')(nn) #10

    # multi-head attention layer
    if mha_layernorm:
        nn = keras.layers.LayerNormalization()(nn)
    nn1, att = MultiHeadAttention(num_heads=mha_heads, dims=mha_dims)(nn, nn, nn)
    nn1 = keras.layers.Dropout(mha_dropout)(nn1)

    # expand back to base-resolution
    for i in range(downsample_factor):
        nn = keras.layers.Conv1DTranspose(filters=decode_filters, kernel_size=decode_kernel_size, strides=2, padding='same')(nn)
        if decode_batchnorm:
            nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation(decode_activation)(nn)
        nn = keras.layers.Dropout(decode_dropout)(nn)

    nn = keras.layers.Conv1D(filters=filters, kernel_size=5, padding='same')(nn)
    if decode_batchnorm:
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(decode_activation)(nn)
    nn = keras.layers.Dropout(decode_dropout)(nn)
    nn = keras.layers.add([nn, nn_connect])

    nn2 = residual_block(nn, 3, activation=dense_activation, dilated=num_resid) 
    nn = keras.layers.add([nn, nn2])

    outputs = []
    for i in range(num_tasks):
        nn2 = keras.layers.Conv1D(filters=output_filters, kernel_size=output_kernel_size, padding='same')(nn)
        nn2 = keras.layers.Activation(decode_activation)(nn2)
        nn2 = keras.layers.Dropout(task_dropout)(nn2)
        nn2 = keras.layers.Dense(1, activation=task_activation)(nn2)
        outputs.append(nn2)
    outputs = tf.concat(outputs, axis=2)
    return inputs, outputs





def residual_block(input_layer, filter_size, activation='relu', dilated=5):

    factor = []
    base = 2
    for i in range(dilated):
        factor.append(base**i)
        
    num_filters = input_layer.shape.as_list()[-1]  

    nn = keras.layers.Conv1D(filters=num_filters,
                                    kernel_size=filter_size,
                                    activation=None,
                                    use_bias=False,
                                    padding='same',
                                    dilation_rate=1,
                                    )(input_layer) 
    nn = keras.layers.BatchNormalization()(nn)
    for f in factor:
        nn = keras.layers.Activation('relu')(nn)
        nn = keras.layers.Dropout(0.1)(nn)
        nn = keras.layers.Conv1D(filters=num_filters,
                                        kernel_size=filter_size,
                                        strides=1,
                                        activation=None,
                                        use_bias=False, 
                                        padding='same',
                                        dilation_rate=f,
                                        )(nn) 
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.add([input_layer, nn])
    return keras.layers.Activation(activation)(nn)
    












