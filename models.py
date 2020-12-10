def DRIB(tensor_input,filters,kernel_size=1,pooling_size=1,dropout=0.5):
    # Defines a diluted residual inception block
    # This block is fixed at four levels named a,b,c,d
    
    filts = int(filters/4)

    a1 = Conv1D(filts,1)(tensor_input)
    a2 = BatchNormalization()(a1)

    
    b1 = Conv1D(filts,1)(tensor_input)
    b2 = BatchNormalization()(b1)
    b3 = ReLU()(b2)
    b4 = Conv1D(filts,3, padding = 'same',dilation_rate = 2)(b3)
    b5 = BatchNormalization()(b4)

    
    c1 = Conv1D(filts,1)(tensor_input)
    c2 = BatchNormalization()(c1)
    c3 = ReLU()(c2)
    c4 = Conv1D(filts,3, padding = 'same',dilation_rate = 4)(c3)
    c5 = BatchNormalization()(c4)

    
    d1 = Conv1D(filts,1)(tensor_input)
    d2 = BatchNormalization()(d1)
    d3 = ReLU()(d2)
    d4 = Conv1D(filts,3, padding = 'same',dilation_rate = 8)(d3)
    d5 = BatchNormalization()(d4)
    
    merge1 = concatenate([a2, b5, c5, d5], axis = 2)

    out = Add()([tensor_input,merge1])
    return out

def unet(pretrained_weights = None,input_size = (200,1),windowSize=200):
    # This is the top level defines the UNET architecture
    
    inputs = Input(input_size)
    n=1 # 7number of filters. This should be pushed back to a parameter.
    
    conv0 = Conv1D(filters=int(1*n), kernel_size=1, strides=1, padding="same")(inputs)               
    bn0 = BatchNormalization()(conv0)
    r0 = LeakyReLU(alpha=0.2)(bn0)
    
    conv1 = Conv1D(filters=int(8*n), kernel_size=4, strides=2, padding="same")(r0)                  
    bn1 = BatchNormalization()(conv1)
    r1 = LeakyReLU(alpha=0.2)(bn1)
    
    conv2 = Conv1D(filters=int(16*n), kernel_size=4, strides=2, padding="same")(r1)              
    bn2 = BatchNormalization()(conv2)
    r2 = LeakyReLU(alpha=0.2)(bn2)
    
    conv3 = Conv1D(filters=int(32*n), kernel_size=4, strides=2, padding="same")(r2)              
    bn3 = BatchNormalization()(conv3)
    r3 = LeakyReLU(alpha=0.2)(bn3)
    
    conv4 = Conv1D(filters=int(64*n), kernel_size=4, strides=2, padding="same")(r3)              
    bn4 = BatchNormalization()(conv4)
    r4 = LeakyReLU(alpha=0.2)(bn4)
    
    conv5 = Conv1D(filters=int(128*n), kernel_size=4, strides=2, padding="same")(r4)               
    bn5 = BatchNormalization()(conv5)
    r5 = LeakyReLU(alpha=0.2)(bn5)
    
    conv6 = Conv1D(filters=int(128*n), kernel_size=1, strides=1, padding="same")(r5)               
    
    drib1 = DRIB(conv6,128*n)
    print('drib1 conv5', drib1.get_shape(), conv5.get_shape())
    u1 = concatenate([drib1, conv5])
    c1 = Conv1D( filters=int(64*n), kernel_size=1, strides=1, activation = 'relu')(u1) #, activation = 'sigmoid'
    
    drib2 = DRIB(c1,64*n)
    u2 = UpSampling1D(size = 2)(drib2)
    print('u2 conv4', u2.get_shape(), conv4.get_shape())
    u2 = concatenate([u2, conv4])
    c2 = Conv1D( filters=int(32*n), kernel_size=1, strides=1, activation = 'relu')(u2) #, activation = 'sigmoid'
    
    drib3 = DRIB(c2,32*n)
    u3 = UpSampling1D(size = 2)(drib3)
    print('u3 conv3', u3.get_shape(), conv3.get_shape())
    u3 = concatenate([u3, conv3])
    c3 = Conv1D( filters=int(16*n), kernel_size=1, strides=1, activation = 'relu')(u3)
    
    drib4 = DRIB(c3,16*n)
    u4 = UpSampling1D(size = 2)(drib4)
    u4 = concatenate([u4, conv2])
    c4 = Conv1D( filters=int(8*n), kernel_size=1, strides=1, activation = 'relu')(u4)
    
    drib5 = DRIB(c4,8*n)
    u5 = UpSampling1D(size = 2)(drib5)
    u5 = concatenate([u5, conv1])
    c5 = Conv1D( filters=int(8*n), kernel_size=1, strides=1, activation = 'relu')(u5)    
    
    #drib5 = DRIB(c7,8*n)
    u10 = UpSampling1D(size = 2)(c5)
    c10 = Conv1D( filters=int(1), kernel_size=1, strides=1, activation = 'relu')(u10) #, activation='sigmoid'

    flat1 = Flatten()(c10)
    dense1 = Dense(1)(flat1)
    
    model = Model(inputs = inputs, outputs = dense1)

    
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def lstm(pretrained_weights = None,input_size = (200,1),windowSize=200):
    inputs = Input(input_size)
    
    lstm1 = tf.keras.layers.LSTM(16,return_sequences=True)(inputs)
    drop1 = Dropout(0.5)(lstm1)
    lstm2 = tf.keras.layers.LSTM(16)(drop1)
    drop2 = Dropout(0.5)(lstm2)
    dense2 = Dense(1)(drop2)
    
    model = Model(inputs = inputs, outputs = dense2)
    
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model