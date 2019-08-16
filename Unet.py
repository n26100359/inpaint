from keras.models import Model, load_model
from keras.layers import Input, Dense, Activation, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Flatten, Reshape, concatenate
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.datasets import mnist
import numpy as np

def build_Unet(input_size):
    inputs = Input(input_size) #32*16*1
    conv1 = Conv2D(10, kernel_size=(3,3), padding='same', activation='relu')(inputs)
    pool1 = MaxPooling2D((2,2))(conv1) #16*8*10
    conv2 = Conv2D(20, kernel_size=(3,3), padding='same', activation='relu')(pool1) 
    pool2 = MaxPooling2D((2,2))(conv2) #8*4*20
    conv3 = Conv2D(30, kernel_size=(3,3), padding='same', activation='relu')(pool2)
    
    flat = Flatten()(conv3) #8*4*30
    dense1 = Dense(30, activation='relu')(flat)
    dense2 = Dense(30, activation='relu')(dense1)
    dense3 = Dense(8*4*30, activation='relu')(dense2)
    reshape = Reshape((8,4,30))(dense3)
    
    cat1 = concatenate([reshape, conv3], axis=3)
    tpconv1 = Conv2DTranspose(30, kernel_size=(3,3), padding='same', activation='relu')(cat1)
    
    up2 = UpSampling2D((2,2))(tpconv1)
    cat2 = concatenate([up2, conv2], axis=3)
    tpconv2 = Conv2DTranspose(20, kernel_size=(3,3), padding='same', activation='relu')(cat2)
    
    up3 = UpSampling2D((2,2))(tpconv2)
    cat3 = concatenate([up3, conv1], axis=3)
    tpconv3 = Conv2DTranspose(10, kernel_size=(3,3), padding='same', activation='relu')(cat3)
    
    out = Conv2DTranspose(1, kernel_size=(3,3), padding='same', activation='sigmoid')(tpconv3)
    
    model = Model(input=inputs, output=out)
    model.compile(loss='mse', optimizer=Adam())
    model.summary()
    
    return model

def train(m, xtrain, ytrain):
    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    checkpoint = ModelCheckpoint('./models/unet-{epoch:02d}.hdf5', monitor='val_loss', save_best_only=False, period=5, verbose=1)
    tensorboard = TensorBoard(log_dir="./logs")
    history = m.fit(
        x=xtrain,
        y=ytrain,
        validation_split=0.2, shuffle=True,
        epochs=300,
        batch_size=500,
        verbose=1,
        callbacks=[checkpoint, tensorboard])
    return history
    

if __name__ == '__main__':

    (rawtrain, ytrain), (rawtest, ytest) = mnist.load_data()
    Ntrain = rawtrain.shape[0]
    Ntest = rawtest.shape[0]
    print(rawtrain.shape, rawtest.shape)

    # zero-pad 28*28 img into 32*32 for convenience
    # and normalize
    rawtrain = np.pad(rawtrain, ((0,0),(2,2),(2,2)), 'constant') / 256.
    rawtest = np.pad(rawtest, ((0,0),(2,2),(2,2)), 'constant') / 256.
    print('after padding: ', rawtrain.shape, rawtest.shape)
    
    #x = left half, y = right half
    xtrain = rawtrain[:, :, :16, np.newaxis]
    ytrain = rawtrain[:, :, 16:, np.newaxis]
    xtest = rawtest[:, :, :16, np.newaxis]
    ytest = rawtest[:, :, 16:, np.newaxis]
    print(xtrain.shape, xtest.shape)
    
    m = build_Unet(input_size=(32, 16, 1))
    train(m, xtrain, ytrain)
    
    
    
    
    