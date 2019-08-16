from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

class CNNmodel():
    
    def __init__(self):
        
        m = Sequential()
        
        #conv
        m.add(Conv2D(10, input_shape=(32,16,1), kernel_size=(3,3), padding='same', activation='relu')) #32*16*10
        m.add(MaxPooling2D((2,2))) #16*8*10
        m.add(Conv2D(20, kernel_size=(3,3), padding='same', activation='relu')) #16*8*20
        m.add(MaxPooling2D((2,2))) #8*4*20
        m.add(Conv2D(30, kernel_size=(3,3), padding='same', activation='relu')) #8*4*30
        
        #FC layers to learn latent feature
        m.add(Flatten())
        m.add(Dense(30, activation='relu'))
        m.add(Dense(60, activation='relu'))
        m.add(Dense(8*4*30, activation='relu'))
        m.add(Reshape((8,4,30))) #8*4*30
        
        #transpose conv
        m.add(Conv2DTranspose(30, kernel_size=(3,3), padding='same', activation='relu')) #8*4*30
        m.add(UpSampling2D((2,2))) #16*8*30
        m.add(Conv2DTranspose(20, kernel_size=(3,3), padding='same', activation='relu')) #16*8*20
        m.add(UpSampling2D((2,2))) #32*16*20
        m.add(Conv2DTranspose(10, kernel_size=(3,3), padding='same', activation='relu')) #32*16*10
        
        #output
        m.add(Conv2DTranspose(1, kernel_size=(3,3), padding='same', activation='sigmoid')) #32*16*1
        
        self.model = m
        
        
    def train(self, xtrain, ytrain):
        #callbacks
        earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        checkpoint = ModelCheckpoint('./models/CNNv2-{epoch:02d}.hdf5', monitor='val_loss', save_best_only=False, period=5, verbose=1)
        tensorboard = TensorBoard(log_dir="./logs")
        
        self.model.compile(loss='mse', optimizer=Adam())
        self.model.summary()
        self.history = self.model.fit(x=xtrain, y=ytrain, validation_split=0.2, shuffle=True, 
                        epochs=300, batch_size=500, verbose=1, callbacks=[checkpoint, tensorboard])
    





