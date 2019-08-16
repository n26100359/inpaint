import tensorflow as tf
import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Reshape, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, UpSampling2D, BatchNormalization, concatenate, Lambda
from keras.optimizers import Adam
import keras.backend as K
import util
import os

class CGANv2:
    
    def __init__(self, img_shape, lr=0.0005, load_G=None, load_D=None):
        self.img_shape = img_shape
        self.left_shape = (img_shape[0], int(img_shape[1]*0.5), img_shape[2])
        self.optimizer = Adam(lr)
        
        if load_G and load_D:
            self.G = load_model(load_G)
            self.D = load_model(load_D)
        else:
            self.G = self.build_G(input_shape=self.left_shape)
            self.D = self.build_D(input_shape=self.img_shape)
            
        self.build_GAN(self.G, self.D)
        
    def build_GAN(self, G, D):
        self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        
        G_input = Input(self.left_shape, name='left_img')
        G_output = self.G(G_input)
        D_input = concatenate([G_input, G_output], axis=2, name='right_img_pred')
        self.D.trainable = False
        D_output = self.D(D_input)
        
        self.GAN = Model(G_input, D_output)
        self.GAN.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        
    def build_D(self, input_shape): # input whole image, output 0~1
        inputs = Input(input_shape)
        
        conv1 = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
        pool1 = MaxPooling2D((2,2))(conv1)
        bn1 = BatchNormalization()(pool1)
        
        conv2 = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(bn1)
        pool2 = MaxPooling2D((2,2))(conv2)
        bn2 = BatchNormalization()(pool2)
        
        conv3 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(bn2)
        flat = Flatten()(conv3)
        
        x = Dropout(.4)(flat)
        x = Dense(64, activation='relu')(x)
        
        x = Dropout(.4)(x)
        x = Dense(32, activation='relu')(x)
        
        x = Dropout(.4)(x)
        x = Dense(16, activation='relu')(x)
        
        x = Dense(8, activation='relu')(x)
        
        out = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, out, name='Discriminator')
        return model
        
    def build_G(self, input_shape): # input left half, output right half
        inputs = Input(input_shape) #32*16*1
        conv1 = Conv2D(16, kernel_size=(3,3), padding='same', activation='relu')(inputs)
        pool1 = MaxPooling2D((2,2))(conv1) #16*8*10
        bn1 = BatchNormalization()(pool1)
        
        conv2 = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu')(bn1) 
        pool2 = MaxPooling2D((2,2))(conv2) #8*4*20
        bn2 = BatchNormalization()(pool2)
      
        conv3 = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(bn2)

        flat = Flatten()(conv3) #8*4*30
        x = BatchNormalization()(flat)
        
        x = Dropout(.4)(x)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        
        x = Dropout(.4)(x)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        
        x = Dropout(.4)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        
        x = Dropout(.4)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        
        x = Dropout(.4)(x)
        x = Dense(8*4*30, activation='relu')(x)
        reshape = Reshape((8,4,30))(x)

        cat1 = concatenate([reshape, conv3], axis=3)
        tpconv1 = Conv2DTranspose(64, kernel_size=(3,3), padding='same', activation='relu')(cat1)

        up2 = UpSampling2D((2,2))(tpconv1)
        cat2 = concatenate([up2, conv2], axis=3)
        tpconv2 = Conv2DTranspose(32, kernel_size=(3,3), padding='same', activation='relu')(cat2)
        bn6 = BatchNormalization()(tpconv2)

        up3 = UpSampling2D((2,2))(bn6)
        cat3 = concatenate([up3, conv1], axis=3)
        tpconv3 = Conv2DTranspose(16, kernel_size=(3,3), padding='same', activation='relu')(cat3)
        bn7 = BatchNormalization()(tpconv3)

        # use tanh as G's output, IDK why
        out = Conv2DTranspose(self.img_shape[2], kernel_size=(3,3), padding='same', activation='tanh')(bn7)
        out_01 = Lambda(lambda x: x * .5 + .5)(out) # [-1:1] to [0:1]

        model = Model(inputs, out_01, name='Generator')
        return model

    def train(self, tr_left, tr_right, batch_size=200, epochs=10000, start_epoch=0, train_D_iter=1, train_G_iter=1, save_model_iter=100, save_path='./models/GAN/', use_fake2=True):
        
        assert tr_left.shape[0] == tr_right.shape[0]
        
        dir_name = f'GAN_batch{batch_size}_diter{train_D_iter}_giter{train_G_iter}_fake2{use_fake2}'
        dir_path = os.path.join(save_path, dir_name)
        try:
            os.mkdir(dir_path)
        except:
            print("Directory", dir_path, "exists.")
        
        for epoch in range(start_epoch, start_epoch + epochs):
            print(f'Epoch {epoch}')

            # ---
            # train G
            # ---
            if epoch % train_G_iter == 0:
                batch_i = np.random.randint(tr_left.shape[0], size=batch_size)
                y = np.ones((batch_size, 1))
                G_loss = self.GAN.train_on_batch(tr_left[batch_i], y)

            # ---
            # train D
            # ---
            if epoch % train_D_iter == 0:
                batch_i = np.random.randint(tr_left.shape[0], size=batch_size)
                batch_i2 = np.random.randint(tr_left.shape[0], size=batch_size)
                
                # real
                real = np.concatenate([
                    tr_left[batch_i],
                    tr_right[batch_i] ], axis=2)

                # match but not realistic
                fake1 = np.concatenate([
                    tr_left[batch_i],
                    self.G.predict(tr_left[batch_i]) ], axis=2)
                
                if use_fake2:
                    # realistic but doesn't match
                    fake2 = np.concatenate([
                        tr_left[batch_i],
                        tr_right[batch_i2] ], axis=2)

                # answers
                real_y = np.ones((batch_size, 1))
                fake_y = np.zeros((batch_size, 1))
                
                if use_fake2:
                    x = np.concatenate([real, fake1, fake2], axis=0)
                    y = np.concatenate([real_y, fake_y, fake_y], axis=0)
                else:
                    x = np.concatenate([real, fake1], axis=0)
                    y = np.concatenate([real_y, fake_y], axis=0)

                D_loss = self.D.train_on_batch(x, y)
            
            print(f'D loss: {D_loss}, G loss: {G_loss}')
            
            if (epoch + 1) % save_model_iter == 0:
                self.G.save(os.path.join(dir_path, f'G_{epoch+1}.hdf5'))
                self.D.save(os.path.join(dir_path, f'D_{epoch+1}.hdf5'))
                print('model saved.')
        
    def write_graph(self, path):
        graph = K.get_session().graph
        writer = tf.compat.v1.summary.FileWriter(logdir=path, graph=graph)


if __name__ == '__main__':
#     xtrain, xtest, ytrain, ytest = util.load_mnist()
#     cgan = CGAN((32,32,1))
#     cgan.write_graph('./logs/cGAN/')
#     cgan.train(xtrain, ytrain, batch_size=200, epochs=40000, train_D_iter=5, save_model_iter=200)

    xtrain, xtest, ytrain, ytest = util.load_cifar10()
    print(xtrain.shape, xtest.shape)
    cgan = CGANv2((32,32,3), lr=0.0001)
    cgan.train(xtrain, ytrain, 
               batch_size=200, 
               epochs=50000, 
               train_D_iter=10, 
               save_model_iter=500, 
               use_fake2=True,
               save_path='./models/GAN_cifar_v2/')
    
    
    
    
    
    
        
