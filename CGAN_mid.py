import tensorflow as tf
import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Reshape, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, UpSampling2D, BatchNormalization, concatenate, Lambda
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import keras.backend as K
import util
import os
import csv
import math
class CGANv2:
    def __init__(self, img_shape, lr=0.0005, load_G=None, load_D=None):

        self.img_shape = img_shape
        self.left_shape = (img_shape[0], img_shape[1], img_shape[2])
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
#       D_input = concatenate([G_input, G_output], axis=2, name='right_img_pred')
        G_out = tf.unstack(G_input)
        G_out[self.left_shape[1]*0.25:self.left_shape[1]*0.75, self.left_shape[2]*0.25:self.left_shape[2]*0.75] = G_output
        D_input = tf.stack(G_out)
        self.D.trainable = False
        D_output = self.D(D_input)

        self.GAN = Model(G_input, D_output)
        self.GAN.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def build_D(self, input_shape): # input whole image, output 0~1
        inputs = Input(input_shape)

#shrink
#        conv01 = Conv2D(16, kernel_size=(3,3), padding='same', activation='relu')(inputs)
#        conv02 = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu')(conv01)
#        conv0 = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(conv01)
#        pool0 = MaxPooling2D((3,3))(conv0)
#        bn0 = BatchNormalization()(pool0)
#shrink
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

        out = Dense(1, activation='sigmoid')(x)#binary_crossentropy

        model = Model(inputs, out, name='Discriminator')
        model.summary()
        return model

    def build_G(self, input_shape): # input left half, output right half
        inputs = Input(input_shape) #32*16*1
#shrink
#        conv01 = Conv2D(16, kernel_size=(3,3), padding='same', activation='relu')(inputs)
#        conv02 = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu')(conv01)
#        conv0 = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(conv01)
#        pool0 = MaxPooling2D((3,3))(conv0)
#        bn0 = BatchNormalization()(pool0)
#shrink
        conv1 = Conv2D(16, kernel_size=(3,3), padding='same', activation='relu')(inputs)
        pool1 = MaxPooling2D((2,2))(conv1) #48*48*10
        bn1 = BatchNormalization()(pool1)

        conv2 = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu')(bn1) 
        pool2 = MaxPooling2D((2,2))(conv2) #24*24*20
        bn2 = BatchNormalization()(pool2)

        conv3 = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(bn2)

        flat = Flatten()(conv3) #24*24*30
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
        x = Dense(24*24*30, activation='relu')(x)#origin 8*4*30 to 2*1*64
        reshape = Reshape((24,24,30))(x)#origin 8*4*30 to 2*1*64

        tpconv1 = Conv2DTranspose(64, kernel_size=(3,3), padding='same', activation='relu')(reshape)

        up2 = UpSampling2D((2,2))(tpconv1)
        tpconv2 = Conv2DTranspose(32, kernel_size=(3,3), padding='same', activation='relu')(up2)
        bn6 = BatchNormalization()(tpconv2)

        tpconv3 = Conv2DTranspose(16, kernel_size=(3,3), padding='same', activation='relu')(bn6)
        bn7 = BatchNormalization()(tpconv3)

#expand
#        up_last = UpSampling2D((3,3))(bn7)
#        cat_last = concatenate([up_last, conv0], axis=3)
#        tpconv_last = Conv2DTranspose(16, kernel_size=(3,3), padding='same', activation='relu')(cat_last)
#        bn_last = BatchNormalization()(tpconv_last)
#expand

        # use tanh as G's output, IDK why
        out = Conv2DTranspose(self.img_shape[2], kernel_size=(3,3), padding='same', activation='tanh')(bn7)
        out_01 = Lambda(lambda x: x * .5 + .5)(out) # [-1:1] to [0:1]

        model = Model(inputs, out_01, name='Generator')
#        model.load_weights('models/GAN_cifar_v2/GAN_batch200_diter10_giter1_fake2True/G_50000.hdf5')

        model.summary()
        return model

    def train(self, tr_left, tr_right, batch_size=200, epochs=10000, start_epoch=0, train_D_iter=1, train_G_iter=1, save_model_iter=100, save_path='./models/GAN/', use_fake2=True):

        assert tr_left.shape[0] == tr_right.shape[0]

        dir_name = f'GAN_batch{batch_size}_diter{train_D_iter}_giter{train_G_iter}_fake2{use_fake2}'
        dir_path = os.path.join(save_path, dir_name)
        G_loss_save = []
        D_loss_save = []
        G_loss_plot = []
        D_loss_plot = []
        for k in range(100):
            G_loss_save.append(99)
            D_loss_save.append(99)
        G_down_count = 0
        D_down_count = 0
        try:
            os.mkdir(dir_path)
        except:
            print("Directory", dir_path, "exists.")

        for epoch in range(start_epoch, start_epoch + epochs):
            print(f'Epoch {epoch}')
            lr_down = 0
            # ---
            # train G
            # ---
            if epoch % train_G_iter == 0:
                batch_i = np.random.randint(tr_left.shape[0], size=batch_size)
                y = np.ones((batch_size, 1))
                print("G_lr=",K.eval(self.GAN.optimizer.lr))
                G_lr_base = K.eval(self.GAN.optimizer.lr)
                G_loss = self.GAN.train_on_batch(tr_left[batch_i], y)

#                if (epoch // train_G_iter) % 100 == 0:
#                    for i in range(100):
#                        if G_loss >= G_loss_save[i]:
#                            lr_down = lr_down + 1
#                    if lr_down == 100:
#                        if G_down_count<3:
#                         K.set_value(self.GAN.optimizer.lr, G_lr_base/10)
#                         G_down_count = G_down_count + 1
#cycliccal
#                if epoch <= 16000:#16/25
#                    G_lr_base = 0.01
#                    G_lr_max = 0.05
#                    G_stepsize = 400
#                elif epoch > 16000 and epoch <= 22000:
#                    G_lr_base = 0.001
#                    G_lr_max = 0.005
#                    G_stepsize = 200
#                elif epoch > 22000:#22/25
#                    G_lr_base = 0.0001
#                    G_lr_max = 0.0005
#                    G_stepsize = 100
#                cycle = math.floor(1 + epoch / (2 * G_stepsize))
#                x = abs(epoch / G_stepsize - 2 * cycle + 1)
#                G_newlr = G_lr_base + (G_lr_max - G_lr_base) * max(0, (1 - x))
#                print('G_lr: %f' % G_newlr)
#                K.set_value(self.GAN.optimizer.lr, G_newlr)
#cycliccal end
                for j in range(100):
                    if (epoch // train_G_iter) % 100 == j:
                        G_loss_save[j] = G_loss

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
                print("D_lr=",K.eval(self.D.optimizer.lr))
                D_lr_base = K.eval(self.D.optimizer.lr)
                D_loss = self.D.train_on_batch(x, y)

#                if (epoch % train_D_iter)% 100 == 0:
#                    for i in range(100):
#                        if D_loss >= D_loss_save[i]:
#                            lr_down = lr_down + 1
#                    if lr_down == 100:
#                        if D_down_count<3:
#                         K.set_value(self.D.optimizer.lr, D_lr_base/10)
#                         D_down_count = D_down_count + 1
#cycliccal 
#                if epoch <= 16000:
#                    D_lr_base = 0.01
#                    D_lr_max = 0.05
#                    D_stepsize = 400
#                elif epoch > 16000 and epoch <= 22000:
#                    D_lr_base = 0.001
#                    D_lr_max = 0.005
#                    D_stepsize = 200
#                elif epoch > 22000:
#                    D_lr_base = 0.0001
#                    D_lr_max = 0.0005
#                    D_stepsize = 100
#                cycle = math.floor(1 + epoch / (2 * D_stepsize))
#                x = abs(epoch / D_stepsize - 2 * cycle + 1)
#                D_newlr = D_lr_base + (D_lr_max - D_lr_base) * max(0, (1 - x))
#                print('D_lr: %f' % D_newlr)
#                K.set_value(self.D.optimizer.lr, D_newlr)
#cycliccal end
                for j in range(100):
                    if (epoch // train_D_iter) % 100 == j:
                        D_loss_save[j] = D_loss

            print(f'D loss: {D_loss}, G loss: {G_loss}')
#做圖                
            G_loss_plot.append(G_loss)
            D_loss_plot.append(D_loss)

          
#            write_log(callback, G_names, G_loss_plot, epoch)
#            write_log(callback, D_names, D_loss_plot, epoch)
            
            if (epoch + 1) % save_model_iter == 0:
                self.G.save(os.path.join(dir_path, f'G_{epoch+1}.hdf5'))
                self.D.save(os.path.join(dir_path, f'D_{epoch+1}.hdf5'))
                G_loss_plot_array=np.array(G_loss_plot)
                D_loss_plot_array=np.array(D_loss_plot)
                np.save('G_loss.npy',G_loss_plot_array)
                np.save('D_loss.npy',D_loss_plot_array)
                print('model saved.')
                
    def write_graph(self, path):
        graph = K.get_session().graph
        writer = tf.compat.v1.summary.FileWriter(logdir=path, graph=graph)


if __name__ == '__main__':
#    xtrain, xtest, ytrain, ytest = util.load_cifar10()
#     cgan = CGAN((32,32,1))
#     cgan.write_graph('./logs/cGAN/')
#     cgan.train(xtrain, ytrain, batch_size=200, epochs=40000, train_D_iter=5, save_model_iter=200)
    xtrain = np.load('./xtrain96_mid.npy',allow_pickle=True) / 255
    ytrain = np.load('./ytrain96_mid.npy',allow_pickle=True) / 255
    xtest = np.load('./xtest96_mid.npy',allow_pickle=True) / 255
    ytest = np.load('./ytest96_mid.npy',allow_pickle=True) / 255
    print(np.shape(xtrain), np.shape(xtest))
    cgan = CGANv2((96,96,3), lr=0.0001)
    cgan.train(xtrain, ytrain,
               batch_size=200,
               epochs=50000,
               train_D_iter=10,
               train_G_iter=1,
               save_model_iter=1000,
               use_fake2=True,
               save_path='./models/GAN_art96_v3/')
