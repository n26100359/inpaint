from keras.datasets import mnist, cifar10
from keras.models import load_model
import keras
import numpy as np
import matplotlib.pyplot as plt

def load_mnist():
    (rawtrain, ytrain), (rawtest, ytest) = mnist.load_data()
    Ntrain = rawtrain.shape[0]
    Ntest = rawtest.shape[0]

    # zero-pad 28*28 img into 32*32 for convenience
    # and normalize
    rawtrain = np.pad(rawtrain, ((0,0),(2,2),(2,2)), 'constant') / 256.
    rawtest = np.pad(rawtest, ((0,0),(2,2),(2,2)), 'constant') / 256.

    #x = left half, y = right half
    xtrain = rawtrain[:, :, :16, np.newaxis]
    ytrain = rawtrain[:, :, 16:, np.newaxis]

    xtest = rawtest[:, :, :16, np.newaxis]
    ytest = rawtest[:, :, 16:, np.newaxis]

    return xtrain, xtest, ytrain, ytest

def load_cifar10():
    (xtrainRaw, _), (xtestRaw, _) = cifar10.load_data()
    
    xtrain = xtrainRaw[:, :, :16, :]/256
    ytrain = xtrainRaw[:, :, 16:, :]/256
    xtest = xtestRaw[:, :, :16, :]/256
    ytest = xtestRaw[:, :, 16:, :]/256
    
    return xtrain, xtest, ytrain, ytest

def plotter(x, y=None, model_dirs=[], rgb=False, cmap=None, figsize=(10,10), title=True):
    imgs = dict()
    for model_dir in model_dirs:
        if model_dir == 'gt':
            ypred = y
        else:
            model = load_model(model_dir)
            ypred = model.predict(x)
        img_pred = np.concatenate([x, ypred], axis=2)
        imgs[model_dir] = img_pred
        print(f'"{model_dir}" prediction done.')
    
    fig, axs = plt.subplots(len(x), len(imgs), figsize=(15,15))
    fig.tight_layout()

    for i in range(len(x)):
        for j, m in enumerate(model_dirs):
            if title and i == 0:
                axs[i, j].set_title(m)
            if rgb:
                axs[i, j].imshow(imgs[m][i,:,:,:], cmap=cmap)
            else:
                axs[i, j].imshow(imgs[m][i,:,:,0], cmap=cmap)
    fig.show()