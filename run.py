from keras.datasets import mnist
import numpy as np
from model import CNNmodel

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

print(xtrain.shape, xtest.shape)

m = CNNmodel()
m.train(xtrain, ytrain)




