from keras.models import load_model
import numpy as np
import os
import time
import tensorflow as tf
from test_tumor import x as x2
from test_normal import x as x1
x1 = np.asarray(x1, dtype=np.float32)
x2 = np.asarray(x2, dtype=np.float32)
print(x1.shape, x2.shape)

model_path = 'weight.hdf5'
m = load_model(model_path, compile=False)
x = np.ones((1, 12634), dtype='float32')
x = np.array(x, dtype='float32')
lables = m.predict(x2) <= 0.5

print(lables)