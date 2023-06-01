from keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
model = load_model('autoenc334.h5',compile = False)
dat = np.load("C:/dataset/train/vort1/1.npy")
dat = np.expand_dims(dat, axis = 1)
res = model.predict(dat)
print(res)
plt.imshow(dat[0,0,:,:])
plt.show()