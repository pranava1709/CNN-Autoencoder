import cv2
from sklearn.model_selection import train_test_split
import glob2
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
import numpy as np
from keras.models import Model
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import ast 


lsttr = []
lstv = []
sphererp = "/home/tiet1/Downloads/dataset/train/sphere1"
vortrp = "/home/tiet1/Downloads/dataset/train/vort1"
norp=  "/home/tiet1/Downloads/dataset/train/no1"
sphere = glob2.glob(sphererp+'/*npy')
vort  = glob2.glob(vortrp+'/*npy')
no = glob2.glob(norp + '/*npy')

labdic = ['SPHERE','VORT','NOSTRUCTURE']
labdf = pd.DataFrame(labdic)
en = LabelEncoder()
tr_labels = en.fit_transform(labdf)
print(tr_labels)

lstlab = []
lstX = []
lstY = []
lstsphere = [(j) for i,j in enumerate(sphere)]
print(lstsphere)

lstvort = [(jj) for ii,jj in enumerate(vort)]
lstno = [(jjj) for iii,jjj in enumerate(no)]
labsp = ['0']*len(lstsphere)
print(labsp)
labvo = ['1']*len(lstvort)
labnp = ['2']*len(lstno)
df = (pd.DataFrame([labsp,lstsphere])).T
df1 =(pd.DataFrame([labvo,lstvort])).T
df2  =(pd.DataFrame([labnp,lstno])).T
combined_df = pd.concat([df,df1,df2],axis = 0)

combined_df.columns = ['labels','Pixel_Values']




		
x = combined_df.iloc[1:,1:2].values
y = combined_df.iloc[1:,0:1].values




for aa in combined_df.iloc[1:,1:2].values:
	aa = str(aa).replace('[','')
	aa = str(aa).replace(']','')
	aa = str(aa).replace("'"," ")
	aa = aa.lstrip()
	aa = aa.rstrip()
	aa = aa.strip()
	aa= np.load(aa)

	lstX.append(aa)
	lstX1 = np.array(lstX)
	print(lstX1.shape)
	lstX1 = lstX1.tolist()
print(len(lstX1))
for bb in combined_df.iloc[1:,0:1].values:
	bb = str(bb).replace('[','')
	bb= str(bb).replace(']','')
	bb = str(bb).replace("'"," ")
	bb = bb.lstrip()
	bb = bb.rstrip()
	bb = bb.strip()


	lstY.append(int(bb))
	lstY1 = np.array(lstY)
	lstY1 = lstY1.tolist()

print(len(lstY1))


dat = np.load("/home/tiet1/Downloads/dataset/train/sphere/1.npy")
print(np.size(dat))

dat1  = np.expand_dims(dat,axis = 1)


inp = keras.Input(shape = dat.shape)
print(inp.shape)

conv1 = layers.Conv2D(150,1,activation  = 'relu')(inp)
bn = layers.BatchNormalization()(conv1)
conv2 = layers.Conv2D(128,1,activation  = 'relu')(bn)
bn1 =layers.BatchNormalization()(conv2)
conv3  = layers.Conv2D(64,1,activation  = 'relu')(bn1)
bn2 =layers.BatchNormalization()(conv3)
conv4  = layers.Conv2D(32,1,activation  = 'relu')(bn2)
bn3 =layers.BatchNormalization()(conv4)
code = layers.Conv2D(16,1,activation  = 'relu')(bn3)

deconv1 = layers.Conv2D(32,1,activation  = 'relu')(code)
bn4 = layers.BatchNormalization()(deconv1)

deconv2 = layers.Conv2D(64,1,activation  = 'relu')(bn4)
bn5 = layers.BatchNormalization()(deconv2)

deconv3= layers.Conv2D(128,1,activation  = 'relu')(bn5)
bn6 = layers.BatchNormalization()(deconv3)


deconv4 = layers.Conv2D(150,1,activation  = 'relu')(bn6)


autoencoder = Model(inp, deconv4)
autoencoder.compile(metrics = ['accuracy'],optimizer='adam',  loss=tf.keras.losses.CategoricalCrossentropy())
autoencoder.fit(lstX1, lstX1, epochs=100)


autoencoder.save("autoenc334.h5")
autoencoder.save("autoenc334.pkl")
