import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#reading the csv file
data = pd.read_csv("musk_csv.csv")
y = data['class'].values
y=y.reshape(-1,1)
data.drop(['ID','molecule_name','conformation_name','class'],axis=1, inplace = True )
print(y)
# to extrct out the columns whose correlation is greater thn 0.9216
corr_matrix = data.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
_drop = [column for column in upper.columns if any(upper[column] > 0.9216)]
data=data.drop(_drop,axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.2, random_state = 20)
print(X_train.shape,X_test.shape,y_test.shape,y_train.shape)

#to reshape the train and testing part in acc to the required input layer

x_train=X_train.values.reshape(X_train.shape[0],19,6,1)
x_test=X_test.values.reshape(X_test.shape[0],19,6,1)


print(x_test.shape)
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


# to make layers for nural network mixture if CNN and ANN
mod=Sequential()
mod.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(19,6,1)))
mod.add(Conv2D(64,(3,3),activation='relu'))
mod.add(MaxPooling2D(pool_size=(2,2)))
mod.add(Dropout(0.25))
mod.add(Flatten())
mod.add(Dense(128,activation='relu'))
mod.add(Dropout(0.5))
mod.add(Dense(1,activation='sigmoid'))

#compiling and forming the model
mod.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
model = mod.fit(x_train,y_train,batch_size=128,epochs=10,validation_data=(x_test,y_test))
score=mod.evaluate(x_test,y_test,verbose=0)

# summarize mainmodel for accuracy
plt.plot(model.history['acc'])
plt.plot(model.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig=plt.figure()
fig.savefig('model accuracy')
# summarize mainmodel for loss
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
fig=plt.figure()
fig.savefig('model loss')
import sklearn
print("f1_score:",sklearn.metrics.f1_score(y_test,mod.predict_classes(x_test)))
print("recall:",sklearn.metrics.recall_score(y_test,mod.predict_classes(x_test)))
print("precision:",sklearn.metrics.average_precision_score(y_test,mod.predict_classes(x_test)))
print("Validation Loss:",score[0])
print("Validation Accuracy:",score[1])


mod.save('mod2shubhdeep.h5')
