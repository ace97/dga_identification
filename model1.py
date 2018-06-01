from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
import numpy as np

maxlen=133926
max_features=2
model = Sequential()


seed =7
np.random.seed(seed)
df=pd.read_csv("mix.csv")
del df['index']

df=df.apply(preprocessing.LabelEncoder().fit_transform)
array=df.values


X_train=array[:,:-1]
y_train=array[:,-1]

#preprocessing
#valid_chars={x:idx+1 for idx,x in enumerate(set(''.join(X)))}
#X=preprocessing.sequence.pad_sequences(X,maxlen=maxlen)

#create training and testing
#X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)

#building model


model.add(Embedding(maxlen, 128))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X_train,y_train,epochs=1,batch_size=10)



scores=model.evaluate(X_train,y_train)

print("%s:%.2f%%"%(model.metrics_names[1],scores[1]*100))

#new
model_json = model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model1.h5")
print("Saved model to disk")