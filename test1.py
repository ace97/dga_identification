from keras.models import model_from_json
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics

seed =7
np.random.seed(seed)
df=pd.read_csv("test.csv")
del df['index']

df=df.apply(preprocessing.LabelEncoder().fit_transform)
array=df.values

#print dataframe.head()

X_test=array[:,:-1]
y_test=array[:,-1]

#create training and testing
#X_train,X_test,y_train,y_test=preprocessing.train_test_split(X,Y,test_size=0.2)

# load json and create model
json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model1.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X_test,y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))