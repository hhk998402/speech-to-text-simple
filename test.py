from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import model_from_json
import os
import json
from decimal import Decimal

'''Second dimension of the feature is dim2'''
feature_dim_2 = 11

X_train, X_test, y_train, y_test = get_train_test()

'''Feature dimension'''
feature_dim_1 = 20
channel = 1

labels = ['11_ConferenceRoom', '10_ConferenceRoom', '3_Washroom', '2_Washroom', '13_Boss', '6_Reception', '12_ConferenceRoom', '14_Boss', '15_Boss', '7_Water', '1_Washroom', '5_Reception', '9_Water', '4_Reception', '8_Water']
label_index = []


testing_output = []

'''Predicts one sample'''
def predict(filepath,each_test,expected,model):
    sample = wav2mfcc(filepath)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    L1 = model.predict(sample_reshaped)
    each_test["predictions"] = {}
    L1 = L1.tolist()
    L1 = L1[0]
    print(L1)
    max1 = -1
    each_test["expected_output"] = labels[expected]
    each_test["expected_output_acc"] = str(L1[expected]*100)+'%'
    for i in range(len(labels)):
        if(max1<L1[i]):
            each_test["predicted_output"] = labels[i]
            each_test["predicted_output_acc"] = str(L1[i]*100)+'%'
            max1 = L1[i]
        each_test["predictions"][labels[i]] = str(L1[i]*100)+'%'
    testing_output.append(each_test)
    print(testing_output)
    return get_labels()[0][
            np.argmax(model.predict(sample_reshaped))
    ]

model = load_model("speech_model.h5")
#model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot))
for j in labels:
    s1 = j.split('_')
    label_index.append(int(s1[0]))

L = os.listdir('./test')
for i in L:
    each_test={}
    each_test["filename"] = i
    if(i=='.DS_Store'):
        continue
    s = i.split('_')
    expected = label_index.index(int(s[0]))
    i = "./test/" + i
    predict(i,each_test,expected,model=model)

output = {}
output["testing_outputs"] = testing_output

json = json.dumps(output)
print(json)
f = open("output.json","w")
f.write(json)
f.close()
