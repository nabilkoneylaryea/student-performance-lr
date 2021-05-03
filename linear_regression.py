import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv('./student-mat.csv', sep=';') # each row is separated by semicolons

attributes = ['G1', 'G2', 'G3', 'studytime', 'failures', 'absences'] # select attributes

data = data[attributes] # select attributes

labels = 'G3' # column we're predicting a.k.a. labels for data

# NEW STUFF
x = np.array(data.drop([labels], 1)) # create an array of inputs
y = np.array(data[labels]) # create array of labels/outputs
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1) # split further into training and testing data
ITERATIONS = 10
best = 0

for iteration in range(ITERATIONS):
    print(f'### ITERATION: {iteration + 1} ###')
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1) # ?? WHY DOUBLED

    linear = linear_model.LinearRegression() # initialize linear regression model from sklearn
    linear.fit(x_train, y_train) # train linear regression model
    accuracy = linear.score(x_test, y_test) # test linear regression model
    print('Accuracy:', accuracy)
    
    if accuracy > best:
        best = accuracy
        with open('student_model.pickle', 'wb') as f:
            pickle.dump(linear, f) # saving model by pickling it
        
pickle_in = open('student_model.pickle', 'rb')
linear = pickle.load(pickle_in)

print('Accuracy:', accuracy)
print('Coefficients:', linear.coef_) # slope values: larger coefficients mean larger impact on result
print('Intercept:', linear.intercept_) # y-intercept

predictions = linear.predict(x_test)
for index, prediction in enumerate(predictions):
    print(prediction, x_test[index], y_test[index])