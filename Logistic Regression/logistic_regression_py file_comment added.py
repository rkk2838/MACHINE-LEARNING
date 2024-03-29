# -*- coding: utf-8 -*-
"""Logistic_Regression.ipynb


# **Implementing Logistic regression from scratch without Scikit Learn**

### Importing the Libraries
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

"""### Defining Sigmoid function"""

# sigmoid Function to calculate the probabillity from the linear function that Z  = w.T + b i.e weights transpose +  bias
def sigmoid(z):
  return 1/(1+np.exp(-z))

"""Create a class of Logistic Regression and pass the paramaters line number of iterations , Learing Rate, Weights and Bias.
- Then Two methods will be created called fit() method and predict method().
- Inside the fit method , optimization function called gradient descent will be implemented which will helps us to find the Optimum weights and bias values which will be used to make the prediction.

### Logistic Regression Class
"""

class LogisticRegression:

  def __init__(self,lr=0.001,n_iterations=1000):
    self.lr = lr
    self.n_iterations = n_iterations
    self.weights = None
    self.bias = None

# fit method to train the model
  def fit(self,X,y):
    n_samples,n_features = X.shape   # Total number of features and total number of samples
    self.weights = np.zeros(n_features)  # Intialising the weights with Zero which is equal to number of features because equation is represented as 
                                          # b0 + b1x1 + b2x2 + ........ + bnXn   so, each feature is getting multiplied with weights so, its should be equal
                                          # to number of features.
    self.bias = 0  # Bias will  be a constant so just intializing it with Zero

    # This is for optimizing the the gradient descent function
    for _ in range(self.n_iterations):
      linear_prediction = np.dot(X,self.weights) + self.bias  # Getting the value of this function b0 + b1x1 + b2x2 + ........ + bnXn 
                                                  # written as X into weights here not doing  transpose of anything because let say X represent matrix of all
                                                  # samples so X is let say m X n matrix where m is the number of samples and n is the number of features
                                                  # since weights W is the same as n i.e. number of features and when we take matrix multiplication of [m,n] for X
                                                  # and [n,1] for weights then no transpose is required because no. of columns of X == number of rows of W
      
      prediction = sigmoid(linear_prediction)  # Once we get the linear function we pass it through the sigmoid function which bounds the range of linear_prediction
                                                # from - inf to inf between [0,1] i.e it shows the probability of samples

      # Cost function for logistic regression is  : -[y log(yhat) + (1-y) log(1-yhat)]
      # y : actual values
      # yhat : prediction , i.e.probability after applying sigmoid function , yhat= sigmoid(Z) = sigmoid(W.T * X + b)

      # Now this cost function should be differentiated to wrt to dw and db , so that cost could be minimised
      # after differentiating by applying chain rule

      # dw = X[A-Y]
      # db = [A-Y]
      # Where X : Matrix of all the samples , A : Prediction ,Y: Actual values

      dw = 1/(n_samples) * np.dot(X.T,(prediction - y))  # Here we have done the trasnpose because X is [m,n] and (prediction-y) is [m,1] as prediction is 
                                                          # is calculated for all the samples passed , so matrix multiplication will not take place and so
                                                          # transpose of X is taken then it will become [n,m]
      db = 1/(n_samples) * np.sum(prediction-y)

      
      # Once the derivatives are calculated the we will updated the value of weights and bias
      self.weights  = self.weights - (self.lr * dw)
      self.bias = self.bias - (self.lr * db)

# This is the prediction function that is passed here.
  def predict(self,X):
      linear_prediction = np.dot(X,self.weights) + self.bias
      prediction = sigmoid(linear_prediction)

    # After getting the precition we check if it is less than 0.5 then 0 else 1
      yhat = [0 if i<=0.5 else 1 for i in prediction]
      return yhat

"""### Accuracy"""

# To Check the accuracy of the model
# accuracy = TP + TN/(TP + FP + TN + FN)
def accuracy(y_pred,y_test):
  accuracy_rate = round((np.sum(y_pred==y_test)/len(y_test))*100,4) # Rounded the values around 4
  print(accuracy_rate)

"""### Loading the data from Scikit Learn """

bc = datasets.load_breast_cancer()

X,y = bc.data,bc.target

# 30 numeric features
X[0]

X.shape # There are 569 samples and 30 features

y.shape

# Getting the training and testing data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state =23)

"""### Creating the Logistic Regression class object"""

clf = LogisticRegression() # Object created without passing any values
clf.fit(X_train,y_train)   # fit method is called
ypred = clf.predict(X_test)  # predict method is called and it's return values is collected in the ypred function
accuracy(ypred,y_test)   # then accuracy function is called to print the accuracy

"""### Creating the object by changing the parameters"""

clf2 = LogisticRegression(lr = 0.01,n_iterations=100) # Obejct created with some parametes
clf2.fit(X_train,y_train)
ypred = clf2.predict(X_test)
accuracy(ypred,y_test)

clf3 = LogisticRegression(lr = 0.001,n_iterations=100)
clf3.fit(X_train,y_train)
ypred = clf3.predict(X_test)
accuracy(ypred,y_test)

clf4 = LogisticRegression(lr = 0.001,n_iterations=100000)
clf4.fit(X_train,y_train)
ypred = clf4.predict(X_test)
accuracy(ypred,y_test)

