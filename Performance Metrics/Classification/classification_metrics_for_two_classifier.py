# -*- coding: utf-8 -*-
"""Classification Metrics for two classifier.ipynb


# Classification Metrics when we have to compare the two or more classifier
"""

# Standard Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Collection Libraries
from collections import Counter

# Preprocessing Libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Algorithm Libraries
from sklearn.linear_model import LogisticRegression

# Metrics Libraries
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
from sklearn.metrics import precision_recall_curve,plot_confusion_matrix,roc_auc_score

# Create a dataset of 10000 dataset
X,y = make_classification(n_samples=10000,random_state=32,weights=[.70])
print(X.shape,y.shape)
print(Counter(y))

# Splitting the dataset
Xtrain , Xtest, ytrain,ytest = train_test_split(X,y,test_size = .2 , random_state=34,stratify=y)
print(Xtrain.shape , Xtest.shape, ytrain.shape,ytest.shape)



"""## Model Training

### Logistic Regression
"""

# importing the logistic Regression
clf1 = LogisticRegression()
# fitting the model
clf1.fit(Xtrain,ytrain)
# getting the predict proba
ypred_proba1 = clf1.predict_proba(Xtest)
# predicting the result
ypred1 = clf1.predict(Xtest)

print(ypred1)
print(ypred_proba1[:,1])

# ypred11 represent that it the prediction of the logistic regression when threshold is set different than the default here it is 0.7
ypred11 = np.array([0 if i<=0.7 else 1 for i in ypred_proba1[:,1]])
print(ypred11)

"""### KNN"""

from sklearn.neighbors import KNeighborsClassifier

# training the KNN model
clf2 = KNeighborsClassifier()
# fitting the model
clf2.fit(Xtrain,ytrain)
# getting the probability
ypred_proba2 = clf2.predict_proba(Xtest)
# Predicting the result
ypred2 = clf2.predict(Xtest)

print(ypred2)
print(ypred_proba2[:,1])

ypred22 = np.array([0 if i<=0.7 else 1 for i in ypred_proba2[:,1]])
print(ypred22)



"""### Performance Measurement

**Now we have Xtrain,ytrain,Xtest,ytest <br>
From logistic regression we have : ypred1,ypred11<br>
From KNN we have : ypred2,ypred22**

**Confusion Matrix**
"""

print("----LR----")
print(confusion_matrix(ytest,ypred1))
print("----KNN----")
print(confusion_matrix(ytest,ypred2))

"""**Plotting the Confusion matrix**"""

fig,ax = plt.subplots(1,2,figsize=(10,5))
plt.suptitle("Confusion Matrix") # setting Super Title

sns.heatmap(confusion_matrix(ytest,ypred1),annot=True,ax=ax[0],fmt='g')

sns.heatmap(confusion_matrix(ytest,ypred2),annot=True,ax=ax[1],fmt='g')

# Setting title for each subplots
ax[0].set_title("LR")
ax[1].set_title("KNN")

# setting labels for each subplots
ax[0].set_ylabel("actual values")
ax[0].set_xlabel("Predicted Values")
ax[1].set_ylabel("actual values")
ax[1].set_xlabel("Predicted Values")

# Plotting the Confusion matrix using the function
 plot_confusion_matrix(clf1,Xtrain,ytrain) # For training data

# With Test Data
plot_confusion_matrix(clf1,Xtest,ytest)
plt.title("confusion Matrix")

"""**Accuracy** """

def accuracy(actual,predicted,comment="comment"):
  print("Accuracy Score for ",comment,"is","  :  ",   accuracy_score(actual,predicted))

accuracy(ytest,ypred1,comment="Logistic Regression with default threshold")
accuracy(ytest,ypred11,comment="Logistic Regression with custom threshold")
print()
accuracy(ytest,ypred2,comment="KNN with default threshold")
accuracy(ytest,ypred22,comment="KNN with custom threshold")

"""**Precision**

True Postive / Predicted Positive = TP/(TP + FP)
"""

def precision(actual,predicted):
  print(precision_score(actual,predicted))

print("-----Logistic Regression------")
precision(ytest,ypred1)
precision(ytest,ypred11)
print('----For KNN------')
precision(ytest,ypred2)
precision(ytest,ypred22)

"""**Recall**

True Positive / Actual Postive = TP/TP+FN
"""

def recall(actual,predicted):
  print(recall_score(actual,predicted))

print("-----Logistic Regression------")
recall(ytest,ypred1)
recall(ytest,ypred11)
print('----For KNN------')
recall(ytest,ypred2)
recall(ytest,ypred22)

"""As we can see that with custom threshold:<br>
Logistic Regression:<br>
precision = 0.9309623430962343 and Recall = 0.7367549668874173<br>
KNN:<br>
precision = 0.9464285714285714 and Recall = 0.5264900662251656<br>
Here the Precision is higher in both cases but Recall have dropped particulary in case of KNN. i.e. Model is able to reduce the number of False Positve(FP) case but unable to reduce the number of False Negative(FN) cases.<br><br>
Now, the confusion is that how to measure which classifier is better than other becasue we can not use the accuracy because dataset is imbalanced, if we use precision then KNN is better than LR but we use Recall then LR is better than KNN.<br><br>
To deal with this confusion we will be using the F1-score and Auc_roc curve.
"""

from sklearn.metrics import f1_score
print("F1-Score")
print("LR - with default threshold",f1_score(ytest,ypred1))
print("LR with custom threshold",f1_score(ytest,ypred11))
print("KNN with default threshold",f1_score(ytest,ypred2))
print("KNN with custom threshold",f1_score(ytest,ypred22))

"""Now as we have the single number which can help us to compare different models

**AUC_ROC** <br>

Returns the Area Under the Curve
"""

print("----FOR LOGISTIC REGRESSION-----")
print("Default threshold :",roc_auc_score(ytest,ypred1))
print("Cusotm threshold :",roc_auc_score(ytest,ypred11))
print("----FOR KNN ----")
print("Default threshold :",roc_auc_score(ytest,ypred2))
print("Cusotm threshold :",roc_auc_score(ytest,ypred22))



"""**--------- Plotting the auc_roc_curve----------**"""

from sklearn.metrics import roc_curve

fpr,tpr,threshold = roc_curve(ytest,ypred1)
print(threshold)

"""**Always pass the second argument to roc_curve as probability because for each unique probabilities it calculates a threshold and on that threhold it calculates the predicted out , compares it to the actual output and return the fpr and tpr for that particular threshold.**"""

fpr,tpr,threshold = roc_curve(ytest,ypred_proba1[:,1])
print(threshold[:7]) # Printing the first 7 threshold

# Creating data using all those fpr,tpr and threshold
d = {'fpr':fpr,
     'tpr':tpr,
     'threshold':threshold}
df_lr = pd.DataFrame(d)
df_lr

#For KNN
fpr2,tpr2,threshold2 = roc_curve(ytest,ypred_proba2[:,1])
print(threshold2[:7]) # Printing the first 7 threshold
# Creating data using all those fpr,tpr and threshold
d2 = {'fpr':fpr2,
     'tpr':tpr2,
     'threshold':threshold2}
df_knn = pd.DataFrame(d2)
df_knn

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(ytest))]
p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)

plt.plot(fpr,tpr,label="LR")
plt.plot(fpr2,tpr2,label='KNN')
plt.plot(p_fpr,p_tpr)

plt.ylabel("True Positive Rate/Recall")
plt.xlabel("False Positive Rate")
plt.title("ROC Curve")
plt.legend()

"""AS we can see that Area under the LR is greater than the KNN. so clf1 is better than the clf2."""

from sklearn.metrics._plot.roc_curve import plot_roc_curve
# Ploting using the function
plot_roc_curve(clf1,Xtest,ytest)
plot_roc_curve(clf2,Xtest,ytest)

"""**We can see a difference that when Area was calculated through roc_auc_score then area was less than when we plotted the graph using the roc_curve function because in roc_auc_score we passed the actual and predicted target values but in case of roc_curve it calculates the area with the best threshold and that's why there is a differnce**

**Classification Report**
"""

from sklearn.metrics import classification_report

lr_default_threshold=classification_report(ytest,ypred1)
lr_custom_threshold=classification_report(ytest,ypred11)

print(lr_default_threshold)

print(lr_custom_threshold)

knn_default_threshold=classification_report(ytest,ypred2)
knn_custom_threshold=classification_report(ytest,ypred22)

print(knn_default_threshold)

print(knn_custom_threshold)

"""--------------------------------------------- X ---------------------------------------------"""
