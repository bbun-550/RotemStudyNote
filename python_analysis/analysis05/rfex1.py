'''
[Randomforest 문제1] 
kaggle.com이 제공하는 'Red Wine quality' 분류 ( 0 - 10)

dataset은 winequality-red.csv 

https://www.kaggle.com/sh6147782/winequalityred?select=winequality-red.csv

 

Input variables (based on physicochemical tests):
 1 - fixed acidity
 2 - volatile acidity
 3 - citric acid
 4 - residual sugar
 5 - chlorides
 6 - free sulfur dioxide
 7 - total sulfur dioxide
 8 - density
 9 - pH
 10 - sulphates
 11 - alcohol
 Output variable (based on sensory data):
 12 - quality (score between 0 and 10)
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('winequality-red.csv')
# print(df.head())

# print(df.isnull().sum())
# df.info()

# print(df['quality'].unique()) # [5 6 7 4 8 3]

x = df.drop(['quality'], axis=1)
y = df['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

logimodel = LogisticRegression(solver='lbfgs', max_iter=500).fit(x_train, y_train)
dtmodel = DecisionTreeClassifier().fit(x_train, y_train)
rfmodel = RandomForestClassifier().fit(x_train, y_train)

logipred = logimodel.predict(x_test)
print(f'logimodel acc : {accuracy_score(y_test,logipred):.5f}')

dtpred = dtmodel.predict(x_test)
print(f'dtmodel acc : {accuracy_score(y_test,dtpred):.5f}')

rfpred = rfmodel.predict(x_test)
print(f'rfmodel acc : {accuracy_score(y_test,rfpred):.5f}')

# 혼동행렬 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, dtpred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

cm = confusion_matrix(y_test, logipred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logimodel.classes_)
disp.plot()
plt.show()

cm = confusion_matrix(y_test, dtpred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dtpred.classes_)
disp.plot()
plt.show()