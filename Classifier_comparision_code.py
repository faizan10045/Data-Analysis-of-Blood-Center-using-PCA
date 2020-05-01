#Import libraries and packages

import pandas
import numpy as np
import scipy
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import KFold
from sklearn import tree
from tpot import TPOTClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
# use seaborn plotting style defaults
import seaborn as sns; sns.set()
from google.colab import files
from IPython.display import  display_html

#upload files into Colaboratory
X = []
Y = []
list_clf = []


uploaded = files.upload()

#read cvs file into dataframe
df = pandas.read_csv('d2.csv', index_col=0)
df = df.values


X = df[:, :(df.shape[1]-1)]
Y = df[:, df.shape[1]-1]

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
tpot.fit(X, Y)

clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_perceptron = Perceptron()
clf_KNN = KNeighborsClassifier()
clf_nb = BernoulliNB()

list_clf = [clf_tree, clf_svm, clf_perceptron, clf_KNN, clf_nb, tpot.fitted_pipeline_]

kf = KFold(n_splits=5)
kf.get_n_splits(X)
c = 1
for clfs in list_clf:
    print(c)
    c += 1
    a = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        clfs.fit(X_test, Y_test)  
        print(clfs.score(X_train, Y_train)) 
        a += clfs.score(X_train, Y_train)
    a = a/5
print("Average=",a,"\n")
print("***************************************************")
print(clfs,"\n")