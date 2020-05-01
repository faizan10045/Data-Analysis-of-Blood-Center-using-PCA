#Import libraries and packages
import pandas
import numpy as np
import scipy
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
# use seaborn plotting style defaults
import seaborn as sns; sns.set()
from google.colab import files
from IPython.display import  display_html

#upload files into Colaboratory
uploaded = files.upload()

#read cvs file into dataframe
df = pandas.read_csv('d2.csv', index_col=0)
print('')
print('***************** STEP - 1 ***************** ')
print('')
print('1. Data Description: - ')
print(df.describe())
print('')
print('2. Data Information: - ')
print(df.info())

# class distribution
print('3. Class Distribution: - ')
print(df.groupby('Class').size())

#normalize data
df1 = df
df2 = df
df1 = (df1 - df1.mean())/df1.std()
print(df1)

#bservations and variables
observations = list(df1.index)
variables = list(df1.columns)

# box and whisker plots
print('')
print('4. Box Plots: - ')
df.plot(kind='box', subplots=True, layout=(2,4), sharex=False, sharey=False, figsize=(18,16))
plt.show()
# histograms
print('')
print('5. Histograms: - ')
df.hist(figsize=(18,16))
plt.show()
# scatter plot matrix
print('')
print('6. Scatter Plot Matrix: - ')
scatter_matrix(df, figsize=(18,16))
plt.show()

#Covariance
print('')
print('7. Correlation and Covariance Matrix: - ')
corr = df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

#Covariance
print('')
dfc = df - df.mean() #centered data
plt. figure()
ax = sns.heatmap(dfc.cov(), cmap='RdYlGn_r', linewidths=0.5, annot=True, 
            cbar=False, square=True)
plt.yticks(rotation=0)
ax.tick_params(labelbottom=False,labeltop=True)
plt.show()

print('8. Principal component analysis: - ')
#Principal component analysis
pca = PCA()
pca.fit(df1)
Z1=pca.transform(df1)
Z = pca.fit_transform(df1)
print(pca.components_)
print(pca.explained_variance_)

plt. figure()
plt.scatter(Z[:,0], Z[:,1], c='r')
plt.xlabel('$Z_1$')
plt.ylabel('$Z_2$')
for label, x, y in zip(observations,Z[:, 0],Z[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(-2, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()

#Eigenvectors
A = pca.components_.T 
plt. figure()
plt.scatter(A[:,0],A[:,1],c='r')
plt.xlabel('$A_1$')
plt.ylabel('$A_2$');
for label, x, y in zip(variables, A[:, 0], A[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(-2, 2), textcoords='offset points', ha='right', va='bottom')
print('')
print('9. Eigen Values and Vectors: - ')
plt. figure()
plt.scatter(A[:, 0],A[:, 1],marker='o',c=A[:, 2],s=A[:, 3]*500,
    cmap=plt.get_cmap('Spectral'))
for label, x, y in zip(variables,A[:, 0],A[:, 1]):
    plt.annotate(label,xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.show()    
#Eigenvalues
Lambda = pca.explained_variance_

#Scree plot
print('')
print('10. Scree plot: - ')
plt. figure()
x = np.arange(len(Lambda)) + 1
Lambda = Lambda / sum(Lambda)
plt.plot(x,Lambda, 'ro-', lw=2)
plt.xticks(x, [""+str(i) for i in x], rotation=0)
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.show()

#Explained variance
print('')
print('11. Explained variance: - ')
ell = pca.explained_variance_ratio_
plt. figure()
ind = np.arange(len(ell))
plt.bar(ind, ell, align='center', alpha=0.5)
plt.plot(np.cumsum(ell))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

#Biplot
# 0,1 denote PC1 and PC2; change values for other PCs
A1 = A[:,0] 
A2 = A[:,1]
Z1 = Z[:,0] 
Z2 = Z[:,1]
plt.show()
plt. figure()
for i in range(len(A1)):
# arrows project features as vectors onto PC axes
    plt.arrow(0, 0, A1[i]*max(Z1), A2[i]*max(Z2),
              color='r', width=0.0005, head_width=0.0025)
    plt.text(A1[i]*max(Z1)*1.2, A2[i]*max(Z2)*1.2,variables[i], color='r')

for i in range(len(Z1)):
# circles project documents (ie rows from csv) as points onto PC axes
    plt.scatter(Z1[i], Z2[i], c='g', marker='o')
    plt.text(Z1[i]*1.2, Z2[i]*1.2, observations[i], color='b')
plt.show()
plt.figure()
comps = pandas.DataFrame(A,columns = variables)
sns.heatmap(comps,cmap='RdYlGn_r', linewidths=0.5, annot=True, 
            cbar=True, square=True)
ax.tick_params(labelbottom=False,labeltop=True)
plt.title('Principal components')
plt.show()


print('')
print('13. Distribution Plots: - ')
X1 = df2.iloc[:, 0:5]
y1 = df2[['Class']]
X_train1, X_cv1, y_train1, y_cv1 = train_test_split(X1, y1, test_size = 0.2, random_state = 0)

X_test1 = df2.iloc[:, 0:5]
X_train1 = X_train1.reset_index(drop = True)
X_cv1 = X_cv1.reset_index(drop = True)
X_test1 = X_test1.reset_index(drop = True)
plt.figure(figsize = (20, 10))
plt.subplot(2, 2, 1)
sns.distplot(X_train1[y_train1.values == 0]['V1'], 
             bins = range(0, 81, 2), color = 'red')
plt.ylabel('Frequency')
plt.title('Distribution of Months Since last Donation for Non-blood Donors')
sns.distplot(X_train1[y_train1.values == 1]['V1'], 
             bins = range(0, 81, 2), color = 'blue')
plt.ylabel('Frequency')
plt.title('Distribution of Months Since last Donation for Blood Donors')
plt.subplot(2, 2, 2)
sns.distplot(X_train1[y_train1.values == 0]['V2'], 
             bins = range(0, 60, 2), color = 'red')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Donations for Non-blood Donors')
sns.distplot(X_train1[y_train1.values == 1]['V2'], 
             bins = range(0, 60, 2), color = 'blue')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Donations for Blood Donors')
plt.subplot(2, 2, 3)
sns.distplot(X_train1[y_train1.values == 0]['V3'], color = 'red')
plt.ylabel('Frequency')
plt.title('Distribution of Total Volume Donated (c.c.) for Non-blood Donors')
sns.distplot(X_train1[y_train1.values == 1]['V3'], color = 'blue')
plt.ylabel('Frequency')
plt.title('Distribution of Total Volume Donated (c.c.) for Blood Donors')
plt.subplot(2, 2, 4)
sns.distplot(X_train1[y_train1.values == 0]['V4'], color = 'red')
plt.ylabel('Frequency')
plt.title('Distribution of Months since First Donation for Non-blood Donors')
sns.distplot(X_train1[y_train1.values == 1]['V4'], color = 'blue')
plt.ylabel('Frequency')
plt.title('Distribution of Months since First Donation for Blood Donors')
plt.show()

X_train1['Average Donation per Month'] = (X_train1['V3']/X_train1['V4'])
X_cv1['Average Donation per Month'] = X_cv1['V3']/X_cv1['V4']
X_test1['Average Donation per Month'] = X_test1['V3']/X_test1['V4']
plt.figure(figsize = (10, 5))

sns.distplot(X_train1[y_train1.values == 0]['Average Donation per Month'], color = 'red')
plt.ylabel('Frequency')
plt.title('Distribution of Months Since last Donation for Non-blood Donors')
sns.distplot(X_train1[y_train1.values == 1]['Average Donation per Month'], color = 'blue')
plt.ylabel('Frequency')
plt.title('Distribution of Months Since last Donation for Blood Donors')
plt.show()

X_train1['Waiting Time'] = ((X_train1['V4'] - X_train1['V1'])/X_train1['V2'])
X_cv1['Waiting Time'] = ((X_cv1['V4'] - X_cv1['V1'])/X_cv1['V2'])
X_test1['Waiting Time'] = ((X_test1['V4'] - X_test1['V1'])/X_test1['V2'])
plt.figure(figsize = (10, 5))
sns.distplot(X_train1[y_train1.values == 0]['Waiting Time'], color = 'red')
plt.ylabel('Frequency')
plt.title('Distribution of Waiting Time for Non-blood Donors')
sns.distplot(X_train1[y_train1.values == 1]['Waiting Time'], color = 'blue')
plt.ylabel('Frequency')
plt.title('Distribution of Waiting Time for Blood Donors')
plt.show()


print('')
print('***************** STEP - 2 ***************** ')
# Split-out validation dataset
print('')
print('STEP - 2 Split-out validation dataset: - ')
array = df.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.30
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
t1= pandas.DataFrame(X_train).describe()
t2= pandas.DataFrame(X_test).describe()
t3= pandas.DataFrame(Y_test).describe()
print(t1)
print(t2)
print(t3)

#Step 3 - Analysis of the Output Column
print('')
print('***************** STEP - 3 ***************** ')
print('')
print('Step 3 - Analysis of the Output Column: - ')
t4= scipy.stats.iqr(Y_train)
print(t4)
t5= scipy.stats.iqr(Y_test)
print(t5)
t6= np.amax(Y_train) - np.amin(Y_train)
print(t6)
t7= np.amax(Y_test) - np.amin(Y_test)
print(t7)

print('')
print('***************** STEP - 4 ***************** ')
#Step 4 - Scale Training and Test dataset
print('')
print('Step 4 - Scale Training and Test dataset: - ')
train_scaler = MinMaxScaler()
train_scaled = pandas.DataFrame(train_scaler.fit_transform(X_train))
t8= train_scaled.describe()
print(t8)
test_scaler = MinMaxScaler()
test_scaled = pandas.DataFrame(test_scaler.fit_transform(X_test))
t9= test_scaled.describe()
print(t9)

print('')
print('***************** STEP - 5 ***************** ')
#Step 5 - Build Predictive Model
print('')
print('Step 5 - Build Predictive Model: - ')
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
#Introduce Algorithms
models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=5, random_state=seed)
    cv_results = model_selection.cross_val_score(model, train_scaled, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

print('')
print('***************** STEP - 6 ***************** ')
#Step 6 - Model Predictions on Training Dataset
print('')
print('Step 6 - Model Predictions on Training Dataset: - ')
# Make predictions on train dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(train_scaled)
# print(accuracy_score(Y_train, predictions))
print(confusion_matrix(Y_train, predictions))

print('')
print('***************** STEP - 7 ***************** ')
#Step 7 - Model Predictions on Test Dataset
print('')
print('Step 7 - Model Predictions on Test Dataset: - ')
# Make predictions on test dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)
# print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))

print('')
print('***************** STEP - 8 ***************** ')
#Step 8 - Model Performance
print('')
print('Step 8 - Model Performance: - ')
# Training Performance
# Make predictions on train dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(train_scaled)
print(accuracy_score(Y_train, predictions))
# print(classification_report(Y_train, predictions))
# Testing Performance
# Make predictions on test dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(classification_report(Y_test, predictions))

print('')
print('***************** STEP - 9 ***************** ')
#Step 9 - Update the Model
print('')
print('Step 9 - Update the Model: - ')
###### STEP 5 ##############
# Update the model         #
# parameters and re-train  #
# the model.               #
###########################

# Introduce Algorithms
models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=20, algorithm='ball_tree', leaf_size=50)))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, train_scaled, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Make predictions on train dataset
knn = KNeighborsClassifier(n_neighbors=20, algorithm='ball_tree', leaf_size=50)
knn.fit(X_train, Y_train)
predictions = knn.predict(train_scaled)
# print(accuracy_score(Y_train, predictions))
print(confusion_matrix(Y_train, predictions))
# print(classification_report(Y_train, predictions))

# Make predictions on test dataset
knn = KNeighborsClassifier(n_neighbors=20, algorithm='ball_tree', leaf_size=50)
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)
# print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
# print(classification_report(Y_test, predictions))

# Training Performance
# Make predictions on train dataset
knn = KNeighborsClassifier(n_neighbors=20, algorithm='ball_tree', leaf_size=50)
knn.fit(X_train, Y_train)
predictions = knn.predict(train_scaled)
print(accuracy_score(Y_train, predictions))
# print(confusion_matrix(Y_train, predictions))
# print(classification_report(Y_train, predictions))

# Testing Performance
# Make predictions on test dataset
knn = KNeighborsClassifier(n_neighbors=20, algorithm='ball_tree', leaf_size=50)
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)
print(accuracy_score(Y_test, predictions))
# print(confusion_matrix(Y_test, predictions))
# print(classification_report(Y_test, predictions))

print('')
print('***************** STEP - 10 ***************** ')
#Step 10 - Change the Error Metric
print('')
print('Step 10 - Change the Error Metric: - ')
# Training Performance
# Make predictions on train dataset
knn = KNeighborsClassifier(n_neighbors=20, algorithm='ball_tree', leaf_size=50)
knn.fit(X_train, Y_train)
predictions = knn.predict(train_scaled)
print(accuracy_score(Y_train, predictions))
print(confusion_matrix(Y_train, predictions))
print(roc_auc_score(Y_train, predictions))
print(classification_report(Y_train, predictions))

# Testing Performance
# Make predictions on test dataset
knn = KNeighborsClassifier(n_neighbors=20, algorithm='ball_tree', leaf_size=50)
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(roc_auc_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
