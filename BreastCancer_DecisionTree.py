#!/usr/bin/env python
# coding: utf-8

# # BREAST CANCER PREDICTION USING DECISION TREE 

# In[1]:


#Importing the libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 
import matplotlib.pyplot as plt # Visualization
import seaborn as sns # Visualization


# In[2]:


# Reading the data from CSV file and making a dataframe
ip = pd.read_csv("D:\SEMESTER I\Intelligent Systems ECE 579\Project Assignment\Breastcancer.csv")


# In[3]:


ip.head()


# In[4]:


ip.tail()


# In[5]:


ip.info()


# In[6]:


ip.isna().sum() 


# In[7]:


len(ip.index), len(ip.columns)


# In[8]:


#Drop the column with all missing values (na, NAN, NaN)
#NOTE: This drops the column Unnamed: 32 column
ip = ip.dropna(axis=1)


# In[9]:


#examine the shape of the data
ip.shape


# In[10]:


ip.describe()


# In[11]:


ip.describe(include="O")


# In[12]:


#Get a count of the number of 'M' & 'B' cells
ip.diagnosis.value_counts()


# In[13]:


#Visualize count of 'M' & 'B' cells
sns.countplot(ip['diagnosis'], palette='husl')


# CLEANING AND PREPROCESSING THE DATA

# In[14]:


#Converting 'M' & 'B' to '1' & '0' respectively
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
ip.iloc[:,1]=labelencoder_Y.fit_transform(ip.iloc[:,1].values)


# In[15]:


ip.head()


# In[16]:


## Drop the column 'id' as it is does not convey any useful info
ip.drop('id',axis=1,inplace=True)


# In[17]:


ip.head()


# In[18]:


## finding the correlation between the attributes
ip.corr()


# In[19]:


#generating scatter plot matrix with the mean columns

col = ['diagnosis',
        'radius_mean', 
        'texture_mean', 
        'perimeter_mean', 
        'area_mean', 
        'smoothness_mean', 
        'compactness_mean', 
        'concavity_mean',
        'concave points_mean', 
        'symmetry_mean', 
        'fractal_dimension_mean']

sns.pairplot(data=ip[col], hue='diagnosis', palette="Set2")


# Almost perfectly linear patterns between the radius, perimeter and area attributes are hinting at the presence of multicollinearity between these variables. (they are highly linearly related) Another set of variables that possibly imply multicollinearity are the concavity, concave_points and compactness.

# In[20]:


## Heatmap to visualize the multicollinearity
corr = ip.corr().round(2)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.tight_layout()


# Removed all the attributes which are having high collinearity by taking threshold value to be above 0.81

# In[21]:


# Dropping all columns related to the worst
cols = ['radius_worst', 
        'texture_worst', 
        'perimeter_worst', 
        'area_worst', 
        'smoothness_worst', 
        'compactness_worst', 
        'concavity_worst',
        'concave points_worst', 
        'symmetry_worst', 
        'fractal_dimension_worst']
ip = ip.drop(cols, axis=1)

# then, drop all columns related to the "perimeter" and "area" attributes
cols = ['perimeter_mean',
        'perimeter_se', 
        'area_mean', 
        'area_se']
ip = ip.drop(cols, axis=1)
# then, drop all columns related to the "concavity" and "concave points" attributes
cols = ['concavity_mean',
        'concavity_se', 
        'concave points_mean',
        'concave points_se']
ip = ip.drop(cols, axis=1)


# In[22]:


# Remaining columns after dropping the columns which has multi-collinearity
ip.columns


# In[23]:


## Heatmap to visualize the collinearity after dropping the columns
corr = ip.corr().round(2)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.tight_layout()


# In[24]:


# Feature columns
X = ip.drop(['diagnosis'],axis=1)

X.head()


# In[25]:


# Class labels
Y = ip.diagnosis
Y.head()


# Model Building

# In[26]:


# splitting the data set into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30,random_state=40, shuffle=True)


# In[27]:


Y_train.value_counts()


# In[28]:


# Featutre Scaling
from sklearn.preprocessing import StandardScaler as sc
X_train = sc().fit_transform(X_train)
X_test = sc().fit_transform(X_test)


# # DECISION TREE

# In[29]:


from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
DT=DecisionTreeClassifier(random_state=42)
model=DT.fit(X_train,Y_train)
prediction=model.predict(X_test)
pd.DataFrame(confusion_matrix(Y_test, prediction), columns=['Predicted Benign', "Predicted Malignant"], index=['Actual Benign', 'Actual Malignant'])


# In[30]:


cm=confusion_matrix(Y_test, prediction)
cm


# In[31]:


# accuracy and classification report
print(classification_report(Y_test, prediction))
print('Accuracy:',accuracy_score(Y_test, prediction))


# In[32]:


# visualizing the decision tree
from sklearn import tree
fig=plt.figure(figsize=(50,15))
tree.plot_tree(DT,feature_names=X.columns,class_names=["Benign","Malignant"],filled=True)
fig.savefig("decistion_tree.png")


# In[33]:


# node count and tree depth count
print(DT.tree_.node_count)
print(DT.tree_.max_depth)


# In[34]:


# Training data and Test data accuracy
DT_test = accuracy_score(Y_test, prediction);
# Evaluating on the training data
y_train_pred = model.predict(X_train);
DT_train = accuracy_score(Y_train, y_train_pred);

print("Training data accuracy is " +  repr(DT_train) + " and test data accuracy is " + repr(DT_test))


# Here, the training data accuracy is 1.0, indicating that the model is overfitted, which we have to avoid it by using the technique of post pruning.
# 

# In[35]:


# Post Pruning- Cost Complexity pruning
path = DT.cost_complexity_pruning_path(X_train, Y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities


# In[36]:


ccp_alphas


# In[37]:


clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, Y_train)
    clfs.append(clf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))


# In[38]:


clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()


# In[39]:


# graph which represents accuracy of training and test data with respect to alpha.
train_scores = [clf.score(X_train, Y_train) for clf in clfs]
test_scores = [clf.score(X_test, Y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()


# In[40]:


# new decision tree after post pruning.
dt = DecisionTreeClassifier(random_state=42, ccp_alpha=0.00753769)
model_after_pruning=dt.fit(X_train,Y_train)
pred=dt.predict(X_test)


# In[41]:


pd.DataFrame(confusion_matrix(Y_test, pred), columns=['Predicted Benign', "Predicted Malignant"], index=['Actual Benign', 'Actual Malignant'])


# In[42]:


print(classification_report(Y_test, pred))
print('Accuracy:',accuracy_score(Y_test, pred))


# In[43]:


fig=plt.figure(figsize=(15,10))
tree.plot_tree(dt,feature_names=X.columns,class_names=["Benign","Malignant"],filled=True)
fig.savefig("decistion_tree1.png")


# In[44]:


print(dt.tree_.node_count)
print(dt.tree_.max_depth)


# In[45]:


# Training and test accuracy after post pruning
dt_test = accuracy_score(Y_test, pred);
# Evaluating on the training data
y_train_pred = model_after_pruning.predict(X_train);
dt_train = accuracy_score(Y_train, y_train_pred);

print("Training data accuracy is " +  repr(dt_train) + " and test data accuracy is " + repr(dt_test))


# In[46]:


# roc_auc_score for DT
from sklearn.metrics import roc_curve, roc_auc_score,auc
y_score1 = dt.predict_proba(X_test)[:,1]


# In[47]:


dt_fpr, dt_tpr, threshold1 = roc_curve(Y_test, y_score1)


# In[48]:


print('roc_auc_score for DecisionTree: ', roc_auc_score(Y_test, y_score1))


# In[49]:


# plotting the ROC Curve
auc_dt=auc(dt_fpr,dt_tpr)
plt.figure(figsize=(5,5),dpi=100)
plt.plot(dt_fpr,dt_tpr,linestyle='-',label='DT (auc=%0.3f)'%auc_dt)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.xlabel('False positive rate -->')
plt.ylabel('True Positive Rate-->')
plt.legend()
plt.show()

