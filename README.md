#Decision tree

Attached dataset along with code, 
For reading the dataset, you can upload the CSV file into your local drive and provide the path of the file in the below line.
ip = pd.read_csv("D:\SEMESTER I\Intelligent Systems ECE 579\Project Assignment\Breastcancer.csv")
or 
You can directly upload the file to jupyter notebook or google collab.
anything is okay at your convenience.

To get 60/40,70/30 and 80/20 as train_test split. 
In the code, the below changes the test_size to 0.40 for 60/40, 0.30 for 70/30, and 0.20 for 80/20.
(X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30,random_state=40, shuffle=True))

To avoid overfitting for the three splits changes the ccp_alpha value as follows in the below line of code.
ccp_alpha = 0.00391007 for 60/40 split.
ccp_alpha = 0.00753769 for 70/30 split.
ccp_alpha = 0.00990645 for 80/20 split.
dt = DecisionTreeClassifier(random_state=42, ccp_alpha=0.00753769)

#SVM

Readme file forThe dataset is read from the Breastcancer.csv file. Plotted the dataset using heatmap. 
The attributes with high multicollinearity of a threshold above 0.81 are dropped. The remaining attributes
are considered for fitting into the SVM model. The Scatter Plot between radius mean and texture mean is plotted. 
The data is split into 70% and 30% only as the model gave high accuracy and high AUC score though the accuracies
and AUC scores of different split data 60% -40% and 80%-20% are written in the project report. In the SVM, to consider
the values of the hyperparameters, the GridSearchCV function is used. Then SVM model is fit using the parameters 
Kernal=’RBF’, C=1, and gamma = 0.1 giving an accuracy of 96.5%. A 3-fold cross-validation method is used for experimenting with the dataset. 
GridSearchCV function is used for getting parameter values in the 3-fold method. Then SVM model gave an accuracy of 
94.3%.with parameters values, Kernal=’Linear’, C=1000, and gamma = 1. The Accuracy values are rounded in the print statements.

