# %% [markdown]
# # CA4 - Stellar object classification

# %% [markdown]
# ### Import libraries 

# %%
# import libraries for plotting and data manipulation
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np

# import classfier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# import model selection and preprocessing tools 
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score


# %% [markdown]
# ### Reading the data 

# %%
# load training and test data 
training_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# %% [markdown]
# ### Data exploration 

# %% [markdown]
# #### Look at the shape of the data

# %%
# check the shape of the data 
training_data.shape, test_data.shape

# %% [markdown]
# We can see by the shape of the data, that the training and test data is split 80/20 

# %% [markdown]
# #### Overview of the data 

# %%
training_data.head()

# %% [markdown]
# The data has 17 columns, where the class column is the class of the stellar object, our target variable y. 

# %% [markdown]
# #### Visualize feature distributions using violinplots 

# %%
# visualize feature distributions 
fig, axes = plt.subplots(4,5, figsize = (12,14)) 
axes = axes.flatten() 

columns = training_data.columns.to_list()
columns = [col for i, col in enumerate(columns) if i != 13]  

colors = sns.color_palette('Set3', n_colors=len(columns)) 

for i, col in enumerate(columns):
    sns.violinplot(x=col, data=training_data, ax=axes[i], color=colors[i])

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Perform descriptive statistics 

# %%
training_data.describe() 

# %% [markdown]
# ### Data cleaning

# %% [markdown]
# #### Look for missing values 

# %%
# identify rows with missing values
NaN = training_data.isna().sum()
print(NaN)

print(f'Missing values in the training data: {NaN[3]}')

# %% [markdown]
# #### Remove missing values from the training data 

# %%
# remove rows with missing values
training_data_clean = training_data.dropna()

# check if there are any missing values left
training_data_clean.isna().sum()

print(f'Missing values left in the training data: {training_data_clean.isna().sum().sum()}')

# %% [markdown]
# #### Remove excessive features from the data set 

# %%
training_data_clean = training_data_clean.drop(columns=['obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID', 'MJD', 'fiber_ID', 'plate', 'fiber_ID'])
training_data_clean.head()

# %%
# remove excessive features from the test data 
test_data_clean = test_data.drop(columns=['obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID', 'MJD', 'fiber_ID', 'plate', 'fiber_ID'])
test_data_clean.head()

# %% [markdown]
# ##### Visualize feature distributions after removing excessive features 

# %%
# visualize feature distributions 
fig, axes = plt.subplots(2,4, figsize = (10,5)) 
axes = axes.flatten() 

columns = training_data_clean.columns.to_list()
columns = [col for i, col in enumerate(columns) if i != 7]  

colors = sns.color_palette('Set3', n_colors=len(columns)) 

for i, col in enumerate(columns):
    sns.boxplot(x=col, data=training_data_clean, ax=axes[i], color=colors[i])

plt.tight_layout()
plt.show()


# %% [markdown]
# ##### Visualization using violinplots 

# %%
# visualize feature distributions 
fig, axes = plt.subplots(2,4, figsize = (10,5)) 
axes = axes.flatten() 

columns = training_data_clean.columns.to_list()
columns = [col for i, col in enumerate(columns) if i != 7]  

colors = sns.color_palette('Set3', n_colors=len(columns)) 

for i, col in enumerate(columns):
    sns.violinplot(x=col, data=training_data_clean, ax=axes[i], color=colors[i])

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Visualize the feature relationships using pairplots from seaborn 

# %%
# make a pairplot of the training data 
sns.pairplot(training_data_clean, hue = 'class', palette= 'BuPu')
plt.show()

# %% [markdown]
# We can see that the features are highly different in their distributions, and therefore should be scaled/standardized. 

# %% [markdown]
# ### Data preprocessing and visualization 

# %% [markdown]
# #### Split the training data into X_train and y_train

# %%
X_train = training_data_clean.drop(columns=['class'])
y_train = training_data_clean['class']
X_train.shape, y_train.shape

# %% [markdown]
# #### Transform the categorical class variable using LabelEncoder 

# %%
le = LabelEncoder() 
# encode the target variable
y_train = le.fit_transform(y_train)
le.classes_

# %%
le.transform(['GALAXY', 'QSO', 'STAR']) 

# %% [markdown]
# #### Look at the target variable distribution 

# %%
(f'The distribution of the target variable is: {np.bincount(y_train)}')

# %%
sns.countplot(x=y_train, palette='BuPu')
plt.title('Distribution of the target variable class')
plt.xlabel('Class')

# %% [markdown]
# We can see from the plot above that the target variable is quite unbalanced, with more than half of the samples in the first class, galaxy. 

# %% [markdown]
# #### Visualize the features before removing outliers

# %%
X_train.hist(figsize=(12, 10), bins=30, color='lightblue', alpha=0.7)

# %% [markdown]
# #### Identify and remove outliers using Z-score 

# %%
# Detect outliers using z-scores 

# Compute Z-scores
z_scores = (X_train - X_train.mean()) / X_train.std()

# Identify Outliers (absolute Z-score > 3)
outliers = (np.abs(z_scores) > 3)

# Count total outliers
num_outliers = outliers.sum().sum()  # Summing over all columns

print(f"Total number of outliers: {num_outliers}")


# %%
if isinstance(y_train, np.ndarray):
    y_train = pd.Series(y_train, index=X_train.index)

# Filter rows where all features have Z-score < 3
X_train_clean = X_train[(np.abs(z_scores) < 3).all(axis=1)]

# Remove corresponding rows from y_train
y_train_clean = y_train.loc[X_train_clean.index]

# Print the shapes of the original and cleaned datasets
print("Original X_train shape:", X_train.shape)
print("Original y_train shape:", y_train.shape)
print("Cleaned X_train shape:", X_train_clean.shape)
print("Cleaned y_train shape:", y_train_clean.shape)

# %% [markdown]
# #### Visualize the features after removing outliers 

# %%
histogram = X_train_clean.hist(figsize=(12, 10), bins=30, color='lightblue', alpha=0.7)

# %% [markdown]
# #### Visualize the data before scaling 

# %%
sns.violinplot(X_train_clean, palette = sns.color_palette('Set3', n_colors= X_train_clean.shape[1])) 
plt.xticks(rotation = 90)
plt.title('Distribution of features before scaling')
plt.show()

# %% [markdown]
# ### Modelling 

# %% [markdown]
# #### Define pipelines 

# %% [markdown]
# ##### SVM 

# %%
# Define the SVC pipeline: 
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
pipe_svc.get_params()

# %% [markdown]
# ##### Logistic regression 

# %%
# Define the logistic regression pipeline:
pipe_logistic = make_pipeline(StandardScaler(), PCA(n_components=7), LogisticRegression(random_state=1))
pipe_logistic.get_params()

# %% [markdown]
# ##### Random Forest

# %%
# Define the random forest pipeline: 
pipe_rf = make_pipeline(RandomForestClassifier(random_state=1)) 
pipe_rf.get_params()

# %% [markdown]
# ##### K-nearest-neighbour (KNN) 

# %%
# Define the KNN pipeline: 
pipe_knn = make_pipeline(StandardScaler(), KNeighborsClassifier())
pipe_knn.get_params()

# %% [markdown]
# #### Evaluate different models and hyperparameters using GridSearchCV with cross-validation 

# %% [markdown]
# ##### SVM 

# %%

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs_svm = GridSearchCV(estimator=pipe_svc,
                  param_grid={'svc__C': [0.1, 1, 10, 100], 'svc__kernel': ['rbf']},
                  scoring='f1_macro',
                  cv=cv_strategy,
                  n_jobs=-1)

gs_svm = gs_svm.fit(X_train_clean, y_train_clean)
print(gs_svm.best_score_)
print(gs_svm.best_params_)

best_model_svm = gs_svm.best_estimator_

# %% [markdown]
# ##### Logistic regression 

# %%
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs_logistic = GridSearchCV(estimator=pipe_logistic, 
                  param_grid={'logisticregression__penalty': ['l2'], 
                              'logisticregression__C': [1,10,100]},
                  scoring='f1_macro',
                  cv=cv_strategy,
                  n_jobs=-1)

gs_logistic = gs_logistic.fit(X_train_clean, y_train_clean)
print(gs_logistic.best_score_)
print(gs_logistic.best_params_)

best_model_logistic = gs_logistic.best_estimator_

# %% [markdown]
# ##### Random forest 

# %%
gs_rf = GridSearchCV(estimator=pipe_rf,
                  param_grid={'randomforestclassifier__n_estimators': [80, 100, 120],
                              'randomforestclassifier__max_features': ['sqrt', 'log2'],
                              'randomforestclassifier__max_depth': [5,10,15],
                              'randomforestclassifier__criterion': ['entropy']}, 
                  scoring='f1_macro',
                  cv=5,
                  n_jobs=-1)

gs_rf = gs_rf.fit(X_train_clean, y_train_clean) 
print(gs_rf.best_score_)
print(gs_rf.best_params_)

best_model_rf = gs_rf.best_estimator_ 

# %% [markdown]
# ##### KNN 

# %%
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs_knn = GridSearchCV(estimator=pipe_knn, 
                  param_grid={'kneighborsclassifier__n_neighbors': [3, 5, 7],
                              'kneighborsclassifier__weights': ['distance'],
                              'kneighborsclassifier__metric': ['euclidean', 'manhattan']},
                  scoring='f1_macro',
                  cv=cv_strategy,
                  n_jobs=-1)

gs_knn = gs_knn.fit(X_train_clean, y_train_clean)
print(gs_knn.best_score_)
print(gs_knn.best_params_)

best_model_knn = gs_knn.best_estimator_

# %% [markdown]
# #### Build classifier based on all training samples using the "optimal parameters"

# %% [markdown]
# The code below is based on the best model of the Random Forest Classifier, since this had the highest score during GridSearch. 
# 
# 
# All fitting, predicting and evaluation is done using this classifier. 

# %%
# fit the best model to the training data (random forest) 
best_model_rf.fit(X_train_clean, y_train_clean) 


# %% [markdown]
# ### Evaluate model performance  

# %% [markdown]
# ##### Confusion matrix 

# %%
# confusion matrix for the random forest model using train/test split on training data
X_train_data, X_test_data, y_train_labels, y_testlabels = train_test_split(X_train_clean, y_train_clean, test_size=0.4, random_state=42) 
y_pred = best_model_rf.predict(X_test_data) 

confusion = confusion_matrix(y_testlabels, y_pred)
confusion 


# %% [markdown]
# ##### Plot the confusion matrix using matplotlib 

# %%
# Code borrowed from lecture 'Chapter_6_part_2b' 

fig, ax = plt.subplots(figsize=(3, 3))
ax.matshow(confusion, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confusion.shape[0]):
    for j in range(confusion.shape[1]):
        ax.text(x=j, y=i, s=confusion[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
plt.show()

# %% [markdown]
# Model performance using classification report in Scikit learn 

# %%
classification_rep = pd.DataFrame(classification_report(y_testlabels, y_pred, output_dict=True)).T
classification_rep

# %% [markdown]
# We can see that the output from the classification report gave the same F1 scores as we manually calculated above.

# %% [markdown]
# ### Kaggle submission 

# %%
y_test = best_model_rf.predict(test_data_clean)
y_test = pd.DataFrame(y_test, columns=["class"])
y_test.index.name = "ID"
y_test[['class']].to_csv("data/sample_submission.csv")


