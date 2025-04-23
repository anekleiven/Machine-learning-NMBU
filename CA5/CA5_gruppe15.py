# %% [markdown]
# # CA5 Pepper Spiciness Prediction 

# %% [markdown]
# ## Import Libraries 

# %%
# import libraries for plotting and data manipulation
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np

# import classifiers 
from sklearn.linear_model import Ridge, Lasso, RANSACRegressor, ElasticNet 
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

# import model selection and preprocessing tools 
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, f1_score
from sklearn.impute import SimpleImputer


# %% [markdown]
# ## Load the Data 

# %%
train_data = pd.read_csv('/Users/tuva/Documents/DAT200/CAs/CA5/train.csv') 
test_data= pd.read_csv('/Users/tuva/Documents/DAT200/CAs/CA5/test.csv') 

# %% [markdown]
# ## Data Exploration

# %% [markdown]
# ### Look at the shape of the data 

# %%
train_data.shape, test_data.shape

# %% [markdown]
# The train and test data includes 14 features (plus one target variable in the test data). 
# 
# The training data has 1000 samples. 
# 
# The test data has 800 samples.

# %% [markdown]
# ### Overview of the Data 

# %%
train_data.head() 

# %% [markdown]
# The data includes the following features: 
# 
# - Length (cm) 
# - Width (cm) 
# - Weight (g) 
# - Pericarp Thickness (mm) 
# - Seed count 
# - Capsaicin Content 
# - Vitamin C Content (mg) 
# - Sugar Content 
# - Moisture Content 
# - Firmness 
# - Color
# - Harvest Time 
# - Average Daily Temperature During Growth (celcius) 
# - Average Temperature During Storage (celcius) 
# 
# The target variable is: 
# 
# - Scoville Heat Units (SHU) 

# %%
test_data.head()

# %% [markdown]
# ### Visualize feature distributions 

# %%
fig, axes = plt.subplots(4, 4, figsize=(14, 12))
axes = axes.flatten()

columns = train_data.columns.to_list()
colors = sns.color_palette('Set3', n_colors=len(columns))

for i, col in enumerate(columns):
    ax = axes[i]
    try:
        if train_data[col].dtype == 'object':  # Kategorisk
            sns.countplot(x=train_data[col], ax=ax, color=colors[i])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        else:  # Numerisk
            sns.histplot(train_data[col], ax=ax, color=colors[i])
        ax.set_title(col, fontsize=9)
    except Exception as e:
        ax.set_visible(False)
        print(f"Kunne ikke plotte '{col}': {e}")

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Perform Descriptive Statistics 

# %%
train_data.describe() 

# %% [markdown]
# ### Visualizing feature correlations 

# %%
# Numerical columns
numerical_data = train_data.select_dtypes(include=['number'])

# Heatmap of feature correlations
plt.figure(figsize=(10, 8))
sns.heatmap(numerical_data.corr(), annot=True, fmt=".2f", cmap="rocket", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# %% [markdown]
# Not a lot of correlation, except from a noticable positive correlation at 0.69 between Weight an Length, which isn't suprising.

# %% [markdown]
# ### Visualize feature relationships with pairplot 

# %%
sns.pairplot(train_data, hue='Scoville Heat Units (SHU)', palette='viridis')
plt.show()

# %% [markdown]
# The pairplot above show how the different features relate to each other. Hard to see a clear pattern here.

# %% [markdown]
# ## Data Cleaning

# %% [markdown]
# ### Look for missing values 

# %%
NaN = train_data.isna().sum()
NaN

# %% [markdown]
# There are missing values in several columns. The most remarkable is 'Average Temperature During Storage (celcius)'. 
# Because of the big number of missing values, all these samples cannot be removed. 
# We have two options: removing the feature or impute another value. 
# The feature will be removed, since 65% of the data set misses this value. 
# 
# Rows with missing samples from other features will be removed from the data set. 

# %% [markdown]
# ### Handle missing values 

# %%
# remove the feature 'Average Temperature During Storage (celcius)'
train_data = train_data.drop(columns=['Average Temperature During Storage (celcius)'])
test_data = test_data.drop(columns=['Average Temperature During Storage (celcius)'])

# remove rows with missing values in other features 
train_data.dropna(inplace=True) 

# check if there are any missing values left
train_data.isna().sum()

# %%
train_data.head()

# %%
train_data.shape 

# %% [markdown]
# The removal of missing values, and imputation of 'Unknown' in the 'Average Temperature During Storage (celcius)' column was successful. 
# 
# The new data set has 990 samples, ten samples were removed. 

# %% [markdown]
# ### Transform categorical variables using OneHotEncoder 

# %% [markdown]
# The data set has two categorical features: 
# - Color 
# - Harvest time
# 
# These will be converted into numerical values using OneHotEncoder from Scikit-learn. 

# %%
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore') 

encoded = ohe.fit_transform(train_data[['color', 'Harvest Time']])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['color', 'Harvest Time']))

# concatenate the encoded features with the original dataframe 
train_data_enc = pd.concat([train_data, encoded_df], axis=1) 

# drop the original categorical columns
train_data_enc.drop(columns=['color', 'Harvest Time'], inplace=True)

# %% [markdown]
# The code below shows the new feature names: 

# %%
train_data_enc.columns

# %% [markdown]
# #### Repeat for the test data 

# %%
encoded_test = ohe.transform(test_data[['color', 'Harvest Time']])
encoded_test_df = pd.DataFrame(encoded_test, columns=ohe.get_feature_names_out(['color', 'Harvest Time'])) 

# concatenate the encoded features with the original dataframe 
test_data_enc = pd.concat([test_data, encoded_test_df], axis=1) 

# drop the original categorical columns
test_data_enc.drop(columns=['color', 'Harvest Time'], inplace=True)

# %%
test_data_enc.columns

# %% [markdown]
# ### Remove outliers from the training data using Z-score 

# %%
# compute Z-scores
z_scores = (train_data_enc - train_data_enc.mean()) / train_data_enc.std()

# identify Outliers (absolute Z-score > 3)
outliers = z_scores.abs() > 3

# count total outliers
num_outliers = outliers.sum().sum() # Summing over all columns

print(f"Total number of outliers: {num_outliers}")


# %%
# remove the detected outliers 
train_data_clean = train_data_enc[~outliers.any(axis=1)]
train_data_clean.shape

# %% [markdown]
# After removing missing values, the feature 'Temperature during storage (celcius)' and outliers, our training data now contains 18 columns and 940 samples. 

# %% [markdown]
# ## Data Preprocessing and Visualization 

# %% [markdown]
# ### Split the training data into X_train and y_train 

# %%
X_train = train_data_clean.drop(columns=['Scoville Heat Units (SHU)']) 
y_train = train_data_clean['Scoville Heat Units (SHU)']
X_train.shape, y_train.shape

# %%
y_train.isna().sum()

# %%
y_train = y_train.dropna()
X_train = X_train.loc[y_train.index]

# %% [markdown]
# 10 samples in y_train was defined with missing values in y_train, these where therefore removed from the data set. 

# %%
y_train.shape, X_train.shape

# %% [markdown]
# ### Visualize the features in X_train (cleaned data) 

# %%
histogram = X_train.hist(figsize=(14, 12), bins=30, color='lightpink', alpha=0.7)

# %% [markdown]
# Most features looks quite normally distributed, except the encoded binary features (either 0 or 1). 

# %% [markdown]
# ### Visualize the unscaled data (X_train)

# %%
sns.violinplot(X_train, palette = sns.color_palette('pastel', n_colors= X_train.shape[1])) 
plt.xticks(rotation = 90)
plt.title('Distribution of features before scaling')
plt.show()

# %% [markdown]
# The features are unevenly distributed and should be scaled. For example, Weight and Seed Count have much higher values than the binary feature 'color'.

# %% [markdown]
# ### Visualize target distribution 

# %%
sns.kdeplot(y_train, fill=True, color='lightpink')
plt.title("Density Plot of Target Variable")
plt.xlabel("Scoville Heat Units (SHU)")
plt.show()

# %% [markdown]
# In the plot above we can see how the target variable is distributed for the samples in the training set. 

# %% [markdown]
# ## Modelling

# %% [markdown]
# ### A) Regression analysis

# %% [markdown]
# #### Random forest regression 

# %%
# define the random forest regression pipeline 
pipe_rf = make_pipeline(
    SimpleImputer(strategy = 'median'),
    StandardScaler(),
    RandomForestRegressor())

pipe_rf

# %%
# get the parameters of the pipeline 
pipe_rf.get_params()

# %%
# perform grid search for random forest regression 
gs_rf = GridSearchCV(pipe_rf, 
                     param_grid = {'randomforestregressor__n_estimators': [100,200], 
                                   'randomforestregressor__max_depth': [None, 20],
                                   'randomforestregressor__min_samples_split': [2, 5, 10], 
                                   'randomforestregressor__max_features': ['sqrt']},
                     cv = 5, 
                     scoring = 'neg_mean_absolute_error')

# fit the model 
gs_rf.fit(X_train, y_train)

# print the best parameters and score 
print(f'The best model parameters are: {gs_rf.best_params_}') 
print(f'The best negative mean error score is: {-gs_rf.best_score_}') 

best_model_rf = gs_rf.best_estimator_

# %% [markdown]
# The classifier found to have the best performance on the data set in task A, was the Random Forest Regressor. 
# 
# The model hyperparameters was tuned using GridSearchCV with 5-fold cross-validation. 
# 
# The best model parameters are displayed in the output above. 

# %% [markdown]
# ### B) Multi-class classification analysis with an ensemble classifier

# %% [markdown]
# Binning the target values was not found efficient during our testing, as the accuracy dropped exponentially with increasing number of bins.

# %% [markdown]
# ### C) A two step analysis 

# %% [markdown]
# #### Convert target variable to a binary variable 

# %%
# convert the target variable to binary classification (0 or 1)
y_binary = (y_train > 0).astype(int)
print(y_binary.head(6))

# 1 is hot peppers 
# 0 is not hot peppers 

# %% [markdown]
# #### Train a Random Forest Classifier on the new target 

# %%
# define the pipeline 
pipe_rfc = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    RandomForestClassifier())

# perform grid search for random forest classifier 
gs_rfc = GridSearchCV(pipe_rfc, 
                       param_grid={'randomforestclassifier__n_estimators': [100, 200], 
                                   'randomforestclassifier__max_depth': [None, 20],
                                   'randomforestclassifier__min_samples_split': [2, 5, 10], 
                                   'randomforestclassifier__max_features': ['sqrt']},
                       cv=5, 
                       scoring='accuracy', 
                       n_jobs=-1)

# fit the model 
gs_rfc.fit(X_train, y_binary) 

# print the best parameters and score
print(f'The best model parameters are: {gs_rfc.best_params_}')
print(f'The best accuracy score is: {gs_rfc.best_score_}')

best_model_rfc = gs_rfc.best_estimator_ 

# %% [markdown]
# #### Predict spicy (1) versus non spicy (0) on the test set 

# %%
is_spicy = best_model_rfc.predict(test_data_enc)  

# %% [markdown]
# #### Create a regression model to estimate the SHU score for the spicy peppers 

# %%
# train regression model on the spicy samples from the training set 
spicy_train = y_binary == 1 
X_train_spicy = X_train[spicy_train]
y_train_spicy = y_train[spicy_train]

# define the random forest regressor pipeline 
pipe_rf_reg = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    RandomForestRegressor())

# perform grid search for random forest regressor
gs_rf_reg = GridSearchCV(pipe_rf_reg, 
                          param_grid={'randomforestregressor__n_estimators': [100, 200], 
                                      'randomforestregressor__max_depth': [None, 20],
                                      'randomforestregressor__min_samples_split': [2, 5, 10], 
                                      'randomforestregressor__max_features': ['sqrt']},
                          cv=5, 
                          scoring='neg_mean_absolute_error', 
                          n_jobs=-1)

# fit the model
gs_rf_reg.fit(X_train_spicy, np.log1p(y_train_spicy)) 

print(f'The best model parameters are: {gs_rf_reg.best_params_}')
print(f'The best negative mean error score is: {-gs_rf_reg.best_score_}')

best_model_rf_reg = gs_rf_reg.best_estimator_

# %% [markdown]
# Since the distribution of SHU is quite skewed, y was log-transformed before fitting. Therefore the MAE score obtained from the Grid Search is also log-transformed, the actual MAE score can be seen in Kaggle.

# %%
sns.histplot(y_train_spicy, kde = True, bins=30, color='lightpink', alpha=0.7)
plt.title("SHU Distribution (Spicy Only)")
plt.show()

# %% [markdown]
# Task C gave us the best performing model, where a Random Forest Classifier first was trained to predict the new target of not spicy (0) and spicy (1) peppers. The best model parameters were: 
# - 'randomforestclassifier__max_depth': None
# - 'randomforestclassifier__max_features': 'sqrt'
# - 'randomforestclassifier__min_samples_split': 5
# - 'randomforestclassifier__n_estimators': 200
# 
# Then a Random Forest Regressor was used to predict the SHU value of the spciy peppers. The best parameters found were:
# - 'randomforestregressor__max_depth': 20
# - 'randomforestregressor__max_features': 'sqrt'
# - 'randomforestregressor__min_samples_split': 2 
# - 'randomforestregressor__n_estimators': 200

# %% [markdown]
# #### Predict the SHU score for the test data and submit to Kaggle

# %%
# create empty array for SHU predictions
full_pred_array = np.zeros(len(test_data_enc))

# predict SHU (log scale) for only the spicy samples
spicy_indices = np.where(is_spicy == 1)[0]
spicy_test_data = test_data_enc.iloc[spicy_indices]

log_shu_pred = best_model_rf_reg.predict(spicy_test_data)
shu_pred = np.expm1(log_shu_pred) # take the inverse of np.log1p

# fill the predicted SHU for spicy samples
full_pred_array[spicy_indices] = shu_pred

# format into DataFrame
pred = pd.DataFrame(full_pred_array, columns=['Scoville Heat Units (SHU)'])
pred.index = test_data_enc.index
pred.index.name = "index"
pred.to_csv('submission.csv', index=True)


