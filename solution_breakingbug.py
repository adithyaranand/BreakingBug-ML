import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from yellowbrick.cluster import KElbowVisualizer
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#sklearn.model_selection contains train_test_split. It should be GridSearchCV and cross_val_score
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score

from sklearn.linear_model import LogisticRegression #It should be LogisticRegression
from sklearn.neighbors import KNeighborsClassifier #KNN should be KNeighborsClassifier
from sklearn.svm import SVC #SVC_Classifier should be SVC
from sklearn.tree import  DecisionTreeClassifier, plot_tree #DecisionTree should be either DecisionTreeClassifier and plot_tree_regressor should be plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestRegressor #RandomForestRegressor, AdaBoostClassifier, and GradientBoostingClassifier should be used for classification, while RandomForestClassifier, AdaBoostRegressor, and GradientBoostingRegressor are for regression tasks.
from xgboost import XGBClassifier #XG should be XGBClassifier
from lightgbm import LGBMClassifier #LGBM should be LGBMClassifier
from sklearn.naive_bayes import GaussianNB #Gaussian should be GaussianNB for Naive Bayes

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#accuracy, confusion, classification should be accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("/content/dataset.csv")

df.head()

df.info()

df.shape

df['id'].min(), df['id'].max()

df['age'].min(), df['age'].max()

df['age'].describe()

import seaborn as sns
custom_colors = ["#FF5733", "#3366FF", "#33FF57"]
sns.histplot(df['age'], kde=True, color="#FF5733", palette=custom_colors)

sns.histplot(df['age'], kde=True)
plt.axvline(df['age'].mean(), color='Red')
plt.axvline(df['age'].median(), color= 'Green')
plt.axvline(df['age'].mode()[0], color='Blue')

print('Mean', df['age'].mean())
print('Median', df['age'].median())
print('Mode', df['age'].mode())

fig = px.histogram(data_frame=df, x='age', color= 'sex')
fig.show()

df['sex'].value_counts()

male_count = 726
female_count = 194
total_count = male_count + female_count

male_percentage = (male_count/total_count)*100
female_percentages = (female_count/total_count)*100

# display the results
print(f'Male percentage i the data: {male_percentage:.2f}%')
print(f'Female percentage in the data : {female_percentages:.2f}%')

difference_percentage = ((male_count - female_count)/female_count) * 100
print(f'Males are {difference_percentage:.2f}% more than female in the data.')

df.groupby('sex')['age'].value_counts()

df['dataset'].value_counts() #dataset and value_counts

fig =px.bar(df, x='dataset', color='sex')
fig.show()

print (df.groupby('sex')['dataset'].value_counts())

fig = px.histogram(data_frame=df, x='age', color= 'dataset')
fig.show()

print("___________________________________________________________")
print("Mean of the dataset: ",df.groupby('dataset')['age'].mean())
print("___________________________________________________________")
print("Median of the dataset: ",df.groupby('dataset')['age'].median())
print("___________________________________________________________")
print("Mode of the dataset: ",df.groupby('dataset')['age'].agg(pd.Series.mode))
print("___________________________________________________________")
#You should use df.groupby('dataset')['age'].mean(), median(), and mode() to calculate and data should be dataset. Also need to use agg for calculating mode.

df['cp'].value_counts()

sns.countplot(df, x='cp', hue= 'sex')

sns.countplot(df,x='cp',hue='dataset')

fig = px.histogram(data_frame=df, x='age', color='cp')
fig.show()

df['trestbps'].describe()

print(f"Percentage of missing values in trestbps column: {df['trestbps'].isnull().sum() /len(df) *100:.2f}%")

imputer1 = IterativeImputer(max_iter=10, random_state=42)

imputer1.fit(df[['trestbps']])

df['trestbps'] = imputer1.transform(df[['trestbps']])

print(f"Missing values in trestbps column: {df['trestbps'].isnull().sum()}")

df.info()

(df.isnull().sum()/ len(df)* 100).sort_values(ascending=False)

imputer2 = IterativeImputer(max_iter=10, random_state=42)

df['ca'] = imputer2.fit_transform(df[['ca']])
df['oldpeak']= imputer2.fit_transform(df[['oldpeak']])
df['chol'] = imputer2.fit_transform(df[['chol']])
df['thalch'] = imputer2.fit_transform(df[['thalch']])
#fit_transform should be used and it should be on 2-d array. So, write df[[column]] instead of df[column]

(df.isnull().sum()/ len(df)* 100).sort_values(ascending=False)

print(f"The missing values in thal column are: {df['thal'].isnull().sum()}")

df['thal'].value_counts()

df.tail()

df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=True)
#null should be replaced with isnull() and also values by sort_values, replace < with > and need to add sum()

missing_data_cols = df.isnull().sum()[df.isnull().sum()>0].index.tolist()

missing_data_cols

cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols

Num_cols = df.select_dtypes(exclude='object').columns.tolist()
Num_cols

print(f'categorical Columns: {cat_cols}')
print(f'numerical Columns: {Num_cols}')

categorical_cols = ['thal', 'ca', 'slope', 'exang', 'restecg','thalch', 'chol', 'trestbps']
bool_cols = ['fbs']
numerical_cols = ['oldpeak','age','restecg','fbs', 'cp', 'sex', 'num']

passed_col = categorical_cols

def impute_categorical_missing_data(df, passed_col, bool_cols, missing_data_cols):

    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]

    X = df_not_null.drop(passed_col, axis=1)
    Y = df_not_null[passed_col]
#Replace y with Y
    other_missing_cols = [col for col in missing_data_cols if col != passed_col]

    label_encoder = LabelEncoder()
    for col in Y.columns:
      if Y[col].dtype == 'object' :
        Y[col] = label_encoder.fit_transform(Y[col].astype(str))
#Label Encoder
    if passed_col in bool_cols:
        y = label_encoder.fit_transform(y)
#InterativeImputer to be used
    imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=16), add_indicator=True)
    for cols in other_missing_cols:
            cols_with_missing_value = Y[col].value.reshape(-100, 100)
            imputed_values = imputer.fit_transform(missing_data_cols)
            #missing_data_cols and imputer.fit
            X[col] = imputed_values[:, 0]
    else:
        pass

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier()

    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)

    print("The feature '"+ passed_col+ "' has been imputed with", round((acc_score * 100), 2), "accuracy\n")

    X = df_null.drop(passed_col, axis=1)

    for cols in Y.columns:
        if Y[col].dtype == 'object' :
            Y[col] = label_encoder.fit_transform(Y[col].astype(str))

    for cols in other_missing_cols:
            cols_with_missing_value = Y[col].value.reshape(-100, 100)
            imputed_values = imputer.fit_transform(missing_data_cols)
            #missing_data_cols and imputer
            X[col] = imputed_values[:, 0]
#Replace passed with passed_col
    if len(df_null) < 0:
        df[passed_col] = rf_classifier.predict(X)
        if passed_col in cols:
            df[passed_col] = df[passed_col].map({0: False, 1: True})
        else:
            pass
    else:
        pass

    df_combined = pd.concat([df_not_null, df_null])

    return df_combined[passed_col]

def impute_continuous_missing_data(df, passed_col, missing_data_cols):

    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]

    X = df_not_null.drop(passed_col, axis=1)
    Y = df_not_null[passed_col]
#Replace y with Y
    other_missing_cols = [col for col in missing_data_cols if col != passed_col]

    label_encoder = LabelEncoder()

    for col in Y.columns:
        if Y[col].dtype == 'object' :
            Y[col] = label_encoder.fit_transform(Y[col].astype(str))
#IterativeImputer to be used
    imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=16), add_indicator=True)

    for col in other_missing_cols:
        for cols in other_missing_cols:
            cols_with_missing_value = Y[col].value.reshape(-100, 100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_regressor = RandomForestRegressor()

    rf_regressor.fit(X_train, y_train)

    y_pred = rf_regressor.predict(X_test)

    print("MAE =", mean_absolute_error(y_test, y_pred), "\n")
    print("RMSE =", mean_squared_error(y_test, y_pred, squared=False), "\n")
    print("R2 =", r2_score(y_test, y_pred), "\n")

    X = df_null.drop(passed_col, axis=1)

    for cols in Y.columns:
        if Y[col].dtype == 'object' :
          Y[col] = label_encoder.fit_transform(Y[col].astype(str))
#label_encoder to be used
    for cols in other_missing_cols:
            cols_with_missing_value = Y[col].value.reshape(-100, 100)
            imputed_values = imputer.transform(missing_data_cols)
            #imputer to be used
            X[col] = imputed_values[:, 0]
    else:
      pass

    if len(df_null) > 0:
        df_not_null[passed_col] = rf_regressor.predict(X_train)
        #passed_col, rf_regressor and Indentation
    else:
        pass

    df_combined = pd.concat([df_not_null, df_null])

    return df_combined[passed_col]

df.isnull().sum().sort_values(ascending=False)

import warnings
warnings.filterwarnings('ignore')

for col in missing_data_cols:
    missing_percentage = (df[col].isnull().sum() / len(df)) * 100
    print(f"Missing Values {col}: {round(missing_percentage, 2)}%")
    """if col in categorical_cols:
      df[col] = impute_categorical_missing_data(df, categorical_cols, bool_cols, col)
    elif col in numerical_cols:
      df[col] = impute_continuous_missing_data(df, numerical_cols, col)
    else:
        pass"""

df.isnull().sum().sort_values(ascending=False)

print("_________________________________________________________________________________________________________________________________________________")
sns.set(rc={"axes.facecolor":"#87CEEB","figure.facecolor":"#EEE8AA"})
palette = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
cmap = ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])
plt.figure(figsize=(10,8))
for i, col in enumerate(col):
    plt.subplot(3,2, i+1)
    sns.boxenplot(color=palette[i % len(palette)])
    plt.title(i)
plt.show()
# subplot(3,2,i+1)

df[df['trestbps']==0]

df= df[df['trestbps']!=0]

sns.set(rc={"axes.facecolor":"#B76E79","figure.facecolor":"#C0C0C0"})
modified_palette = ["#C44D53", "#B76E79", "#DDA4A5", "#B3BCC4", "#A2867E", "#F3AB60"]
cmap = ListedColormap(modified_palette)
plt.figure(figsize=(10,8))
for i, col in enumerate(col):
    plt.subplot(3,2, i+1)
    sns.boxenplot( color=palette[i % len(palette)])
    plt.title(col)
plt.show()
#subplot(i+1), col

df.trestbps.describe()

df.describe()

print("___________________________________________________________________________________________________________________________________________________________________")
sns.set(rc={"axes.facecolor": "#FFF9ED", "figure.facecolor": "#FFF9ED"})
night_vision_palette = ["#00FF00", "#FF00FF", "#00FFFF", "#FFFF00", "#FF0000", "#0000FF"]
plt.figure(figsize=(10, 8))
for i, col in enumerate(col):
    plt.subplot(3,2, i+1)
    sns.boxenplot( color=palette[i % len(palette)])
    plt.title(col)
plt.show()

df.age.describe()

palette = ["#999999", "#666666", "#333333"]

sns.histplot(data=df,
             x='trestbps',
             kde=True,
             color=palette[0])

plt.title('Resting Blood Pressure')
plt.xlabel('Pressure (mmHg)')
plt.ylabel('Count')

plt.style.use('default')
plt.rcParams['figure.facecolor'] = palette[1]
plt.rcParams['axes.facecolor'] = palette[2]

sns.histplot(df, x='trestbps', kde=True, palette = "Spectral", hue ='sex')

df.info()

df.columns

df.head()

X= df.drop('num', axis=1)
y = df['num']

Label_Encoder = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = Label_Encoder.fit_transform(X[col].astype(str))
    else:
        pass
#X[col], col, Label_Encode should be changed

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

from sklearn.linear_model import LogisticRegression #It should be LogisticRegression
from sklearn.neighbors import KNeighborsClassifier #KNN should be KNeighborsClassifier
from sklearn.svm import SVC #SVC_Classifier should be SVC
from sklearn.tree import  DecisionTreeClassifier, plot_tree #DecisionTree should be either DecisionTreeClassifier and plot_tree_regressor should be plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier #RandomForestRegressor, AdaBoostClassifier, and GradientBoostingClassifier should be used for classification, while RandomForestClassifier, AdaBoostRegressor, and GradientBoostingRegressor are for regression tasks.
from xgboost import XGBClassifier #XG should be XGBClassifier
from lightgbm import LGBMClassifier #LGBM should be LGBMClassifier
from sklearn.naive_bayes import GaussianNB #Gaussian should be GaussianNB for Naive Bayes
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

models = [
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('KNeighbors Classifier', KNeighborsClassifier()),
    ('Decision Tree Classifier', DecisionTreeClassifier(random_state=42)),
    ('AdaBoost Classifier',AdaBoostClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('XGboost Classifier', XGBClassifier(random_state=42)),

    ('Support Vector Machine', SVC(random_state=42)),

    ('Naye base Classifier', GaussianNB())


]
#Logistic Regression, KNeighborsClassifier, RandomForestClassifier, GaussianNB, AdaBoostClassifier, DecisionTreeClassifier, GradientBoostingClassifier names should be mentioned correctly and random should be replaced by random_state

best_model = None
best_accuracy = 0.0

for name, model in models:
    pipeline = Pipeline([
        ('model',model)
    ])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    mean_accuracy = scores.mean()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model", model)
    print("Cross Validatino accuracy: ", mean_accuracy)
    print("Test Accuracy: ", accuracy)
    print()
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline

# name replaced by model, cross_val, x_train and y_train should be used for cross_val, mean() instead of avg()

print("Best Model: ", best_model)

categorical_cols = ['thal', 'ca', 'slope', 'exang', 'restecg','fbs', 'cp', 'sex', 'num']

def evaluate_classification_models(X, y, categorical_columns):
    X_encoded = X.copy()
    label_encoders = {}
    for col in categorical_columns:
      X_encoded[col] = Label_Encoder.fit_transform(X_encoded[col])

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "NB": GaussianNB(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier()
    }
    #All Models should be named correctly
    results = {}
    best_model = None
    best_accuracy = 0.0
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = name

    return results, best_model

X = df[categorical_cols]
y = df['num']
results, best_model = evaluate_classification_models(X, y, categorical_cols)
print("Model accuracies:", results)
print("Best model:", best_model)

def hyperparameter_tuning(X, y, categorical_columns, models):
    results = {}
    X_encoded = X.copy()
    for col in categorical_columns:
        X_encoded[col] = Label_Encoder.fit_transform(X_encoded[col])
        #Label encoder to be used and X_encoded
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    for model_name, model in models.items():
        param_grid = {}
    if model_name == 'Logistic Regression':
        param_grid = {'C': [0.1, 1, 10, 100]}
    elif model_name == 'KNN':
        param_grid = {'n_neighbors': [3, 5, 7, 9]}
    elif model_name == 'NB':
        param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}
    elif model_name == 'SVM':
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100]}
    elif model_name == 'Decision Tree':
        param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
    elif model_name == 'Random Forest':
        param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
    elif model_name == 'XGBoost':
        parameter_grid = {'learning_rates': [0.01, 0.1, 0.2], 'num_estimators': [100, 200, 300], 'depths': [3, 5, 7]}
    elif model_name == 'GradientBoosting':
        parameter_grid = {'learning_rates': [0.01, 0.1, 0.2], 'num_estimators': [100, 200, 300], 'depths': [3, 5, 7]}
    elif model_name == 'AdaBoost':
        param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200]}



        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results[model_name] = {'best_params': best_params, 'accuracy': accuracy}

    return results

models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "NB": GaussianNB(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

results = hyperparameter_tuning(X, y, categorical_cols, models)
for model_name, result in results.items():
    print("Model:", model_name)
    print("Best hyperparameters:", result['best_params'])
    print("Accuracy:", result['accuracy'])
    print()