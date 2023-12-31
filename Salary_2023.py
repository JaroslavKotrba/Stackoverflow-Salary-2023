# SALARY APP 2023
# https://insights.stackoverflow.com/survey
# shiny

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
import matplotlib.pyplot as plt
import seaborn as sn

import os
path = "/Users/HP/OneDrive/Documents/PythonAnaconda/Shiny_Salary_App/Salary_2023"
os.chdir(path)
os.listdir()

import warnings
warnings.filterwarnings("ignore")

# Import
data = pd.read_csv('survey_results_public_2023.csv')
data.columns
data.shape
data

# Select
df = data[['ConvertedCompYearly', 'Employment', 'Country', 'EdLevel', 'YearsCode', 'YearsCodePro', 'DevType', 'OrgSize', 'OpSysProfessional use', 'Age', 'LanguageHaveWorkedWith', 'NEWCollabToolsHaveWorkedWith']] # 'DatabaseHaveWorkedWith'

# $Salary
df.rename({'ConvertedCompYearly':'Salary'}, axis=1, inplace=True)
df = df[df["Salary"].notnull()] # Salary cannot be null

# Changes
df = df[(df['Salary'] <= 520000) & (df['Salary'] >= 12000)].sort_values('Salary', ascending=False)

# $Employment
df = df[df["Employment"] == "Employed, full-time"]
df = df.drop("Employment", axis=1)
df.info()

# $Country
df['Country'].value_counts()

df['Country'] = df['Country'].str.replace('United Kingdom of Great Britain and Northern Ireland', 'UK')
df['Country'] = df['Country'].str.replace('United States of America', 'USA')
df['Country'] = df['Country'].str.replace('Russian Federation', 'Russia')
df['Country'] = df['Country'].str.replace('Iran, Islamic Republic of...', 'Iran')
df['Country'] = df['Country'].str.replace('Venezuela, Bolivarian Republic of...', 'Venezuela')
df['Country'] = df['Country'].str.replace('The former Yugoslav Republic of Macedonia', 'Macedonia')
df['Country'] = df['Country'].str.replace('United Republic of Tanzania', 'Tanzania')

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

country_map = shorten_categories(df.Country.value_counts(), 50)
df['Country'] = df['Country'].map(country_map)

df = df[df['Country'] != 'Other']

# column = 'Country'
def viz(column):
    sum_row = pd.DataFrame(df[column].value_counts()).reset_index().rename(columns={'count':'Count'})

    plt.figure(figsize=(12,8))
    plt.bar(sum_row[column], sum_row['Count'])
    plt.xticks(rotation='vertical')

viz('Country');

# $Education
df['EdLevel'].value_counts()

def clean_education(x):
    if pd.isna(x): # np.isnan(x) df[df['EdLevel'].isna()]
        return 'Less than a Bachelor'
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelor'
df['EdLevel'] = df['EdLevel'].apply(clean_education)

viz('EdLevel');

# $YearsCode
df['YearsCode'].value_counts()

def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)
df['YearsCode'] = df['YearsCode'].apply(clean_experience)

viz('YearsCode');

# $YearsCodePro
df['YearsCodePro'].value_counts()

def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)
df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)

viz('YearsCodePro');

# $DevType
df['DevType'].value_counts().head(20)

df['DevType'] = df['DevType'].str.split(';').str.get(0)

def clean_developer(x):
    if pd.isna(x): 
        return 'Developer, not-specified'
    if 'Other (please specify):' in x:
        return 'Developer, not-specified'
    return x
df['DevType'] = df['DevType'].apply(clean_developer)

# $OrgSize
df['OrgSize'].value_counts()

df['OrgSize'] = np.where(df['OrgSize'] == 'I don’t know', 'Just me - I am a freelancer, sole proprietor, etc.', df['OrgSize'])

sum_row = df.groupby('OrgSize').agg({'Salary':'mean'}).reset_index().round().sort_values('Salary', ascending=False); sum_row

plt.figure(figsize=(12,8))
plt.bar(sum_row['OrgSize'], sum_row['Salary'])
plt.xticks(rotation='vertical')

# $OpSys
df.rename({'OpSysProfessional use':'OpSys'}, axis=1, inplace=True)
df['OpSys'].value_counts()

def clean_os(x):
    if pd.isna(x): # np.isnan(x) df[df['EdLevel'].isna()]
        return 'Other'
    if 'Windows' in x:
        return 'Windows'
    if 'MacOS' in x:
        return 'MacOS'
    if 'Linux-based' in x or 'Windows Subsystem for Linux' in x:
        return 'Linux-based'
    return 'Other'
df['OpSys'] = df['OpSys'].apply(clean_os)

# $Age
df['Age'].value_counts()

def clean_age(x):
    if x == '65 years or older':
        return '65+'
    if x == '55-64 years old':
        return '55-64'
    if x == '45-54 years old':
        return '45-54'
    if x == '35-44 years old':
        return '35-44'
    if x == '25-34 years old':
        return '25-34'
    if x == '18-24 years old':
        return '18-24'
    if x == 'Under 18 years old':
        return 'Under 18'
    if x == 'Prefer not to say':
        return '35-44 years old'
    return float(x)
df['Age'] = df['Age'].apply(clean_age)

df['Age'] = np.where(df['Age'] == '35-44 years old', '35-44', df['Age'])

# $LanguageHaveWorkedWith
df['LanguageHaveWorkedWith'].value_counts()

df = pd.concat([df, df['LanguageHaveWorkedWith'].str.get_dummies(sep=';')], axis=1); df
df.drop(columns=['LanguageHaveWorkedWith'], inplace=True)

sum_row = pd.DataFrame({'Count':df.iloc[:, 13:].sum()}).sort_values('Count', ascending=False).reset_index()
sum_row.index = np.arange(1, len(sum_row) + 1)
sum_row.rename(columns={'index':'Language'}, inplace=True); sum_row

plt.figure(figsize=(12,8))
plt.bar(sum_row['Language'], sum_row['Count'])
plt.xticks(rotation='vertical');

# $NEWCollabToolsHaveWorkedWith
df['NEWCollabToolsHaveWorkedWith'].value_counts()
df.rename({'NEWCollabToolsHaveWorkedWith':'CollabToolsHaveWorkedWith'}, axis=1, inplace=True)

df = pd.concat([df, df['CollabToolsHaveWorkedWith'].str.get_dummies(sep=';')], axis=1); df
df.drop(columns=['CollabToolsHaveWorkedWith'], inplace=True)

sum_row = pd.DataFrame({'Count':df.iloc[:, 68:].sum()}).sort_values('Count', ascending=False).reset_index()
sum_row.index = np.arange(1, len(sum_row) + 1)
sum_row.rename(columns={'index':'IDE'}, inplace=True); sum_row

plt.figure(figsize=(12,8))
plt.bar(sum_row['IDE'], sum_row['Count'])
plt.xticks(rotation='vertical');

# Outliers categorical q (extreme outliers)
def remove_outliers(df):
    out = pd.DataFrame()
    for key, subset in df.groupby('Country'):
        q1 = subset.Salary.quantile(.25)
        q3 = subset.Salary.quantile(.75)
        iqr = q3 - q1
        reduced_df = subset[(subset.Salary>(q1-(1.5*iqr))) & (subset.Salary<=(q3+(1.5*iqr)))]
        out = pd.concat([out, reduced_df], ignore_index=True)
    return out
df = remove_outliers(df)

# Salary histogram
df.hist(column='Salary', bins=25, grid=True, figsize=(12,8), color='#86bf91', rwidth=0.8);

# Visualisation by country
fig, ax = plt.subplots(1,1,figsize=(16,12))
df.boxplot('Salary', 'Country', ax=ax, color=dict(boxes='b', whiskers='b', medians='r', caps='b'))
plt.suptitle('Salary (USD) by Country')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()

# Visualisation by country plotly
import plotly.express as px
fig = px.box(df.sort_values('Country'), x="Country", y="Salary", 
                                template='simple_white',
                                title=f"<b>Global</b> - Salary based on country (USD)")
fig.update_xaxes(tickangle=90, tickmode = 'array', tickvals = df.sort_values('Country')['Country'])
fig.show()

# Visualisation by experience plotly
import plotly.express as px

object = "Czech Republic"

country = df[df['Country'] == object]
country['EdLevel'] = pd.Categorical(country['EdLevel'], ['Less than a Bachelor', 'Bachelor’s degree', 'Master’s degree', 'Post grad'])
country = country.sort_values('EdLevel')

fig = px.scatter(country, x="YearsCodePro", y="Salary", trendline="ols", color="EdLevel", symbol='EdLevel', opacity=0.8,
                                marginal_x="histogram", 
                                marginal_y="rug",
                                template='simple_white', hover_data=['Salary', 'YearsCodePro', 'Age', 'OrgSize'],
                                title=f" <b>{object}</b> - Salary based on experience & education (USD)")
fig.update_traces(marker=dict(size=8, line=dict(width=1, color='black')), selector=dict(mode='markers'))
fig.show()

# Table with mean and median
table = df.groupby('Country')[['Salary', 'YearsCodePro']].mean().round()
table = table.rename(columns={"Salary" : "SalaryMean", "YearsCodePro" : "YearsCodeProMean"})

table = table.join(df.groupby('Country')[["Salary", "YearsCodePro"]].median().round(), on=['Country']) # on index
table = table.rename(columns={"Salary" : "SalaryMedian", "YearsCodePro" : "YearsCodeProMedian"})

table = table.join(df.groupby('Country').count()[['Salary']], on=['Country']) # on index
table = table.rename(columns={"Salary" : "Count"})

table['Country'] = table.index # index into column
table = table[['Country', 'Count', 'SalaryMean','SalaryMedian', "YearsCodeProMean", "YearsCodeProMedian"]].sort_values('SalaryMean', ascending=False).reset_index(drop=True)

table

# Remove NAs
print(df.isnull().sum())
df = df.dropna()

# Droping columns (multicollinearity)
df.drop(columns=['YearsCode'], inplace=True)

# Droping IDE
df.drop(columns=['Android Studio',
        'Atom', 'BBEdit', 'CLion', 'Code::Blocks', 'DataGrip', 'Eclipse',
       'Emacs', 'Fleet', 'Geany', 'Goland', 'Helix', 'IPython',
       'IntelliJ IDEA', 'Jupyter Notebook/JupyterLab', 'Kate', 'Micro', 'Nano',
       'Neovim', 'Netbeans', 'Notepad++', 'Nova', 'PhpStorm', 'PyCharm',
       'Qt Creator', 'RStudio', 'Rad Studio (Delphi, C++ Builder)', 'Rider',
       'RubyMine', 'Spyder', 'Sublime Text', 'TextMate', 'VSCodium', 'Vim',
       'Visual Studio', 'Visual Studio Code', 'WebStorm', 'Xcode', 'condo'], inplace=True)

# SAVE
df.to_csv("survey_clean.csv", index=False)

# ---------------------------------------------------------------------------------------------------------------
# 1. Modeling pipe based ----------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
import matplotlib.pyplot as plt
import seaborn as sn

import os
path = "/Users/HP/OneDrive/Documents/PythonAnaconda/Shiny_Salary_App/Salary_2023"
os.chdir(path)
os.listdir()

import warnings
warnings.filterwarnings("ignore")

# Import
df = pd.read_csv('survey_clean.csv'); df

# Splitting
X = df.drop(columns=['Salary']); X
y = df.Salary; y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Preprocessing
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
column_trans = make_column_transformer((OneHotEncoder(sparse=False),
['Country', 'EdLevel', 'DevType', 'OrgSize', 'OpSys', 'Age']), # non-numeric
remainder='passthrough')

# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# ---------------------------------------------------------------------------------------------------------------
# Pipe & Sample -------------------------------------------------------------------------------------------------

def print_regression_metrics(model, X_train, X_test, y_train, y_test):
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(column_trans, scaler, model)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    outcome = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})
    outcome['difference'] = outcome['y_test'] - outcome['y_pred']
    outcome['difference_percentage'] = round(outcome.difference/(outcome.y_test/100),6)

    model_name = str(model).split("(")[0]
    if "Regression" in model_name:
        model_name = model_name.replace("Regression", "")

    print(f"{model_name} regression model:")
    print('PROC: ', round(outcome.difference_percentage.abs().mean(),2),'%')
    print('MAE: ', round(mean_absolute_error(y_test, y_pred),4))
    print('RMSE: ', round(np.sqrt(mean_squared_error(y_test, y_pred)),4))
    print('R2:', round(r2_score(y_test, y_pred),4))

    return pipe

# Sample
def sample(pipe, X_test):
    X_sample = np.array(["Czech Republic", "Master’s degree", "3", "Developer, back-end", "10,000 or more employees", "Windows", "25-34",
                           "0",   "0",   "0",     "0",                 "1",           "0",  "1",  "0",     "0",      "0",      "0",      "0",     "0",      "0",      "0",    "0",   "0",     "0",       "0",     "0",    "0",       "1",       "0",       "0",       "0",       "0",     "0",      "0",   "0",   "0",     "0",    "0",        "0",       "0",   "0",       "0",         "0",     "1",    "1",  "0",    "0",    "0",    "0",   "1",    "0",      "0",       "0",        "0",      "0",            "0",           "0"])
                        # 'APL', 'Ada', 'Apex' 'Assembly', 'Bash/Shell (all shells)', 'C', 'C#', 'C++', 'Clojure', 'Cobol', 'Crystal', 'Dart', 'Delphi', 'Elixir', 'Erlang', 'F#', 'Flow', 'Fortran', 'GDScript', 'Go', 'Groovy', 'HTML/CSS', 'Haskell', 'Java', 'JavaScript', 'Julia', 'Kotlin', 'Lisp', 'Lua' 'MATLAB', 'Nim', 'OCaml', 'Objective-C', 'PHP', 'Perl', 'PowerShell', 'Prolog', 'Python', 'R', 'Raku', 'Ruby', 'Rust', 'SAS', 'SQL', 'Scala', 'Solidity', 'Swift', 'TypeScript', 'VBA', 'Visual Basic (.Net)', 'Zig'

    X_sample = pd.DataFrame(X_sample.reshape(1,-1))
    X_sample.columns = X_test.columns

    y_pred = pipe.predict(X_sample)
    y_pred = np.where(y_pred < 0, 0, y_pred) # not to be negative

    print(f"Sample: {round(y_pred[0])} USD")

# ---------------------------------------------------------------------------------------------------------------
# MODEL Linear regression ---------------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
model = LinearRegression()

pipe = print_regression_metrics(model, X_train, X_test, y_train, y_test)

sample(pipe, X_test)

# ---------------------------------------------------------------------------------------------------------------
# MODEL Lasso regression ----------------------------------------------------------------------------------------

from sklearn.linear_model import Lasso
model = Lasso()

pipe = print_regression_metrics(model, X_train, X_test, y_train, y_test)

sample(pipe, X_test)

# ---------------------------------------------------------------------------------------------------------------
# MODEL Ridge regression ----------------------------------------------------------------------------------------

from sklearn.linear_model import Ridge
model = Ridge()

pipe = print_regression_metrics(model, X_train, X_test, y_train, y_test)

sample(pipe, X_test)

# ---------------------------------------------------------------------------------------------------------------
# MODEL Decision tree regression --------------------------------------------------------------------------------

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=0)

pipe = print_regression_metrics(model, X_train, X_test, y_train, y_test)

sample(pipe, X_test)

# ---------------------------------------------------------------------------------------------------------------
# MODEL Random forest regression --------------------------------------------------------------------------------

import timeit
start = timeit.default_timer()

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(
    n_estimators=256,               # Number of trees in the forest.
    criterion='friedman_mse',       # The function to measure the quality of a split: 'absolute_error', 'poisson', 'friedman_mse', 'squared_error'
    max_depth=None,                 # The maximum depth of the tree.
    min_samples_split=2,            # The minimum number of samples required to split an internal node.
    min_samples_leaf=1,             # The minimum number of samples required to be at a leaf node.
    min_weight_fraction_leaf=0.0,   # The minimum weighted fraction of the sum total of weights required to be at a leaf node.
    max_features='auto',            # The number of features to consider when looking for the best split.
    max_leaf_nodes=None,            # Grow trees with max_leaf_nodes in best-first fashion.
    min_impurity_decrease=0.0,      # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    bootstrap=True,                 # Whether bootstrap samples are used when building trees.
    oob_score=False,                # Whether to use out-of-bag samples to estimate the R^2 on unseen data.
    n_jobs=None,                    # The number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context.
    random_state=0,                 # Controls both the randomness of the bootstrapping of the samples used when building trees.
    verbose=0,                      # Controls the verbosity when fitting and predicting.
    warm_start=False,               # When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble.
    ccp_alpha=0.0,                  # Complexity parameter used for Minimal Cost-Complexity Pruning.
    max_samples=None                # If bootstrap is True, the number of samples to draw from X to train each base estimator.
)

pipe = print_regression_metrics(model, X_train, X_test, y_train, y_test)

sample(pipe, X_test)

stop = timeit.default_timer()
print('Time: ', stop - start)

# 2 min calculation

# ---------------------------------------------------------------------------------------------------------------
# MODEL XGBoost -------------------------------------------------------------------------------------------------

# Initialize model before tunning
import xgboost as xgb
model = xgb.XGBRegressor(
    objective='reg:squarederror',  # Objective function to be used
    n_estimators=128,              # Number of gradient boosted trees
    max_depth=6,                   # Maximum tree depth for base learners
    learning_rate=0.3,             # Boosting learning rate
    gamma=0,                       # Minimum loss reduction required to make a further partition on a leaf node - prevent overfitting
    min_child_weight=1,            # Minimum sum of instance weight (hessian) needed in a child
    subsample=1,                   # Subsample ratio of the training instance
    colsample_bytree=1,            # Subsample ratio of columns when constructing each tree
    colsample_bylevel=1,           # Subsample ratio of columns for each level
    colsample_bynode=1,            # Subsample ratio of columns for each split
    reg_alpha=1,                   # L1 regularization term on weights
    reg_lambda=1,                  # L2 regularization term on weights
    scale_pos_weight=1,            # Balancing of positive and negative weights
    base_score=0.5,                # The initial prediction score of all instances, global bias
    booster='gbtree',              # Which booster to use: gbtree, gblinear or dart
    random_state=0                 # Random number seed for reproducibility
)

pipe = print_regression_metrics(model, X_train, X_test, y_train, y_test)

sample(pipe, X_test)

# Tunning
def tunning_model(model, param_grid, X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=1)

    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(column_trans, scaler, grid_search)
    pipe.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")

    return best_params

# Define the parameter grid
param_grid = {
    'n_estimators': [120, 180, 240],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.3],
}

# The best params
best_params = tunning_model(model, param_grid, X_train, y_train)

# Use the best params
model = xgb.XGBRegressor(**best_params)

pipe = print_regression_metrics(model, X_train, X_test, y_train, y_test)

sample(pipe, X_test)

# Shap
def shap_model(pipe, X_train, X_test):
    import shap
    xgb_model = pipe.named_steps['xgbregressor']
    X_encoded = pipe.named_steps['columntransformer'].transform(X_test)
    X_encoded = pipe.named_steps['standardscaler'].transform(X_encoded)

    explainer = shap.Explainer(xgb_model)
    shap_values = explainer.shap_values(X_encoded)

    categorical_columns = ['Country', 'EdLevel', 'DevType', 'OrgSize', 'OpSys', 'Age']
    X_train_dummies = pd.get_dummies(X_train, columns=categorical_columns)

    shap.summary_plot(shap_values, X_encoded, feature_names=X_train_dummies.columns, max_display = 40)

shap_model(pipe, X_train, X_test)

# ---------------------------------------------------------------------------------------------------------------
# MODEL LightGBM ------------------------------------------------------------------------------------------------

# conda install lightgbm
import lightgbm as lgb
model = lgb.LGBMRegressor(
    boosting_type='gbdt',            # The boosting type (gbdt, rf, dart, goss)
    num_leaves=31,                   # Maximum tree leaves for base learners
    max_depth=-1,                    # Maximum tree depth for base learners, <=0 means no limit
    learning_rate=0.05,              # Boosting learning rate
    n_estimators=128,                # Number of boosted trees to fit
    subsample_for_bin=200000,        # Number of samples for constructing bins
    objective='regression',          # Specify the learning task and the corresponding learning objective
    class_weight=None,               # Weights associated with classes in the form {class_label: weight}
    min_split_gain=0.0,              # Minimum loss reduction required to make a further partition
    min_child_weight=0.001,          # Minimum sum of instance weight (hessian) needed in a child
    min_child_samples=20,            # Minimum number of data needed in a child
    subsample=1.0,                   # Subsample ratio of the training instance
    subsample_freq=0,                # Frequency of subsample, <=0 means no enable
    colsample_bytree=1.0,            # Subsample ratio of columns when constructing each tree
    reg_alpha=0.0,                   # L1 regularization term on weights
    reg_lambda=0.0,                  # L2 regularization term on weights
    random_state=None,               # Random number seed
    n_jobs=-1,                       # Number of parallel threads
    importance_type='split',         # Type of feature importance to be filled into feature_importances_
    force_row_wise=True              # You can set `force_row_wise=true` to remove the overhead.
)

pipe = print_regression_metrics(model, X_train, X_test, y_train, y_test)

sample(pipe, X_test)

# ---------------------------------------------------------------------------------------------------------------
# MODEL CatBoost ------------------------------------------------------------------------------------------------

# conda install catboost
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    iterations=128,               # Equivalent to n_estimators in XGBoost
    depth=6,                      # Equivalent to max_depth
    learning_rate=0.3,            # Same as in XGBoost
    l2_leaf_reg=1,                # Equivalent to reg_lambda
    border_count=254,             # Similar to max_bin in XGBoost
    random_seed=0,                # Equivalent to random_state
    verbose=False                 # To keep the output clean
)

pipe = print_regression_metrics(model, X_train, X_test, y_train, y_test)

sample(pipe, X_test)
