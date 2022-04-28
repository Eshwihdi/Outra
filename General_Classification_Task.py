#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
# To show the full dataframe:
pd.set_option("display.max_rows", None, "display.max_columns", None)


# # Klib Data Cleaning
# https://klib.readthedocs.io/en/latest/
# 

# In[2]:


# For GCP:
# !pip install --target=$nb_path klib
# Normal:
# !pip install klib 
import klib


# import klib
# 
# import pandas as pd
# 
# df = pd.DataFrame(data)
# 
# # klib.describe - functions for visualizing datasets
# - klib.cat_plot(df) # returns a visualization of the number and frequency of categorical features
# - klib.corr_mat(df) # returns a color-encoded correlation matrix
# - klib.corr_plot(df) # returns a color-encoded heatmap, ideal for correlations
# - klib.dist_plot(df) # returns a distribution plot for every numeric feature
# - klib.missingval_plot(df) # returns a figure containing information about missing values
# 
# # klib.clean - functions for cleaning datasets
# - klib.data_cleaning(df) # performs datacleaning (drop duplicates & empty rows/cols, adjust dtypes,...)
# - klib.clean_column_names(df) # cleans and standardizes column names, also called inside data_cleaning()
# - klib.convert_datatypes(df) # converts existing to more efficient dtypes, also called inside data_cleaning()
# - klib.drop_missing(df) # drops missing values, also called in data_cleaning()
# - klib.mv_col_handling(df) # drops features with high ratio of missing vals based on informational content
# - klib.pool_duplicate_subsets(df) # pools subset of cols based on duplicates with min. loss of information
# 
# # klib.preprocess - functions for data preprocessing (feature selection, scaling, ...)
# - klib.train_dev_test_split(df) # splits a dataset and a label into train, optionally dev and test sets
# - klib.feature_selection_pipe() # provides common operations for feature selection
# - klib.num_pipe() # provides common operations for preprocessing of numerical data
# - klib.cat_pipe() # provides common operations for preprocessing of categorical data
# - klib.preprocess.ColumnSelector() # selects num or cat columns, ideal for a Feature Union or Pipeline
# - klib.preprocess.PipeInfo() # prints out the shape of the data at the specified step of a Pipeline

# In[3]:


import klib
print('klib version: ' + klib.__version__)


# In[4]:


#Change working directory on Jupiter NoteBook
import os
print('Path at terminal Before:')
print(os.getcwd() + "\n")


# ## Using GCP :

# In[8]:


get_ipython().system('pwd')


# In[9]:


get_ipython().run_line_magic('ls', '')


# In[ ]:


get_ipython().run_line_magic('cd', '..')
get_ipython().run_line_magic('ls', '')


# In[ ]:


get_ipython().run_line_magic('cd', 'content/')
get_ipython().run_line_magic('ls', '')


# In[ ]:


get_ipython().run_line_magic('cd', 'content')
get_ipython().run_line_magic('ls', '')


# In[ ]:


# Access Google Drive:
from google.colab import drive 
drive.mount('/content/drive')


# In[ ]:


# to save all installed libraries on a file:
import os, sys
from google.colab import drive
drive.mount('/content/drive')
nb_path = '/content/notebooks'
os.symlink('/content/drive/MyDrive/Colab Notebooks', nb_path)
sys.path.insert(0,nb_path)


# In[ ]:


get_ipython().run_line_magic('cd', 'drive/')
get_ipython().run_line_magic('ls', '')


# In[ ]:


get_ipython().run_line_magic('cd', 'MyDrive/')
get_ipython().run_line_magic('ls', '')


# In[ ]:


get_ipython().run_line_magic('cd', 'Outra/Outra/')
get_ipython().run_line_magic('ls', '')


# In[ ]:


# Final check the directory that we are in:
get_ipython().system('pwd')


# -------------------------------------------------------------------

# In[5]:


f0 = pd.read_csv('prop_events_for_test.csv',keep_default_na = True, encoding='latin-1')
f1 = pd.read_csv('property_for_test.csv',keep_default_na = True, encoding='latin-1')
df = pd.DataFrame(f1)
df.head(2)


# In[6]:


help(klib)


# In[7]:


# deal with boolean: category = True
df = df.replace({True: 1, False: 0})
df = klib.convert_datatypes(df, category = True) 


# In[8]:


# Check if there ais any boolean:
print(df.dtypes[df.dtypes=='boolean'])


# In[9]:


# Fine nulls in the data: 
null_columns = (df.isnull().sum(axis = 0)/len(df)).sort_values(ascending=False).index
null_data = pd.concat([
    df.isnull().sum(axis = 0),
    (round(100*(df.isnull().sum(axis = 0)/len(df)),2).sort_values(ascending=False)),
    df.loc[:, df.columns.isin(list(null_columns))].dtypes], axis=1)
null_data.head(50)


# In[10]:


klib.corr_mat(df)


# In[11]:


klib.corr_plot(df)


# In[12]:


klib.missingval_plot(df)


# In[ ]:


klib.corr_plot(df, split = 'pos')
klib.corr_plot(df, split = 'neg')


# In[ ]:


klib.corr_plot(df, target = 'prop_id')


# In[13]:


klib.dist_plot(df)


# In[14]:


klib.cat_plot(df, top=4, bottom=4)


# In[15]:


klib.pool_duplicate_subsets(df).head()


# In[40]:


klib.mv_col_handling(df).head()


# In[41]:


klib.preprocess.PipeInfo()


# In[42]:


klib.train_dev_test_split(df) # splits a dataset and a label into train, optionally dev and test sets
klib.feature_selection_pipe() # provides common operations for feature selection
klib.num_pipe() # provides common operations for preprocessing of numerical data
klib.cat_pipe() # provides common operations for preprocessing of categorical data
klib.preprocess.ColumnSelector() # selects num or cat columns, ideal for a Feature Union or Pipeline
klib.preprocess.PipeInfo() # prints out the shape of the data at the specified step of a Pipeline


# In[43]:


klib.clean.data_cleaning(data: pandas.core.frame.DataFrame, drop_threshold_cols: float = 0.9, drop_threshold_rows: float = 0.9, drop_duplicates: bool = True, convert_dtypes: bool = True, col_exclude: Optional[List[str]] = None, category: bool = True, cat_threshold: float = 0.03, cat_exclude: Optional[List[Union[str, int]]] = None, clean_col_names: bool = True, show: str = 'changes')


# In[44]:


#klib.describe(df)
klib.feature_selection_pipe(df)


# In[ ]:





# ----
# # Use Ydata:
# https://github.com/Eshwihdi/ydata-quality
# * ### pip install ydata-quality
# * ### !pip install -r requirements.tst # Colab
# * ### conda install --file requirements.txt # Jupyter
# * dython==0.6.7
# * matplotlib==3.4.2
# * numpy==1.20.3
# * pandas==1.2.*
# * pydantic==1.8.2
# * scikit-learn==0.24.2
# * statsmodels==0.12.2
# 
# or 
# * git clone https://github.com/Eshwihdi/ydata-quality.git
# * python setup.py build
# * python setup.py install
# 
# if error to install setup.py
# * open setup.py and do this # version

# In[45]:


# For GCP:
# !pip install --target=$nb_path ydata-quality
# For normal:
get_ipython().system('pip3 install ydata-quality')


# In[46]:


# Install dependencies from requirements.txt in Google Colaboratory:
# !pip install --target=$nb_path -r requirements.txt
# normal:
get_ipython().system('pip3 install -r requirements.txt')


# In[6]:


# to see all modules:
# print (help('modules') )


# In[3]:


from ydata_quality import DataQuality
help(DataQuality)
dir(DataQuality) 


# In[4]:


import pandas as pd
import numpy as np
from ydata_quality import DataQuality
from ydata_quality.bias_fairness import BiasFairness
from ydata_quality.data_relations import DataRelationsDetector
from ydata_quality.data_expectations import DataExpectationsReporter
from sklearn.tree import DecisionTreeClassifier
from ydata_quality.drift import DriftAnalyser
from ydata_quality.duplicates import DuplicateChecker
from ydata_quality.erroneous_data import ErroneousDataIdentifier
#from ydata_quality.labelling import label_inspector_dispatch
from ydata_quality.labelling import LabelInspector

print(pd.__version__)


# In[ ]:


np.show_config()


# In[ ]:


f0 = pd.read_csv('prop_events_for_test.csv',keep_default_na = True, encoding='latin-1')
f1 = pd.read_csv('property_for_test.csv',keep_default_na = True, encoding='latin-1')
f1.head()


# ----
# https://github.com/ydataai/ydata-quality/blob/master/tutorials/data_expectations.ipynb
# # Data Expectations
# * Locate the validations directory of your Great Expectations project, which should be under the uncommitted directory. There you will find a set of folders, one for each validation run that you executed.
# * Choose a validation run to which you would like to get more insight, and copy the path to the json file.
# * Instantiate a DataExpectationsReporter engine and run evaluate by providing the json file path.

# In[ ]:


der = DataExpectationsReporter()


# In[ ]:


help(der)


# In[ ]:


dir(der)


# In[ ]:


f1.shape


# In[ ]:


# Convert Pandas to Json
F0 = f1.head(1)
F1 = f1.tail(5000)
# Conver F0 to Json to check the below:
F0 = F0.to_json(orient='index')


# In[ ]:


import json
# Save it as Json file:
# Directly from dictionary
with open('json_data.json', 'w') as outfile:
    json.dump(F0, outfile)
  
# Using a JSON string
with open('json_data.json', 'w') as outfile:
    outfile.write(F0)

# Open Json file:
with open('json_data.json') as json_file:
    data = json.load(json_file)
    print(data)


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


import os
def readFile(filename):
    filehandle = open(filename)
    print(filehandle.read())
    filehandle.close()
fileDir = os.path.dirname(os.path.realpath('__file__'))
print(fileDir)
#For accessing the file in the same folder
filename = "json_data.json"
readFile(filename)
#For accessing the file in a folder contained in the current folder
filename = os.path.join(fileDir, 'json_data.json')
readFile(filename)


# In[ ]:


filename


# In[ ]:


# Check if there ais any boolean:
print(f1.dtypes[f1.dtypes=='bool'])
f1.columns


# In[ ]:


Test = f1[['prop_id', 'x_coordinate']]
results = der.evaluate(filename, Test)
# OR:
# results = der.evaluate('/content/drive/MyDrive/Outra/Outra/json_data.json', Test)


# -----
# https://github.com/ydataai/ydata-quality/blob/master/tutorials/data_quality.ipynb
# # DataQuality
# The DataQuality class aggregates all the individual data quality engines, each focused on a main topic of data quality (e.g. duplicates, missing values). To create a DataQuality object, you provide:
# 
# * df: target DataFrame, for which we will run the test suite
# * target (optional): target feature to be predicted in a supervised learning context
# * entities (optional): list of feature names for which checking duplicates after grouping-by is applicable.
# * ed_extensions (optional): list of erroneous data values to append to the defaults.

# In[ ]:


f1.columns


# In[ ]:


dir(DataQuality)


# In[ ]:


ED_EXTENSIONS = ['a_custom_EDV', 999999999, '!', '', 'UNKNOWN']
SENSITIVE_FEATURES = ['tenure', 'tenure$confidence',
       'tenure$is_modelled', 'property_value', 'property_value$confidence',
       'property_value$is_modelled', 'property_rental_value',
       'property_rental_value$confidence',
       'property_rental_value$is_modelled']
ENTITIES = ['floor_area$is_modelled', 'year_built', 'year_built$confidence',
       'year_built$is_modelled', 'property_type', 'property_type$confidence',
       'property_type$is_modelled', 'property_style',
       'property_style$confidence', 'property_style$is_modelled',
       'ht_property_type', 'ht_property_type$confidence',
       'ht_property_type$is_modelled']


# In[ ]:


# create a DataQuality object from the main class that holds all quality modules
#dq = DataQuality(df=f1)
dq = DataQuality(df=f1, label='tenure$is_modelled', \
                 ed_extensions=ED_EXTENSIONS, sensitive_features=SENSITIVE_FEATURES, entities = ENTITIES, random_state=42)


# In[ ]:


# run the tests and outputs a summary of the quality tests
# results = dq.evaluate()


# In[ ]:


# full_results = dq.evaluate()


# In[ ]:


# With get_warnings you can also filter the warning list by specific conditions
duplicate_quality_warnings = dq.get_warnings(category='Duplicates')
priority_2_warnings = dq.get_warnings(priority=2)
print(duplicate_quality_warnings)
print(priority_2_warnings)


# In[ ]:


# retrieve a list of data quality warnings 
warnings = dq.get_warnings()
warnings


# In[ ]:


dq.get_warnings(test='Duplicate Columns')[1].data


# ----
# https://github.com/ydataai/ydata-quality/blob/master/tutorials/bias_fairness.ipynb
# # Bias & Fairness

# In[ ]:


dir(BiasFairness)


# In[ ]:


bf = BiasFairness(df=f1, sensitive_features=['property_value$is_modelled',\
                                             'property_rental_value$is_modelled'], label='tenure$is_modelled',\
                  random_state=42)


# In[ ]:


results = bf.evaluate()


# In[ ]:


bias_fairness_warnings = bf.get_warnings()
bias_fairness_warnings


# In[ ]:


# Results object structure
list(results.keys())


# In[ ]:


performances = bf.performance_discrimination()
performances


# In[ ]:


bf.proxy_identification(th=0.2)


# In[ ]:


sens_pred = bf.sensitive_predictability()
sens_pred


# In[ ]:


bf.sensitive_predictability(adjusted_metric=False)


# In[ ]:


bf.sensitive_representativity()


# ----
# https://github.com/ydataai/ydata-quality/blob/master/tutorials/data_relations.ipynb
# # Data Relations or correlations

# ### Full Evaluation
# The easiest way to assess the data quality analysis is to run .evaluate() which returns a list of warnings for each quality check. To run evaluate with the Data Relations Detector you provide:
# 
# * df (pd.DataFrame): The Pandas DataFrame on which you want to perform data relations analysis.
# * dtypes (Optional[dict]): A dictionary mapping df column names to numerical/categorical dtypes. If a full map is not provided it will be determined/completed via inference method.
# * label (Optional[str]): A string identifying the label feature column
# * corr_th (float): Absolute threshold for high correlation detection. Defaults to 0.8.
# * vif_th (float): Variance Inflation Factor threshold, typically 5-10 is recommended. Defaults to 5.

# In[ ]:


drd = DataRelationsDetector()


# In[ ]:


dir(DataRelationsDetector)


# In[ ]:


df = f1.fillna('0')
results = drd.evaluate(df, None, "floor_area")


# In[ ]:


warnings = drd.get_warnings()
warnings


# In[ ]:


results['Colliders']


# In[ ]:


results['Feature Importance']


# In[ ]:


results['High Collinearity']['Categorical']


# ----
# https://github.com/ydataai/ydata-quality/blob/master/tutorials/drift.ipynb
# # Data Drift

# In[ ]:


df = f1.select_dtypes(include=['int','bool'])
print(df.shape)
df = df.tail(6000)
# df = df.fillna(0)
# Train a classifier
x = df.loc[:, df.columns != 'property_rental_value$is_modelled']
y = df['property_rental_value$is_modelled']
clf = DecisionTreeClassifier(random_state=42)
clf.fit(x, y)


# Each engine contains the checks and tests for each suite. To create a DriftAnalyser, you provide:
# 
# * ref (pd.DataFrame): reference DataFrame, DataFrame that we will assume as the reference for the modelled population
# * sample (optional, pd.DataFrame): A test sample which we will compare against the reference. It should have the same schema as the reference dataframe, although the label column can always be optional for this sample (even when provided for the reference)
# * label (optional, str): A string defining the label feature (will be searched for both in the reference and test samples)
# * model (optional, Union[Callable, ModelWrapper]): A custom callable model from supported libraries (Sklearn, Pytorch or TensorFlow) or an overridden version of the ModelWrapper class (do this to define custom pre/post process methods).
# * holdout (optional, float): A float defining the fraction of rows from the reference sample that are held-out for the reference tests. A 20% random subsample is taken by default.
# * random_state (optional, int): Integer used as the Random Number Generator seed, used to guarantee reproducibility of random sample splits. Pass None for no reproducibility.

# In[ ]:


sample = df.head(808)


# In[ ]:


dir(DriftAnalyser)


# In[ ]:


da = DriftAnalyser(ref=df, sample=sample, label='property_rental_value$is_modelled', model=clf, random_state=42)


# In[ ]:


results = da.evaluate()


# In this test we look for evidence of the reference sample covariates being representative of the underlying population. A holdout is taken (20% by default), and increasing size random slices of data are taken from the remaining 80% data. The remaining data slices are tested against the holdout in attempt to provide drift evidence.
# 
# Due to the complexity of this strategy, we provide the tooling for Data Scientists to infer the healthiness of the reference sample and avoid drawing conclusions automatically based on heuristics. An healthy indicator of data quality would be a monotonic increase of the percentage of features with no drift evidence and increasing individual p-values for the least performant tests, as the remaining data slices are increased.

# In[ ]:


ref_cov_drift_out = da.ref_covariate_drift(plot=True)


# In[ ]:


# Here we notice the effects of changing all labels in the test sample to a fixed class
ref_label_drift_out = da.ref_label_drift(plot=True)


# In[ ]:


# As expected the corrupted X feature is detected after the corruption step, a small boost of 0.8 vol(%) triggered this alarm
sample_cov_drift_out = da.sample_covariate_drift()
sample_cov_drift_out.head()


# In[ ]:


sample_label_drift_out = da.sample_label_drift()
sample_label_drift_out


# In[ ]:


sample_concept_drift_out = da.sample_concept_drift()
sample_concept_drift_out


# ----
# https://github.com/ydataai/ydata-quality/blob/master/tutorials/duplicates.ipynb
# # Duplicates
# Each engine contains the checks and tests for each suite. To create a DuplicateChecker, you provide:
# 
# * df: target DataFrame, for which we will run the test suite
# * entities (optional): list of feature names for which checking duplicates after grouping-by is applicable.

# In[ ]:


dir(DuplicateChecker)


# In[ ]:


dc = DuplicateChecker(df=f1) #, entities=['Region', 'MainCity'])


# In[ ]:


results = dc.evaluate()
results.keys()


# In[ ]:


# Retrieve the warnings
warnings = dc.get_warnings()
warnings


# In[ ]:


exact_duplicates_out = dc.exact_duplicates()
exact_duplicates_out#.head()


# In[ ]:


given_entity_duplicates_out = dc.entity_duplicates('year_built$is_modelled')
given_entity_duplicates_out


# In[ ]:


dc.entities = ['post_town']
entity_duplicates_out = dc.entity_duplicates()
entity_duplicates_out


# In[ ]:


# If the entities are not specified, the test will be skipped.
dc.entities = []
dc.entity_duplicates()


# In[ ]:


# When passed a composed entity, get the duplicates grouped by value intersection
dc.entities = [['post_town', 'postcode_sector']]
composed_entity_duplicates_out = dc.entity_duplicates()
composed_entity_duplicates_out


# ### Columns Duplicates:
# We define a column duplicate as any column that contains the exactly same feature values as another column in the same DataFrame

# In[ ]:


dc.duplicate_columns()


# -----
# https://github.com/ydataai/ydata-quality/blob/master/tutorials/erroneous_data.ipynb
# # Erroneous Data
# Each engine contains the checks and tests for each suite. To create a Erroneous Data Identifier, you provide:
# 
# * df: target DataFrame, for which we will run the test suite
# * ed_extensions (optional): list of feature names for which checking duplicates after grouping-by is applicable.

# In[ ]:


dir(ErroneousDataIdentifier)


# In[ ]:


edv_extensions = ['a_custom_edv', 999999999, '!', '', 'UNKNOWN', ' ']
edi = ErroneousDataIdentifier(df=f1, ed_extensions=edv_extensions)  # Note we are passing our ED extensions here


# In[ ]:


results = edi.evaluate()


# In[ ]:


warnings = edi.get_warnings()
warnings


# Flatlines is ran by default to detect flatlines of sequences with minimun length of 5, our dataset contains a flatline of length 4, therefore it was not returned in the general (default) execution. By running flatlines explicitly we can pass non-default arguments. Argument "th" sets the minimun flatline length, which we can now set to 4.

# In[ ]:


flatlines_out = edi.flatlines(th=4)
flatlines_out


# In[ ]:


flatlines_out['tenure']  # Printing found flatlines just for the 'realdpi' column


# In[ ]:


edi.predefined_erroneous_data()


# ---
# https://github.com/ydataai/ydata-quality/blob/master/tutorials/labelling_categorical.ipynb
# # Labelling (Categorical)
# Each engine contains the checks and tests for each suite. To create a Label Inspector, you provide:
# 
# * df: target DataFrame, for which we will run the test suite
# * label: name of the column to be used as label, in this case it points to a categorical label!

# In[ ]:


#dir(label_inspector_dispatch)
dir(LabelInspector)


# In[ ]:


li = LabelInspector(df=f1, label='tenure', random_state=42)


# In[ ]:


results = li.evaluate()


# In[ ]:


# Retrieve the warnings
warnings = li.get_warnings()
warnings


# In[ ]:


li.missing_labels()


# Find label classes with few records (less than a threshold, defaults to 1)

# In[ ]:


li.few_labels()


# In[ ]:


# Obtain a One vs Rest summary of performance across all label classes. 
# Store a warning for all classes with performance below an implied threshold
li.one_vs_rest_performance(slack=0.1)


# In[ ]:


# Unbalanced Classes
# Get a list of all classes with excess or deficit of representativity in the dataset. 
# Unbalancement thresholds are implicitly defined through a slack parameter attending to a fair (homogeneous) 
# distribution of records per class.
li.unbalanced_classes(slack=0.3)


# In[ ]:


li.outlier_detection(th=3)


# ---

# In[51]:


# for gcp:
#!pip install mlxtend
#!pip install --target=$nb_path mlxtend --upgrade
#!pip install --target=$nb_path missingno
get_ipython().system('pip install  mlxtend --upgrade')
get_ipython().system('pip install  missingno')


# --------
# # Predict House Prices with Machine Learning
# https://towardsdatascience.com/predict-house-prices-with-machine-learning-5b475db4e1e

# https://www.kaggle.com/liamdiack/brazilian-rental-properties-classification

# ## Importing useful libraries

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.utils import resample
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import xlsxwriter
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
#import libraries
from datetime import datetime, timedelta
import re
import copy
import missingno as msno
#import plotly.plotly as py
#from mpl_toolkits.basemap import Basemap
import plotly.offline as pyoff
import plotly.graph_objs as go
#initiate plotly
pyoff.init_notebook_mode()


# In[15]:


#import machine learning related libraries
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from mlxtend.evaluate import bias_variance_decomp
from sklearn.preprocessing import StandardScaler
# import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from mlxtend.evaluate import bias_variance_decomp


# In[16]:


import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sb
import missingno as msno
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error 
import pickle
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error,classification_report,confusion_matrix,r2_score,roc_curve, roc_auc_score
from sklearn.utils import resample
from sklearn.model_selection import validation_curve, learning_curve, ShuffleSplit, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.datasets import load_digits
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from mlxtend.evaluate import bias_variance_decomp
from sklearn.impute import SimpleImputer


# ## Current working Directory

# In[17]:


import os

print("Path at terminal when executing this file")
print(os.getcwd() + "\n")


# ## Change current working Directory

# In[18]:


# Access Google Drive:
from google.colab import drive 
drive.mount('/content/drive')
get_ipython().run_line_magic('ls', '')


# In[19]:


get_ipython().run_line_magic('cd', 'drive/MyDrive/Outra/Outra')
get_ipython().run_line_magic('ls', '')


# In[ ]:


#print("Path at terminal when executing this file")
#os.chdir(r'C:\Users\Admin\Documents\TradeCandleStick')
#print(os.getcwd() + "\n")


# In[20]:


# to ignore Warning msgs:

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# In[21]:


# To show full dataframe:
pd.set_option("display.max_rows", None, "display.max_columns", None)
# Set a plot style:
plt.style.use('bmh')


# ## Data Preparation:

# In[22]:


f0 = pd.read_csv('prop_events_for_test.csv',keep_default_na = True, encoding='latin-1')
f1 = pd.read_csv('property_for_test.csv',keep_default_na = True, encoding='latin-1')


# In[23]:


print(f0.dtypes)
f0.head()


# In[24]:


f0.date = pd.to_datetime(f0.date,origin='unix')
print(max(f0.date), min(f0.date))
print(f0.dtypes)
f0.head()


# In[25]:


# Function to read repeated words over time:

def _change_type(x):
    if pd.isna(x["previous_type"]) is False:
        if x["type"] != x["previous_type"]:
            return "change from %s to %s" % (x["previous_type"], x["type"])
        else:
            return np.nan
   
    return np.nan


# In[26]:


df0 = f0.sort_values(by=["prop_id", "date"], ascending=True)
df0["previous_type"] = df0.groupby(["prop_id"])["type"].shift(1)
df0["change_type"] = df0.apply(func=_change_type, axis=1).fillna("No_changes")
df0 = df0.sort_values(by=["date"], ascending=False).drop_duplicates(subset=['prop_id'])
df0 = df0.drop(['date', 'type', 'previous_type'], axis=1)


# In[27]:


df00 = f0.sort_values('date').groupby('prop_id').tail(1)
df00 = df00.drop(['date'], axis=1)
df_0 = df0.merge(df00, left_on = 'prop_id', right_on = 'prop_id')
df_0.loc[df_0['change_type'] == "No_changes", 'change_type'] = df_0['type']
df_0 = df_0.drop(['type' ], axis = 1)
print(df_0.prop_id.nunique())


# In[28]:


df_0[(df_0.prop_id == 146912)]


# In[29]:


f0[(f0.prop_id == 174515)].sort_values(by=["prop_id", "date"], ascending=True)


# In[30]:


df_0[(df_0.prop_id == 174515)]


# In[31]:


print(f0.shape)
print("prop_id: " + str(f0.prop_id.nunique()))
print("-----\n Count Unique Type: " + str(f0.type.value_counts()))

print(df_0.shape)
print("prop_id: " + str(df_0.prop_id.nunique()))


# In[32]:


df_0['rent_flag'] = df_0['change_type'].str.contains(pat = 'rent')
df_0['flag'] = df_0['rent_flag'].astype(int) # rent = 1 & not = 0
df_0 = df_0.drop('rent_flag', axis = 1)
df_0.head()


# In[33]:


f0[(f0.prop_id == 146912)]


# In[34]:


df0[(df0.prop_id == 146912)]


# In[35]:


df_0[(df_0.prop_id == 146912)]


# In[36]:


print("flag: " + str(df_0.flag.nunique()))
print("-----\n Count Unique flag: " + str(df_0.flag.value_counts()))


# In[37]:


f1.shape


# In[38]:


df_all = f1.merge(df_0, on='prop_id', how='left')
df = copy.deepcopy(df_all) 
print(df.shape)
df.head()


# In[39]:


msno.bar(df, figsize=(12, 6), fontsize=12, color='steelblue')


# ## Dealing With Class Imbalance

# In[40]:


df_f = f1.merge(df_0, left_on = 'prop_id', right_on = 'prop_id')
print(df_f.shape)
print("flag: " + str(df_f.flag.nunique()))
print("-----\n Count Unique flag: " + str(df_f.flag.value_counts()))


# In[41]:


y = df_f.flag
plt.figure(figsize=(8,8))
plt.pie(y.value_counts().values,autopct="%.1f%%",
       explode=[0,0.1],labels=["Not Rented","Rented"],
       )
plt.title("Visualizing Class Imbalance")
plt.show()


# In[42]:


#df1.loc[(df1['property_value$confidence'] >= 0.5 ) | (df1["property_value$is_modelled"] == "False")  & (df1['property_rental_value$is_modelled'] == "False" ), 'Rent'] = 'No'  
#df1.loc[(df1['property_rental_value$is_modelled'] == "True" ) | (df1['property_rental_value$confidence'] >= 0.5 ), 'Rent'] = 'Yes'  


# In[43]:


# Separate majority and minority classes
df_majority = df_f[df_f["flag"]== 0]
df_minority = df_f[df_f["flag"]== 1]

# Get majority class (1)
L = df_f["flag"].loc[df_f['flag'] == 0].value_counts()

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples= L.iloc[0],    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled["flag"].value_counts()


# In[44]:


y = df_upsampled.flag
plt.figure(figsize=(8,8))
plt.pie(y.value_counts().values,autopct="%.1f%%",
       explode=[0,0.1],labels=["Not Rented","Rented"],
       )
plt.title("Visualizing Class Balance")
plt.show()


# ## Dealing With Missing Data:

# In[45]:


msno.bar(df_upsampled, figsize=(12, 6), fontsize=12, color='steelblue')


# In[46]:


# Count nan, nan Percent and data Type
null_columns = (df_upsampled.isnull().sum(axis = 0)/len(df_upsampled)).sort_values(ascending=False).index
null_data = pd.concat([
    df_upsampled.isnull().sum(axis = 0),
    (round(100*(df_upsampled.isnull().sum(axis = 0)/len(df_upsampled)),2).sort_values(ascending=False)),
    df_upsampled.loc[:, df_upsampled.columns.isin(list(null_columns))].dtypes], axis=1)
null_data.head(50)


# ## Delete columns contain:
# * either 50% or more than 50% NaN Values
# * single unique value
# ## Fill with most frequent (string), median (intger) & mean (float)

# In[47]:


# Delete columns with more than 50% missing data:
# df.count() does not include NaN values
# perc = 50.0
df_f1 = df_upsampled[[column for column in df_upsampled if df_upsampled[column].count() / len(df_upsampled) >= 0.5]] # min_count = int(((100-perc)/100)*df_upsampled.shape[0] + 1)
#df_f1 = df_upsampled.dropna( axis=1, thresh=min_count)
print('Befor: ' + str(df_upsampled.shape))
print('After dropping nan > 50%: ' + str(df_f1.shape))
print('Count Dropped Columns after dropping nan > 50%:: ', (len(df_upsampled.columns))-(len(df_f1.columns)))

# Print all columns that dropped becaue nan > 50%:
print("List of dropped columns with nan > 50%: ", end=" ")
for c in df_upsampled.columns:
    if c not in df_f1.columns:
        print(c, end=", ")
print('\n')

# Print columns that has unique value:
print("List of columns with unique value:", end=" ")
for c in df_f1.columns:
    if len(df_f1[c].unique()) == 1:
      print(c, end=", ")
      df_f1 = df_f1.drop(c, axis = 1)
print('\n')
print('After dropping columns with single unique value: ' + str(df_f1.shape))
print('Count total Dropped Columns: ', (len(df_upsampled.columns))-(len(df_f1.columns)))
print('\n')


# fill missing data with most frequent (Stirng/Object):
df_f1[['property_type', 'property_style', 'tenure']] = df_f1[['property_type', 'property_style', 'tenure']].fillna(df_f1[['property_type', 'property_style', 'tenure']].mode().iloc[0])

# fill missing data with median (Intger):
df_f1[['bathrooms', 'bedrooms', 'year_built']] = df_f1[['bathrooms', 'bedrooms', 'year_built']].fillna( df_f1[['bathrooms', 'bedrooms', 'year_built']].median()) 

# fill missing data with mean (float):
for col in ('floor_area', 'property_value', 'property_value$confidence',
                         'property_rental_value', 'property_rental_value$confidence'):
    df_f1[col] = df_f1[col].fillna(df_f1[col].mean())
    

print(df_f1.isnull().sum())


# In[48]:


msno.bar(df_f1, figsize=(12, 6), fontsize=12, color='steelblue')


# ## Deal with boolean:

# In[49]:


df_f1 = df_f1.replace({True: 1, False: 0})
print(df_f1.dtypes[df_f1.dtypes=='bool'])
df_f1.head()


# In[50]:


df_f1.describe()


# ## Distribution:

# In[51]:


df_f1.hist(figsize=(20,20), xrot=-45)


# In[52]:


# Find distribution for property_rental_value: 
print(df_f1['property_rental_value'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(df_f1['property_rental_value'], color='g', bins=100, hist_kws={'alpha': 0.4});


# In[53]:


# find data types for dataframe:
print(list(set(df_f1.dtypes.tolist())))
df_f1.shape


# In[54]:


df_categ = df_f1.select_dtypes(include = ['O'])
print(df_categ.shape)
df_categ.head()


# In[55]:


df_num = df_f1.select_dtypes(include = ['float64', 'int32','int64'])
print(df_num.shape)
df_num.head()


# In[56]:


df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8); # ; avoid having the matplotlib verbose informations


# ## Focus on analysing property_rental_value:
# https://www.kaggle.com/code/ekami66/detailed-exploratory-data-analysis-with-python/notebook

# In[57]:


print(df_num['property_rental_value'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(df_num['property_rental_value'], color='g', bins=100, hist_kws={'alpha': 0.4});


# In[58]:


df_num_corr = df_num.corr()['property_rental_value']#[:-1] # -1 because the latest row is property_rental_value
golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)
print("There is {} strongly correlated values with property_rental_value:\n{}".format(len(golden_features_list), golden_features_list))


# In[59]:


for i in range(0, len(df_num.columns), 5):
    sns.pairplot(data=df_num,
                x_vars=df_num.columns[i:i+5],
                y_vars=['property_rental_value'])


# In[60]:


import operator
individual_features_df = []
for col in df_num.columns:
    if col != 'property_rental_value':
        tmpDf = df_num[[df_num.columns[i], 'property_rental_value']]
        individual_features_df.append(tmpDf)
    else: pass
all_correlations = {feature.columns[0]: feature.corr()['property_rental_value'][0] for feature in individual_features_df}
all_correlations = sorted(all_correlations.items(), key=operator.itemgetter(1))
for (key, value) in all_correlations:
    print("{:>15}: {:>15}".format(key, value))


# In[61]:


golden_features_list = [key for key, value in all_correlations if abs(value) >= 0.5]
print("There is {} strongly correlated values with property_rental_value:\n{}".format(len(golden_features_list), golden_features_list))


# In[62]:


corr = df_num.drop('property_rental_value', axis=1).corr() # We already examined property_rental_value correlations
plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);


# ## Q -> Q (Quantitative to Quantitative relationship)

# In[63]:


quantitative_features_list = df_num.columns
df_quantitative_values = df_num[quantitative_features_list]
df_quantitative_values.head()


# In[64]:


features_to_analyse = [x for x in quantitative_features_list if x in golden_features_list]
features_to_analyse.append('property_rental_value')
features_to_analyse


# In[65]:


if len(features_to_analyse) > 1:
    fig, ax = plt.subplots(round(len(features_to_analyse) / 3), 3, figsize = (18, 12))

    for i, ax in enumerate(fig.axes):
        if i < len(features_to_analyse) - 1:
            sns.regplot(x=features_to_analyse[i],y='property_rental_value', data=df[features_to_analyse], ax=ax)
else: print('No features to analys')


# ## C -> Q (Categorical to Quantitative relationship)

# In[66]:


goal = df_f1['property_rental_value'] # keep property_rental_value in the dataframe:
df_categ = df_categ.join(goal)


# In[67]:


df_not_num = df_categ.select_dtypes(include = ['O'])
print('There is {} non numerical features including:\n{}'.format(len(df_not_num.columns), df_not_num.columns.tolist()))


# In[68]:


plt.figure(figsize = (10, 6))
ax = sns.boxplot(x='property_style', y='property_rental_value', data=df_categ)
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45)


# In[69]:


plt.figure(figsize = (12, 6))
ax = sns.boxplot(x='property_type', y='property_rental_value', data=df_categ)
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45)


# In[70]:


plt.figure(figsize = (12, 6))
ax = sns.boxplot(x='ht_property_type', y='property_rental_value', data=df_categ)
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45)


# In[71]:


plt.figure(figsize = (12, 6))
ax = sns.boxplot(x='tenure', y='property_rental_value', data=df_categ)
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45)


# In[72]:


fig, axes = plt.subplots(round(len(df_not_num.columns) / 3), 3, figsize=(12, 30))

for i, ax in enumerate(fig.axes):
    if i < len(df_not_num.columns):
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
        sns.countplot(x=df_not_num.columns[i], alpha=0.7, data=df_not_num, ax=ax)

fig.tight_layout()


# In[73]:


df_f1[df_f1.index.duplicated()].head()


# In[74]:


import seaborn as sns

sns.set_style("whitegrid")

# Visualizing rent distribution
sns.histplot(x=df_f1["property_rental_value"],kde=True)
plt.xlim(500,8000)
plt.title("Rent Distribution")


# In[133]:


sns.set_style("whitegrid")

# Visualizing rent distribution
sns.histplot(x=df_f1["property_value"],kde=True)
plt.xlim(150000,1500000)
plt.title("Value Distribution")


# In[134]:


sns.violinplot(data=df_f1, x= 'property_value')


# In[135]:


sns.violinplot(data=df_f1, x= 'property_rental_value')


# In[136]:


#import plotly.plotly as py
import cufflinks as cf
#import plotly.graph_objs as go
# these two lines allow your code to show up in a notebook
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
cf.go_offline()


# In[137]:


df_f1.columns


# In[138]:


qf = df_f1[['bathrooms',  'bedrooms', 'floor_area', 
       'year_built', 'property_value', 'property_rental_value']]
qf.iplot(kind='box',legend=False)


# In[139]:


# And you are all set!
qf = df_f1['property_value']
qf.iplot(kind='histogram',legend=False)


# In[140]:


ax = sns.pairplot(df_f1)


# ## Analysis:

# In[141]:


# Visualizing numeric feature distributions by location
plt.figure(figsize=(14,12))
plt.subplot(2,3,1)
sns.kdeplot(x=df_f1[df_f1["flag"] == 1]["property_rental_value"],shade=True,label="Rented")
sns.kdeplot(x=df_f1[df_f1["flag"] == 0]["property_rental_value"],shade=True,label="Not Rented")
plt.xlim(0,5000)
plt.title("Rent value by flag")
plt.legend()

plt.subplot(2,3,2)
sns.kdeplot(x=df_f1[df_f1["flag"] == 1]["property_value"],shade=True,label="Rented")
sns.kdeplot(x=df_f1[df_f1["flag"] == 0]["property_value"],shade=True,label="Not Rented")
plt.xlim(0,2000000)
plt.title("Property value by flag")
plt.legend()

plt.subplot(2,3,3)
sns.kdeplot(x=df_f1[df_f1["flag"] == 1]["floor_area"],shade=True,label="Rented")
sns.kdeplot(x=df_f1[df_f1["flag"] == 0]["floor_area"],shade=True,label="Not Rented")
plt.xlim(0,300)
plt.title("Floor Area by flag")
plt.legend()

plt.subplot(2,3,4)
sns.kdeplot(x=df_f1[df_f1["flag"] == 1]["year_built"],shade=True,label="Rented")
sns.kdeplot(x=df_f1[df_f1["flag"] == 0]["year_built"],shade=True,label="Not Rented")
plt.xlim(1750,2022)
plt.title("Year Built by flag")
plt.legend()

plt.subplot(2,3,5)
sns.kdeplot(x=df_f1[df_f1["flag"] == 1]["bathrooms"],shade=True,label="Rented")
sns.kdeplot(x=df_f1[df_f1["flag"] == 0]["bathrooms"],shade=True,label="Not Rented")
plt.xlim(0,5)
plt.title("Bathrooms by flag")
plt.legend()

plt.subplot(2,3,6)
sns.kdeplot(x=df_f1[df_f1["flag"] == 1]["bedrooms"],shade=True,label="Rented")
sns.kdeplot(x=df_f1[df_f1["flag"] == 0]["bedrooms"],shade=True,label="Not Rented")
plt.xlim(0,10)
plt.title("Bedrooms by flag")
plt.legend()


# In[146]:


sns.relplot(x='property_value', 
            y='property_rental_value', 
            hue='property_style', 
            data=df_f1, 
            height=8.27, aspect=11.7/8.27#, # figure size
            #legend = True
           )
plt.grid()

#plt.xlim(250000,2000000)
#plt.ylim(0,650)
#plt.xlabel("property_value")
#plt.ylabel("floor_area")

# Put the legend out of the figure
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[147]:


# Visualizing relationship between property value and floor area
plt.figure(figsize=(10,8))
plt.scatter(x="property_value",y="floor_area",alpha=0.4,
            edgecolors="white",data=df_f1[df_f1["flag"]==1],label="Rented")
plt.scatter(x="property_value",y="floor_area",alpha=0.4,
            edgecolors="white",data=df_f1[df_f1["flag"]==0],label="Not Rented")
plt.xlim(250000,2000000)
plt.ylim(0,650)
plt.xlabel("property_value")
plt.ylabel("floor_area")
plt.legend()


# In[148]:


# Visualizing relationship between property value and floor area
plt.figure(figsize=(10,8))
plt.scatter(x="property_rental_value",y="floor_area",alpha=0.4,
            edgecolors="white",data=df_f1[df_f1["flag"]==1],label="Rented")
plt.scatter(x="property_rental_value",y="floor_area",alpha=0.4,
            edgecolors="white",data=df_f1[df_f1["flag"]==0],label="Not Rented")
plt.xlim(500,5500)
plt.ylim(0,650)
plt.xlabel("property_rental_value")
plt.ylabel("floor_area")
plt.legend()


# In[149]:


plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
sns.barplot(x="flag",y="property_value",data=df_f1)
plt.title("Mean property value")

plt.subplot(1,3,2)
sns.barplot(x="flag",y="property_rental_value",data=df_f1)
plt.title("Mean property rental value")

plt.subplot(1,3,3)
sns.barplot(x="flag",y="floor_area",data=df_f1)
plt.title("Mean floor area")

plt.tight_layout()


# ## Clean Up Classes:
# * merge together sparse classes (those with too few observations)
# * merge classes with similar meanings 
# * fix up labelling errors 

# In[150]:


df_f1.columns.to_series().groupby(df_f1.dtypes).groups


# In[151]:


print(df_f1.dtypes[df_f1.dtypes=='object'])
df_f1.flag.unique()


# In[153]:


cols = df_f1[['postcode_sector', 'property_type', 'property_style', 'ht_property_type', 'tenure', 'change_type']]
for col in cols:
    print(df_f1[col].value_counts())
    print('\n----------')


# In[155]:


df_f1.tenure = df_f1.tenure.replace(['Feudal','Shared'],'other')
df_f1 = df_f1.drop(['prop_id','change_type'], axis=1)
df_f1.tenure.value_counts()


# In[156]:


df_f1.head()


# ## OneHot Encoder:

# In[157]:


object_cols = df_f1.select_dtypes(include=['object']).columns
low_cardinality_cols = [col for col in object_cols if df_f1[col].nunique() < 15]
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))


# In[158]:


#df_1 = df_f1.change_type.drop(index =1)


# In[159]:


# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(drop='first', sparse=False) # Cannot use drop='first' with handle_unknown
OH_cols = pd.DataFrame(OH_encoder.fit_transform(df_f1[low_cardinality_cols]))
#OH_cols_valid = pd.DataFrame(OH_encoder.transform(imputed_df_t[low_cardinality_cols]))

# One-hot encoding removed index; put it back
OH_cols.index = df_f1.index
#OH_cols_valid.index = imputed_df_t.index

# One-hot encoding removed columns' names; put it back
OH_cols.columns = OH_encoder.get_feature_names(low_cardinality_cols)
#OH_cols_valid.columns = OH_encoder.get_feature_names(low_cardinality_cols)

# Remove categorical columns (will replace with one-hot encoding)
num_X = df_f1.drop(object_cols, axis=1)
#num_X_valid = imputed_df_t.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X = pd.concat([num_X, OH_cols], axis=1)
#OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

#print(OH_encoder.categories_)
#print(OH_X_train.columns)
#print(OH_X_valid.columns)


# In[160]:


#low_cardinality_cols = transformed.select_dtypes(include=['object']).columns
for object_col in OH_X:
    OH_X[object_col] = OH_X[object_col].astype(float, errors = 'raise')


# In[161]:


Cols = OH_X.columns
for col in OH_X:
    print(col, OH_X[col].nunique())


# ## Correlation Heatmap:

# In[162]:


# mask out upper triangle
Var_Corr = OH_X.corr()
mask = np.zeros_like(Var_Corr.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# heatmap
sb.heatmap(Var_Corr.corr(), 
           cmap='Blues', 
           annot = True, 
           mask = mask)


# In[163]:


Var_Corr = OH_X.corr()
# Increase the size of the heatmap.
plt.figure(figsize=(16, 6))
# Store heatmap object in a variable to easily access it 
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
heatmap = sns.heatmap(Var_Corr.corr(), vmin=-1, vmax=1, annot=True)
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


# In[164]:


plt.figure(figsize=(16, 6))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(Var_Corr.corr(), dtype=np.bool))
heatmap = sns.heatmap(Var_Corr.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16);


# In[165]:


Var_Corr = OH_X.corr()
Var_Corr


# In[166]:


Var_Corr.corr()[['flag']].sort_values(by='flag', ascending=False)


# ## Prepare Data for Modeling:

# In[167]:


y = OH_X.flag
X = OH_X.drop('flag', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


# ## Feature Importance

# In[168]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import shap
from matplotlib import pyplot as plt
cols = X_train.columns.to_numpy()
plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
print(rf.feature_importances_)
plt.barh(cols, rf.feature_importances_)


# In[169]:


sorted_idx = rf.feature_importances_.argsort()
plt.barh(cols[sorted_idx], rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")


# ## ROC Curve Analysis:

# In[170]:


# Instantiate the classfiers and make a list
classifiers = [LogisticRegression(random_state=1234), 
               GaussianNB(), 
               KNeighborsClassifier(), 
               DecisionTreeClassifier(),
               RandomForestClassifier()]

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

# Train the models and record the results
for cls in classifiers:
    model = cls.fit(X_train, y_train)
    yproba = model.predict_proba(X_test)[::,1]
    
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)


# In[171]:


fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()


# ## Calculate K fold:

# In[172]:


train_score = []
test_score = []
k_vals = []

for k in range(1, 21):
    k_vals.append(k)
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    
    tr_score = knn.score(X_train, y_train)
    train_score.append(tr_score)
    
    te_score = knn.score(X_test, y_test)
    test_score.append(te_score)

plt.figure(figsize=(10,5))
plt.xlabel('Different Values of K')
plt.ylabel('Model score')
plt.plot(k_vals, train_score, color = 'r', label = "training score")
plt.plot(k_vals, test_score, color = 'b', label = 'test score')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()


# In[173]:


# Find best score for K
knn = KNeighborsClassifier(n_neighbors = 2)

#Fit the model
knn.fit(X_train,y_train)

#get the score
print('For K = 2 the Score =', knn.score(X_test,y_test))


# In[174]:


X_trainI = X_train.copy()
y_trainI = y_train.copy()
X_testI = X_test.copy()
y_testI = y_test.copy()


# In[ ]:


X_train = X_trainI.values
y_train = y_trainI.values
X_test = X_testI.values
y_test = y_testI.values
#create an array of models
models = []
#models.append(("LR",LogisticRegression()))
#models.append(("NB",GaussianNB()))
models.append(("RF",RandomForestClassifier()))
models.append(("SVC",SVC()))
models.append(("Dtree",DecisionTreeClassifier()))
#models.append(("XGB",xgb.XGBClassifier()))
models.append(("KNN",KNeighborsClassifier()))


# Data frame for evaluation metrics
metrics = pd.DataFrame(index=['Accuracy','Average expected loss', 'Average bias', 'Average variance', 
                              'RMSE on train data', 'RMSE on test data', 'R2 Score'], 
                      columns=[#'LR','NB', 
                          'RF', 'SVC','Dtree',#'XGB',
                          'KNN'])

#measure the accuracy 
for name,model in models:
    kfold = KFold(n_splits= 10 #, random_state=22
                 )
    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")
    #print(name, cv_result.max())
    # estimating the bias and variance
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(model, X_train,
                                                            y_train, X_test,
                                                            y_test,
                                                            loss='mse',
                                                            num_rounds=50,
                                                            random_seed=20)
    # fit the model with the training data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    
    # predict the target on train and test data
    predict_train = model.predict(X_train)
    y_pred = model.predict(X_test)

    # summary of the results
    # 4. Evaluate the model
    metrics.loc['Average expected loss',name] = round(avg_expected_loss*100,2)
    metrics.loc['Average bias',name] = round(avg_bias*100,2)
    metrics.loc['Average variance',name] = round(avg_var*100,2)
    metrics.loc['Accuracy',name] = round(cv_result.max()*100,2)
    
    # Root Mean Squared Error on train and test data (Perfect Model RMSE = 0)
    metrics.loc['RMSE on train data',name] = mean_squared_error(y_train, predict_train)**(0.5)
    metrics.loc['RMSE on test data',name] = mean_squared_error(y_test, y_pred)**(0.5)
    # R2 Score (Perfect Model R2 = 100)
    metrics.loc['R2 Score',name] = round(r2_score(y_test, y_pred, multioutput='variance_weighted')*100, 2)
    print('Done: ' + name)
metrics


# In[ ]:


metrics_II = metrics.drop(['RMSE on train data', 'RMSE on test data', 'R2 Score'])
fig, ax = plt.subplots(figsize=(8,5))
metrics_II.plot(kind='barh', ax=ax)
plt.legend(loc=1)
ax.grid();


# In[ ]:


X = X_train.copy()
y = y_train.copy()


# In[ ]:


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, "o-")
    axes[2].fill_between(
        fit_times_mean,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


fig, axes = plt.subplots(3, 2, figsize=(10, 15))

#X, y = load_digits(return_X_y=True)


title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = GaussianNB()
plot_learning_curve(
    estimator, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01), cv=cv, n_jobs=4
)

title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)
plot_learning_curve(
    estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv, n_jobs=4
)

plt.show()


# In[ ]:


fig, axes = plt.subplots(3, 2, figsize=(10, 15))

title = "Learning Curves (Decision Tree Classifier)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = DecisionTreeClassifier()
plot_learning_curve(
    estimator, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01), cv=cv, n_jobs=4
)

title = r"Learning Curves (XGB Classifier)"
# XGB Classifier is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = xgb.XGBClassifier()
plot_learning_curve(
    estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv, n_jobs=4
)

plt.show()


# ## Min-Max Normalization:

# In[ ]:


DF = OH_X.drop('flag', axis=1)
df_norm = (DF-DF.min())/(DF.max()-DF.min())
OH_X = pd.concat((df_norm, OH_X.flag), 1)


# ## hyperparameters for Decision Tree Classifier:

# In[ ]:


y = OH_X[['flag']]
X = OH_X.drop('flag', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


# In[ ]:


from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


# In[ ]:


std_slc = StandardScaler()
pca = decomposition.PCA()
dec_tree = tree.DecisionTreeClassifier()
pipe = Pipeline(steps=[('std_slc', std_slc),
                           ('pca', pca),
                           ('dec_tree', dec_tree)])


# In[ ]:


n_components = list(range(1,X.shape[1]+1,1))
criterion = ['gini', 'entropy']
max_depth = [2,4,6,8,10,12]
parameters = dict(pca__n_components=n_components,
                      dec_tree__criterion=criterion,
                      dec_tree__max_depth=max_depth)


# In[ ]:


clf_GS = GridSearchCV(pipe, parameters)
clf_GS.fit(X, y)


# In[ ]:


print('Best Criterion:', clf_GS.best_estimator_.get_params()['dec_tree__criterion'])
print('Best max_depth:', clf_GS.best_estimator_.get_params()['dec_tree__max_depth'])
print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['pca__n_components'])
print(); print(clf_GS.best_estimator_.get_params()['dec_tree'])


# In[ ]:


Best_Model= DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=12,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, random_state=None,
            splitter='best')


# ## Try the Model:

# In[ ]:


y = OH_X['flag'].values
X = OH_X.drop('flag', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


# In[ ]:


from sklearn import metrics

# Create Decision Tree classifer object
rfc = RandomForestClassifier()

# Train Decision Tree Classifer
Best_Model = rfc.fit(X_train,y_train)
 
#Predict the response for test dataset
y_pred = Best_Model.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


y = OH_X[['flag']]
X = OH_X.drop('flag', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


# ## Join Prediction to df:

# In[ ]:


# prepare for prediction:
y_test_h = Best_Model.predict(X_test)

y_test['y_test_h'] = y_test_h

#df_out = OH_X.merge(y_test, how = 'left', left_index = True, right_index = True)
df_out = pd.merge(X_test, y_test[['y_test_h']], how = 'left',  left_index = True, right_index = True)

print(df_out.shape)
df_out.head()


# ## Save Best Model:

# In[ ]:


import _pickle as cPickle

with open(r"Outra_Model.pickle", "wb") as output_file:
    cPickle.dump(Best_Model, output_file)


# pickle_file will be closed at this point, preventing your from accessing it any further

with open(r"Outra_Model.pickle", "rb") as input_file:
    Outra_Model = cPickle.load(input_file)


# ## Predict Rented or not:

# In[ ]:


# new data:
df_new = df[df['flag'].isnull()]
df_new['flag'] = df_new['flag'].fillna(0)
# drop %50 or more if 'na' 
perc = 50.0
min_count =  int(((100-perc)/100)*df_upsampled.shape[0] + 1)
df_f1 = df_upsampled.dropna( axis=1, 
                thresh=min_count)
# filling missing data:
df_f1[['property_type', 'property_style', 'tenure']] = df_f1[['property_type', 'property_style', 'tenure']].fillna(df_f1[['property_type', 'property_style', 'tenure']].mode().iloc[0])
df_f1[['bathrooms', 'bedrooms', 'year_built']] = df_f1[['bathrooms', 'bedrooms', 'year_built']].fillna( df_f1[['bathrooms', 'bedrooms', 'year_built']].median()) 
for col in ('floor_area', 'property_value', 'property_value$confidence',
                         'property_rental_value', 'property_rental_value$confidence'):
    df_f1[col] = df_f1[col].fillna(df_f1[col].mean())  
# deal with boolean:
df_f1 = df_f1.replace({True: 1, False: 0})
# clean up classes:    
df_f1.tenure = df_f1.tenure.replace(['Feudal','Shared'],'other')
df_f1 = df_f1.drop(['prop_id', 'post_town'], axis=1)
# one hot encoder:
object_cols = df_f1.select_dtypes(include=['object']).columns
low_cardinality_cols = [col for col in object_cols if df_f1[col].nunique() < 15]
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))
# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(drop='first' , sparse=False) # Cannot use drop='first' with handle_unknown='ignore'
OH_cols = pd.DataFrame(OH_encoder.fit_transform(df_f1[low_cardinality_cols]))
# One-hot encoding removed index; put it back
OH_cols.index = df_f1.index
# One-hot encoding removed columns' names; put it back
OH_cols.columns = OH_encoder.get_feature_names(low_cardinality_cols)
# Remove categorical columns (will replace with one-hot encoding)
num_X = df_f1.drop(object_cols, axis=1)
OH_X = pd.concat([num_X, OH_cols], axis=1)
# Min-Max Normalize scale data:
DF = OH_X.drop('flag', axis=1)
df_norm = (DF-DF.min())/(DF.max()-DF.min())
OH_X = pd.concat((df_norm, OH_X.flag), 1)
# prepare for prediction:
y_n = OH_X[['flag']]
X_n = OH_X.drop('flag', axis=1)


# In[ ]:


#Predict the response for test dataset
y_pred = Outra_Model.predict(X_n)
y_n['y_test_h'] = y_pred

#df_out = OH_X.merge(y_test, how = 'left', left_index = True, right_index = True)
df_out = pd.merge(df, y_n[['y_test_h']], how = 'left',  left_index = True, right_index = True)

print(df_out.shape)
df_out.head()


# ## Save Results:

# In[ ]:


df_all = df_out.copy(deep=True)
df_all.to_csv('C:/Users/Admin/Documents/Outra/df_All.csv', index=False)


# #   End of Property Prediction Task
# ----

# *********

# ## Modelling
# Regression models:
# * regularised linear regression (Ridge, Lasso & Elastic Net)
# * random forests
# * gradient-boosted trees

# ##  Pipeline object

# In[ ]:


#create an array of models
models = []
models.append(("lesso",Lasso(random_state=123)))
models.append(("ridge",Ridge(random_state=123)))
models.append(("enet",ElasticNet(random_state=123)))
models.append(("rf",RandomForestRegressor(random_state=123)))
models.append(("rf",RandomForestRegressor(random_state=123)))
#models.append(("drf",DecisionTreRegressor(random_state=123)))
models.append(("gb",GradientBoostingRegressor(random_state=123)))


# In[ ]:


fitted_models = {}
X_train = X_train.fillna(0)
y_train = y_train.fillna(0)
#measure the accuracy 
for name,model in models:
    model.fit(X_train, y_train)
    #y_pred = model.predict(X_test)
    fitted_models[name] = model
    '''print(classification_report(y_test, y_pred))
    confusion = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix\n')
    print(confusion)
    print('------------------------------------')'''


# In[ ]:


pipelines = {
    'lasso' : make_pipeline(StandardScaler(),
              Lasso(random_state=123)),
    'ridge' : make_pipeline(StandardScaler(),
              Ridge(random_state=123)),
    'enet' :  make_pipeline(StandardScaler(),
              ElasticNet(random_state=123)),
    'rf' :    make_pipeline(
              RandomForestRegressor(random_state=123)),
    'gb' :    make_pipeline(
              GradientBoostingRegressor(random_state=123))
}


# ##  hyperparameters for each algorithm.
# For all three regularised regressions, tune alpha (L1 & L2 penalty strengths), along with the l1_ratio for Elastic Net (i.e. weighting between L1 & L2 penalties).

# In[ ]:


lasso_hyperparameters = {
    'lasso__alpha' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
ridge_hyperparameters = {
    'ridge__alpha' : [0.001, 0.005, 0.01, 0.1, 0.5, 1, 5, 10]}
enet_hyperparameters = { 
    'elasticnet__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 1, 5, 10], 
    'elasticnet__l1_ratio' : [0.1, 0.3, 0.5, 0.7, 0.9]}

rf_hyperparameters = {
     'randomforestregressor__n_estimators' : [100, 200],
     'randomforestregressor__max_features' : ['auto', 'sqrt', 0.33],
     'randomforestregressor__min_samples_leaf' : [1, 3, 5, 10]}

gb_hyperparameters = {
      'gradientboostingregressor__n_estimators' : [100, 200],
      'gradientboostingregressor__learning_rate' : [0.05, 0.1, 0.2],
      'gradientboostingregressor__max_depth' : [1, 3, 5]}


# In[ ]:


name


# In[ ]:


fitted_models = {}
for name, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, 
                         hyperparameters[name], 
                         cv=10, 
                         n_jobs=-1)
    model.fit(X_train, y_train)
    fitted_models[name] = model


# ## Evaluation
# Performance scores
# We’ll start by printing the cross-validation scores. This is the average performance across the 10 hold-out folds and is a way to get a reliable estimate of the model performance using only your training data.

# In[ ]:


for name, model in fitted_models.items():
    print( name, model.coef_)


# In[ ]:


X_test = X_test.fillna(0)
y_test = y_test.fillna(0)
for name, model in fitted_models.items():
    pred = model.predict(X_test)
    print(name)
    print(' — — — — ')
    print('R²:', r2_score(y_test, pred))
    print('MAE:', mean_absolute_error(y_test, pred))
    print()


# In[ ]:


RandomForestRegressor(bootstrap=True, 
                      criterion='mse',
                      max_depth=None,
                      max_features='auto',
                      max_leaf_nodes=None,
                      min_impurity_decrease=0.0,
                      min_impurity_split=None,
                      min_samples_leaf=10, 
                      min_samples_split=2,
                      min_weight_fraction_leaf=0.0,
                      n_estimators=200, 
                      n_jobs=None,
                      oob_score=False, 
                      random_state=123,
                      verbose=0, 
                      warm_start=False)


# ##  Feature importances

# In[ ]:


coef =  RandomForestRegressor.feature_importances_
ind = np.argsort(-coef)
for i in range(X_train.shape[1]):
    print("%d. %s (%f)" % (i + 1, X.columns[ind[i]], coef[ind[i]]))
#x = range(X_train.shape[1])
x = range(X_train.shape[1])
y = coef[ind][:X_train.shape[1]]
plt.title("Feature importances")
ax = plt.subplot()
plt.barh(x, y, color='red')
ax.set_yticks(x)
ax.set_yticklabels(X.columns[ind])
plt.gca().invert_yaxis()


# In[ ]:


import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
X_train = X_train.fillna(0)
y_train = y_train.fillna(0)
feature_names = [f"feature {i}" for i in range(X_train.shape[1])]
forest = RandomForestRegressor(random_state=0)
forest.fit(X_train, y_train)


start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

import pandas as pd

forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()


# In[ ]:


col = list(df.columns)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import shap
from matplotlib import pyplot as plt
cols = X_train.columns.to_numpy()
plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
print(rf.feature_importances_)
plt.barh(cols, rf.feature_importances_)


# In[ ]:


sorted_idx = rf.feature_importances_.argsort()
plt.barh(cols[sorted_idx], rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")


# ## Permutation Based Feature Importance (with scikit-learn)

# ## Missing values:

# In[ ]:


f1.select_dtypes(include=['object']).isnull().sum()


# In[ ]:


for feat in f1.select_dtypes(include=['object']):
    f1[feat] = f1[feat].fillna('Missing')


# In[ ]:


for feat in f1.dtypes[f1.dtypes == 'object'].index:
    sb.countplot(y=feat, data=f1)
    plt.show()


# In[ ]:





# ## Segmentations

# In[ ]:


for feat in df.dtypes[df.dtypes=='object'].index:
    sb.boxplot(data=df, x = 'property_rental_value', y = '{}'.format(feat))


# In[ ]:


f1.groupby('property_type').agg(['mean','median'])


# In[ ]:


f1.groupby('ht_property_type').agg(['mean','median']) 


# ##  Feature engineering

# In[ ]:


df['two_and_two'] = ((df.beds == 2) & (df.baths == 2)).astype(int)
df['during_recesion'] = ((df.tx_year >= 2010) & 
                          (df.tx_year <= 2013))
                          .astype(int)
df['property_age'] = df.tx_year - df.year_built
df.drop(['tx_year', 'year_built'], axis=1, inplace=True)
df['school_score'] = df.num_schools * df.median_school


# In[ ]:


df.two_and_two.mean()
df.during_recession.mean()


# In[ ]:


X_train = X_train.fillna(0)
y_train = y_train.fillna(0)
perm_importance = permutation_importance(rf, X_test, y_test)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(cols[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")


# ## Feature Importance Computed with SHAP Values

# In[ ]:


explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)


# In[ ]:


shap.summary_plot(shap_values, X_test, plot_type="bar")


# In[ ]:


shap.summary_plot(shap_values, X_test)


# In[ ]:





# The top two predictors by far are
# * cost of monthly homeowner’s insurance and
# * monthly property tax.

# ## Deployment

# In[ ]:


with open('final_model.pkl', 'wb') as f:
    pickle.dump(fitted_models['rf'].best_estimator_, f)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


df = f0.merge(f1, left_on = 'CountryID', right_on = 'ID')
df = df.drop(['ID_y' ], axis = 1)
df.rename(columns={'ID_x': 'ID', 'Code': 'Country_Code', 'Name':'Country'}, inplace=True)
print(df.shape)
df.head(2)


# In[ ]:


df = df.merge(f2, left_on = 'CustomerID', right_on = 'ID')
df = df.drop(['ID_y','CountryID_y' ], axis = 1)
df.rename(columns={'ID_x': 'ID', 'CountryID_x':'CountryID','Code':'Customer_Code', 'Name':'Customer_Name'}, inplace=True)
print(df.shape)
df.head(2).sort_values(by=['ID'])


# In[ ]:


df = df.merge(f3, left_on = 'ProductID', right_on = 'ID')
df = df.drop(['ID_y' ], axis = 1)
df.rename(columns={'ID_x': 'ID','Name':'Product_Name','Code':'Product_Code'}, inplace=True)
print(df.shape)
df.head(2)


# In[ ]:


df.columns


# In[ ]:


# Drop duplicate rows and keep the first one:
df = df.drop_duplicates()
df.shape


# In[ ]:


df.columns.to_series().groupby(df.dtypes).groups


# In[ ]:


df.describe()


# In[ ]:


df.corr()


# In[ ]:


print('Products: ' , df.ProductID.nunique())
print('Countries: ' , df.Country_Code.nunique())
print('Customer_Code: '+ str(df.Customer_Code.nunique()))
print('Customer_Name: ', (df.Customer_Name.nunique()))


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dtypes


# In[ ]:


df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
print(df.dtypes)
df.head()


# # Make a copy:

# In[ ]:


df_all = df.copy(deep=True)
df_all.to_csv('C:/Users/User/OneDrive/GAC/df_all.csv')


# In[ ]:


df.hist(figsize=(20,20))
plt.show()


# #### Listings price distribution after removing outliers

# In[ ]:


df.loc[(df.Price <= 20) & (df.Price > 0)].Price.hist(bins=50)
plt.ylabel('Count')
plt.xlabel('Listing price')
plt.title('Histogram of listing prices');


# In[ ]:


print('Max Product Price: ', df.Price.max())


# In[ ]:


df.loc[(df.Price >= 2000) & (df.Price > 0)].Price.hist(bins=50)
plt.ylabel('Count')
plt.xlabel('Listing price')
plt.title('Histogram of listing prices');


# In[ ]:


corr_matrix = df.corr()
corr_matrix["CustomerID"].sort_values(ascending=False)


# In[ ]:


import seaborn as sns

Var_Corr = df.corr()
Var_Corr


# In[ ]:


# Increase the size of the heatmap.
plt.figure(figsize=(16, 6))
# Store heatmap object in a variable to easily access it when you want to include more features (such as title).
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
heatmap = sns.heatmap(Var_Corr.corr(), vmin=-1, vmax=1, annot=True)
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


# In[ ]:


plt.figure(figsize=(16, 6))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(Var_Corr.corr(), dtype=np.bool))
heatmap = sns.heatmap(Var_Corr.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16);


# In[ ]:


Var_Corr.corr()[['Quantity']].sort_values(by='Quantity', ascending=False)


# In[ ]:


plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(Var_Corr.corr()[['Quantity']].sort_values(by='Quantity', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with Sales Price', fontdict={'fontsize':18}, pad=16);


# In[ ]:


# use groupby to aggregate sales by CustomerID
customer_df = df.groupby('CustomerID').agg({'Amount': sum, 
                                            'InvoiceNumber': lambda x: x.nunique()})

# Select the columns we want to use
customer_df.columns = ['TotalSales', 'OrderCount'] 

# create a new column 'AvgOrderValu'
customer_df['AvgOrderValue'] = customer_df['TotalSales'] / customer_df['OrderCount']

customer_df.head()


# In[ ]:


rank_df = customer_df.rank(method='first')
normalized_df = (rank_df - rank_df.mean()) / rank_df.std()
normalized_df.head(5)


# In[ ]:


df.shape


# In[ ]:


result_df = df.drop_duplicates()
result_df.shape


# ## Monthly Revenue

# In[ ]:


#converting the type of Invoice Date Field from string to datetime.
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

#creating YearMonth field for the ease of reporting and visualization
df['InvoiceYearMonth'] = df['InvoiceDate'].map(lambda date: 100*date.year + date.month)

#calculate Revenue for each row and create a new dataframe with YearMonth - Revenue columns
df['Revenue'] = df['Price'] * df['Quantity']
df_revenue = df.groupby(['InvoiceYearMonth'])['Revenue'].sum().reset_index()
df_revenue


# In[ ]:


#import plotly.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go


# In[ ]:


#X and Y axis inputs for Plotly graph. We use Scatter for line graphs
plot_data = [
    go.Scatter(
        x=df_revenue['InvoiceYearMonth'],
        y=df_revenue['Revenue'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Montly Revenue'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[ ]:


#using pct_change() function to see monthly percentage change
df_revenue['MonthlyGrowth'] = df_revenue['Revenue'].pct_change()

#showing first 5 rows
df_revenue.head()

#visualization - line graph
plot_data = [
    go.Scatter(
        x=df_revenue.query("InvoiceYearMonth < 201112")['InvoiceYearMonth'],
        y=df_revenue.query("InvoiceYearMonth < 201112")['MonthlyGrowth'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Montly Growth Rate'
    )

fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[ ]:


df.head()


# In[ ]:


#creating a new dataframe with UK customers only
df_uk = df.query("Country_Code=='GB'").reset_index(drop=True)

#creating monthly active customers dataframe by counting unique Customer IDs
df_monthly_active = df_uk.groupby('InvoiceYearMonth')['CustomerID'].nunique().reset_index()

#print the dataframe
df_monthly_active

#plotting the output
plot_data = [
    go.Bar(
        x=df_monthly_active['InvoiceYearMonth'],
        y=df_monthly_active['CustomerID'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Active Customers'
    )

fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[ ]:


#print the dataframe
df_monthly_active


# In[ ]:


#create a new dataframe for no. of order by using quantity field
df_monthly_sales = df_uk.groupby('InvoiceYearMonth')['Quantity'].sum().reset_index()

#print the dataframe
df_monthly_sales

#plot
plot_data = [
    go.Bar(
        x=df_monthly_sales['InvoiceYearMonth'],
        y=df_monthly_sales['Quantity'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Total # of Order'
    )

fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[ ]:


#print the dataframe
df_monthly_sales


# In[ ]:


# create a new dataframe for average revenue by taking the mean of it
df_monthly_order_avg = df_uk.groupby('InvoiceYearMonth')['Revenue'].mean().reset_index()

#print the dataframe
df_monthly_order_avg

#plot the bar chart
plot_data = [
    go.Bar(
        x=df_monthly_order_avg['InvoiceYearMonth'],
        y=df_monthly_order_avg['Revenue'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Order Average'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[ ]:


#print the dataframe
df_monthly_order_avg


# In[ ]:


#create a dataframe contaning CustomerID and first purchase date
tx_min_purchase = df_uk.groupby('CustomerID').InvoiceDate.min().reset_index()
tx_min_purchase.columns = ['CustomerID','MinPurchaseDate']
tx_min_purchase['MinPurchaseYearMonth'] = tx_min_purchase['MinPurchaseDate'].map(lambda date: 100*date.year + date.month)

#merge first purchase date column to our main dataframe (tx_uk)
df_uk = pd.merge(df_uk, tx_min_purchase, on='CustomerID')

df_uk.head()

#create a column called User Type and assign Existing 
#if User's First Purchase Year Month before the selected Invoice Year Month
df_uk['UserType'] = 'New'
df_uk.loc[df_uk['InvoiceYearMonth']>df_uk['MinPurchaseYearMonth'],'UserType'] = 'Existing'

#calculate the Revenue per month for each user type
tx_user_type_revenue = df_uk.groupby(['InvoiceYearMonth','UserType'])['Revenue'].sum().reset_index()

#filtering the dates and plot the result
tx_user_type_revenue = tx_user_type_revenue.query("InvoiceYearMonth != 201012 and InvoiceYearMonth != 201112")
plot_data = [
    go.Scatter(
        x=tx_user_type_revenue.query("UserType == 'Existing'")['InvoiceYearMonth'],
        y=tx_user_type_revenue.query("UserType == 'Existing'")['Revenue'],
        name = 'Existing'
    ),
    go.Scatter(
        x=tx_user_type_revenue.query("UserType == 'New'")['InvoiceYearMonth'],
        y=tx_user_type_revenue.query("UserType == 'New'")['Revenue'],
        name = 'New'
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='New vs Existing'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[ ]:


tx_user_type_revenue = tx_user_type_revenue.query("InvoiceYearMonth != 201012 and InvoiceYearMonth != 201112")
tx_user_type_revenue


# In[ ]:


df_uk.head()


# In[ ]:


#create a dataframe that shows new user ratio - we also need to drop NA values (first month new user ratio is 0)
tx_user_ratio = df_uk.query("UserType == 'New'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique()/df_uk.query("UserType == 'Existing'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique() 
tx_user_ratio = tx_user_ratio.reset_index()
tx_user_ratio = tx_user_ratio.dropna()

#print the dafaframe
tx_user_ratio

#plot the result

plot_data = [
    go.Bar(
        x=tx_user_ratio.query("InvoiceYearMonth>201101 and InvoiceYearMonth<201112")['InvoiceYearMonth'],
        y=tx_user_ratio.query("InvoiceYearMonth>201101 and InvoiceYearMonth<201112")['CustomerID'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='New Customer Ratio'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[ ]:


#print the dafaframe
tx_user_ratio


# In[ ]:


#identify which users are active by looking at their revenue per month
tx_user_purchase = df_uk.groupby(['CustomerID','InvoiceYearMonth'])['Revenue'].sum().reset_index()

#create retention matrix with crosstab
tx_retention = pd.crosstab(tx_user_purchase['CustomerID'], tx_user_purchase['InvoiceYearMonth']).reset_index()

tx_retention.head()

#create an array of dictionary which keeps Retained & Total User count for each month
months = tx_retention.columns[2:]
retention_array = []
for i in range(len(months)-1):
    retention_data = {}
    selected_month = months[i+1]
    prev_month = months[i]
    retention_data['InvoiceYearMonth'] = int(selected_month)
    retention_data['TotalUserCount'] = tx_retention[selected_month].sum()
    retention_data['RetainedUserCount'] = tx_retention[(tx_retention[selected_month]>0) & (tx_retention[prev_month]>0)][selected_month].sum()
    retention_array.append(retention_data)
    
#convert the array to dataframe and calculate Retention Rate
tx_retention = pd.DataFrame(retention_array)
tx_retention['RetentionRate'] = tx_retention['RetainedUserCount']/tx_retention['TotalUserCount']

#plot the retention rate graph
plot_data = [
    go.Scatter(
        x=tx_retention.query("InvoiceYearMonth<201112")['InvoiceYearMonth'],
        y=tx_retention.query("InvoiceYearMonth<201112")['RetentionRate'],
        name="organic"
    )
    
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Retention Rate'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[ ]:


#identify which users are active by looking at their revenue per month
tx_user_purchase = df_uk.groupby(['CustomerID','InvoiceYearMonth'])['Revenue'].sum().reset_index()
tx_user_purchase.head()


# # Cohort Analysis with Python

# In[ ]:


dft19 = df
def daily(x):     
    return dt.datetime(x.year, x.month, x.day, 1)
dft19['BillDay'] = dft19['InvoiceDate'].apply(daily)
df_sum19 = dft19.groupby('BillDay').sum() # .drop('CustomerID', axis = 1)
df_sum19 = df_sum19.rename(columns={'Amount' : 'GrossProfit'})

plt.figure(figsize=(16,7))
sns.lineplot(x = df_sum19.index, y = df_sum19['GrossProfit'])
plt.title("Revenue")
#plt.ylim([0, 1450000])
plt.show()


# In[ ]:


dft19 = df
def monthly(x):     
    return dt.datetime(x.year, x.month, 1)
dft19['BillMonth'] = dft19['InvoiceDate'].apply(monthly)
df_sum = dft19.groupby('BillMonth').sum() # .drop('CustomerID', axis = 1)
df_sum = df_sum.rename(columns={'Amount' : 'GrossProfit'})
plt.figure(figsize=(16,7))
sns.lineplot(x = df_sum.index, y = df_sum['GrossProfit'])
plt.title("Revenue")
#plt.ylim([0, 1450000])
plt.show()


# In[ ]:


df.columns


# In[ ]:


g = dft19.groupby('CustomerID')['BillMonth']
dft19['CohortMonth'] = g.transform('min')
dft19.head(50)


# In[ ]:


def get_int(dft19, column):
    year = dft19[column].dt.year
    month = dft19[column].dt.month
    return year, month
billYear, billMonth = get_int(dft19, 'BillMonth')
cohortYear, cohortMonth = get_int(dft19, 'CohortMonth')
diffYear = billYear - cohortYear
diffMonth = billMonth - cohortMonth
dft19['Month_Index'] = diffYear * 12 + diffMonth + 1
dft19.head()


# In[ ]:


dft19['CohortMonth'] = dft19['CohortMonth'].apply(dt.datetime.date)
g = dft19.groupby(['CohortMonth', 'Month_Index'])
cohortData = g['CustomerID'].apply(pd.Series.nunique).reset_index()
cohortCounts = cohortData.pivot(index = 'CohortMonth', columns = 'Month_Index', values = 'CustomerID')
cohortSizes = cohortCounts.iloc[:, 0]
retention = cohortCounts.divide(cohortSizes, axis = 0) * 100
retention.round(2)


# In[ ]:


month_list = ["Jan '10", "Feb '10", "Mar '10", "Apr '10", "May '10", "Jun '10", "Jul '10", "Aug '10", "Sep '10", "Oct '10", "Nov '10", "Dec '10"
              , "Jan '11", "Feb '11", "Mar '11", "Apr '11", "May '11", "Jun '11", "Jul '11", "Aug '11", "Sep '11", "Oct '11", "Nov '11", "Dec '11"]
plt.figure(figsize = (20,10))
plt.title('Retention by Monthly Cohorts')
sns.heatmap(retention.round(2), annot = True, cmap = "Blues", vmax = list(retention.max().sort_values(ascending = False))[1]+3, fmt = '.1f', linewidth = 0.3, yticklabels=month_list)
plt.show()


# ## Select the optimal number of clusters

# ### Silhouette method

# In[ ]:


# Use silhouette coefficient to determine the best number of clusters
from sklearn.metrics import silhouette_score

for n_cluster in [4,5,6,7,8]:
    kmeans = KMeans(n_clusters=n_cluster).fit(
        normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']])
    
    silhouette_avg = silhouette_score(
        normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']], 
        kmeans.labels_)
    
    print('Silhouette Score for %i Clusters: %0.4f' % (n_cluster, silhouette_avg))


# ### The Elbow Method with the Sum of Squared Errors (SSE)

# In[ ]:


from sklearn import cluster
import numpy as np

sse = []
krange = list(range(2,11))
X = normalized_df[['TotalSales','OrderCount','AvgOrderValue']].values
for n in krange:
    model = cluster.KMeans(n_clusters=n, random_state=3)
    model.fit_predict(X)
    cluster_assignments = model.labels_
    centers = model.cluster_centers_
    sse.append(np.sum((X - centers[cluster_assignments]) ** 2))

plt.plot(krange, sse)
plt.grid()
plt.xlabel("$K$")
plt.ylabel("Sum of Squares")
plt.show()


# Based on the graph above, it looks like K=4, or 4 clusters is the optimal number of clusters for this analysis. Now let’s interpret the customer segments provided by these clusters.
# 
# 

# # Interpreting Customer Segments

# In[ ]:


kmeans = KMeans(n_clusters=4).fit(normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']])

four_cluster_df = normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']].copy(deep=True)
four_cluster_df['Cluster'] = kmeans.labels_

four_cluster_df.head(10)


# Now let’s group the cluster metrics and see what we can gather from the normalized data for each cluster.

# In[ ]:


cluster1_metrics = kmeans.cluster_centers_[0]
cluster2_metrics = kmeans.cluster_centers_[1]
cluster3_metrics = kmeans.cluster_centers_[2]
cluster4_metrics = kmeans.cluster_centers_[3]

data = [cluster1_metrics, cluster2_metrics, cluster3_metrics, cluster4_metrics]
cluster_center_df = pd.DataFrame(data)

cluster_center_df.columns = four_cluster_df.columns[0:3]
cluster_center_df


# # Visualizing Clusters

# In[ ]:


plt.scatter(
four_cluster_df.loc[four_cluster_df['Cluster'] == 0]['OrderCount'], 
four_cluster_df.loc[four_cluster_df['Cluster'] == 0]['TotalSales'],
c='blue')
plt.scatter(
four_cluster_df.loc[four_cluster_df['Cluster'] == 1]['OrderCount'], 
four_cluster_df.loc[four_cluster_df['Cluster'] == 1]['TotalSales'],
c='red')
plt.scatter(
four_cluster_df.loc[four_cluster_df['Cluster'] == 2]['OrderCount'], 
four_cluster_df.loc[four_cluster_df['Cluster'] == 2]['TotalSales'],
c='orange')
plt.scatter(
four_cluster_df.loc[four_cluster_df['Cluster'] == 3]['OrderCount'], 
four_cluster_df.loc[four_cluster_df['Cluster'] == 3]['TotalSales'],
c='green')
plt.title('TotalSales vs. OrderCount Clusters')
plt.xlabel('Order Count')
plt.ylabel('Total Sales')
plt.grid()
plt.show()
plt.scatter(
four_cluster_df.loc[four_cluster_df['Cluster'] == 0]['OrderCount'], 
four_cluster_df.loc[four_cluster_df['Cluster'] == 0]['AvgOrderValue'],
c='blue')
plt.scatter(
four_cluster_df.loc[four_cluster_df['Cluster'] == 1]['OrderCount'], 
four_cluster_df.loc[four_cluster_df['Cluster'] == 1]['AvgOrderValue'],
c='red')
plt.scatter(
four_cluster_df.loc[four_cluster_df['Cluster'] == 2]['OrderCount'], 
four_cluster_df.loc[four_cluster_df['Cluster'] == 2]['AvgOrderValue'],
c='orange')
plt.scatter(
four_cluster_df.loc[four_cluster_df['Cluster'] == 3]['OrderCount'], 
four_cluster_df.loc[four_cluster_df['Cluster'] == 3]['AvgOrderValue'],
c='green')
plt.title('AvgOrderValue vs. OrderCount Clusters')
plt.xlabel('Order Count')
plt.ylabel('Avg Order Value')
plt.grid()
plt.show()
plt.scatter(
four_cluster_df.loc[four_cluster_df['Cluster'] == 0]['TotalSales'], 
four_cluster_df.loc[four_cluster_df['Cluster'] == 0]['AvgOrderValue'],
c='blue')
plt.scatter(
four_cluster_df.loc[four_cluster_df['Cluster'] == 1]['TotalSales'], 
four_cluster_df.loc[four_cluster_df['Cluster'] == 1]['AvgOrderValue'],
c='red')
plt.scatter(
four_cluster_df.loc[four_cluster_df['Cluster'] == 2]['TotalSales'], 
four_cluster_df.loc[four_cluster_df['Cluster'] == 2]['AvgOrderValue'],
c='orange')
plt.scatter(
four_cluster_df.loc[four_cluster_df['Cluster'] == 3]['TotalSales'], 
four_cluster_df.loc[four_cluster_df['Cluster'] == 3]['AvgOrderValue'],
c='green')
plt.title('AvgOrderValue vs. TotalSales Clusters')
plt.xlabel('Total Sales')
plt.ylabel('Avg Order Value')
plt.grid()
plt.show()


# In the first plot, customers in orange have low total sales AND low order count, meaning they are all-around low-value customers. On the other hand, the customers in red have high total sales AND high order counts, indicating they are the highest value customers.
# 
# In the second plot, we’re looking at the average order value vs the order count. Once again, the customers in orange are the lowest value customers and the customers in red are the highest value customers.
# 
# In the last plot, we have the average order value versus total sales clusters. This plot further substantiates the previous 2 plots in identifying the red cluster as the highest value customers, orange as the lowest value customers, and the blue and green as high opportunity customers.

# # Find the best-selling item by segment
# We can better understand the customer segments is to identify which items are the best-selling within each segment. Highest Cluster is the red 'Cluster = 1'

# In[ ]:


high_value_cluster = four_cluster_df.loc[four_cluster_df['Cluster'] == 0]
pd.DataFrame(df.loc[df['CustomerID'].isin(high_value_cluster.index)].groupby(
'Product_Name').count()['Product_Code'].sort_values(ascending=False).head(10))


# Based on this information, we now know that the WHITE HANGING HEART T-LIGHT HOLDER is the best-selling item for our highest-value cluster. With that information in hand, we can make recommendations of __*Other Items You Might Like*__ to customers within this segment. These actions can be taken to another level of specificity with Association Rule Mining and Market Basket Analysis which I’ll cover below.

# In[ ]:


# from the above we can put a threshold number to avoid numbers e.g. max 25
print(four_cluster_df['Cluster'].value_counts())


# In[ ]:


print('Customers in Blue Cluster %' , 1383/5772*100)
print('Customers in Red Cluster %' ,1538/5772*100)
print('Customers in Orange Cluster %' ,1334/5772*100)
print('Customers in Green Cluster %' ,1517/5772*100)


# ## Customer Lifetime Value

# In[ ]:


#read data from csv and redo the data work we done before
tx_data = df_all.copy()
#tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])
tx_uk = tx_data.query("Country_Code=='GB'").reset_index(drop=True)

#create 3m and 6m dataframes
tx_3m = tx_uk[(tx_uk.InvoiceDate < datetime(2011,6,1)) & (tx_uk.InvoiceDate >= datetime(2011,3,1))].reset_index(drop=True)
tx_6m = tx_uk[(tx_uk.InvoiceDate >= datetime(2011,6,1)) & (tx_uk.InvoiceDate < datetime(2011,12,1))].reset_index(drop=True)

#create tx_user for assigning clustering
tx_user = pd.DataFrame(tx_3m['CustomerID'].unique())
tx_user.columns = ['CustomerID']

#order cluster method
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final


#calculate recency score
tx_max_purchase = tx_3m.groupby('CustomerID').InvoiceDate.max().reset_index()
tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']
tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days
tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')

kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Recency']])
tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])

tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,False)

#calcuate frequency score
tx_frequency = tx_3m.groupby('CustomerID').InvoiceDate.count().reset_index()
tx_frequency.columns = ['CustomerID','Frequency']
tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')

kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Frequency']])
tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])

tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)

#calcuate revenue score
tx_3m['Revenue'] = tx_3m['Price'] * tx_3m['Quantity']
tx_revenue = tx_3m.groupby('CustomerID').Revenue.sum().reset_index()
tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')

kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Revenue']])
tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])
tx_user = order_cluster('RevenueCluster', 'Revenue',tx_user,True)


#overall scoring
tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']
tx_user['Segment'] = 'Low-Value'
tx_user.loc[tx_user['OverallScore']>1,'Segment'] = 'Mid-Value' 
tx_user.loc[tx_user['OverallScore']>3,'Segment'] = 'High-Value' 
tx_user.loc[tx_user['OverallScore']>5,'Segment'] = 'Very-High-Value'


# In[ ]:


#calculate revenue and create a new dataframe for it
tx_6m['Revenue'] = tx_6m['Price'] * tx_6m['Quantity']
tx_user_6m = tx_6m.groupby('CustomerID')['Revenue'].sum().reset_index()
tx_user_6m.columns = ['CustomerID','m6_Revenue']


#plot LTV histogram
plot_data = [
    go.Histogram(
        x=tx_user_6m.query('m6_Revenue < 10000')['m6_Revenue']
    )
]

plot_layout = go.Layout(
        title='6m Revenue'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[ ]:


tx_merge = pd.merge(tx_user, tx_user_6m, on='CustomerID', how='left')
tx_merge = tx_merge.fillna(0)

tx_graph = tx_merge.query("m6_Revenue < 30000")

plot_data = [
    go.Scatter(
        x=tx_graph.query("Segment == 'Low-Value'")['OverallScore'],
        y=tx_graph.query("Segment == 'Low-Value'")['m6_Revenue'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'Mid-Value'")['OverallScore'],
        y=tx_graph.query("Segment == 'Mid-Value'")['m6_Revenue'],
        mode='markers',
        name='Mid',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'High-Value'")['OverallScore'],
        y=tx_graph.query("Segment == 'High-Value'")['m6_Revenue'],
        mode='markers',
        name='High',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'orange',
            opacity= 0.9
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'Very-High-Value'")['OverallScore'],
        y=tx_graph.query("Segment == 'Very-High-Value'")['m6_Revenue'],
        mode='markers',
        name='Very_High',
        marker= dict(size= 14,
            line= dict(width=1),
            color= 'red',
            opacity= 0.7
           )
    )
]

plot_layout = go.Layout(
        yaxis= {'title': "6m LTV"},
        xaxis= {'title': "RFM Score"},
        title='LTV'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[ ]:


#remove outliers
'''What are Outliers?
An outlier is an observation that is unlike the other observations.
It is rare, or distinct, or does not fit in some way.
We will generally define outliers as samples that are exceptionally far from the mainstream of the data.

Outliers can have many causes, such as:
Measurement or input error.
Data corruption.
True outlier observation (e.g. Michael Jordan in basketball).
'''
tx_merge = tx_merge[tx_merge['m6_Revenue']<tx_merge['m6_Revenue'].quantile(0.99)]


#creating 3 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(tx_merge[['m6_Revenue']])
tx_merge['LTVCluster'] = kmeans.predict(tx_merge[['m6_Revenue']])

#order cluster number based on LTV
tx_merge = order_cluster('LTVCluster', 'm6_Revenue',tx_merge,True)

#creatinga new cluster dataframe
tx_cluster = tx_merge.copy()

#see details of the clusters
tx_cluster.groupby('LTVCluster')['m6_Revenue'].describe()


# ## Data engineering for ML

# # Transfer data to training and testing set:

# In[ ]:


#ss = StandardScaler()
#data_df = ss.fit_transform(tx_class)
#X = tx_class.drop(['LTVCluster','m6_Revenue'],axis=1)
#y = tx_class['LTVCluster']


# In[ ]:


#convert categorical columns to numerical
tx_class = pd.get_dummies(tx_cluster)

#create X and y, X will be feature set and y is the label - LTV
#X = tx_class.drop(['LTVCluster','m6_Revenue'],axis=1)
#y = tx_class['LTVCluster']


data = tx_class.copy()
y = data["LTVCluster"].values
X = data[['CustomerID', 'Recency', 'RecencyCluster', 'Frequency',
       'FrequencyCluster', 'Revenue', 'RevenueCluster', 'OverallScore',
       'Segment_High-Value', 'Segment_Low-Value',
       'Segment_Mid-Value']].values


#split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56) # 5% training and 30% test
print(X.shape)
print(y.shape)


# In[ ]:


#calculate and show correlations
corr_matrix = tx_class.corr()
corr_matrix['LTVCluster'].sort_values(ascending=False)


# In[ ]:


tx_cluster.LTVCluster.unique()


# In[ ]:


tx_cluster.head()


# In[ ]:


tx_class.head()


# from sklearn.preprocessing import StandardScaler
# ss = StandardScaler()
# data_df = ss.fit_transform(tx_class)
# X = tx_class.drop(['LTVCluster','m6_Revenue'],axis=1)
# y = tx_class['LTVCluster']
# #Divide into training and test data
# Xa_train, Xa_test, ya_train, ya_test = train_test_split(Xa, ya, test_size=0.2, random_state=56) # 5% training and 30% test

# # import the necessary packages
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.datasets import fetch_california_housing
# from mlxtend.evaluate import bias_variance_decomp

# tx_class.columns
# X = tx_class.drop(['LTVCluster','m6_Revenue'],axis=1)
# y = tx_class['LTVCluster']
# 

# In[ ]:


#create an array of models
models = []
models.append(("LR",LogisticRegression()))
models.append(("NB",GaussianNB()))
models.append(("RF",RandomForestClassifier()))
models.append(("SVC",SVC()))
models.append(("Dtree",DecisionTreeClassifier()))
models.append(("XGB",xgb.XGBClassifier()))
models.append(("KNN",KNeighborsClassifier()))

#measure the accuracy 
for name,model in models:
    kfold = KFold(n_splits=3, random_state=22)
    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")
    print(name, cv_result.max())
    # estimating the bias and variance
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(model, X_train,
                                                            y_train, X_test,
                                                            y_test,
                                                            loss='mse',
                                                            num_rounds=50,
                                                            random_seed=20)
 
    # summary of the results
    print('Average expected loss: %.3f' % avg_expected_loss)
    print('Average bias: %.3f' % avg_bias)
    print('Average variance: %.3f' % avg_var)
    print('-----------------------------------')


# In[ ]:


#create an array of models
models = []
models.append(("LR",LogisticRegression()))
models.append(("NB",GaussianNB()))
models.append(("RF",RandomForestClassifier()))
models.append(("SVC",SVC()))
models.append(("Dtree",DecisionTreeClassifier()))
models.append(("XGB",xgb.XGBClassifier()))
models.append(("KNN",KNeighborsClassifier()))


#measure the accuracy 
for name,model in models:
    kfold = KFold(n_splits=3, random_state=22)
    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")
    print(name, cv_result.max())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    confusion = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix\n')
    print(confusion)
    print('------------------------------------')


# In[ ]:


# preparing the dataset into inputs (feature matrix) and outputs (target vector)
data = tx_class.copy()
ya = data["LTVCluster"].values
Xa = data[['CustomerID', 'Recency', 'RecencyCluster', 'Frequency',
       'FrequencyCluster', 'Revenue', 'RevenueCluster', 'OverallScore',
       'Segment_High-Value', 'Segment_Low-Value',
       'Segment_Mid-Value']].values

#create an array of models
models = []
models.append(("LR",LogisticRegression()))
models.append(("NB",GaussianNB()))
models.append(("RF",RandomForestClassifier()))
models.append(("SVC",SVC()))
models.append(("Dtree",DecisionTreeClassifier()))
models.append(("XGB",xgb.XGBClassifier()))
models.append(("KNN",KNeighborsClassifier()))


# Data frame for evaluation metrics
metrics = pd.DataFrame(index=['Accuracy','Average expected loss', 'Average bias', 'Average variance'], 
                      columns=['LR','NB', 'RF', 'SVC','Dtree','XGB','KNN'])

#measure the accuracy 
for name,model in models:
    kfold = KFold(n_splits=3, random_state=22)
    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")
    #print(name, cv_result.max())
    # estimating the bias and variance
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(model, X_train,
                                                            y_train, X_test,
                                                            y_test,
                                                            loss='mse',
                                                            num_rounds=50,
                                                            random_seed=20)
 
    # summary of the results
    # 4. Evaluate the model
    metrics.loc['Average expected loss',name] = round(avg_expected_loss*100,2)
    metrics.loc['Average bias',name] = round(avg_bias*100,2)
    metrics.loc['Average variance',name] = round(avg_var*100,2)
    metrics.loc['Accuracy',name] = round(cv_result.max()*100,2)
    
metrics


# In[ ]:


fig, ax = plt.subplots(figsize=(8,5))
metrics.plot(kind='barh', ax=ax)
plt.legend(loc=1)
ax.grid();


# ## Hyperparameter Tuning

# In[ ]:


from sklearn.model_selection import GridSearchCV
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier(), 
param_grid = param_test1, scoring='accuracy',n_jobs=-1,iid=False, cv=2)
gsearch1.fit(X_train,y_train)
gsearch1.best_params_, gsearch1.best_score_


# In[ ]:


#XGBoost Multiclassification Model
xgb_model = xgb.XGBClassifier(max_depth=3, min_child_weight = 1).fit(X_train, y_train)
print('Accuracy of XGB classifier on training set: {:.2f}'
       .format(xgb_model.score(X_train, y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'
       .format(xgb_model.score(X_test, y_test)))

y_pred = xgb_model.predict(X_test)
print(classification_report(y_test, y_pred))
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(xgb_model, X_train,
                                                            y_train, X_test,
                                                            y_test,
                                                            loss='mse',
                                                            num_rounds=50,
                                                            random_seed=20)
 
# summary of the results
# 4. Evaluate the model
print('Average expected loss',round(avg_expected_loss*100,2))
print('Average bias' ,round(avg_bias*100,2))
print('Average variance',round(avg_var*100,2))
print('Accuracy', round(cv_result.max()*100,2))
    


# In[ ]:


#XGBoost Multiclassification Model
ltv_xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1,objective= 'multi:softprob',n_jobs=-1).fit(X_train, y_train)

print('Accuracy of XGB classifier on training set: {:.2f}'
       .format(ltv_xgb_model.score(X_train, y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'
       .format(ltv_xgb_model.score(X_test, y_test)))

y_pred = ltv_xgb_model.predict(X_test)
print(classification_report(y_test, y_pred))
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(ltv_xgb_model, X_train,
                                                            y_train, X_test,
                                                            y_test,
                                                            loss='mse',
                                                            num_rounds=50,
                                                            random_seed=20)
# summary of the results
# 4. Evaluate the model
print('Average expected loss',round(avg_expected_loss*100,2))
print('Average bias' ,round(avg_bias*100,2))
print('Average variance',round(avg_var*100,2))
print('Accuracy', round(cv_result.max()*100,2))


# In[ ]:


#Logistic Regression Multiclassification Model
ltv_LR_model = LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(X_train, y_train)

print('Accuracy of LR classifier on training set: {:.2f}'
       .format(ltv_LR_model.score(X_train, y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'
       .format(ltv_LR_model.score(X_test, y_test)))

y_pred = ltv_LR_model.predict(X_test)
print(classification_report(y_test, y_pred))
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(ltv_LR_model, X_train,
                                                            y_train, X_test,
                                                            y_test,
                                                            loss='mse',
                                                            num_rounds=50,
                                                            random_seed=20)
# summary of the results
# 4. Evaluate the model
print('Average expected loss',round(avg_expected_loss*100,2))
print('Average bias' ,round(avg_bias*100,2))
print('Average variance',round(avg_var*100,2))
print('Accuracy', round(cv_result.max()*100,2))


# In[ ]:


train_score = []
test_score = []
k_vals = []

for k in range(1, 21):
    k_vals.append(k)
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    
    tr_score = knn.score(X_train, y_train)
    train_score.append(tr_score)
    
    te_score = knn.score(X_test, y_test)
    test_score.append(te_score)


# In[ ]:


plt.figure(figsize=(10,5))
plt.xlabel('Different Values of K')
plt.ylabel('Model score')
plt.plot(k_vals, train_score, color = 'r', label = "training score")
plt.plot(k_vals, test_score, color = 'b', label = 'test score')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()


# ## Normalize data

# In[ ]:


rank_df_n = tx_class[['CustomerID', 'Recency', 'RecencyCluster', 'Frequency',
       'FrequencyCluster', 'Revenue', 'RevenueCluster', 'OverallScore',
       'm6_Revenue', 'Segment_High-Value', 'Segment_Low-Value',
       'Segment_Mid-Value']].rank(method='first')
normalized_df_n = (rank_df_n - rank_df_n.mean()) / rank_df_n.std()
print('Max & Min = ' , normalized_df_n.max().max() , normalized_df_n.min().min())
normalized_df_n.head(5)


# In[ ]:


tx_class['LTVCluster'].unique()


# In[ ]:


#create X and y, X will be feature set and y is the label - LTV

# preparing the dataset into inputs (feature matrix) and outputs (target vector)
yn = tx_class["LTVCluster"].values
# Exclude m6_Revenue
Xn = normalized_df_n[['CustomerID', 'Recency', 'RecencyCluster', 'Frequency',
       'FrequencyCluster', 'Revenue', 'RevenueCluster', 'OverallScore',
       'Segment_High-Value', 'Segment_Low-Value',
       'Segment_Mid-Value']].values


#split training and test sets
Xn_train, Xn_test, yn_train, yn_test = train_test_split(Xn, yn, test_size=0.2, random_state=56)


# In[ ]:


#measure the accuracy 
for name,model in models:
    kfold = KFold(n_splits=3, random_state=22)
    cv_result = cross_val_score(model,Xn_train,yn_train, cv = kfold,scoring = "accuracy")
    print(name, cv_result.max())
    model.fit(Xn_train, yn_train)
    yn_pred = model.predict(Xn_test)
    print('Accuracy on training set: {:.2f}'
       .format(model.score(Xn_train, yn_train)))
    print('Accuracy on test set: {:.2f}'
       .format(model.score(Xn_test, yn_test)))
    print(classification_report(yn_test, yn_pred))
    confusion = confusion_matrix(yn_test, yn_pred)
    print('Confusion Matrix\n')
    print(confusion)
    print('------------------------------------')



# In[ ]:


#measure the accuracy 
for name,model in models:
    kfold = KFold(n_splits=3, random_state=22)
    cv_result = cross_val_score(model,Xn_train,yn_train, cv = kfold,scoring = "accuracy")
    print(name, cv_result.max())
    model.fit(Xn_train, yn_train)
    yn_pred = model.predict(Xn_test)
    print(classification_report(yn_test, yn_pred))
    confusion = confusion_matrix(yn_test, yn_pred)
    print('Confusion Matrix\n')
    print(confusion)
    print('------------------------------------')


# In[ ]:


# Data frame for evaluation metrics
metrics = pd.DataFrame(index=['Accuracy','Average expected loss', 'Average bias', 'Average variance'], 
                      columns=['LR','NB', 'RF', 'SVC','Dtree','XGB','KNN'])

#measure the accuracy 
for name,model in models:
    kfold = KFold(n_splits=3, random_state=22)
    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")
    #print(name, cv_result.max())
    # estimating the bias and variance
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(model, Xn_train,
                                                            yn_train, Xn_test,
                                                            yn_test,
                                                            loss='mse',
                                                            num_rounds=50,
                                                            random_seed=20)
 
    # summary of the results
    # 4. Evaluate the model
    metrics.loc['Average expected loss',name] = round(avg_expected_loss*100,2)
    metrics.loc['Average bias',name] = round(avg_bias*100,2)
    metrics.loc['Average variance',name] = round(avg_var*100,2)
    metrics.loc['Accuracy',name] = round(cv_result.max()*100,2)
    
metrics


# In[ ]:


fig, ax = plt.subplots(figsize=(8,5))
metrics.plot(kind='barh', ax=ax)
plt.legend(loc=1)
ax.grid();


# In[ ]:


print(models)


# In[ ]:


#Pickling the models
import pickle


# In[ ]:


import _pickle as cPickle

#d = models

with open(r"GAC_Model.pickle", "wb") as output_file:
    cPickle.dump(models, output_file)


# pickle_file will be closed at this point, preventing your from accessing it any further

with open(r"GAC_Model.pickle", "rb") as input_file:
    GAC_Model = cPickle.load(input_file)





# In[ ]:


print (GAC_Model)


# In[ ]:


# from the above we can put a threshold number to avoid numbers e.g. max 25
#column = 'ACCT_FULL_NM'
i = 'Country'
if df[i].nunique()<= 40:
    print(i)
    print(df[i].value_counts())


# ## Predicting Next Purchase Day

# In[ ]:


df_uk.InvoiceDate


# In[ ]:


df_uk.InvoiceDate < datetime(2011,9,1)


# In[ ]:


tx_6m = df_uk[(df_uk.InvoiceDate < datetime (2011,9,1)) & (df_uk.InvoiceDate >= datetime (2011,3,1))].reset_index(drop=True)
tx_next = df_uk[(df_uk.InvoiceDate >= datetime (2011,9,1)) & (df_uk.InvoiceDate < datetime (2011,12,1))].reset_index(drop=True)


# In[ ]:


tx_user = pd.DataFrame(tx_6m['CustomerID'].unique())
tx_user.columns = ['CustomerID']


# In[ ]:


#create a dataframe with customer id and first purchase date in tx_next
tx_next_first_purchase = tx_next.groupby('CustomerID').InvoiceDate.min().reset_index()
tx_next_first_purchase.columns = ['CustomerID','MinPurchaseDate']

#create a dataframe with customer id and last purchase date in tx_6m
tx_last_purchase = tx_6m.groupby('CustomerID').InvoiceDate.max().reset_index()
tx_last_purchase.columns = ['CustomerID','MaxPurchaseDate']

#merge two dataframes
tx_purchase_dates = pd.merge(tx_last_purchase,tx_next_first_purchase,on='CustomerID',how='left')

#calculate the time difference in days:
tx_purchase_dates['NextPurchaseDay'] = (tx_purchase_dates['MinPurchaseDate'] - tx_purchase_dates['MaxPurchaseDate']).dt.days

#merge with tx_user 
tx_user = pd.merge(tx_user, tx_purchase_dates[['CustomerID','NextPurchaseDay']],on='CustomerID',how='left')

#print tx_user
tx_user.head()


# In[ ]:


tx_user.isna().sum()


# In[ ]:


tx_user['NextPurchaseDay'] = tx_user['NextPurchaseDay'].fillna(0)


# In[ ]:


tx_user.isna().sum()


# In[ ]:


#get max purchase date for Recency and create a dataframe
tx_max_purchase = tx_6m.groupby('CustomerID').InvoiceDate.max().reset_index()
tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']

#find the recency in days and add it to tx_user
tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days
tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')

#plot recency
plot_data = [
    go.Histogram(
        x=tx_user['Recency']
    )
]

plot_layout = go.Layout(
        title='Recency'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)

#clustering for Recency
kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Recency']])
tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])

#order cluster method
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final


#order recency clusters
tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,False)

#print cluster characteristics
tx_user.groupby('RecencyCluster')['Recency'].describe()


#get total purchases for frequency scores
tx_frequency = tx_6m.groupby('CustomerID').InvoiceDate.count().reset_index()
tx_frequency.columns = ['CustomerID','Frequency']

#add frequency column to tx_user
tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')

#plot frequency
plot_data = [
    go.Histogram(
        x=tx_user.query('Frequency < 1000')['Frequency']
    )
]

plot_layout = go.Layout(
        title='Frequency'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)

#clustering for frequency
kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Frequency']])
tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])

#order frequency clusters and show the characteristics
tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)
tx_user.groupby('FrequencyCluster')['Frequency'].describe()


#calculate monetary value, create a dataframe with it
tx_6m['Revenue'] = tx_6m['Amount']
tx_revenue = tx_6m.groupby('CustomerID').Revenue.sum().reset_index()

#add Revenue column to tx_user
tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')

#plot Revenue
plot_data = [
    go.Histogram(
        x=tx_user.query('Revenue < 10000')['Revenue']
    )
]

plot_layout = go.Layout(
        title='Monetary Value'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)

#Revenue clusters 
kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Revenue']])
tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])

#ordering clusters and who the characteristics
tx_user = order_cluster('RevenueCluster', 'Revenue',tx_user,True)
tx_user.groupby('RevenueCluster')['Revenue'].describe()


#building overall segmentation
tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']

#assign segment names
tx_user['Segment'] = 'Low-Value'
tx_user.loc[tx_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
tx_user.loc[tx_user['OverallScore']>4,'Segment'] = 'High-Value' 

#plot revenue vs frequency
tx_graph = tx_user.query("Revenue < 50000 and Frequency < 2000")

plot_data = [
    go.Scatter(
        x=tx_graph.query("Segment == 'Low-Value'")['Frequency'],
        y=tx_graph.query("Segment == 'Low-Value'")['Revenue'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'Mid-Value'")['Frequency'],
        y=tx_graph.query("Segment == 'Mid-Value'")['Revenue'],
        mode='markers',
        name='Mid',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'High-Value'")['Frequency'],
        y=tx_graph.query("Segment == 'High-Value'")['Revenue'],
        mode='markers',
        name='High',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'red',
            opacity= 0.9
           )
    ),
]

plot_layout = go.Layout(
        yaxis= {'title': "Revenue"},
        xaxis= {'title': "Frequency"},
        title='Segments'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)

#plot revenue vs recency
tx_graph = tx_user.query("Revenue < 50000 and Frequency < 2000")

plot_data = [
    go.Scatter(
        x=tx_graph.query("Segment == 'Low-Value'")['Recency'],
        y=tx_graph.query("Segment == 'Low-Value'")['Revenue'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'Mid-Value'")['Recency'],
        y=tx_graph.query("Segment == 'Mid-Value'")['Revenue'],
        mode='markers',
        name='Mid',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'High-Value'")['Recency'],
        y=tx_graph.query("Segment == 'High-Value'")['Revenue'],
        mode='markers',
        name='High',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'red',
            opacity= 0.9
           )
    ),
]

plot_layout = go.Layout(
        yaxis= {'title': "Revenue"},
        xaxis= {'title': "Recency"},
        title='Segments'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[ ]:


tx_user.head()


# In[ ]:


df.head()


# ### Convert whole dataframe from upper case to lower case with Pandas

# df = df.applymap(lambda s:s.lower() if type(s) == str else s)
# df.head()

# df['Amount'].min()

# df.shape

# In[ ]:


# in each column of dataframe
uniqueValues = df.nunique()
 
print('Count of unique value sin each column :')
print(uniqueValues)


# In[ ]:


# from the above we can put a threshold number to avoid numbers e.g. max 25
#column = 'ACCT_FULL_NM'
for i in df:
    if df[i].nunique()< 20:
        print(i)
        print(df[i].value_counts())


# df.shape

# In[ ]:


#To remove all rows where column 'CAMPAIGN' is not needed
#df = df.drop(df[(df.CAMPAIGN == 20190317) | (df.CAMPAIGN == 20180317) | (df.CAMPAIGN == 20180318)].index)


# df.dtypes

# df.columns
# #list(df.columns) 

# In[ ]:


#df = df.drop(['BillMonth' ], axis = 1)


# In[ ]:


print('{:,} rows; {:,} columns'.format(df.shape[0], df.shape[1]))


# In[ ]:


print('{:,} Customers don\'t have a Customer_Code'.format(df[df.Customer_Code.isnull()].shape[0]))
print('{:,} Products don\'t have a Product_Code'.format(df[df.Product_Code.isnull()].shape[0]))
print('{:,} Countries don\'t have a Country_Code'.format(df[df.Country_Code.isnull()].shape[0]))
print('{:,} Invoices don\'t have a InvoiceNumber'.format(df[df.InvoiceNumber.isnull()].shape[0]))

