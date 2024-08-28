import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#import statsmodels.api as sm
#from thefuzz import process
#from thefuzz import fuzz
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
#import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

from scipy.stats import pearsonr
from itertools import combinations

from utils import processing_df, filter_shapefile_columns, train_and_evaluate_rf


povery_metric = 1
data_type = 1
model_type = 3

mobile_path = '/lirneasia/data/sei_mapping/raw/mobile_features.csv'
satellite_path = '/lirneasia/data/sei_mapping/raw/satellite_features.csv'
pca_path = '/lirneasia/data/sei_mapping/raw/pca_census.csv'
urban_rural = '/lirneasia/data/sei_mapping/raw/GND_urban_rural_classification.csv'
shape_file = '/lirneasia/data/sei_mapping/sri_lanka_gnd_shape_files/sri_lanka_gnd.shp'


data_frame = processing_df(pca_path, mobile_path, satellite_path, povery_metric, data_type, urban_rural)

data_shapefile = gpd.read_file(shape_file)
data_shapefile.set_index("code_7", inplace=True)
data_shapefile = data_shapefile.to_crs(epsg=3857)
merged = data_frame.merge(data_shapefile, left_on=data_frame.index, right_on='code_7', how='inner')

n_iterations = 10
results = []

# 1. Sample 20% of the data for the testing and leaving it untouched

df, test = train_test_split(merged, test_size=0.2, stratify=merged['dsd_name'], random_state= 42)
test = filter_shapefile_columns(test)


for i in range(n_iterations):

    
    # 2. Sample 20% for the validation, such that 60% from the complete dataset will be available for training

    train, val = train_test_split(df, test_size=0.25, stratify=df['dsd_name'], random_state=i)

    train = filter_shapefile_columns(train)
    val = filter_shapefile_columns(val)


    train_and_evaluate_rf(train, val, 1000)

    print(f"Training the model, iteration: {i + 1}")


performance = pd.read_csv('/lirneasia/projects/sei_mapping/notebooks_codespaces/output.csv', header=None)
values = performance.mean()

print("Mean Values for Precision and Recall at Each 10% Increment:")
for i, (key, value) in enumerate(values.items()):
    metric = 'Precision' if i % 2 == 0 else 'Recall'
    print(f"{key}: {metric} = {value:.2f}")

    

'''

1.⁠ ⁠Sample 60% of the data, then sample 20% from the rest (don't sample the test set with replacement)
2.⁠ ⁠Train the model with 60%, validate with the 20%
3.⁠ ⁠Repeat 1 -2 for n iterations

For each iteration, let's calculate the precision and recall at different thresholds
Let's say 10%, 20%, ..., 100%
All 10 percents
So you would store 20 numbers
For each one iteration
Your results CSV could have these columns
    Iteration number, metric, threshold, value

Your final CSV would have n_iterations * 20 rows
'''
