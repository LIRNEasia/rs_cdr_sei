import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
#from thefuzz import process
#from thefuzz import fuzz
#import geopandas as gpd
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
#from scipy.stats import pearsonr
from itertools import combinations
import csv
import os
import matplotlib.pyplot as plt



def processing_df(pca_path, mobile_path, satellite_path, povery_metric, data_type, urban_rural):

    data_pca = pd.read_csv(pca_path)
    missing_gnd_ids = data_pca.loc[data_pca['PC1'].isnull(), 'gnd_id']
    count = missing_gnd_ids.count()
    data_raw = pd.read_csv(mobile_path).merge(pd.read_csv(satellite_path), on='gnd_id').merge(pd.read_csv(pca_path), on='gnd_id').rename(columns={"PC1": "pc1"}).set_index('gnd_id')

    data_raw = data_raw[~data_raw.index.isin(missing_gnd_ids)]
    urb_rur = pd.read_csv(urban_rural)
    
    urban_gnds = urb_rur.loc[urb_rur['urbanity'] == 'Urban', 'gnd_id']
    rural_gnds = urb_rur.loc[urb_rur['urbanity'] == 'Rural', 'gnd_id']

    if povery_metric == 1:
        pass
    elif povery_metric == 2:
        data_raw = data_raw[data_raw.index.isin(urban_gnds)]
    elif povery_metric == 3:
        data_raw = data_raw[data_raw.index.isin(rural_gnds)]


    data_unskewed = data_raw[[

        #Call detail records
        "call_count",
        "avg_call_duration",
        "nighttime_call_count",
        "avg_nighttime_call_duration",
        "incoming_call_count",
        "avg_incoming_call_duration",
        "radius_of_gyration",
        "unique_tower_count",
        "spatial_entropy",
        "avg_call_count_per_contact",
        "avg_call_duration_per_contact",
        "contact_count",
        "social_entropy",

        #Remote Sensing data
        "travel_time_major_cities",
        "population_count_worldpop",
        "population_count_ciesin",
        "population_density",
        "aridity_index",
        "evapotranspiration",
        "nighttime_lights",
        "elevation",
        "vegetation",
        "distance_roadways_motorway",
        "distance_roadways_trunk",
        "distance_roadways_primary",
        "distance_roadways_secondary",
        "distance_roadways_tertiary",
        "distance_waterways",
        "urban_rural_fb",
        "urban_rural_ciesin",
        "global_human_settlement",
        "protected_areas",
        "land_cover_woodland",
        "land_cover_grassland",
        "land_cover_cropland",
        "land_cover_wetland",
        "land_cover_bareland",
        "land_cover_urban",
        "land_cover_water",
        "pregnancies",
        "births",
        "precipitation",
        "temperature",

        "pc1"
    ]].copy()

    data_unskewed.fillna(data_unskewed.mean(), inplace=True)

    # Log transform skewed variables
    data_unskewed.loc[:, "radius_of_gyration_log"] = np.log(data_unskewed["radius_of_gyration"]+ 1) #
    data_unskewed.loc[:, "travel_time_major_cities_log"] = np.log(data_unskewed["travel_time_major_cities"] + 1)
    data_unskewed.loc[:, "population_count_worldpop_log"] = np.log(data_unskewed["population_count_worldpop"] + 1)
    data_unskewed.loc[:, "population_count_ciesin_log"] = np.log(data_unskewed["population_count_ciesin"] + 1)
    data_unskewed.loc[:, "population_density_log"] = np.log(data_unskewed["population_density"] + 1) #
    data_unskewed.loc[:, "elevation_log"] = np.log(data_unskewed["elevation"] + 1)
    data_unskewed.loc[:, "distance_roadways_trunk_log"] = np.log(data_unskewed["distance_roadways_trunk"] + 1)
    data_unskewed.loc[:, "distance_roadways_primary_log"] = np.log(data_unskewed["distance_roadways_primary"] + 1)
    data_unskewed.loc[:, "distance_roadways_secondary_log"] = np.log(data_unskewed["distance_roadways_secondary"] + 1)
    data_unskewed.loc[:, "distance_roadways_tertiary_log"] = np.log(data_unskewed["distance_roadways_tertiary"] + 1)
    data_unskewed.loc[:, "distance_waterways_log"] = np.log(data_unskewed["distance_waterways"] + 1)
    data_unskewed.loc[:, "urban_rural_fb_log"] = np.log(data_unskewed["urban_rural_fb"] + 1)
    data_unskewed.loc[:, "global_human_settlement_log"] = np.log(data_unskewed["global_human_settlement"] + 1)
    data_unskewed.loc[:, "protected_areas_log"] = np.log(data_unskewed["protected_areas"] + 1)
    data_unskewed.loc[:, "land_cover_grassland_log"] = np.log(data_unskewed["land_cover_grassland"] + 1)
    data_unskewed.loc[:, "land_cover_wetland_log"] = np.log(data_unskewed["land_cover_wetland"] + 1)
    data_unskewed.loc[:, "land_cover_bareland_log"] = np.log(data_unskewed["land_cover_bareland"] + 1)
    data_unskewed.loc[:, "land_cover_water_log"] = np.log(data_unskewed["land_cover_water"] + 1)
    data_unskewed.loc[:, "pregnancies_log"] = np.log(data_unskewed["pregnancies"] + 1)#
    data_unskewed.loc[:, "births_log"] = np.log(data_unskewed["births"] + 1) #
    data_unskewed.loc[:, "nighttime_lights_log"] = np.log(data_unskewed["nighttime_lights"] + 1)#

    if data_type == 1:
        data_unskewed = data_unskewed.drop(columns=["radius_of_gyration", "travel_time_major_cities", "population_count_worldpop", "population_count_ciesin", "population_density", "elevation", "distance_roadways_trunk", "distance_roadways_primary", "distance_roadways_secondary", "distance_roadways_tertiary", "distance_waterways", "urban_rural_fb", "global_human_settlement", "protected_areas", "land_cover_grassland", "land_cover_wetland", "land_cover_bareland", "land_cover_water", "pregnancies", "births", "nighttime_lights"])

    elif data_type == 2:    
        
        rs_only = data_unskewed[[
            "travel_time_major_cities_log",
            "population_count_worldpop_log",
            "population_count_ciesin_log",
            "population_density_log",
            "aridity_index",
            "evapotranspiration",
            "nighttime_lights",
            "elevation_log",
            "vegetation",
            "distance_roadways_motorway",
            "distance_roadways_trunk_log",
            "distance_roadways_primary_log",
            "distance_roadways_secondary_log",
            "distance_roadways_tertiary_log",
            "distance_waterways_log",
            "urban_rural_fb_log",
            "urban_rural_ciesin",
            "global_human_settlement_log",
            "protected_areas_log",
            "land_cover_woodland",
            "land_cover_grassland_log",
            "land_cover_cropland",
            "land_cover_wetland_log",
            "land_cover_bareland_log",
            "land_cover_urban",
            "land_cover_water_log",
            "pregnancies_log",
            "births_log",
            "precipitation",
            "temperature",

            "pc1"
        ]].copy()

        data_unskewed = rs_only.copy()

    elif data_type == 3:
        
        cdr_only = data_unskewed[[
            "call_count",
                "avg_call_duration",
                "nighttime_call_count",
                "avg_nighttime_call_duration",
                "incoming_call_count",
                "avg_incoming_call_duration",
                "radius_of_gyration",
                "unique_tower_count",
                "spatial_entropy",
                "avg_call_count_per_contact",
                "avg_call_duration_per_contact",
                "contact_count",
                "social_entropy",
                "pc1" 
        ]].copy()

        data_unskewed = cdr_only.copy()


    data_unskewed['pc1'] = np.where(data_unskewed['pc1'] <= -3, 1, 0)
    return data_unskewed


def filter_shapefile_columns(df):

    df.drop(['prov_name', 'dist_name', 'dsd_name', 'gnd_name', 'geometry'], axis=1, inplace=True)
    df.rename(columns={"code_7": "gnd_id"}, inplace=True)
    df.set_index('gnd_id', inplace=True)

    return df


def train_and_evaluate_rf(df, val, n_estimators):
    
    #training using df,  would incremental or online learning suit?
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_classifier.fit(df.drop(columns=["pc1"]), df["pc1"])
    
    #validating using val
    #predictions = rf_classifier.predict(val.drop(columns = ["pc1"]))
    pred_prob = rf_classifier.predict_proba(val.drop(columns = ["pc1"]))[:, 1] 
    #val['predictions'] = predictions
    val['pred_prob'] = pred_prob
    
    # sorting by predictions to find the performance at different thresholds

    val = val.sort_values('pred_prob', ascending=False)
    
    n_label_positives = len(val[val['pc1'] == 1])    
    metrics = list()


    for i in range(5, 101, 5):
    
        threshold = round((len(val)/100)*i)

        top_threshold = val.iloc[:threshold]

        #tp = len(top_threshold[(top_threshold['predictions'] == 1) & (top_threshold['pc1'] == 1)])  # Count of true positives
        
        #fp = len(top_threshold[(top_threshold['predictions'] == 1) & (top_threshold['pc1'] == 0)])  # Count of false positives

        #fn = len(top_threshold[(top_threshold['predictions'] == 0) & (top_threshold['pc1'] == 1)])  # Count of false negatives

        #precision    
        #precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        precision = top_threshold['pc1'].mean()  # This is TP / (TP + FP)
        
        #Recall
        #recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall = top_threshold['pc1'].sum() / n_label_positives  # This is TP / (TP + FN)

        d = dict()
        d['threshold'] = threshold
        d['precision'] = precision
        d['recall'] = recall
        metrics.append(d)
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('performance.csv', mode='a', header=False, index=False)
    #precision_recall_curve(metrics_df)

    #with open("output.csv", "a", newline='') as file:
    #   writer = csv.writer(file)
    #    writer.writerow(row)


def precision_recall_curve(metrics_df):

    output_dir = "pr_graphs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_filename = "precision_recall_curve.png"
    file_path = os.path.join(output_dir, base_filename)

    if os.path.exists(file_path):
        i = 1
        while os.path.exists(file_path):
            file_path = os.path.join(output_dir, f"precision_recall_curve_{i}.png")
            i += 1

    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['threshold'], metrics_df['precision'], label='Precision', marker='o')
    plt.plot(metrics_df['threshold'], metrics_df['recall'], label='Recall', marker='o')

    #plt.axvline(x=2800, color='red', linestyle='--', label='Threshold = 270')
    plt.xlabel('Thresholds')
    plt.ylabel('Score')
    plt.title('Precision and Recall Curves')
    plt.legend()
    plt.grid(True)

    plt.savefig(file_path)
    plt.close()