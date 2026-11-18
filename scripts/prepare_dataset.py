import os
import sys
import time
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath('src'))
from data_processor import load_trip_data, build_spatial_network, get_ground_truth_arcs
from feature_engineering import extract_features_for_trip

BASE_DIR = r'Student_folder/Student folder'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'student_test')
ARTIFACTS_DIR = 'artifacts'

def extract_and_save_data(num_train_trips=30):
    os.makedirs(os.path.join(ARTIFACTS_DIR, 'metrics'), exist_ok=True)
    
    train_trips = sorted(os.listdir(TRAIN_DIR))[:num_train_trips]
    test_trips = sorted(os.listdir(TEST_DIR))
    
    print(f"Extracting features for {len(train_trips)} training trips and {len(test_trips)} student testing trips...")
    
    # 1. Training features
    train_dfs = []
    t0 = time.time()
    for idx, t in enumerate(train_trips):
        trip_t0 = time.time()
        track, nodes, arcs, route = load_trip_data(TRAIN_DIR, t)
        net = build_spatial_network(nodes, arcs, track['lat'].mean())
        true_arcs = get_ground_truth_arcs(track, nodes, arcs, route, net)
        df_feat, _ = extract_features_for_trip(track, nodes, arcs, net, true_arcs=true_arcs)
        df_feat['trip_id'] = t
        train_dfs.append(df_feat)
        if (idx + 1) % 5 == 0 or idx == len(train_trips) - 1:
            print(f"  Processed {idx + 1}/{len(train_trips)} train trips ({time.time() - t0:.1f}s)")
            
    df_train = pd.concat(train_dfs, ignore_index=True)
    train_out_path = os.path.join(ARTIFACTS_DIR, 'metrics', 'train_features.parquet')
    df_train.to_parquet(train_out_path, index=False)
    print(f"Saved {train_out_path}: {df_train.shape[0]} rows, {df_train['label'].sum()} positives ({df_train['label'].mean()*100:.1f}%)")
    
    # 2. Student Testing features
    test_dfs = []
    t1 = time.time()
    for idx, t in enumerate(test_trips):
        track, nodes, arcs, route = load_trip_data(TEST_DIR, t)
        net = build_spatial_network(nodes, arcs, track['lat'].mean())
        true_arcs = get_ground_truth_arcs(track, nodes, arcs, route, net)
        df_feat, _ = extract_features_for_trip(track, nodes, arcs, net, true_arcs=true_arcs)
        df_feat['trip_id'] = t
        test_dfs.append(df_feat)
        print(f"  Processed {idx + 1}/{len(test_trips)} test trip {t}")
        
    df_test = pd.concat(test_dfs, ignore_index=True)
    test_out_path = os.path.join(ARTIFACTS_DIR, 'metrics', 'test_features.parquet')
    df_test.to_parquet(test_out_path, index=False)
    print(f"Saved {test_out_path}: {df_test.shape[0]} rows, {df_test['label'].sum()} positives ({df_test['label'].mean()*100:.1f}%)")
    print(f"Feature preparation completed in {time.time() - t0:.1f}s")

if __name__ == '__main__':
    extract_and_save_data(num_train_trips=30)

