import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix

sys.path.append(os.path.abspath('src'))
from models import get_model_definitions
from feature_engineering import FEATURE_COLUMNS
from data_processor import load_trip_data

BASE_DIR = r'Student_folder/Student folder'
TEST_DIR = os.path.join(BASE_DIR, 'student_test')
ARTIFACTS_DIR = 'artifacts'

def load_network_connectivity(test_dir):
    """Loads arc connectivity mappings for all student testing trips."""
    connectivity = {}
    test_trips = sorted(os.listdir(test_dir))
    for t in test_trips:
        arcs_f = os.path.join(test_dir, t, f"{t}.arcs")
        arcs = pd.read_csv(arcs_f, sep=r'\s+', header=None, names=['u', 'v'])
        u_arr = arcs['u'].values
        v_arr = arcs['v'].values
        
        # map u -> outgoing arcs, v -> incoming arcs
        connectivity[t] = {
            'u': u_arr,
            'v': v_arr,
            'n_arcs': len(arcs)
        }
    return connectivity

def compute_route_continuity(pred_arcs, trip_conn):
    """Computes fraction of consecutive steps where arcs are identical or directly connected."""
    if len(pred_arcs) <= 1:
        return 1.0
    u_arr = trip_conn['u']
    v_arr = trip_conn['v']
    
    connected_steps = 0
    total_steps = len(pred_arcs) - 1
    
    for i in range(total_steps):
        a1 = pred_arcs[i]
        a2 = pred_arcs[i+1]
        if a1 == a2:
            connected_steps += 1
        else:
            # Check if to_node of a1 == from_node of a2
            if v_arr[a1] == u_arr[a2]:
                connected_steps += 1
                
    return connected_steps / max(1, total_steps)

def run_model_experiments():
    print("=== Step 3: Model Experimentation and Comparison ===")
    
    df_train = pd.read_parquet(os.path.join(ARTIFACTS_DIR, 'metrics', 'train_features.parquet'))
    df_test = pd.read_parquet(os.path.join(ARTIFACTS_DIR, 'metrics', 'test_features.parquet'))
    
    X_train = df_train[FEATURE_COLUMNS].values
    y_train = df_train['label'].values
    groups_train = df_train['trip_id'].values
    
    X_test = df_test[FEATURE_COLUMNS].values
    y_test = df_test['label'].values
    
    test_connectivity = load_network_connectivity(TEST_DIR)
    
    models_dict = get_model_definitions()
    
    # 1. 5-Fold GroupKFold Cross-Validation on Training Trips
    print("\n--- Running 5-Fold GroupKFold Cross-Validation ---")
    gkf = GroupKFold(n_splits=5)
    
    cv_results = {m_name: {'macro_f1': [], 'log_loss': [], 'top1_acc': []} for m_name in models_dict}
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
        print(f"\nFold {fold + 1}/5:")
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        df_val = df_train.iloc[val_idx]
        
        for m_name, m_spec in models_dict.items():
            clf = m_spec['pipeline']
            clf.fit(X_tr, y_tr)
            
            y_val_prob = clf.predict_proba(X_val)[:, 1]
            y_val_pred = (y_val_prob >= 0.5).astype(int)
            
            f1_val = f1_score(y_val, y_val_pred, average='macro')
            ll_val = log_loss(y_val, y_val_prob)
            
            # Compute Fold Top-1 Observation Accuracy
            df_val_eval = df_val.copy()
            df_val_eval['prob'] = y_val_prob
            
            # Group by trip and point_idx to find top candidate
            top_cands = df_val_eval.sort_values(['trip_id', 'point_idx', 'prob'], ascending=[True, True, False]).drop_duplicates(subset=['trip_id', 'point_idx'], keep='first')
            top1_acc = top_cands['label'].mean()
            
            cv_results[m_name]['macro_f1'].append(f1_val)
            cv_results[m_name]['log_loss'].append(ll_val)
            cv_results[m_name]['top1_acc'].append(top1_acc)
            print(f"  {m_name:20s} | Macro F1: {f1_val:.4f} | Log Loss: {ll_val:.4f} | Top-1: {top1_acc:.4f}")

    # 2. Train on Full Training Set and Evaluate on Student Test Trips
    print("\n--- Training on Full Training Set and Evaluating on Student Test Set ---")
    comparison_records = []
    test_predictions = {}
    
    confusion_matrices = {}
    mlp_loss_history = None
    
    for m_name, m_spec in models_dict.items():
        print(f"\nTraining {m_name}...")
        clf = m_spec['pipeline']
        
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0
        
        # Save model artifact
        model_path = os.path.join(ARTIFACTS_DIR, 'models', f"{m_name}.joblib")
        joblib.dump(clf, model_path)
        print(f"  Saved model artifact to {model_path} (Training time: {train_time:.2f}s)")
        
        if m_name == 'Neural_Network_MLP':
            mlp_model = clf.named_steps['classifier']
            if hasattr(mlp_model, 'loss_curve_'):
                mlp_loss_history = list(mlp_model.loss_curve_)
        
        # Test inference
        t1 = time.time()
        test_probs = clf.predict_proba(X_test)[:, 1]
        inference_time = time.time() - t1
        
        test_preds = (test_probs >= 0.5).astype(int)
        test_predictions[m_name] = test_probs
        
        # Candidate-level metrics
        test_acc = accuracy_score(y_test, test_preds)
        test_prec = precision_score(y_test, test_preds, zero_division=0)
        test_rec = recall_score(y_test, test_preds, zero_division=0)
        test_macro_f1 = f1_score(y_test, test_preds, average='macro')
        test_ll = log_loss(y_test, test_probs)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, test_preds).tolist()
        confusion_matrices[m_name] = cm
        
        # Observation-level Top-1 Accuracy and Route Continuity on Student Test Set
        df_test_eval = df_test.copy()
        df_test_eval['prob'] = test_probs
        
        # For each point, select top-1 candidate
        top_cands_test = df_test_eval.sort_values(['trip_id', 'point_idx', 'prob'], ascending=[True, True, False]).drop_duplicates(subset=['trip_id', 'point_idx'], keep='first')
        test_top1_acc = float(top_cands_test['label'].mean())
        
        # Route Continuity per trip
        continuity_list = []
        for t_id, group in top_cands_test.groupby('trip_id'):
            group_sorted = group.sort_values('point_idx')
            pred_arcs = group_sorted['arc_id'].values
            cont = compute_route_continuity(pred_arcs, test_connectivity[t_id])
            continuity_list.append(cont)
        mean_continuity = float(np.mean(continuity_list))
        
        cv_macro_f1_mean = float(np.mean(cv_results[m_name]['macro_f1']))
        cv_macro_f1_std = float(np.std(cv_results[m_name]['macro_f1']))
        cv_top1_mean = float(np.mean(cv_results[m_name]['top1_acc']))
        cv_log_loss_mean = float(np.mean(cv_results[m_name]['log_loss']))
        
        record = {
            'model': m_name,
            'algorithm_family': m_spec['family'],
            'configuration': m_spec['config'],
            'loss_objective': m_spec['objective'],
            'cv_macro_f1_mean': cv_macro_f1_mean,
            'cv_macro_f1_std': cv_macro_f1_std,
            'cv_top1_accuracy': cv_top1_mean,
            'cv_log_loss': cv_log_loss_mean,
            'test_accuracy': float(test_acc),
            'test_precision': float(test_prec),
            'test_recall': float(test_rec),
            'test_macro_f1': float(test_macro_f1),
            'test_top1_accuracy': test_top1_acc,
            'test_route_continuity': mean_continuity,
            'test_log_loss': float(test_ll),
            'training_time_s': float(train_time),
            'inference_time_s': float(inference_time)
        }
        comparison_records.append(record)
        
        print(f"  {m_name} Results:")
        print(f"    CV Macro F1: {cv_macro_f1_mean:.4f} +/- {cv_macro_f1_std:.4f}")
        print(f"    Test Top-1 Accuracy: {test_top1_acc*100:.2f}% | Macro F1: {test_macro_f1:.4f} | Continuity: {mean_continuity*100:.2f}%")
        print(f"    Log Loss: {test_ll:.4f} | Training: {train_time:.2f}s | Inference: {inference_time*1000:.1f}ms")

    # Save Comparison Records
    df_comp = pd.DataFrame(comparison_records)
    df_comp.to_csv(os.path.join(ARTIFACTS_DIR, 'metrics', 'model_comparison.csv'), index=False)
    
    with open(os.path.join(ARTIFACTS_DIR, 'metrics', 'model_comparison.json'), 'w') as f:
        json.dump({
            'models': comparison_records,
            'cv_results': cv_results,
            'confusion_matrices': confusion_matrices,
            'mlp_loss_history': mlp_loss_history
        }, f, indent=4)
    print("\nSaved model_comparison.csv and model_comparison.json")

    # Generate Visualizations (Compulsory per guidelines)
    generate_comparison_plots(df_comp, confusion_matrices, mlp_loss_history)
    print("=== Step 3 Complete! ===")

def generate_comparison_plots(df_comp, confusion_matrices, mlp_loss_history):
    os.makedirs(os.path.join(ARTIFACTS_DIR, 'plots'), exist_ok=True)
    
    # 1. CV Macro F1 vs Test Macro F1
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df_comp))
    width = 0.35
    plt.bar(x - width/2, df_comp['cv_macro_f1_mean'], width, yerr=df_comp['cv_macro_f1_std'], capsize=5, label='5-Fold CV Macro F1', color='#3498db')
    plt.bar(x + width/2, df_comp['test_macro_f1'], width, label='Student Test Macro F1', color='#2ecc71')
    plt.xticks(x, df_comp['model'], rotation=15, ha='right', fontsize=11)
    plt.ylabel('Macro F1-Score', fontsize=12)
    plt.title('Cross-Validation vs Student-Test Macro F1 Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0.5, 1.0)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'plots', 'cv_vs_test_macro_f1.png'), dpi=300)
    plt.close()
    
    # 2. Observation-Level Top-1 Accuracy and Route Continuity
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, df_comp['test_top1_accuracy'] * 100, width, label='Top-1 Accuracy (%)', color='#e67e22')
    plt.bar(x + width/2, df_comp['test_route_continuity'] * 100, width, label='Route Continuity (%)', color='#9b59b6')
    plt.xticks(x, df_comp['model'], rotation=15, ha='right', fontsize=11)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.title('Observation-Level Top-1 Accuracy & Route Continuity (Student Test)', fontsize=14, fontweight='bold')
    plt.ylim(60, 100)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'plots', 'top1_accuracy_and_continuity.png'), dpi=300)
    plt.close()
    
    # 3. Log Loss Comparison
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df_comp['model'], df_comp['test_log_loss'], color='#e74c3c', width=0.5)
    plt.xticks(rotation=15, ha='right', fontsize=11)
    plt.ylabel('Binary Logarithmic Loss', fontsize=12)
    plt.title('Student-Testing Log Loss Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'plots', 'log_loss_comparison.png'), dpi=300)
    plt.close()
    
    # 4. Training and Inference Time
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = '#2980b9'
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Training Time (s)', color=color, fontsize=12)
    bars1 = ax1.bar(x - width/2, df_comp['training_time_s'], width, label='Training Time (s)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_comp['model'], rotation=15, ha='right', fontsize=11)
    
    ax2 = ax1.twinx()
    color = '#d35400'
    ax2.set_ylabel('Inference Time (ms)', color=color, fontsize=12)
    bars2 = ax2.bar(x + width/2, df_comp['inference_time_s'] * 1000, width, label='Inference Time (ms)', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Computational Efficiency: Training vs Inference Time', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'plots', 'computational_efficiency.png'), dpi=300)
    plt.close()
    
    # 5. Confusion Matrices for all 5 models
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, (m_name, cm) in enumerate(confusion_matrices.items()):
        cm_arr = np.array(cm)
        sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(m_name.replace('_', ' '), fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Predicted Label')
        if i == 0:
            axes[i].set_ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'plots', 'confusion_matrices.png'), dpi=300)
    plt.close()
    
    # 6. MLP Training Loss Curve
    if mlp_loss_history:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(mlp_loss_history) + 1), mlp_loss_history, marker='o', color='#8e44ad', linewidth=2)
        plt.xlabel('Epoch / Iteration', fontsize=12)
        plt.ylabel('Cross-Entropy Loss', fontsize=12)
        plt.title('Neural Network (MLP) Training Loss Curve', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACTS_DIR, 'plots', 'mlp_training_loss_curve.png'), dpi=300)
        plt.close()

if __name__ == '__main__':
    run_model_experiments()

