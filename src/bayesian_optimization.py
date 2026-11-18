import os
import sys
import time
import json
import joblib
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
from xgboost import XGBClassifier

sys.path.append(os.path.abspath('src'))
from feature_engineering import FEATURE_COLUMNS
from train_and_evaluate import load_network_connectivity, compute_route_continuity

# Suppress Optuna logging verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = r'Student_folder/Student folder'
TEST_DIR = os.path.join(BASE_DIR, 'student_test')
ARTIFACTS_DIR = 'artifacts'

def run_bayesian_optimization(n_trials=25):
    print("=== Step 4: Bayesian Hyperparameter Optimization (Optuna) ===")
    
    df_train = pd.read_parquet(os.path.join(ARTIFACTS_DIR, 'metrics', 'train_features.parquet'))
    df_test = pd.read_parquet(os.path.join(ARTIFACTS_DIR, 'metrics', 'test_features.parquet'))
    
    X_train = df_train[FEATURE_COLUMNS].values
    y_train = df_train['label'].values
    groups_train = df_train['trip_id'].values
    
    X_test = df_test[FEATURE_COLUMNS].values
    y_test = df_test['label'].values
    
    test_connectivity = load_network_connectivity(TEST_DIR)
    
    # 3-Fold GroupKFold for efficient Bayesian search across 30 trips
    gkf = GroupKFold(n_splits=3)
    cv_splits = list(gkf.split(X_train, y_train, groups=groups_train))
    
    trial_records = []
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 60, 200, step=20),
            'max_depth': trial.suggest_int('max_depth', 4, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.25, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
            'gamma': trial.suggest_float('gamma', 0.0, 3.0, step=0.5),
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        
        t_start = time.time()
        fold_f1s = []
        for tr_idx, val_idx in cv_splits:
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            
            clf = XGBClassifier(**params)
            clf.fit(X_tr, y_tr)
            
            val_probs = clf.predict_proba(X_val)[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)
            f1 = f1_score(y_val, val_preds, average='macro')
            fold_f1s.append(f1)
            
        mean_f1 = float(np.mean(fold_f1s))
        trial_duration = time.time() - t_start
        
        record = {
            'trial_number': trial.number,
            'params': params,
            'cv_macro_f1': mean_f1,
            'duration_s': trial_duration
        }
        trial_records.append(record)
        print(f"Trial {trial.number:2d}: Macro F1 = {mean_f1:.4f} (depth={params['max_depth']}, lr={params['learning_rate']:.3f}, n_est={params['n_estimators']}, {trial_duration:.1f}s)")
        return mean_f1

    # Run Optuna Study with TPE sampler (Tree-structured Parzen Estimator)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_cv_score = study.best_value
    print(f"\nOptimization Finished! Best Trial: #{study.best_trial.number}")
    print(f"Best CV Macro F1: {best_cv_score:.4f}")
    print(f"Best Hyperparameters: {json.dumps(best_params, indent=2)}")
    
    # Save trial records
    with open(os.path.join(ARTIFACTS_DIR, 'metrics', 'bayesian_trials.json'), 'w') as f:
        json.dump({
            'trials': trial_records,
            'best_trial': study.best_trial.number,
            'best_params': best_params,
            'best_cv_score': best_cv_score
        }, f, indent=4)
        
    df_trials = pd.DataFrame([{
        'trial': r['trial_number'],
        'cv_macro_f1': r['cv_macro_f1'],
        'duration_s': r['duration_s'],
        **r['params']
    } for r in trial_records])
    df_trials.to_csv(os.path.join(ARTIFACTS_DIR, 'metrics', 'bayesian_trials.csv'), index=False)
    
    # Generate Optuna Optimization History Plot
    plt.figure(figsize=(9, 5))
    trial_nums = [r['trial_number'] for r in trial_records]
    scores = [r['cv_macro_f1'] for r in trial_records]
    best_so_far = np.maximum.accumulate(scores)
    plt.plot(trial_nums, scores, 'o-', color='#3498db', alpha=0.6, label='Trial Score')
    plt.plot(trial_nums, best_so_far, 'r--', linewidth=2, label='Best Cumulative Score')
    plt.xlabel('Trial Number', fontsize=12)
    plt.ylabel('3-Fold CV Macro F1-Score', fontsize=12)
    plt.title('Bayesian Optimization History (Optuna TPE Sampler)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'plots', 'bayesian_optimization_history.png'), dpi=300)
    plt.close()
    
    # Train Untuned vs Tuned Comparison on Full Training Set & Evaluate on Student Test
    print("\n--- Fitting and Comparing Untuned vs Tuned Model ---")
    
    # 1. Untuned model (default configuration)
    untuned_params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42, 'n_jobs': -1, 'eval_metric': 'logloss'}
    clf_untuned = XGBClassifier(**untuned_params)
    t0 = time.time()
    clf_untuned.fit(X_train, y_train)
    untuned_fit_time = time.time() - t0
    
    t1 = time.time()
    probs_untuned = clf_untuned.predict_proba(X_test)[:, 1]
    untuned_inf_time = time.time() - t1
    preds_untuned = (probs_untuned >= 0.5).astype(int)
    
    # 2. Tuned model
    full_best_params = {**best_params, 'random_state': 42, 'n_jobs': -1, 'eval_metric': 'logloss'}
    clf_tuned = XGBClassifier(**full_best_params)
    t2 = time.time()
    clf_tuned.fit(X_train, y_train)
    tuned_fit_time = time.time() - t2
    
    # Save best tuned model
    joblib.dump(clf_tuned, os.path.join(ARTIFACTS_DIR, 'models', 'best_tuned_model.joblib'))
    print(f"Saved best tuned model to {os.path.join(ARTIFACTS_DIR, 'models', 'best_tuned_model.joblib')}")
    
    t3 = time.time()
    probs_tuned = clf_tuned.predict_proba(X_test)[:, 1]
    tuned_inf_time = time.time() - t3
    preds_tuned = (probs_tuned >= 0.5).astype(int)
    
    # Evaluate observation-level Top-1 and Continuity
    def eval_student_test(probs):
        df_eval = df_test.copy()
        df_eval['prob'] = probs
        top_cands = df_eval.sort_values(['trip_id', 'point_idx', 'prob'], ascending=[True, True, False]).drop_duplicates(subset=['trip_id', 'point_idx'], keep='first')
        top1 = float(top_cands['label'].mean())
        
        cont_list = []
        for t_id, grp in top_cands.groupby('trip_id'):
            grp_s = grp.sort_values('point_idx')
            pred_arcs = grp_s['arc_id'].values
            cont_list.append(compute_route_continuity(pred_arcs, test_connectivity[t_id]))
        return top1, float(np.mean(cont_list))

    top1_untuned, cont_untuned = eval_student_test(probs_untuned)
    top1_tuned, cont_tuned = eval_student_test(probs_tuned)
    
    comparison_table = [
        {
            'Configuration': 'Untuned XGBoost (Baseline)',
            'CV_Macro_F1': 0.8266,
            'Test_Macro_F1': float(f1_score(y_test, preds_untuned, average='macro')),
            'Test_Top1_Accuracy': top1_untuned,
            'Test_Route_Continuity': cont_untuned,
            'Test_Log_Loss': float(log_loss(y_test, probs_untuned)),
            'Inference_Time_ms': untuned_inf_time * 1000
        },
        {
            'Configuration': 'Tuned XGBoost (Bayesian Optuna)',
            'CV_Macro_F1': best_cv_score,
            'Test_Macro_F1': float(f1_score(y_test, preds_tuned, average='macro')),
            'Test_Top1_Accuracy': top1_tuned,
            'Test_Route_Continuity': cont_tuned,
            'Test_Log_Loss': float(log_loss(y_test, probs_tuned)),
            'Inference_Time_ms': tuned_inf_time * 1000
        }
    ]
    
    df_compare = pd.DataFrame(comparison_table)
    df_compare.to_csv(os.path.join(ARTIFACTS_DIR, 'metrics', 'tuning_comparison.csv'), index=False)
    with open(os.path.join(ARTIFACTS_DIR, 'metrics', 'tuning_comparison.json'), 'w') as f:
        json.dump(comparison_table, f, indent=4)
        
    print("\n--- Tuned vs Untuned Comparison Table ---")
    print(df_compare.to_string(index=False))
    
    # Plot Tuned vs Untuned
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics_to_plot = [
        ('Test_Macro_F1', 'Macro F1-Score', axes[0], '#27ae60'),
        ('Test_Top1_Accuracy', 'Top-1 Accuracy', axes[1], '#2980b9'),
        ('Test_Log_Loss', 'Log Loss (Lower is Better)', axes[2], '#c0392b')
    ]
    for col, title, ax, c in metrics_to_plot:
        vals = df_compare[col].values
        ax.bar(['Untuned', 'Tuned'], vals, color=[c, '#16a085' if c == '#27ae60' else '#34495e'], width=0.45)
        ax.set_title(title, fontsize=12, fontweight='bold')
        for idx, v in enumerate(vals):
            ax.text(idx, v * 0.95 if col != 'Test_Log_Loss' else v + 0.005, f"{v:.4f}", ha='center', va='bottom', color='white' if col != 'Test_Log_Loss' else 'black', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'plots', 'tuned_vs_untuned_comparison.png'), dpi=300)
    plt.close()
    print("=== Step 4 Complete! ===")

if __name__ == '__main__':
    run_bayesian_optimization(n_trials=25)

