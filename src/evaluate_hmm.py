import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath('src'))
from data_processor import load_trip_data, build_spatial_network, get_ground_truth_arcs
from feature_engineering import extract_features_for_trip, FEATURE_COLUMNS
from hmm_decoder import HMMRouteDecoder
from train_and_evaluate import compute_route_continuity

BASE_DIR = r'Student_folder/Student folder'
TEST_DIR = os.path.join(BASE_DIR, 'student_test')
ARTIFACTS_DIR = 'artifacts'

def evaluate_all_test_trips():
    print("=== Step 5: HMM and Viterbi Route Reconstruction Evaluation ===")
    
    # Load best tuned model
    model_path = os.path.join(ARTIFACTS_DIR, 'models', 'best_tuned_model.joblib')
    clf = joblib.load(model_path)
    print(f"Loaded tuned classifier from {model_path}")
    
    test_trips = sorted(os.listdir(TEST_DIR))
    
    trip_eval_results = []
    
    total_points = 0
    total_clf_correct = 0
    total_hmm_correct = 0
    total_misses = 0
    
    for t_idx, trip_id in enumerate(test_trips):
        t0 = time.time()
        track, nodes, arcs, route = load_trip_data(TEST_DIR, trip_id)
        net = build_spatial_network(nodes, arcs, track['lat'].mean())
        true_arcs = get_ground_truth_arcs(track, nodes, arcs, route, net)
        
        df_feat, cands_per_point = extract_features_for_trip(track, nodes, arcs, net, true_arcs=true_arcs)
        
        # Predict candidate probabilities
        X_trip = df_feat[FEATURE_COLUMNS].values
        probs = clf.predict_proba(X_trip)[:, 1]
        
        # Attach probs back to cands_per_point
        row_idx = 0
        miss_count = 0
        for i, cands in enumerate(cands_per_point):
            true_present = False
            for c in cands:
                c['prob'] = float(probs[row_idx])
                if c['arc_id'] == true_arcs[i]:
                    true_present = True
                row_idx += 1
            if not true_present:
                miss_count += 1
                
        # 1. Classifier-Only Prediction (Greedy Top-1)
        clf_pred_arcs = [max(cands, key=lambda c: c['prob'])['arc_id'] for cands in cands_per_point]
        clf_pred_arcs = np.array(clf_pred_arcs)
        
        # 2. Classifier + HMM Viterbi Decoding
        # Step distances and dt for transition costs
        scale = net['scale']
        track_m = track[['lon', 'lat']].values * scale
        diff_track = np.linalg.norm(track_m[1:] - track_m[:-1], axis=1)
        times = track['time'].values
        dts = np.maximum(times[1:] - times[:-1], 1e-3)
        
        decoder = HMMRouteDecoder(arcs, nodes, beta=20.0, same_arc_prob=0.85, direct_prob=1.0)
        hmm_pred_arcs = np.array(decoder.decode(cands_per_point, step_distances=diff_track, dts=dts))
        
        # Compute metrics
        N = len(true_arcs)
        clf_acc = float(np.mean(clf_pred_arcs == true_arcs))
        hmm_acc = float(np.mean(hmm_pred_arcs == true_arcs))
        
        # Connectivity dict
        trip_conn = {'u': arcs['u'].values, 'v': arcs['v'].values, 'n_arcs': len(arcs)}
        clf_cont = compute_route_continuity(clf_pred_arcs, trip_conn)
        hmm_cont = compute_route_continuity(hmm_pred_arcs, trip_conn)
        
        miss_rate = miss_count / N
        
        total_points += N
        total_clf_correct += int(np.sum(clf_pred_arcs == true_arcs))
        total_hmm_correct += int(np.sum(hmm_pred_arcs == true_arcs))
        total_misses += miss_count
        
        elapsed = time.time() - t0
        print(f"Trip {trip_id} (N={N:4d}): Miss Rate: {miss_rate*100:.2f}% | Clf Acc: {clf_acc*100:.2f}%, Cont: {clf_cont*100:.2f}% | HMM Acc: {hmm_acc*100:.2f}%, Cont: {hmm_cont*100:.2f}% ({elapsed:.2f}s)")
        
        trip_eval_results.append({
            'trip_id': trip_id,
            'observations': N,
            'candidate_miss_rate': float(miss_rate),
            'clf_top1_accuracy': float(clf_acc),
            'clf_route_continuity': float(clf_cont),
            'hmm_top1_accuracy': float(hmm_acc),
            'hmm_route_continuity': float(hmm_cont),
            'accuracy_gain': float(hmm_acc - clf_acc),
            'continuity_gain': float(hmm_cont - clf_cont),
            'processing_time_s': float(elapsed)
        })

    overall_clf_acc = total_clf_correct / total_points
    overall_hmm_acc = total_hmm_correct / total_points
    overall_miss_rate = total_misses / total_points
    mean_clf_cont = float(np.mean([r['clf_route_continuity'] for r in trip_eval_results]))
    mean_hmm_cont = float(np.mean([r['hmm_route_continuity'] for r in trip_eval_results]))

    summary = {
        'total_observations': total_points,
        'overall_candidate_miss_rate': overall_miss_rate,
        'overall_clf_top1_accuracy': overall_clf_acc,
        'overall_hmm_top1_accuracy': overall_hmm_acc,
        'mean_clf_continuity': mean_clf_cont,
        'mean_hmm_continuity': mean_hmm_cont,
        'accuracy_improvement_pct': (overall_hmm_acc - overall_clf_acc) * 100,
        'continuity_improvement_pct': (mean_hmm_cont - mean_clf_cont) * 100,
        'trip_results': trip_eval_results
    }
    
    print("\n--- OVERALL RESULTS (10 Student Test Trips) ---")
    print(f"Total Observations: {total_points}")
    print(f"Candidate Miss Rate: {overall_miss_rate*100:.2f}%")
    print(f"Classifier-Only Top-1 Accuracy:  {overall_clf_acc*100:.2f}% | Continuity: {mean_clf_cont*100:.2f}%")
    print(f"Classifier + HMM Top-1 Accuracy: {overall_hmm_acc*100:.2f}% | Continuity: {mean_hmm_cont*100:.2f}%")
    print(f"Improvement: Accuracy: +{(overall_hmm_acc - overall_clf_acc)*100:.2f} percentage points | Continuity: +{(mean_hmm_cont - mean_clf_cont)*100:.2f} percentage points")

    # Save metrics
    with open(os.path.join(ARTIFACTS_DIR, 'metrics', 'hmm_comparison.json'), 'w') as f:
        json.dump(summary, f, indent=4)
        
    df_results = pd.DataFrame(trip_eval_results)
    df_results.to_csv(os.path.join(ARTIFACTS_DIR, 'metrics', 'hmm_comparison.csv'), index=False)
    
    # Generate Visualizations
    generate_hmm_plots(df_results, summary)
    print("=== Step 5 Complete! ===")

def generate_hmm_plots(df_results, summary):
    os.makedirs(os.path.join(ARTIFACTS_DIR, 'plots'), exist_ok=True)
    
    # 1. Per-Trip Accuracy Comparison: Classifier vs HMM
    plt.figure(figsize=(12, 5))
    x = np.arange(len(df_results))
    width = 0.35
    plt.bar(x - width/2, df_results['clf_top1_accuracy'] * 100, width, label='Classifier-Only Top-1', color='#e67e22')
    plt.bar(x + width/2, df_results['hmm_top1_accuracy'] * 100, width, label='Classifier + HMM Viterbi', color='#27ae60')
    plt.xticks(x, df_results['trip_id'], rotation=25, ha='right', fontsize=11)
    plt.ylabel('Top-1 Accuracy (%)', fontsize=12)
    plt.title('Pointwise Top-1 Accuracy: Classifier-Only vs Classifier + HMM', fontsize=14, fontweight='bold')
    plt.ylim(50, 100)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'plots', 'classifier_vs_hmm_accuracy.png'), dpi=300)
    plt.close()
    
    # 2. Route Continuity Comparison
    plt.figure(figsize=(12, 5))
    plt.bar(x - width/2, df_results['clf_route_continuity'] * 100, width, label='Classifier-Only Continuity', color='#e74c3c')
    plt.bar(x + width/2, df_results['hmm_route_continuity'] * 100, width, label='Classifier + HMM Continuity', color='#2980b9')
    plt.xticks(x, df_results['trip_id'], rotation=25, ha='right', fontsize=11)
    plt.ylabel('Route Continuity (%)', fontsize=12)
    plt.title('Route Continuity: Classifier-Only vs Classifier + HMM', fontsize=14, fontweight='bold')
    plt.ylim(70, 100)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'plots', 'classifier_vs_hmm_continuity.png'), dpi=300)
    plt.close()

if __name__ == '__main__':
    evaluate_all_test_trips()

