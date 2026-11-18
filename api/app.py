import os
import sys
import time
import json
import logging
import joblib
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

# Setup logging to console so evaluators can inspect live endpoint requests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PathSnap API] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PathSnapAPI")

# Resolve workspace root and src directory dynamically
API_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(API_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

for path in [PROJECT_ROOT, SRC_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from src.data_processor import load_trip_data, build_spatial_network, get_ground_truth_arcs
from src.feature_engineering import extract_features_for_trip, FEATURE_COLUMNS
from src.hmm_decoder import HMMRouteDecoder
from src.folium_visualizer import generate_interactive_folium_map
from src.train_and_evaluate import compute_route_continuity

BASE_DIR = os.path.join(PROJECT_ROOT, 'Student_folder', 'Student folder')
TEST_DIR = os.path.join(BASE_DIR, 'student_test')
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'models', 'best_tuned_model.joblib')
UI_DIR = os.path.join(PROJECT_ROOT, 'ui')

app = FastAPI(
    title="PathSnap Distributed Map-Matching & Route Reconstruction API",
    version="1.0.0",
    description="Evaluation server hosting tuned classifier, HMM/Viterbi engine, and ground-truth validation."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model into memory on startup
model = None

@app.on_event("startup")
def startup_event():
    global model
    logger.info("Initializing PathSnap Evaluation Server...")
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info(f"Successfully loaded trained model artifact from {MODEL_PATH}")
    else:
        logger.error(f"Model artifact not found at {MODEL_PATH}!")
    logger.info(f"Target test directory: {TEST_DIR}")
    logger.info("Server ready to process evaluation requests.")

@app.get("/")
def read_root():
    return {
        "service": "PathSnap Map-Matching API",
        "status": "online",
        "model_loaded": model is not None,
        "evaluator_ui_url": "/ui/index.html",
        "docs_url": "/docs"
    }

@app.get("/api/trips")
def get_available_trips():
    """Lists all 10 reserved evaluation trips available on the server."""
    if not os.path.exists(TEST_DIR):
        raise HTTPException(status_code=500, detail="Test directory not found")
        
    trips = sorted(os.listdir(TEST_DIR))
    trip_list = []
    
    for t in trips:
        t_path = os.path.join(TEST_DIR, t)
        track_f = os.path.join(t_path, f"{t}.track")
        nodes_f = os.path.join(t_path, f"{t}.nodes")
        arcs_f = os.path.join(t_path, f"{t}.arcs")
        
        with open(track_f) as f:
            n_track = sum(1 for _ in f)
        with open(nodes_f) as f:
            n_nodes = sum(1 for _ in f)
        with open(arcs_f) as f:
            n_arcs = sum(1 for _ in f)
            
        trip_list.append({
            "trip_id": t,
            "observations": n_track,
            "road_nodes": n_nodes,
            "road_arcs": n_arcs
        })
        
    logger.info(f"Dispatched trip catalog ({len(trip_list)} evaluation trips)")
    return {"trips": trip_list}

@app.post("/api/evaluate/{trip_id}")
def evaluate_trip(trip_id: str):
    """
    Core evaluation endpoint:
    Processes the raw GPS observations through candidate generation,
    feature extraction, tuned classifier scoring, and HMM Viterbi route reconstruction.
    Calculates server-side accuracy against the internal reference route.
    """
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on server")
        
    trip_path = os.path.join(TEST_DIR, trip_id)
    if not os.path.exists(trip_path):
        logger.warning(f"Evaluation request for unknown trip ID: {trip_id}")
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found in reserved test set")
        
    logger.info(f">>> INCOMING EVALUATION REQUEST: Trip {trip_id}")
    t_start = time.time()
    
    # 1. Load data
    track, nodes, arcs, route = load_trip_data(TEST_DIR, trip_id)
    n_obs = len(track)
    logger.info(f"Loaded {n_obs} GPS observations and {len(arcs)} road arcs for Trip {trip_id}")
    
    # 2. Build spatial network & server ground truth
    net = build_spatial_network(nodes, arcs, track['lat'].mean())
    true_arcs = get_ground_truth_arcs(track, nodes, arcs, route, net)
    
    # 3. Extract candidate features
    df_feat, cands_per_point = extract_features_for_trip(track, nodes, arcs, net, true_arcs=true_arcs)
    X_trip = df_feat[FEATURE_COLUMNS].values
    
    # 4. Predict candidate probabilities
    probs = model.predict_proba(X_trip)[:, 1]
    
    # Attach probs to candidate objects
    row_idx = 0
    misses = 0
    for i, cands in enumerate(cands_per_point):
        has_true = False
        for c in cands:
            c['prob'] = float(probs[row_idx])
            if c['arc_id'] == true_arcs[i]:
                has_true = True
            row_idx += 1
        if not has_true:
            misses += 1
            
    # 5. Classifier-Only Prediction (Greedy Top-1)
    clf_pred_arcs = [max(cands, key=lambda c: c['prob'])['arc_id'] for cands in cands_per_point]
    clf_pred_arcs = np.array(clf_pred_arcs)
    
    # 6. Classifier + HMM Viterbi Route Reconstruction
    scale = net['scale']
    track_m = track[['lon', 'lat']].values * scale
    diff_track = np.linalg.norm(track_m[1:] - track_m[:-1], axis=1)
    times = track['time'].values
    dts = np.maximum(times[1:] - times[:-1], 1e-3)
    
    decoder = HMMRouteDecoder(arcs, nodes, beta=20.0, same_arc_prob=0.85, direct_prob=1.0)
    hmm_pred_arcs = np.array(decoder.decode(cands_per_point, step_distances=diff_track, dts=dts))
    
    # 7. Calculate server-side metrics
    clf_acc = float(np.mean(clf_pred_arcs == true_arcs))
    hmm_acc = float(np.mean(hmm_pred_arcs == true_arcs))
    
    trip_conn = {'u': arcs['u'].values, 'v': arcs['v'].values, 'n_arcs': len(arcs)}
    clf_cont = compute_route_continuity(clf_pred_arcs, trip_conn)
    hmm_cont = compute_route_continuity(hmm_pred_arcs, trip_conn)
    
    total_time = time.time() - t_start
    miss_rate = misses / max(1, n_obs)
    
    # Generate/ensure Folium map exists
    map_filename = f"map_reconstruction_{trip_id}.html"
    map_filepath = os.path.join(ARTIFACTS_DIR, 'maps', map_filename)
    if not os.path.exists(map_filepath):
        generate_interactive_folium_map(trip_id, map_filepath)
        
    logger.info(f"Evaluation Complete for Trip {trip_id} in {total_time:.2f}s:")
    logger.info(f"  Clf Top-1 Acc: {clf_acc*100:.2f}% | Continuity: {clf_cont*100:.2f}%")
    logger.info(f"  HMM Top-1 Acc: {hmm_acc*100:.2f}% | Continuity: {hmm_cont*100:.2f}%")
    logger.info(f"  Candidate Miss Rate: {miss_rate*100:.2f}%")
    
    return {
        "status": "success",
        "trip_id": trip_id,
        "observations_count": n_obs,
        "candidate_miss_rate": round(miss_rate, 4),
        "clf_top1_accuracy": round(clf_acc, 4),
        "clf_route_continuity": round(clf_cont, 4),
        "hmm_top1_accuracy": round(hmm_acc, 4),
        "hmm_route_continuity": round(hmm_cont, 4),
        "accuracy_gain_pct": round((hmm_acc - clf_acc) * 100, 2),
        "continuity_gain_pct": round((hmm_cont - clf_cont) * 100, 2),
        "processing_time_s": round(total_time, 3),
        "map_url": f"/api/map/{trip_id}",
        "predicted_route_sample": hmm_pred_arcs[:30].tolist(),
        "total_predicted_arcs": len(hmm_pred_arcs)
    }

@app.get("/api/map/{trip_id}")
def get_map(trip_id: str):
    """Serves the interactive Folium HTML visualization for the trip."""
    map_filepath = os.path.join(ARTIFACTS_DIR, 'maps', f"map_reconstruction_{trip_id}.html")
    if not os.path.exists(map_filepath):
        generate_interactive_folium_map(trip_id, map_filepath)
    return FileResponse(map_filepath, media_type="text/html")

# Mount UI static files
os.makedirs(UI_DIR, exist_ok=True)
app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

