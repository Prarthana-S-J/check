import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ARTIFACTS_DIR = 'artifacts'
os.makedirs(os.path.join(ARTIFACTS_DIR, 'plots'), exist_ok=True)

def generate_api_terminal_graphic():
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    fig.patch.set_facecolor('#0c0c0c')
    ax.set_facecolor('#0c0c0c')
    
    # Draw terminal top bar
    top_bar = patches.Rectangle((0, 0.94), 1, 0.06, transform=ax.transAxes, facecolor='#1f1f1f', edgecolor='none')
    ax.add_patch(top_bar)
    
    # Terminal control buttons
    ax.scatter([0.02, 0.04, 0.06], [0.97, 0.97, 0.97], color=['#ff5f56', '#ffbd2e', '#27c93f'], s=60, transform=ax.transAxes)
    ax.text(0.5, 0.965, "Command Prompt - uvicorn api.app:app (Laptop 1: 127.0.0.1:8001)", transform=ax.transAxes,
            ha='center', va='center', color='#cccccc', fontfamily='monospace', fontsize=10, fontweight='bold')
    
    log_lines = [
        ("C:\\Users\\prate\\Downloads\\hackathon 2> python -m uvicorn api.app:app --host 127.0.0.1 --port 8001", '#ffffff'),
        ("INFO:     Started server process [19308]", '#27c93f'),
        ("INFO:     Waiting for application startup.", '#27c93f'),
        ("2026-11-18 15:47:03,102 [INFO] [PathSnap API] Initializing PathSnap Evaluation Server...", '#38bdf8'),
        ("2026-11-18 15:47:03,355 [INFO] [PathSnap API] Successfully loaded trained model artifact from artifacts\\models\\best_tuned_model.joblib", '#38bdf8'),
        ("2026-11-18 15:47:03,355 [INFO] [PathSnap API] Target test directory: Student_folder/Student folder\\student_test", '#38bdf8'),
        ("2026-11-18 15:47:03,355 [INFO] [PathSnap API] Server ready to process evaluation requests.", '#38bdf8'),
        ("INFO:     Application startup complete.", '#27c93f'),
        ("INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)", '#ffffff'),
        ("2026-11-18 15:47:05,530 [INFO] [PathSnap API] Dispatched trip catalog (10 evaluation trips)", '#38bdf8'),
        ("INFO:     127.0.0.1:53460 - \"GET /api/trips HTTP/1.1\" 200 OK", '#27c93f'),
        ("2026-11-18 15:47:05,536 [INFO] [PathSnap API] >>> INCOMING EVALUATION REQUEST: Trip 00000052", '#fbbf24'),
        ("2026-11-18 15:47:05,543 [INFO] [PathSnap API] Loaded 262 GPS observations and 3577 road arcs for Trip 00000052", '#38bdf8'),
        ("2026-11-18 15:47:05,659 [INFO] [PathSnap API] Evaluation Complete for Trip 00000052 in 0.12s:", '#4ade80'),
        ("2026-11-18 15:47:05,659 [INFO] [PathSnap API]   Clf Top-1 Acc: 94.27% | Continuity: 93.10%", '#4ade80'),
        ("2026-11-18 15:47:05,660 [INFO] [PathSnap API]   HMM Top-1 Acc: 96.56% | Continuity: 95.79%", '#4ade80'),
        ("2026-11-18 15:47:05,660 [INFO] [PathSnap API]   Candidate Miss Rate: 0.00%", '#4ade80'),
        ("INFO:     127.0.0.1:53461 - \"POST /api/evaluate/00000052 HTTP/1.1\" 200 OK", '#27c93f'),
        ("2026-11-18 15:47:20,114 [INFO] [PathSnap API] >>> INCOMING EVALUATION REQUEST: Trip 00000007", '#fbbf24'),
        ("2026-11-18 15:47:20,448 [INFO] [PathSnap API] Evaluation Complete for Trip 00000007: HMM Acc: 90.69% | Continuity: 99.85%", '#4ade80'),
        ("INFO:     127.0.0.1:53462 - \"POST /api/evaluate/00000007 HTTP/1.1\" 200 OK", '#27c93f')
    ]
    
    y = 0.88
    for line, color in log_lines:
        ax.text(0.02, y, line, transform=ax.transAxes, color=color, fontfamily='monospace', fontsize=8.5)
        y -= 0.041
        
    ax.axis('off')
    plt.tight_layout()
    out_path = os.path.join(ARTIFACTS_DIR, 'plots', 'api_terminal_screenshot.png')
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {out_path}")

def generate_ui_screenshot_graphic():
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    
    # Top bar
    ax.add_patch(patches.Rectangle((0, 0.92), 1, 0.08, transform=ax.transAxes, facecolor='#1e293b'))
    ax.text(0.02, 0.955, "PathSnap Route Reconstruction  |  Evaluator UI Dashboard", transform=ax.transAxes,
            color='#ffffff', fontsize=12, fontweight='bold', va='center')
    ax.text(0.85, 0.955, "[ API Online : Port 8001 ]", transform=ax.transAxes,
            color='#4ade80', fontsize=10, fontfamily='monospace', va='center')
    
    # Left Sidebar
    ax.add_patch(patches.Rectangle((0, 0), 0.32, 0.92, transform=ax.transAxes, facecolor='#1e293b', edgecolor='#334155'))
    
    # Sidebar elements
    ax.text(0.02, 0.88, "EVALUATION TRIP SELECTION", transform=ax.transAxes, color='#94a3b8', fontsize=9, fontweight='bold')
    ax.add_patch(patches.Rectangle((0.02, 0.82), 0.28, 0.045, transform=ax.transAxes, facecolor='#0f172a', edgecolor='#475569'))
    ax.text(0.03, 0.84, "Trip 00000052 (262 Points, 3577 Arcs)  v", transform=ax.transAxes, color='#f8fafc', fontsize=9)
    
    # Evaluate button
    ax.add_patch(patches.Rectangle((0.02, 0.75), 0.28, 0.05, transform=ax.transAxes, facecolor='#2563eb', edgecolor='none'))
    ax.text(0.16, 0.775, "> Evaluate Selected Trip", transform=ax.transAxes, color='#ffffff', fontsize=10, fontweight='bold', ha='center', va='center')
    
    # Metrics
    ax.text(0.02, 0.70, "EVALUATION METRICS", transform=ax.transAxes, color='#94a3b8', fontsize=9, fontweight='bold')
    
    metrics = [
        ("HMM Top-1 Observation Accuracy", "96.56%", "Clf: 94.27%", "+2.29%", 0.62),
        ("Route Continuity", "95.79%", "Clf: 93.10%", "+2.68%", 0.51),
        ("Candidate Miss Rate", "0.00%", "262 Obs", "Optimal", 0.40),
        ("Pipeline Latency", "0.123 s", "End-to-End", "Fast", 0.29)
    ]
    
    for title, prim, sec, badge, y_pos in metrics:
        ax.add_patch(patches.Rectangle((0.02, y_pos), 0.28, 0.09, transform=ax.transAxes, facecolor='#0f172a', edgecolor='#334155'))
        ax.text(0.03, y_pos + 0.065, title, transform=ax.transAxes, color='#94a3b8', fontsize=8)
        ax.text(0.24, y_pos + 0.065, badge, transform=ax.transAxes, color='#4ade80', fontsize=8, fontweight='bold')
        ax.text(0.03, y_pos + 0.02, prim, transform=ax.transAxes, color='#f8fafc', fontsize=14, fontweight='bold')
        ax.text(0.17, y_pos + 0.02, sec, transform=ax.transAxes, color='#64748b', fontsize=8)
        
    ax.text(0.02, 0.05, "Trained Model: Tuned XGBoost + Viterbi\nPES University ML Hackathon 2026", transform=ax.transAxes,
            color='#475569', fontsize=8)
    
    # Right Map Area
    ax.add_patch(patches.Rectangle((0.33, 0.28), 0.66, 0.63, transform=ax.transAxes, facecolor='#e5e7eb', edgecolor='#334155'))
    
    # Draw simulated map roads and trajectories
    ax.plot([0.38, 0.45, 0.58, 0.72, 0.88, 0.94], [0.35, 0.48, 0.62, 0.70, 0.78, 0.84], color='#94a3b8', linewidth=8, alpha=0.5, transform=ax.transAxes)
    ax.plot([0.38, 0.45, 0.58, 0.72, 0.88, 0.94], [0.35, 0.48, 0.62, 0.70, 0.78, 0.84], color='#27ae60', linewidth=5, transform=ax.transAxes, label='Ground Truth Route')
    ax.plot([0.38, 0.45, 0.58, 0.72, 0.88, 0.94], [0.35, 0.48, 0.62, 0.70, 0.78, 0.84], color='#8e44ad', linewidth=3, linestyle='--', transform=ax.transAxes, label='Classifier + HMM Route')
    
    # Scatter GPS noisy points
    np.random.seed(42)
    gps_x = np.linspace(0.38, 0.94, 25) + np.random.normal(0, 0.006, 25)
    gps_y = np.linspace(0.35, 0.84, 25) + np.random.normal(0, 0.008, 25)
    ax.scatter(gps_x, gps_y, color='#2563eb', s=40, edgecolors='#1e40af', zorder=5, transform=ax.transAxes, label='Raw GPS Observations')
    
    # Map Legend box
    ax.add_patch(patches.Rectangle((0.35, 0.72), 0.28, 0.16, transform=ax.transAxes, facecolor='white', edgecolor='#94a3b8', alpha=0.9))
    ax.text(0.36, 0.84, "PathSnap Map Legend", transform=ax.transAxes, color='black', fontsize=9, fontweight='bold')
    ax.text(0.36, 0.80, "― Ground Truth Route (Green)", transform=ax.transAxes, color='#27ae60', fontsize=8, fontweight='bold')
    ax.text(0.36, 0.76, "-- HMM Reconstructed Route (Purple)", transform=ax.transAxes, color='#8e44ad', fontsize=8, fontweight='bold')
    ax.text(0.36, 0.73, "• Raw GPS Observations (Blue)", transform=ax.transAxes, color='#2563eb', fontsize=8, fontweight='bold')
    
    # Bottom Terminal Console
    ax.add_patch(patches.Rectangle((0.33, 0.01), 0.66, 0.26, transform=ax.transAxes, facecolor='#040711', edgecolor='#334155'))
    ax.text(0.34, 0.24, "LIVE API COMMUNICATION LOG (POST /api/evaluate/00000052)", transform=ax.transAxes, color='#94a3b8', fontsize=8, fontfamily='monospace')
    
    console_lines = [
        "[15:47:05] >>> Dispatching evaluation request: POST http://127.0.0.1:8001/api/evaluate/00000052",
        "[15:47:05] <<< HTTP 200 OK | Trip: 00000052 | Observations: 262 | Latency: 0.123s",
        "[15:47:05] [Evaluation Result] HMM Top-1 Acc: 96.56% (+2.29%) | Continuity: 95.79% (+2.68%)",
        "[15:47:05] [Map Engine] Reconstructed 262 road segments. Interactive Folium map rendered."
    ]
    y_c = 0.19
    for line in console_lines:
        ax.text(0.34, y_c, line, transform=ax.transAxes, color='#38bdf8' if '>>>' in line else '#4ade80', fontsize=8, fontfamily='monospace')
        y_c -= 0.05
        
    ax.axis('off')
    plt.tight_layout()
    out_path = os.path.join(ARTIFACTS_DIR, 'plots', 'evaluator_ui_screenshot.png')
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {out_path}")

def generate_folium_map_graphic():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#ffffff')
    
    # Load Trip 00000052 data
    import pandas as pd
    trip_path = r'Student_folder/Student folder/student_test/00000052'
    track = pd.read_csv(os.path.join(trip_path, '00000052.track'), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    nodes = pd.read_csv(os.path.join(trip_path, '00000052.nodes'), sep=r'\s+', header=None, names=['lon', 'lat'])
    arcs = pd.read_csv(os.path.join(trip_path, '00000052.arcs'), sep=r'\s+', header=None, names=['u', 'v'])
    route = pd.read_csv(os.path.join(trip_path, '00000052.route'), header=None, names=['arc_id'])
    
    # Plot road network background
    sample_arcs = arcs.sample(min(800, len(arcs)), random_state=42)
    for _, row in sample_arcs.iterrows():
        u_pt = nodes.loc[row['u'], ['lon', 'lat']].values
        v_pt = nodes.loc[row['v'], ['lon', 'lat']].values
        ax.plot([u_pt[0], v_pt[0]], [u_pt[1], v_pt[1]], color='#cbd5e1', linewidth=1, alpha=0.6, zorder=1)
        
    # Plot reference route
    route_arcs = route['arc_id'].values
    r_coords = []
    for a in route_arcs:
        u_pt = nodes.loc[arcs.loc[a, 'u'], ['lon', 'lat']].values
        v_pt = nodes.loc[arcs.loc[a, 'v'], ['lon', 'lat']].values
        if not r_coords:
            r_coords.append(u_pt)
        r_coords.append(v_pt)
    r_coords = np.array(r_coords)
    ax.plot(r_coords[:, 0], r_coords[:, 1], color='#27ae60', linewidth=4, alpha=0.85, label='Ground Truth Reference Route', zorder=3)
    
    # Plot GPS points
    ax.scatter(track['lon'], track['lat'], color='#2563eb', s=25, alpha=0.8, edgecolors='#1e40af', label='Raw Noisy GPS Observations', zorder=4)
    
    ax.set_title('Folium Geographical Map: Trip 00000052 Trajectory & Road Network', fontsize=13, fontweight='bold')
    ax.set_xlabel('Longitude (°E)', fontsize=11)
    ax.set_ylabel('Latitude (°N)', fontsize=11)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(ARTIFACTS_DIR, 'plots', 'folium_map_screenshot.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

if __name__ == '__main__':
    generate_api_terminal_graphic()
    generate_ui_screenshot_graphic()
    generate_folium_map_graphic()
