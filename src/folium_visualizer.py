import os
import sys
import pandas as pd
import numpy as np
import folium
from folium import FeatureGroup, LayerControl
import joblib

sys.path.append(os.path.abspath('src'))
from data_processor import load_trip_data, build_spatial_network, get_ground_truth_arcs
from feature_engineering import extract_features_for_trip, FEATURE_COLUMNS
from hmm_decoder import HMMRouteDecoder

BASE_DIR = r'Student_folder/Student folder'
TEST_DIR = os.path.join(BASE_DIR, 'student_test')
ARTIFACTS_DIR = 'artifacts'

def generate_interactive_folium_map(trip_id, output_path=None):
    if output_path is None:
        output_path = os.path.join(ARTIFACTS_DIR, 'maps', f"map_reconstruction_{trip_id}.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Load data
    track, nodes, arcs, route = load_trip_data(TEST_DIR, trip_id)
    net = build_spatial_network(nodes, arcs, track['lat'].mean())
    true_arcs = get_ground_truth_arcs(track, nodes, arcs, route, net)
    
    # 2. Run inference
    model_path = os.path.join(ARTIFACTS_DIR, 'models', 'best_tuned_model.joblib')
    clf = joblib.load(model_path)
    
    df_feat, cands_per_point = extract_features_for_trip(track, nodes, arcs, net, true_arcs=true_arcs)
    probs = clf.predict_proba(df_feat[FEATURE_COLUMNS].values)[:, 1]
    
    row_idx = 0
    for cands in cands_per_point:
        for c in cands:
            c['prob'] = float(probs[row_idx])
            row_idx += 1
            
    scale = net['scale']
    track_m = track[['lon', 'lat']].values * scale
    diff_track = np.linalg.norm(track_m[1:] - track_m[:-1], axis=1)
    times = track['time'].values
    dts = np.maximum(times[1:] - times[:-1], 1e-3)
    
    decoder = HMMRouteDecoder(arcs, nodes, beta=20.0, same_arc_prob=0.85, direct_prob=1.0)
    hmm_pred_arcs = decoder.decode(cands_per_point, step_distances=diff_track, dts=dts)
    clf_pred_arcs = [max(cands, key=lambda c: c['prob'])['arc_id'] for cands in cands_per_point]
    
    # 3. Create Map centered at trip
    center_lat = track['lat'].mean()
    center_lon = track['lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='OpenStreetMap')
    
    # Layer 1: Ground Truth Route (Green)
    fg_truth = FeatureGroup(name='Ground Truth Route (Reference)', show=True)
    route_arcs = route['arc_id'].values
    truth_coords = []
    for a in route_arcs:
        u_pt = [nodes.loc[arcs.loc[a, 'u'], 'lat'], nodes.loc[arcs.loc[a, 'u'], 'lon']]
        v_pt = [nodes.loc[arcs.loc[a, 'v'], 'lat'], nodes.loc[arcs.loc[a, 'v'], 'lon']]
        if not truth_coords:
            truth_coords.append(u_pt)
        truth_coords.append(v_pt)
        
    folium.PolyLine(
        locations=truth_coords,
        color='#27ae60',
        weight=6,
        opacity=0.8,
        tooltip=f"Ground Truth Route ({len(route_arcs)} arcs)"
    ).add_to(fg_truth)
    fg_truth.add_to(m)
    
    # Layer 2: HMM Reconstructed Route (Purple dashed)
    fg_hmm = FeatureGroup(name='Reconstructed Route (Classifier + HMM)', show=True)
    hmm_coords = []
    for a in hmm_pred_arcs:
        u_pt = [nodes.loc[arcs.loc[a, 'u'], 'lat'], nodes.loc[arcs.loc[a, 'u'], 'lon']]
        v_pt = [nodes.loc[arcs.loc[a, 'v'], 'lat'], nodes.loc[arcs.loc[a, 'v'], 'lon']]
        if not hmm_coords:
            hmm_coords.append(u_pt)
        hmm_coords.append(v_pt)
        
    folium.PolyLine(
        locations=hmm_coords,
        color='#8e44ad',
        weight=4,
        dash_array='8, 8',
        opacity=0.9,
        tooltip='Reconstructed Route (HMM Viterbi)'
    ).add_to(fg_hmm)
    fg_hmm.add_to(m)
    
    # Layer 3: Raw GPS Observations (Blue Dots)
    fg_gps = FeatureGroup(name='Raw Noisy GPS Observations', show=True)
    # sample every 2nd point for crispness
    for idx in range(0, len(track), 2):
        row = track.iloc[idx]
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=3.5,
            color='#1f77b4',
            fill=True,
            fill_color='#3498db',
            fill_opacity=0.75,
            popup=f"GPS #{idx}<br>Time: {row['time']:.0f}s<br>Lat: {row['lat']:.5f}<br>Lon: {row['lon']:.5f}"
        ).add_to(fg_gps)
    fg_gps.add_to(m)
    
    # Add Legend HTML
    legend_html = '''
    <div style="position: fixed; 
                bottom: 30px; left: 30px; width: 260px; height: 160px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:13px; font-family: Arial, sans-serif; padding: 12px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3); border-radius: 8px;">
      <b>PathSnap Route Reconstruction</b><br>
      <i style="color:#27ae60; font-size:16px;">&#9644;&#9644;&#9644;</i> <b>Ground Truth Route</b><br>
      <i style="color:#8e44ad; font-size:16px;">&#9644; &#9644; &#9644;</i> <b>Classifier + HMM Route</b><br>
      <i style="color:#3498db; font-size:16px;">&bull;</i> <b>Raw GPS Points (Noisy)</b><br>
      <hr style="margin: 6px 0;">
      <small>Trip: ''' + trip_id + ''' | Method: XGBoost + Viterbi</small>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    LayerControl(collapsed=False).add_to(m)
    m.save(output_path)
    print(f"Generated interactive Folium map for Trip {trip_id}: {output_path}")
    return output_path

if __name__ == '__main__':
    # Generate maps for key demonstration trips
    for tid in ['00000052', '00000007', '00000000', '00000091']:
        generate_interactive_folium_map(tid)

