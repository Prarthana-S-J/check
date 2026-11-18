import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def project_points_to_segments(p_arr, u_arr, v_arr):
    vec = v_arr - u_arr
    L2 = np.sum(vec**2, axis=-1, keepdims=True)
    L2 = np.maximum(L2, 1e-6)
    diff = p_arr - u_arr
    t = np.clip(np.sum(diff * vec, axis=-1, keepdims=True) / L2, 0.0, 1.0)
    proj = u_arr + t * vec
    dists = np.linalg.norm(p_arr - proj, axis=-1)
    return dists, t.squeeze(-1)

def load_trip_data(trip_dir, trip_id):
    track_f = os.path.join(trip_dir, trip_id, f"{trip_id}.track")
    nodes_f = os.path.join(trip_dir, trip_id, f"{trip_id}.nodes")
    arcs_f = os.path.join(trip_dir, trip_id, f"{trip_id}.arcs")
    route_f = os.path.join(trip_dir, trip_id, f"{trip_id}.route")
    
    track = pd.read_csv(track_f, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    nodes = pd.read_csv(nodes_f, sep=r'\s+', header=None, names=['lon', 'lat'])
    arcs = pd.read_csv(arcs_f, sep=r'\s+', header=None, names=['u', 'v'])
    route = pd.read_csv(route_f, header=None, names=['arc_id']) if os.path.exists(route_f) else None
    
    return track, nodes, arcs, route

def build_spatial_network(nodes, arcs, lat_center):
    scale = np.array([111320.0 * np.cos(np.radians(lat_center)), 111320.0])
    u_idx = arcs['u'].values
    v_idx = arcs['v'].values
    
    u_m = nodes.loc[u_idx, ['lon', 'lat']].values * scale
    v_m = nodes.loc[v_idx, ['lon', 'lat']].values * scale
    vec_m = v_m - u_m
    lengths = np.linalg.norm(vec_m, axis=1)
    
    sample_pts = []
    sample_arc_ids = []
    
    for arc_id in range(len(arcs)):
        u_pt = u_m[arc_id]
        v_pt = v_m[arc_id]
        seg_vec = vec_m[arc_id]
        seg_len = lengths[arc_id]
        n_samples = max(2, int(np.ceil(seg_len / 25.0)) + 1)
        fractions = np.linspace(0.0, 1.0, n_samples)
        pts = u_pt + fractions[:, None] * seg_vec
        sample_pts.append(pts)
        sample_arc_ids.extend([arc_id] * n_samples)
        
    sample_pts = np.vstack(sample_pts)
    sample_arc_ids = np.array(sample_arc_ids)
    tree = cKDTree(sample_pts)
    
    return {
        'scale': scale,
        'u_m': u_m,
        'v_m': v_m,
        'vec_m': vec_m,
        'lengths': lengths,
        'sample_pts': sample_pts,
        'sample_arc_ids': sample_arc_ids,
        'tree': tree
    }

def get_ground_truth_arcs(track, nodes, arcs, route, net):
    route_arcs = route['arc_id'].values
    scale = net['scale']
    u_nodes = arcs.loc[route_arcs, 'u'].values
    v_nodes = arcs.loc[route_arcs, 'v'].values
    u_m = nodes.loc[u_nodes, ['lon', 'lat']].values * scale
    v_m = nodes.loc[v_nodes, ['lon', 'lat']].values * scale
    vec_m = v_m - u_m
    L2 = np.maximum(np.sum(vec_m**2, axis=1), 1e-6)
    
    track_m = track[['lon', 'lat']].values * scale
    true_arcs = []
    curr_idx = 0
    M = len(route_arcs)
    
    for i in range(len(track_m)):
        p = track_m[i]
        win_start = max(0, curr_idx - 2)
        win_end = min(M, curr_idx + 30)
        
        u_win = u_m[win_start:win_end]
        v_win = v_m[win_start:win_end]
        vec_win = vec_m[win_start:win_end]
        L2_win = L2[win_start:win_end]
        
        diff = p - u_win
        t = np.clip(np.sum(diff * vec_win, axis=1) / L2_win, 0.0, 1.0)
        proj = u_win + t[:, None] * vec_win
        dists = np.linalg.norm(p - proj, axis=1)
        best_local = np.argmin(dists)
        curr_idx = win_start + best_local
        true_arcs.append(route_arcs[curr_idx])
        
    return np.array(true_arcs)

