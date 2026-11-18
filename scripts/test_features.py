import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

trip_path = r'Student_folder/Student folder/train/00000001'
track = pd.read_csv(os.path.join(trip_path, '00000001.track'), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
nodes = pd.read_csv(os.path.join(trip_path, '00000001.nodes'), sep=r'\s+', header=None, names=['lon', 'lat'])
arcs = pd.read_csv(os.path.join(trip_path, '00000001.arcs'), sep=r'\s+', header=None, names=['u', 'v'])
route = pd.read_csv(os.path.join(trip_path, '00000001.route'), header=None, names=['arc_id'])

route_arcs = route['arc_id'].values
u_nodes = arcs.loc[route_arcs, 'u'].values
v_nodes = arcs.loc[route_arcs, 'v'].values
u_coords = nodes.loc[u_nodes, ['lon', 'lat']].values
v_coords = nodes.loc[v_nodes, ['lon', 'lat']].values

lat_mid = np.radians(track['lat'].mean())
scale = np.array([111320.0 * np.cos(lat_mid), 111320.0])

u_m = u_coords * scale
v_m = v_coords * scale
vec_m = v_m - u_m
L2 = np.sum(vec_m**2, axis=1)
L2[L2 == 0] = 1e-6
track_m = track[['lon', 'lat']].values * scale

# Ground truth matching
true_arcs = []
curr_idx = 0
M = len(route_arcs)
for i in range(len(track_m)):
    p = track_m[i]
    win_start = max(0, curr_idx - 2)
    win_end = min(M, curr_idx + 25)
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
true_arcs = np.array(true_arcs)

all_u = arcs['u'].values
all_v = arcs['v'].values
all_u_m = nodes.loc[all_u, ['lon', 'lat']].values * scale
all_v_m = nodes.loc[all_v, ['lon', 'lat']].values * scale
all_vec_m = all_v_m - all_u_m
all_L2 = np.sum(all_vec_m**2, axis=1)
all_L2[all_L2 == 0] = 1e-6
all_lengths = np.sqrt(all_L2)

# Build KDTree with sampled points
sample_pts = []
sample_arc_ids = []
for arc_id in range(len(arcs)):
    u_pt = all_u_m[arc_id]
    v_pt = all_v_m[arc_id]
    seg_vec = v_pt - u_pt
    seg_len = all_lengths[arc_id]
    n_samples = max(2, int(np.ceil(seg_len / 25.0)) + 1)
    fractions = np.linspace(0.0, 1.0, n_samples)
    pts = u_pt + fractions[:, None] * seg_vec
    sample_pts.append(pts)
    sample_arc_ids.extend([arc_id] * n_samples)

sample_pts = np.vstack(sample_pts)
sample_arc_ids = np.array(sample_arc_ids)
tree = cKDTree(sample_pts)

# Compute vehicle velocity vectors
diff_track = np.zeros_like(track_m)
diff_track[1:] = track_m[1:] - track_m[:-1]
diff_track[0] = diff_track[1] if len(track_m) > 1 else np.array([0, 0])

times = track['time'].values
dt = np.ones_like(times)
dt[1:] = np.maximum(times[1:] - times[:-1], 1e-3)
dt[0] = dt[1] if len(times) > 1 else 1.0

speeds = np.linalg.norm(diff_track, axis=1) / dt
veh_heading = np.arctan2(diff_track[:, 1], diff_track[:, 0])

# Feature extraction
rows = []
for i in range(len(track_m)):
    p = track_m[i]
    pt_idxs = tree.query_ball_point(p, r=50.0)
    if len(pt_idxs) == 0:
        _, nearest_idx = tree.query(p, k=1)
        c_arcs = np.array([sample_arc_ids[nearest_idx]])
    else:
        c_arcs = np.unique(sample_arc_ids[pt_idxs])
    
    u_cand = all_u_m[c_arcs]
    v_cand = all_v_m[c_arcs]
    vec_cand = all_vec_m[c_arcs]
    L2_cand = all_L2[c_arcs]
    len_cand = all_lengths[c_arcs]
    
    diff = p - u_cand
    t_proj = np.clip(np.sum(diff * vec_cand, axis=1) / L2_cand, 0.0, 1.0)
    proj = u_cand + t_proj[:, None] * vec_cand
    perp_dists = np.linalg.norm(p - proj, axis=1)
    
    # Sort by perp dist and take top 8
    order = np.argsort(perp_dists)[:8]
    c_arcs = c_arcs[order]
    perp_dists = perp_dists[order]
    t_proj = t_proj[order]
    len_cand = len_cand[order]
    vec_cand = vec_cand[order]
    u_cand = u_cand[order]
    v_cand = v_cand[order]
    
    dist_u = np.linalg.norm(p - u_cand, axis=1)
    dist_v = np.linalg.norm(p - v_cand, axis=1)
    
    seg_heading = np.arctan2(vec_cand[:, 1], vec_cand[:, 0])
    angle_diff = np.abs(veh_heading[i] - seg_heading)
    angle_diff = np.where(angle_diff > np.pi, 2 * np.pi - angle_diff, angle_diff)
    cos_angle = np.cos(angle_diff)
    
    min_d = perp_dists[0]
    
    for rank in range(len(c_arcs)):
        arc_id = c_arcs[rank]
        is_true = 1 if arc_id == true_arcs[i] else 0
        rows.append({
            'point_idx': i,
            'arc_id': arc_id,
            'label': is_true,
            'perp_dist': perp_dists[rank],
            'dist_u': dist_u[rank],
            'dist_v': dist_v[rank],
            'seg_length': len_cand[rank],
            'proj_ratio': t_proj[rank],
            'angle_diff': angle_diff[rank],
            'cos_angle': cos_angle[rank],
            'speed': speeds[i],
            'rank_dist': rank,
            'dist_diff_min': perp_dists[rank] - min_d,
            'dist_ratio': perp_dists[rank] / (min_d + 1.0)
        })

df_feat = pd.DataFrame(rows)
print("Features DataFrame shape:", df_feat.shape)
print("Label distribution:\n", df_feat['label'].value_counts())
print("\nCorrelation with label:")
print(df_feat.drop(columns=['point_idx', 'arc_id']).corr()['label'].sort_values(ascending=False))

