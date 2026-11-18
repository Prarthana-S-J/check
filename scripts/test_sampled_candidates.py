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

# Strategy A: Sample points along each arc every 25m
all_u = arcs['u'].values
all_v = arcs['v'].values
all_u_m = nodes.loc[all_u, ['lon', 'lat']].values * scale
all_v_m = nodes.loc[all_v, ['lon', 'lat']].values * scale

sample_pts = []
sample_arc_ids = []

for arc_id in range(len(arcs)):
    u_pt = all_u_m[arc_id]
    v_pt = all_v_m[arc_id]
    seg_vec = v_pt - u_pt
    seg_len = np.linalg.norm(seg_vec)
    n_samples = max(2, int(np.ceil(seg_len / 25.0)) + 1)
    fractions = np.linspace(0.0, 1.0, n_samples)
    pts = u_pt + fractions[:, None] * seg_vec
    sample_pts.append(pts)
    sample_arc_ids.extend([arc_id] * n_samples)

sample_pts = np.vstack(sample_pts)
sample_arc_ids = np.array(sample_arc_ids)

print(f"Total sampled points in network: {len(sample_pts)}")
tree = cKDTree(sample_pts)

for radius in [30.0, 50.0, 75.0, 100.0]:
    misses = 0
    candidate_counts = []
    for i in range(len(track_m)):
        p = track_m[i]
        # query all points within radius
        pt_idxs = tree.query_ball_point(p, r=radius)
        c_arcs = set(sample_arc_ids[pt_idxs]) if len(pt_idxs) > 0 else set()
        candidate_counts.append(len(c_arcs))
        if true_arcs[i] not in c_arcs:
            misses += 1
    mean_c = np.mean(candidate_counts)
    print(f"Radius {radius:.0f}m: Misses = {misses}/{len(track)} ({misses/len(track)*100:.2f}%), Avg candidates per point = {mean_c:.1f}")

