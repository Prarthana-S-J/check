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

all_u = arcs['u'].values
all_v = arcs['v'].values
all_u_m = nodes.loc[all_u, ['lon', 'lat']].values * scale
all_v_m = nodes.loc[all_v, ['lon', 'lat']].values * scale
all_vec_m = all_v_m - all_u_m
all_L2 = np.sum(all_vec_m**2, axis=1)
all_L2[all_L2 == 0] = 1e-6

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
tree = cKDTree(sample_pts)

def get_candidates(p, radius=50.0, max_k=8):
    pt_idxs = tree.query_ball_point(p, r=radius)
    if len(pt_idxs) == 0:
        # Fallback to nearest point
        _, nearest_idx = tree.query(p, k=1)
        c_arcs = np.array([sample_arc_ids[nearest_idx]])
    else:
        c_arcs = np.unique(sample_arc_ids[pt_idxs])
    
    # Compute exact perpendicular distance to each candidate arc segment
    u_cand = all_u_m[c_arcs]
    v_cand = all_v_m[c_arcs]
    vec_cand = all_vec_m[c_arcs]
    L2_cand = all_L2[c_arcs]
    diff = p - u_cand
    t = np.clip(np.sum(diff * vec_cand, axis=1) / L2_cand, 0.0, 1.0)
    proj = u_cand + t[:, None] * vec_cand
    perp_dists = np.linalg.norm(p - proj, axis=1)
    
    order = np.argsort(perp_dists)[:max_k]
    return c_arcs[order], perp_dists[order]

for max_k in [3, 5, 8, 10]:
    misses = 0
    top1_closest = 0
    for i in range(len(track_m)):
        p = track_m[i]
        c_arcs, c_dists = get_candidates(p, radius=50.0, max_k=max_k)
        if true_arcs[i] not in c_arcs:
            misses += 1
        if c_arcs[0] == true_arcs[i]:
            top1_closest += 1
    print(f"Max-K {max_k:2d}: Candidate misses = {misses}/{len(track)} ({misses/len(track)*100:.2f}%), Top-1 closest Euclidean accuracy = {top1_closest}/{len(track)} ({top1_closest/len(track)*100:.2f}%)")

