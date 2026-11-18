import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

trip_path = r'Student_folder/Student folder/train/00000001'
track = pd.read_csv(os.path.join(trip_path, '00000001.track'), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
nodes = pd.read_csv(os.path.join(trip_path, '00000001.nodes'), sep=r'\s+', header=None, names=['lon', 'lat'])
arcs = pd.read_csv(os.path.join(trip_path, '00000001.arcs'), sep=r'\s+', header=None, names=['u', 'v'])
route = pd.read_csv(os.path.join(trip_path, '00000001.route'), header=None, names=['arc_id'])

# Ground truth arcs per point via monotonic matching
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

# Now, build spatial index for ALL arcs in the trip's road network
all_u_coords = nodes.loc[arcs['u'].values, ['lon', 'lat']].values
all_v_coords = nodes.loc[arcs['v'].values, ['lon', 'lat']].values
all_mid_coords = (all_u_coords + all_v_coords) / 2.0
all_mid_m = all_mid_coords * scale

tree = cKDTree(all_mid_m)

# Test different K and radius
for K in [5, 8, 10, 15]:
    misses = 0
    for i in range(len(track_m)):
        p = track_m[i]
        # Query top K nearest arc midpoints (or with radius)
        dists, idxs = tree.query(p, k=K)
        # Check if true_arcs[i] is in idxs
        if true_arcs[i] not in idxs:
            misses += 1
    print(f"Top-{K} nearest arc midpoints: Misses = {misses}/{len(track)} ({misses/len(track)*100:.2f}%)")

