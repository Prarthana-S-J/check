import os
import pandas as pd
import numpy as np

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

# Monotonic matching: for point t, search in window [current_idx - 1, current_idx + K]
matched_indices = []
matched_dists = []
curr_idx = 0
M = len(route_arcs)

for i in range(len(track_m)):
    p = track_m[i]
    # search window from max(0, curr_idx - 2) to min(M, curr_idx + 15)
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
    best_global = win_start + best_local
    curr_idx = best_global
    matched_indices.append(best_global)
    matched_dists.append(dists[best_local])

matched_dists = np.array(matched_dists)
matched_indices = np.array(matched_indices)

print(f"Monotonic matching distances: mean={np.mean(matched_dists):.2f}m, median={np.median(matched_dists):.2f}m, max={np.max(matched_dists):.2f}m")
print(f"Final route index matched: {matched_indices[-1]} / {M-1}")
print(f"Number of distinct route arcs traversed: {len(np.unique(matched_indices))} / {M}")

