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

# For each GPS point, find the closest arc in route
# To maintain physical consistency and avoid backward jumps if roads loop,
# we can search in a forward window or globally. Let's compare global closest vs forward-constrained closest.

lat_mid = np.radians(track['lat'].mean())
scale = np.array([111320.0 * np.cos(lat_mid), 111320.0])

u_m = u_coords * scale
v_m = v_coords * scale
vec_m = v_m - u_m
L2 = np.sum(vec_m**2, axis=1)
L2[L2 == 0] = 1e-6

track_m = track[['lon', 'lat']].values * scale

# Method 1: Global closest route arc for each GPS point
global_closest_arc = []
global_min_dists = []
global_route_indices = []

for i in range(len(track_m)):
    p = track_m[i]
    diff = p - u_m
    t = np.clip(np.sum(diff * vec_m, axis=1) / L2, 0.0, 1.0)
    proj = u_m + t[:, None] * vec_m
    dists = np.linalg.norm(p - proj, axis=1)
    best_idx = np.argmin(dists)
    global_route_indices.append(best_idx)
    global_closest_arc.append(route_arcs[best_idx])
    global_min_dists.append(dists[best_idx])

global_min_dists = np.array(global_min_dists)
global_route_indices = np.array(global_route_indices)

print(f"Global closest distances to route: mean={np.mean(global_min_dists):.2f}m, median={np.median(global_min_dists):.2f}m, max={np.max(global_min_dists):.2f}m, 95th={np.percentile(global_min_dists, 95):.2f}m")

# Check monotonicity: how often does route index go backward?
backwards = np.diff(global_route_indices) < 0
print(f"Backward index jumps: {np.sum(backwards)} out of {len(track)-1}")
if np.sum(backwards) > 0:
    jump_sizes = np.diff(global_route_indices)[backwards]
    print(f"Backward jump sizes: min={np.min(jump_sizes)}, max={np.max(jump_sizes)}, median={np.median(jump_sizes)}")

