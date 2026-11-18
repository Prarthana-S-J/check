import os
import pandas as pd
import numpy as np

trip_path = r'Student_folder/Student folder/train/00000001'
track = pd.read_csv(os.path.join(trip_path, '00000001.track'), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
nodes = pd.read_csv(os.path.join(trip_path, '00000001.nodes'), sep=r'\s+', header=None, names=['lon', 'lat'])
arcs = pd.read_csv(os.path.join(trip_path, '00000001.arcs'), sep=r'\s+', header=None, names=['u', 'v'])
route = pd.read_csv(os.path.join(trip_path, '00000001.route'), header=None, names=['arc_id'])

route_arcs = route['arc_id'].values

# Let's inspect the length of each arc in route
u_nodes = arcs.loc[route_arcs, 'u'].values
v_nodes = arcs.loc[route_arcs, 'v'].values

u_coords = nodes.loc[u_nodes, ['lon', 'lat']].values
v_coords = nodes.loc[v_nodes, ['lon', 'lat']].values

# Calculate physical lengths of each arc in meters
lat_mid = np.radians(u_coords[:, 1])
scale = np.column_stack([111320.0 * np.cos(lat_mid), np.full_like(lat_mid, 111320.0)])
vec_m = (v_coords - u_coords) * scale
arc_lengths = np.linalg.norm(vec_m, axis=1)

print(f"Total route arcs: {len(route_arcs)}")
print(f"Arc lengths (m): min={np.min(arc_lengths):.1f}, max={np.max(arc_lengths):.1f}, mean={np.mean(arc_lengths):.1f}, median={np.median(arc_lengths):.1f}")
print(f"Total route length: {np.sum(arc_lengths):.1f} m ({np.sum(arc_lengths)/1000:.2f} km)")

# Calculate track length
track_coords = track[['lon', 'lat']].values
lat_mid_t = np.radians(track_coords[:-1, 1])
scale_t = np.column_stack([111320.0 * np.cos(lat_mid_t), np.full_like(lat_mid_t, 111320.0)])
diff_t = np.diff(track_coords, axis=0) * scale_t
step_dists = np.linalg.norm(diff_t, axis=1)
print(f"Track points: {len(track)}")
print(f"Track step distances (m): mean={np.mean(step_dists):.1f}, median={np.median(step_dists):.1f}, sum={np.sum(step_dists):.1f} m")

# Speeds
dt = np.diff(track['time'].values)
speeds = step_dists / np.maximum(dt, 1e-3)
print(f"Speeds (m/s): mean={np.mean(speeds):.1f}, max={np.max(speeds):.1f}")

