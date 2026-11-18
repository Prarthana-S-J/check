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

def project_point_to_segments(p, seg_u, seg_v):
    lat_mid = np.radians(p[1])
    scale = np.array([111320.0 * np.cos(lat_mid), 111320.0])
    p_m = p * scale
    u_m = seg_u * scale
    v_m = seg_v * scale
    vec = v_m - u_m
    L2 = np.sum(vec**2, axis=1)
    L2[L2 == 0] = 1e-6
    diff = p_m - u_m
    dot = np.sum(diff * vec, axis=1)
    t = np.clip(dot / L2, 0.0, 1.0)
    proj = u_m + t[:, None] * vec
    dists = np.linalg.norm(p_m - proj, axis=1)
    return dists

for t_idx in range(0, len(track), 150):
    p = track.iloc[t_idx][['lon', 'lat']].values
    dists = project_point_to_segments(p, u_coords, v_coords)
    best_idx = np.argmin(dists)
    t_sec = track.iloc[t_idx]['time']
    print(f"GPS t={t_idx:4d} (time={t_sec:4.0f}s): closest route idx = {best_idx:3d}/{len(route_arcs)} (arc_id {route_arcs[best_idx]}), dist = {dists[best_idx]:.2f}m")

