import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    'perp_dist',
    'dist_u',
    'dist_v',
    'seg_length',
    'proj_ratio',
    'angle_diff',
    'cos_angle',
    'speed',
    'rank_dist',
    'dist_diff_min',
    'dist_ratio',
    'speed_along_segment',
    'speed_perp_segment',
    'time_delta'
]

def extract_features_for_trip(track, nodes, arcs, net, true_arcs=None, radius=50.0, max_k=8):
    scale = net['scale']
    u_m = net['u_m']
    v_m = net['v_m']
    vec_m = net['vec_m']
    lengths = net['lengths']
    sample_arc_ids = net['sample_arc_ids']
    tree = net['tree']
    
    track_m = track[['lon', 'lat']].values * scale
    N = len(track_m)
    
    # Compute vehicle motion kinematics
    diff_track = np.zeros_like(track_m)
    if N > 1:
        diff_track[1:] = track_m[1:] - track_m[:-1]
        diff_track[0] = diff_track[1]
    
    times = track['time'].values
    dt = np.ones(N)
    if N > 1:
        dt[1:] = np.maximum(times[1:] - times[:-1], 1e-3)
        dt[0] = dt[1]
    
    speeds = np.linalg.norm(diff_track, axis=1) / dt
    veh_heading = np.arctan2(diff_track[:, 1], diff_track[:, 0])
    
    records = []
    candidates_per_point = []
    
    for i in range(N):
        p = track_m[i]
        pt_idxs = tree.query_ball_point(p, r=radius)
        if len(pt_idxs) == 0:
            _, nearest_idx = tree.query(p, k=1)
            c_arcs = np.array([sample_arc_ids[nearest_idx]])
        else:
            c_arcs = np.unique(sample_arc_ids[pt_idxs])
            
        u_cand = u_m[c_arcs]
        v_cand = v_m[c_arcs]
        vec_cand = vec_m[c_arcs]
        len_cand = lengths[c_arcs]
        L2_cand = np.maximum(np.sum(vec_cand**2, axis=1), 1e-6)
        
        diff = p - u_cand
        t_proj = np.clip(np.sum(diff * vec_cand, axis=1) / L2_cand, 0.0, 1.0)
        proj = u_cand + t_proj[:, None] * vec_cand
        perp_dists = np.linalg.norm(p - proj, axis=1)
        
        order = np.argsort(perp_dists)[:max_k]
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
        
        # Velocity components along and perpendicular to road segment
        seg_unit = vec_cand / np.maximum(len_cand[:, None], 1e-6)
        v_veh = diff_track[i] / dt[i]
        v_along = np.sum(v_veh * seg_unit, axis=1)
        v_perp = np.abs(v_veh[0] * seg_unit[:, 1] - v_veh[1] * seg_unit[:, 0])
        
        min_d = perp_dists[0]
        K_cand = len(c_arcs)
        
        point_cands = []
        for rank in range(K_cand):
            arc_id = int(c_arcs[rank])
            is_true = 1 if (true_arcs is not None and arc_id == true_arcs[i]) else 0
            
            feat_dict = {
                'point_idx': i,
                'arc_id': arc_id,
                'label': is_true,
                'perp_dist': float(perp_dists[rank]),
                'dist_u': float(dist_u[rank]),
                'dist_v': float(dist_v[rank]),
                'seg_length': float(len_cand[rank]),
                'proj_ratio': float(t_proj[rank]),
                'angle_diff': float(angle_diff[rank]),
                'cos_angle': float(cos_angle[rank]),
                'speed': float(speeds[i]),
                'rank_dist': int(rank),
                'dist_diff_min': float(perp_dists[rank] - min_d),
                'dist_ratio': float(perp_dists[rank] / (min_d + 1.0)),
                'speed_along_segment': float(v_along[rank]),
                'speed_perp_segment': float(v_perp[rank]),
                'time_delta': float(dt[i])
            }
            records.append(feat_dict)
            point_cands.append(feat_dict)
            
        candidates_per_point.append(point_cands)
        
    df_features = pd.DataFrame(records)
    return df_features, candidates_per_point

