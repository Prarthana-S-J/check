import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.dpi'] = 300

BASE_DIR = r'Student_folder/Student folder'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'student_test')
ARTIFACTS_DIR = 'artifacts'

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000.0 # meters
    return c * r

def run_eda():
    print("=== Starting Exploratory Data Analysis ===")
    
    trip_records = []
    
    partitions = {'train': TRAIN_DIR, 'student_test': TEST_DIR}
    
    all_sampling_intervals = []
    all_speeds = []
    all_step_distances = []
    
    for part_name, part_dir in partitions.items():
        trips = sorted(os.listdir(part_dir))
        for t in trips:
            t_path = os.path.join(part_dir, t)
            track_f = os.path.join(t_path, f"{t}.track")
            nodes_f = os.path.join(t_path, f"{t}.nodes")
            arcs_f = os.path.join(t_path, f"{t}.arcs")
            route_f = os.path.join(t_path, f"{t}.route")
            
            # Read files
            track = pd.read_csv(track_f, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
            nodes = pd.read_csv(nodes_f, sep=r'\s+', header=None, names=['lon', 'lat'])
            arcs = pd.read_csv(arcs_f, sep=r'\s+', header=None, names=['u', 'v'])
            route = pd.read_csv(route_f, header=None, names=['arc_id'])
            
            n_track = len(track)
            n_nodes = len(nodes)
            n_arcs = len(arcs)
            n_route = len(route)
            
            # Check missing / invalid
            track_missing = track.isnull().sum().sum()
            
            # Compute time deltas
            time_vals = track['time'].values
            dt = np.diff(time_vals)
            all_sampling_intervals.extend(dt.tolist())
            
            # Compute step distances
            coords = track[['lon', 'lat']].values
            step_dists = haversine_np(coords[:-1, 0], coords[:-1, 1], coords[1:, 0], coords[1:, 1])
            all_step_distances.extend(step_dists.tolist())
            
            # Speeds (m/s)
            speeds = step_dists / np.maximum(dt, 1e-4)
            all_speeds.extend(speeds.tolist())
            
            total_duration_s = time_vals[-1] - time_vals[0] if len(time_vals) > 1 else 0
            total_dist_km = np.sum(step_dists) / 1000.0
            
            # Bounding box
            min_lon, max_lon = track['lon'].min(), track['lon'].max()
            min_lat, max_lat = track['lat'].min(), track['lat'].max()
            
            # Road network connectivity
            deg_out = np.bincount(arcs['u'].values, minlength=n_nodes)
            deg_in = np.bincount(arcs['v'].values, minlength=n_nodes)
            isolated_nodes = np.sum((deg_out == 0) & (deg_in == 0))
            
            trip_records.append({
                'trip_id': t,
                'partition': part_name,
                'track_points': n_track,
                'nodes_count': n_nodes,
                'arcs_count': n_arcs,
                'route_arcs': n_route,
                'missing_values': int(track_missing),
                'duration_sec': float(total_duration_s),
                'distance_km': float(total_dist_km),
                'mean_speed_kmh': float(np.mean(speeds) * 3.6) if len(speeds) > 0 else 0,
                'max_speed_kmh': float(np.max(speeds) * 3.6) if len(speeds) > 0 else 0,
                'mean_dt_sec': float(np.mean(dt)) if len(dt) > 0 else 0,
                'pct_1s_dt': float(np.mean(dt == 1.0) * 100) if len(dt) > 0 else 0,
                'isolated_nodes': int(isolated_nodes),
                'min_lon': float(min_lon), 'max_lon': float(max_lon),
                'min_lat': float(min_lat), 'max_lat': float(max_lat)
            })

    df_inventory = pd.DataFrame(trip_records)
    df_inventory.to_csv(os.path.join(ARTIFACTS_DIR, 'metrics', 'trip_inventory.csv'), index=False)
    print("Saved trip_inventory.csv")
    
    # Compute aggregate stats
    sampling_intervals = np.array(all_sampling_intervals)
    step_distances = np.array(all_step_distances)
    speeds_arr = np.array(all_speeds)
    
    summary_stats = {
        'total_trips': len(df_inventory),
        'train_trips': int((df_inventory['partition'] == 'train').sum()),
        'student_test_trips': int((df_inventory['partition'] == 'student_test').sum()),
        'total_track_points': int(df_inventory['track_points'].sum()),
        'train_track_points': int(df_inventory[df_inventory['partition'] == 'train']['track_points'].sum()),
        'student_test_track_points': int(df_inventory[df_inventory['partition'] == 'student_test']['track_points'].sum()),
        'total_route_arcs': int(df_inventory['route_arcs'].sum()),
        'missing_values_total': int(df_inventory['missing_values'].sum()),
        'dt_stats': {
            'mean': float(np.mean(sampling_intervals)),
            'median': float(np.median(sampling_intervals)),
            'pct_exactly_1s': float(np.mean(sampling_intervals == 1.0) * 100),
            'max': float(np.max(sampling_intervals)),
            'p99': float(np.percentile(sampling_intervals, 99))
        },
        'step_dist_stats': {
            'mean_m': float(np.mean(step_distances)),
            'median_m': float(np.median(step_distances)),
            'max_m': float(np.max(step_distances)),
            'p95_m': float(np.percentile(step_distances, 95))
        },
        'speed_stats': {
            'mean_kmh': float(np.mean(speeds_arr) * 3.6),
            'median_kmh': float(np.median(speeds_arr) * 3.6),
            'p95_kmh': float(np.percentile(speeds_arr, 95) * 3.6),
            'max_kmh': float(np.max(speeds_arr) * 3.6)
        },
        'trip_length_stats': {
            'mean_km': float(df_inventory['distance_km'].mean()),
            'median_km': float(df_inventory['distance_km'].median()),
            'min_km': float(df_inventory['distance_km'].min()),
            'max_km': float(df_inventory['distance_km'].max())
        }
    }
    
    with open(os.path.join(ARTIFACTS_DIR, 'metrics', 'eda_summary.json'), 'w') as f:
        json.dump(summary_stats, f, indent=4)
    print("Saved eda_summary.json")
    
    # Generate EDA Figures
    # 1. Distribution of Sampling Intervals and Speeds
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # dt distribution
    sns.histplot(sampling_intervals[sampling_intervals <= 10], bins=20, ax=axes[0, 0], color='#2b5c8f', kde=False)
    axes[0, 0].set_title('GPS Sampling Interval Distribution (dt <= 10s)', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Sampling Interval $\Delta t$ (seconds)')
    axes[0, 0].set_ylabel('Observation Count')
    axes[0, 0].axvline(1.0, color='red', linestyle='--', label=f'Nominal 1s ({summary_stats["dt_stats"]["pct_exactly_1s"]:.1f}%)')
    axes[0, 0].legend()
    
    # Speeds distribution
    sns.histplot(speeds_arr[speeds_arr <= 50] * 3.6, bins=40, ax=axes[0, 1], color='#e67e22', kde=True)
    axes[0, 1].set_title('Vehicle Speed Distribution (km/h)', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Speed (km/h)')
    axes[0, 1].set_ylabel('Observation Count')
    
    # Track points per trip by partition
    sns.boxplot(data=df_inventory, x='partition', y='track_points', ax=axes[1, 0], palette=['#3498db', '#2ecc71'])
    axes[1, 0].set_title('GPS Observations per Trip by Partition', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Dataset Partition')
    axes[1, 0].set_ylabel('Track Points per Trip')
    
    # Trip Distance vs Duration
    sns.scatterplot(data=df_inventory, x='duration_sec', y='distance_km', hue='partition', ax=axes[1, 1], palette=['#3498db', '#e74c3c'], s=60)
    axes[1, 1].set_title('Trip Distance vs Duration', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Duration (seconds)')
    axes[1, 1].set_ylabel('Trajectory Distance (km)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'plots', 'eda_distributions.png'), dpi=300)
    plt.close()
    print("Saved eda_distributions.png")
    
    # 2. Road Network Size vs Trajectory Extent
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df_inventory, x='nodes_count', y='arcs_count', hue='partition', size='track_points', sizes=(30, 200), palette=['#2980b9', '#27ae60'], alpha=0.8)
    plt.title('Road Network Graph Complexity (Nodes vs Arcs)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Road Network Nodes $|V|$')
    plt.ylabel('Number of Road Network Arcs $|E|$')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'plots', 'eda_road_network_complexity.png'), dpi=300)
    plt.close()
    print("Saved eda_road_network_complexity.png")

    # 3. Create Representative Folium Interactive Map for Trip 00000001
    generate_representative_folium_map('00000001', TRAIN_DIR, os.path.join(ARTIFACTS_DIR, 'maps', 'eda_trip_00000001_map.html'))
    print("=== EDA Complete! ===")

def generate_representative_folium_map(trip_id, base_dir, out_html):
    trip_path = os.path.join(base_dir, trip_id)
    track = pd.read_csv(os.path.join(trip_path, f"{trip_id}.track"), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    nodes = pd.read_csv(os.path.join(trip_path, f"{trip_id}.nodes"), sep=r'\s+', header=None, names=['lon', 'lat'])
    arcs = pd.read_csv(os.path.join(trip_path, f"{trip_id}.arcs"), sep=r'\s+', header=None, names=['u', 'v'])
    route = pd.read_csv(os.path.join(trip_path, f"{trip_id}.route"), header=None, names=['arc_id'])
    
    center_lat = track['lat'].mean()
    center_lon = track['lon'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='cartodbpositron')
    
    # Reference route polyline (Green)
    route_arcs = route['arc_id'].values
    u_nodes = arcs.loc[route_arcs, 'u'].values
    v_nodes = arcs.loc[route_arcs, 'v'].values
    
    # Build continuous route line coordinates
    route_coords = []
    for u, v in zip(u_nodes, v_nodes):
        u_pt = [nodes.loc[u, 'lat'], nodes.loc[u, 'lon']]
        v_pt = [nodes.loc[v, 'lat'], nodes.loc[v, 'lon']]
        if not route_coords:
            route_coords.append(u_pt)
        route_coords.append(v_pt)
        
    folium.PolyLine(
        locations=route_coords,
        color='#2ecc71',
        weight=5,
        opacity=0.85,
        tooltip='Reference Route (Ground Truth)'
    ).add_to(m)
    
    # Raw GPS points (Blue circles) - sample every 2nd point for crisp rendering
    for idx, row in track.iloc[::2].iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=3,
            color='#2980b9',
            fill=True,
            fill_color='#3498db',
            fill_opacity=0.7,
            popup=f"GPS Point #{idx}<br>Time: {row['time']:.0f}s"
        ).add_to(m)
        
    folium.LayerControl().add_to(m)
    m.save(out_html)
    print(f"Saved Folium map to {out_html}")

if __name__ == '__main__':
    run_eda()

