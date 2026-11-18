import os
import glob
import pandas as pd
import numpy as np

train_dir = r'Student_folder/Student folder/train'
test_dir = r'Student_folder/Student folder/student_test'

def inspect_partition(partition_dir, name):
    trips = sorted(os.listdir(partition_dir))
    print(f"=== Partition {name}: {len(trips)} trips ===")
    
    track_lens = []
    route_lens = []
    nodes_lens = []
    arcs_lens = []
    time_deltas = []
    
    for t in trips:
        t_path = os.path.join(partition_dir, t)
        track_f = os.path.join(t_path, f"{t}.track")
        nodes_f = os.path.join(t_path, f"{t}.nodes")
        arcs_f = os.path.join(t_path, f"{t}.arcs")
        route_f = os.path.join(t_path, f"{t}.route")
        
        # Fast line count
        with open(track_f) as f:
            t_len = sum(1 for _ in f)
        with open(nodes_f) as f:
            n_len = sum(1 for _ in f)
        with open(arcs_f) as f:
            a_len = sum(1 for _ in f)
        with open(route_f) as f:
            r_len = sum(1 for _ in f)
            
        track_lens.append(t_len)
        route_lens.append(r_len)
        nodes_lens.append(n_len)
        arcs_lens.append(a_len)
        
    print(f"Track points: min={min(track_lens)}, max={max(track_lens)}, mean={np.mean(track_lens):.1f}, total={sum(track_lens)}")
    print(f"Route arcs:   min={min(route_lens)}, max={max(route_lens)}, mean={np.mean(route_lens):.1f}, total={sum(route_lens)}")
    print(f"Nodes count:  min={min(nodes_lens)}, max={max(nodes_lens)}, mean={np.mean(nodes_lens):.1f}")
    print(f"Arcs count:   min={min(arcs_lens)}, max={max(arcs_lens)}, mean={np.mean(arcs_lens):.1f}")
    print(f"Are nodes/arcs identical across trips? Unique node counts: {len(set(nodes_lens))}, unique arc counts: {len(set(arcs_lens))}")

inspect_partition(train_dir, "TRAIN")
inspect_partition(test_dir, "STUDENT_TEST")

