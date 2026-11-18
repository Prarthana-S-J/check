import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict, deque

class HMMRouteDecoder:
    """
    Hidden Markov Model (HMM) with Viterbi decoding for map-matching
    and route reconstruction.
    
    States: Candidate road segments at each GPS observation.
    Emissions: Classifier-predicted candidate likelihoods.
    Transitions: Road-network topology and spatial routing distance consistency.
    """
    def __init__(self, arcs_df, nodes_df, beta=20.0, same_arc_prob=0.85, direct_prob=1.0, penalty_prob=1e-5):
        self.arcs_df = arcs_df
        self.nodes_df = nodes_df
        self.beta = beta
        self.same_arc_prob = same_arc_prob
        self.direct_prob = direct_prob
        self.penalty_prob = penalty_prob
        
        # Build adjacency graph
        self.u_arr = arcs_df['u'].values
        self.v_arr = arcs_df['v'].values
        self.num_arcs = len(arcs_df)
        
        # node -> list of outgoing arc_ids
        self.node_outgoing = defaultdict(list)
        for arc_id in range(self.num_arcs):
            self.node_outgoing[self.u_arr[arc_id]].append(arc_id)
            
    def get_transition_cost(self, a1, a2, step_dist, dt):
        """
        Calculates log transition probability from arc a1 to arc a2.
        """
        if a1 == a2:
            # Same arc
            prob = self.same_arc_prob
        elif self.v_arr[a1] == self.u_arr[a2]:
            # Direct network successor
            prob = self.direct_prob
        else:
            # Check 2-hop or 3-hop connection
            target_u = self.u_arr[a2]
            v1 = self.v_arr[a1]
            
            # Check 2-hop: is there an intermediate arc connecting v1 to target_u?
            two_hop = False
            for mid_arc in self.node_outgoing[v1]:
                if self.v_arr[mid_arc] == target_u:
                    two_hop = True
                    break
                    
            if two_hop:
                prob = 0.25
            else:
                # If large time gap (e.g. dt > 10s), vehicle could have traversed unobserved path
                if dt > 10.0:
                    prob = 0.05
                else:
                    prob = self.penalty_prob
                    
        return np.log(max(prob, 1e-9))

    def decode(self, candidates_per_point, step_distances=None, dts=None):
        """
        Performs Viterbi decoding over candidate road segments for each GPS observation.
        """
        T = len(candidates_per_point)
        if T == 0:
            return []
        if T == 1:
            best_cand = max(candidates_per_point[0], key=lambda c: c.get('prob', 0.5))
            return [best_cand['arc_id']]
            
        # Viterbi table: V[t, k]
        # Backpointers: BP[t, k]
        V = []
        BP = []
        
        # Initialization (t=0)
        cands_0 = candidates_per_point[0]
        K_0 = len(cands_0)
        
        # Normalized emissions
        emissions_0 = np.array([c.get('prob', 0.5) for c in cands_0], dtype=float)
        emissions_0 = np.maximum(emissions_0, 1e-6)
        emissions_0 /= np.sum(emissions_0)
        
        V_0 = np.log(emissions_0)
        V.append(V_0)
        BP.append(np.zeros(K_0, dtype=int))
        
        # Forward pass
        for t in range(1, T):
            cands_prev = candidates_per_point[t-1]
            cands_curr = candidates_per_point[t]
            K_prev = len(cands_prev)
            K_curr = len(cands_curr)
            
            step_d = step_distances[t-1] if step_distances is not None else 15.0
            dt = dts[t-1] if dts is not None else 1.0
            
            emissions_curr = np.array([c.get('prob', 0.5) for c in cands_curr], dtype=float)
            emissions_curr = np.maximum(emissions_curr, 1e-6)
            emissions_curr /= np.sum(emissions_curr)
            log_emissions = np.log(emissions_curr)
            
            V_t = np.zeros(K_curr)
            BP_t = np.zeros(K_curr, dtype=int)
            
            for j in range(K_curr):
                a_curr = cands_curr[j]['arc_id']
                
                # Compute transition from each previous state
                scores = np.zeros(K_prev)
                for i in range(K_prev):
                    a_prev = cands_prev[i]['arc_id']
                    log_trans = self.get_transition_cost(a_prev, a_curr, step_d, dt)
                    scores[i] = V[t-1][i] + log_trans
                    
                best_prev = np.argmax(scores)
                BP_t[j] = best_prev
                V_t[j] = scores[best_prev] + log_emissions[j]
                
            V.append(V_t)
            BP.append(BP_t)
            
        # Backtracking
        best_last_idx = np.argmax(V[-1])
        optimal_path = [candidates_per_point[-1][best_last_idx]['arc_id']]
        
        curr_bp = best_last_idx
        for t in range(T - 1, 0, -1):
            curr_bp = BP[t][curr_bp]
            optimal_path.append(candidates_per_point[t-1][curr_bp]['arc_id'])
            
        optimal_path.reverse()
        return optimal_path

