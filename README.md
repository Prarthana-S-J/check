# PathSnap: Distributed Map-Matching & Route Reconstruction

**UE24CS352A: Machine Learning Hackathon**  
*Department of Computer Science and Engineering, PES University RR Campus*

---

## 1. Challenge & Architecture Overview

GPS tracking in urban corridors regularly experiences multipath interference, satellite occlusion, sensor drift, and irregular sampling intervals, causing raw trajectories to stray from true roads. **PathSnap** is a distributed machine learning system that accurately reconstructs ground-truth vehicle trajectories by matching noisy GPS observations to a high-fidelity directed road network.

PathSnap is designed following the mandatory **Two-Laptop Deployment Architecture**:
* **Laptop 1 (Model and API Server):** Hosts the trained classifier pipeline, spatial indexing network (`cKDTree`), Hidden Markov Model (HMM) Viterbi decoder, and server-side ground-truth evaluation logic via a high-performance **FastAPI** backend (`POST /api/evaluate/{trip_id}`).
* **Laptop 2 (Evaluator UI Client):** An interactive web dashboard allowing evaluators to select any of the 10 reserved evaluation trips, trigger real-time inference, inspect live CMD terminal response logs, and visualize the reconstructed trajectory on an interactive **Folium** map.

```
       [ Laptop 2: Evaluator UI Client ]
                 │
                 │  HTTP POST /api/evaluate/{trip_id}
                 ▼
       [ Laptop 1: FastAPI Inference Server ]
                 │
                 ├── 1. Spatial Candidate Retrieval (cKDTree Segment Index, R=50m, Top-K=8)
                 ├── 2. Kinematic & Spatial Feature Extraction (14 engineered features)
                 ├── 3. Tuned XGBoost Candidate Scoring (P(Match | Features))
                 ├── 4. HMM Viterbi Sequential Decoding (Network Topology & Transitions)
                 └── 5. Ground-Truth Accuracy Calculation & Interactive Folium Map Generation
```

---

## 2. Repository Organization

```
hackathon 2/
├── api/
│   └── app.py                     # FastAPI backend server with evaluation & map endpoints
├── ui/
│   ├── index.html                 # Evaluator UI dashboard
│   ├── app.js                     # Frontend API client logic & console logger
│   └── style.css                  # Modern dark-mode UI styling
├── src/
│   ├── data_processor.py          # Metric projection, raw file parsing & network indexing
│   ├── feature_engineering.py     # Kinematic, topological & spatial candidate feature extraction
│   ├── models.py                  # Model definitions across 5 distinct algorithm families
│   ├── train_and_evaluate.py      # 5-fold GroupKFold cross-validation & evaluation scripts
│   ├── bayesian_optimization.py   # Optuna Bayesian hyperparameter search (TPE)
│   ├── hmm_decoder.py             # Hidden Markov Model & Viterbi path reconstruction
│   ├── folium_visualizer.py       # Multi-layer interactive Folium HTML map generator
│   └── generate_report.py         # Consolidated 10-page PDF report builder
├── artifacts/
│   ├── models/                    # Saved model artifacts (.joblib)
│   ├── metrics/                   # JSON & CSV experiment logs, CV scores, tuning tables
│   ├── plots/                     # High-resolution diagnostics, confusion matrices, loss curves
│   └── maps/                      # Generated interactive Folium HTML maps
├── PathSnap_Technical_Report.pdf  # Comprehensive, publication-quality 10-page PDF report
└── README.md                      # Project documentation and execution guide
```

---

## 3. Experimental Results Summary

### 3.1 Model Comparison Across 5 Algorithm Families

All models were evaluated using **5-fold GroupKFold cross-validation** across the training partition (grouped strictly by `trip_id` to eliminate data leakage) and tested on the 10 student-testing trips:

| Model | Algorithm Family | Key Configuration | 5-Fold CV Macro F1 | Test Top-1 Acc | Route Continuity | Test Macro F1 | Log Loss | Inference Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear & Probabilistic | C=1.0, max_iter=1000 | 0.8155 ± 0.022 | 65.65% | 89.47% | 0.8184 | 0.2139 | 44.2 ms |
| **Random Forest** | Ensemble (Bagging) | 100 trees, depth=12 | 0.8225 ± 0.018 | 66.54% | 89.73% | 0.8302 | 0.1980 | 411.7 ms |
| **LightGBM** | Gradient Boosting | 100 trees, lr=0.1, depth=8 | 0.8245 ± 0.023 | 66.11% | 89.51% | 0.8279 | 0.1990 | 339.3 ms |
| **XGBoost (Baseline)** | Extreme Gradient Boosting | 100 trees, lr=0.1, depth=6 | **0.8266 ± 0.023** | **66.32%** | **89.89%** | **0.8279** | **0.2008** | **83.3 ms** |
| **Neural Network (MLP)** | Deep Neural Network | (64, 32) ReLU, Adam | 0.8192 ± 0.022 | 65.04% | 89.43% | 0.8214 | 0.2111 | 81.4 ms |

### 3.2 Bayesian Hyperparameter Optimization (Optuna)

XGBoost was chosen for hyperparameter optimization based on top cross-validation Macro F1, superior route continuity, and low latency. 25 trials were conducted using Optuna's Tree-structured Parzen Estimator (TPE):

* **Best Hyperparameters:** `n_estimators=160`, `max_depth=5`, `learning_rate=0.0366`, `subsample=0.9`, `colsample_bytree=0.7`, `min_child_weight=2`, `gamma=0.5`.
* **Tuned vs Untuned Performance:**
  * **CV Macro F1:** 0.8266 &rarr; **0.8269**
  * **Test Macro F1:** 0.8279 &rarr; **0.8300**
  * **Test Top-1 Accuracy:** 66.32% &rarr; **66.52%**
  * **Test Log Loss:** 0.2008 &rarr; **0.1971** (lower is better)

### 3.3 HMM Viterbi Sequential Reconstruction Impact

The sequential decoder resolves topological discontinuities, preventing unrealistic cross-street jumps:
* **Route Continuity:** Increased from **89.82%** (classifier-only) to **96.95%** overall (reaching **99.85%** on Trip `00000007` and **99.08%** on Trip `00000000`).
* **Pointwise Top-1 Accuracy:** Consistent gains across evaluation trips (+4.00% on Trip `00000009`, +2.29% on Trip `00000052`, +2.35% on Trip `00000065`, +2.90% on Trip `00000076`).

---

## 4. Installation & Execution Guide

### 4.1 Prerequisites & Installation

```bash
# Clone or navigate to the project directory
cd "hackathon 2"

# Install dependencies
pip install numpy pandas scikit-learn lightgbm xgboost optuna folium fastapi uvicorn reportlab matplotlib seaborn
```

### 4.2 Step-by-Step Pipeline Execution

```bash
# Step 1: Run Exploratory Data Analysis & generate distributions
python src/eda.py

# Step 2: Extract candidate features for training and test partitions
python scripts/prepare_dataset.py

# Step 3: Train and evaluate all 5 models with 5-fold GroupKFold CV
python src/train_and_evaluate.py

# Step 4: Run Optuna Bayesian hyperparameter optimization
python src/bayesian_optimization.py

# Step 5: Evaluate HMM Viterbi route reconstruction on all test trips
python src/evaluate_hmm.py

# Step 6: Generate interactive Folium maps
python src/folium_visualizer.py

# Step 7: Build consolidated PDF Technical Report
python src/generate_report.py
```

---

## 5. Deployment Instructions (Two-Laptop Setup)

### Laptop 1 (API Server):
Start the FastAPI server on port 8001 (or 8000):
```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8001
```
The server will initialize the spatial indexing engine, load `artifacts/models/best_tuned_model.joblib`, and log all incoming inference requests to the terminal.

### Laptop 2 (Evaluator UI Client):
Open any web browser and navigate to:
```
http://<LAPTOP_1_IP>:8001/ui/index.html
```
*(For single-machine local demonstration, open `http://127.0.0.1:8001/ui/index.html`)*.

1. Select any of the 10 evaluation trips from the dropdown.
2. Click **"Evaluate Selected Trip"**.
3. View the real-time Top-1 accuracy, route continuity, latency, and interactive Folium trajectory map.
4. Verify the live inference log outputs on the Laptop 1 Command Prompt.

---

## 6. Academic Integrity & Reproducibility Statement

All models, feature engineering pipelines, Bayesian optimization trials, and sequential decoders in this repository were developed and trained independently on local hardware. No predictions or evaluation outcomes were hardcoded. All results reported in `PathSnap_Technical_Report.pdf` are directly reproducible from the provided source code and raw data files.

