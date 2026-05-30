# Challenge 5 — Unsupervised Learning: Clustering EPA Air Quality Data

**Course:** Machine Learning — Universidad Distrital Francisco José de Caldas  
**Group:** 2 — Environment & Air Quality  
**Authors:** Angelo Ibañez · David Ariza · Cristhian Truque  

---

## Overview

Comparative study of three clustering algorithms applied to **144,815 EPA county-day records** from the U.S. Air Quality System (AQS) for 2025. All algorithms run on the same preprocessed feature matrix built in Challenge 2.

| Algorithm | Strategy | Best config | Silhouette |
|---|---|---|---|
| K-Means | MiniBatchKMeans (full 144k) | k=2 | 0.2313 ± 0.025 |
| DBSCAN | PCA-10 + 30k subsample + kNN propagation | ε=1.5, min_samples=20 | 0.3774 |
| Hierarchical | Stratified 10k subsample | average linkage, k=2 | 0.6007 |

Key finding: the data has a **unimodal distribution with a heavy tail** — K-Means captures the global pollution gradient, DBSCAN identifies isolated spikes (11.4% noise), and Hierarchical clustering isolates 173 extreme county-days consistent with wildfire events.

---

## Repository Structure

```
challenge-5_group2/
├── challenge5_group2.ipynb      # Main notebook (all experiments)
├── final_dataset_all_elements.csv  # Preprocessed input (from Challenge 2)
├── requirements.txt
├── README.md
├── CHECKLIST.md
├── artifacts/
│   ├── km_labels_full.npy       # K-Means labels for all 144k records (→ Challenge 6)
│   ├── db_labels_full.npy       # DBSCAN labels (full dataset via kNN propagation)
│   ├── cluster_labels_ch5.csv   # Both label arrays as CSV
│   ├── scaler_ch5.pkl           # Fitted StandardScaler
│   ├── pca_ch5.pkl              # Fitted PCA (90% variance)
│   ├── kmeans_profiles.csv      # Per-cluster mean feature values
│   ├── dbscan_profiles.csv
│   ├── hierarchical_profiles.csv
│   └── metrics_comparison.csv   # Summary metrics table
└── figures/
    ├── fig1_kmeans_elbow_silhouette.png
    ├── fig2_dbscan_knn_distance.png
    ├── fig3_dendrogram.png
    └── fig4_cluster_pca2d.png
```

---

## Setup and Reproduction

### 1. Clone and enter the repository
```bash
git clone https://github.com/<your-org>/challenge-5_group2.git
cd challenge-5_group2
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Place the input file
Copy `final_dataset_all_elements.csv` (produced by Challenge 2) into the project root.

### 5. Run the notebook
```bash
jupyter lab challenge5_group2.ipynb
```

Run all cells in order. Total estimated runtime on CPU: **25–40 minutes** (DBSCAN sweep and silhouette score computation are the bottlenecks).

---

## Reproducing Individual Experiments

| Section | Cell range | Runtime |
|---|---|---|
| Data loading & PCA | Cells 5–8 | < 1 min |
| K-Means elbow + silhouette sweep | Cells 9–10 | ~8–12 min |
| K-Means best model (3 seeds) | Cell 11 | ~3 min |
| DBSCAN k-NN plot | Cell 13 | ~2 min |
| DBSCAN parameter sweep (30k) | Cell 14 | ~5 min |
| DBSCAN final fit + kNN propagation | Cell 15 | ~3 min |
| Hierarchical dendrogram | Cell 18 | ~1 min |
| Hierarchical linkage comparison | Cell 19 | ~2 min |
| All figures | Cells 20–21 | ~2 min |
| Cluster profiles & metrics table | Cells 22–24 | < 1 min |
| Label export | Cell 26 | < 1 min |

---

## Random Seeds

All stochastic operations use seeds `{42, 7, 123}`. The primary seed for single-run outputs is **42**. Seeds are set via `set_seed(s)` which fixes `random`, `numpy.random`, and `sklearn` where applicable.

---

## Notes on Scale

With 144,815 records:
- `silhouette_score` is always called with `sample_size=5000` to avoid O(N²) memory
- DBSCAN is fitted on a 30k subsample (MemoryError on full dataset); labels are propagated to all 144k via `KNeighborsClassifier(n_neighbors=5)`
- Hierarchical clustering uses a stratified 10k subsample
- `n_jobs=1` is used throughout for stability on shared environments

---

## Output for Challenge 6

The file `artifacts/km_labels_full.npy` contains K-Means cluster labels for all 144,815 records and is loaded by the Challenge 6 notebook to colour latent space visualisations.
