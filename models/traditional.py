# models/traditional.py

### Library Imports and Setup ###
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from tqdm import tqdm
import numpy as np

### Main Function ###
import time

class TraditionalAnomalyDetector:
    def __init__(self, method="iforest"):
        if method == "iforest":
            self.model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        elif method == "lof":
            self.model = LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True)
        elif method == "ocsvm":
            self.model = OneClassSVM(kernel="rbf", gamma="auto")
        else:
            raise ValueError("Unsupported method: {}".format(method))
        
        self.method = method

    def fit(self, X):
        if self.method == "lof":
            if X.shape[0] > 20000:
                print(f"[WARN] Downsampling LOF input from {X.shape[0]} to 20,000 samples...")
                idx = np.random.choice(X.shape[0], 20000, replace=False)
                X = X[idx]
        
        print(f"[INFO] Fitting {self.method.upper()} model on {X.shape[0]} samples...")
        start = time.time()
        self.model.fit(X)
        print(f"[INFO] {self.method.upper()} fit completed in {time.time() - start:.2f}s")

    def score(self, X):
        if self.method in {"lof", "ocsvm", "iforest"}:
            print(f"[INFO] Scoring with {self.method.upper()} on {X.shape[0]} samples...")
            scores = []
            batch_size = 1000  # You can adjust this based on available memory

            for i in tqdm(range(0, X.shape[0], batch_size), desc="Scoring"):
                batch = X[i:i + batch_size]
                batch_scores = -self.model.decision_function(batch)
                scores.append(batch_scores)

            return np.concatenate(scores)