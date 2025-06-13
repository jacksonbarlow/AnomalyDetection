import torch
import os

CONFIG = {
    # ========== PATHS ==========
    "TRAJECTORY_PATH": "trajectory_data/I-80-Emeryville-CA/0400pm-0415pm/trajectories-0400-0415.csv",
    "SEQUENCE_FILE": "cache/Sequences.npz",
    "LANE_CHANGE_MASK_PATH": "cache/lane_change_mask.npy",
    "CONTEXT_CACHE_PATH": "cache/context_cache.pkl",
    "SCALED_FEATURES_PATH": "cache/scaled_features.parquet",
    "SEQ_PATH": "cache/",
    "SCALER_PATH": "cache/scaler.pkl",

    # Checkpoints & Models
    "MODEL_PATH": "checkpoints/autoencoder.pt",
    "TRADITIONAL_MODEL_PATH": "checkpoints/autoencoder_epoch50.pt",
    "CHECKPOINT_DIR": "checkpoints/",

    # Plot Output Directories
    "PLOTS_DIR": "plots/",
    "PREPROCESSING_PLOTS_DIR": "plots/preprocessing/",
    "CONTEXT_PLOTS_DIR": "plots/context/",
    "EVALUATION_PLOTS_DIR": "plots/evaluation/",

    # ========== FEATURE SETTINGS ==========
    "FEATURES": ['v_Vel', 'v_Acc', 'Space_Headway', 'Time_Headway'],
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    # ========== MODEL HYPERPARAMETERS ==========
    "WINDOW_SIZE": 50,
    "STEP_SIZE": 1,
    "BATCH_SIZE": 64,
    "HIDDEN_DIM": 256,
    "LATENT_DIM": 128,
    "NUM_LAYERS": 4,
    "EPOCHS": 50,
    "LEARNING_RATE": 1e-3,
    "SUBSET_FRACTION": 0.1,  # 10% of total sequences; adjust as needed
    "CONTEXT_DIM": 24,
    "STATIC_DIM": 3,
    "EGO_DIM": 4,

    # ========== HYBRID SCORING WEIGHTS ==========
    "HYBRID_WEIGHTS": {
        "reconstruction": 0.3,
        "mahalanobis": 0.3,
        "isolation_forest": 0.2,
        "lof": 0.2
    },

    # ========== PLOTTING OPTIONS ==========
    "PLOT_SMOOTHING": True,
    "PLOT_SMOOTHING_FIRST_ONLY": True,
}

# ========== DERIVED PATHS ==========
CONFIG["SEQUENCE_DIR"] = os.path.dirname(CONFIG["SEQUENCE_FILE"])
CONFIG["CONTEXT_DIR"] = os.path.dirname(CONFIG["CONTEXT_CACHE_PATH"])
CONFIG["PLOTS_EVAL_DIR"] = CONFIG["EVALUATION_PLOTS_DIR"]
