### main.py

import argparse
from config import CONFIG
from train import train_model
from evaluate import evaluate_model
from detect import detect_anomalies
from utils.lane_change import run_lane_change_analysis


def main():
    parser = argparse.ArgumentParser(description="Context-Aware Anomaly Detection CLI")

    parser.add_argument("--mode", choices=["train", "eval", "detect", "lane"], help="Mode to run")
    parser.add_argument("--model", choices=["autoencoder", "iforest", "lof"], default="autoencoder", help="Model type to use")
    parser.add_argument("--load", action="store_true", help="Load saved model if available")
    parser.add_argument("--window", type=int, default=CONFIG['WINDOW_SIZE'], help="Sequence window size")
    parser.add_argument("--latent_dim", type=int, default=CONFIG['LATENT_DIM'], help="Latent space dimensionality")
    parser.add_argument("--run_lane_analysis", action="store_true", help="Run lane change anomaly correlation")
    parser.add_argument("--build_data", action="store_true", help="Regenerate sequences from raw CSV")
    parser.add_argument("--epochs", type=int, default=CONFIG["EPOCHS"], help="Number of training epochs")
    parser.add_argument("--checkpoint", type=str, default=CONFIG["MODEL_PATH"], help="Custom checkpoint path to load/save")
    parser.add_argument("--trajectory_file", type=str, help="Path to raw trajectory CSV (overrides config)")

    args = parser.parse_args()

    # Override model path and optionally trajectory path
    CONFIG["MODEL_PATH"] = args.checkpoint
    if args.trajectory_file:
        CONFIG["TRAJECTORY_PATH"] = args.trajectory_file

    # Build data if specified
    if args.build_data:
        from data.build_data import build_and_save_sequences
        print("[INFO] Regenerating sequence data from raw CSV...")
        build_and_save_sequences()
        return  # Stop after building data

    # Require mode unless just building data
    if not args.mode:
        parser.error("You must specify --mode unless using --build_data")

    # Run the selected mode
    if args.mode == "train":
        train_model(
            model_type=args.model,
            load=args.load,
            window=args.window,
            latent_dim=args.latent_dim,
            epochs=args.epochs
        )

    elif args.mode == "eval":
        evaluate_model(model_type=args.model)

    elif args.mode == "detect":
        detect_anomalies(model_type=args.model)

    elif args.mode == "lane":
        run_lane_change_analysis()


if __name__ == "__main__":
    main()
