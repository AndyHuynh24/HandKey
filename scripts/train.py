#!/usr/bin/env python
"""Train unified gesture recognition model. Supports resume from checkpoints."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from tensorflow import keras

from handflow.data.loader import (
    check_processed_data_valid,
    load_processed_data,
)
from handflow.features import FeatureEngineer
from handflow.models import Trainer, build_model
from handflow.utils import load_config
from handflow.utils.logging import setup_logging, get_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train HandFlow unified gesture model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train from scratch
    python scripts/train.py --architecture tcn --epochs 100

    # Resume training from a checkpoint with lower learning rate
    python scripts/train.py --resume models/hand_action.keras --epochs 50 --lr 0.00001

    # Resume from a specific checkpoint
    python scripts/train.py --resume models/checkpoints/tcn_20260203_best.h5 --lr 0.0001 --epochs 100
        """
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        choices=["lstm", "gru", "cnn1d", "transformer", "tcn"],
        help="Model architecture (ignored if --resume is used)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=Path("config/config.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to model checkpoint to resume training from (.keras or .h5)",
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=None,
        dest="learning_rate",
        help="Starting learning rate (overrides config)",
    )
    # --- AkashTrainer sweep-friendly knobs (all optional; only override config when set) ---
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
        help="Override config training.batch_size",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Override config model.dropout (if your architecture uses it)",
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        default=None,
        dest="hidden_units",
        help="Override config model.hidden_units (if your architecture uses it)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        dest="num_layers",
        help="Override config model.num_layers (if your architecture uses it)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        help="Write trained model + results.json here (AkashTrainer mounts /output and pushes it to a GitHub branch)",
    )
    return parser.parse_args()


def _maybe_setattr(obj, name: str, value) -> bool:
    """Set obj.name = value only if value is not None AND obj already has the attribute.

    Returns True if set, False otherwise. Used to apply sweep CLI overrides without
    crashing when the active config schema doesn't define a given knob.
    """
    if value is None:
        return False
    if not hasattr(obj, name):
        return False
    setattr(obj, name, value)
    return True


def apply_feature_engineering(
    sequences: np.ndarray, feature_engineer: FeatureEngineer
) -> np.ndarray:
    """Apply feature engineering to all sequences."""
    enhanced = []
    for seq in sequences:
        enhanced.append(feature_engineer.transform(seq))
    return np.array(enhanced)


def load_training_data(
    config
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load processed data from cache for both hands and merge them.
    Uses flip canonicalization so both hands use the same model.

    Args:
        config: Configuration object

    Returns:
        Tuple of (x_train, x_val, y_train, y_val, actions)
    """
    processed_dir = Path("data/processed")
    
    # Paths for both hands
    right_train_path = processed_dir / "right_train.npz"
    right_val_path = processed_dir / "right_val.npz"
    left_train_path = processed_dir / "left_train.npz"
    left_val_path = processed_dir / "left_val.npz"

    actions = config.model.gestures

    # Check if all cached data files are valid
    all_paths = [right_train_path, right_val_path, left_train_path, left_val_path]
    all_valid = all(check_processed_data_valid(p, config) for p in all_paths)

    if not all_valid:
        raise FileNotFoundError(
            f"Processed data not found: {processed_dir}\n"
            "Run preprocessing first: python scripts/preprocess.py"
        )

    # Load data from both hands
    x_train_right, y_train_right, _ = load_processed_data(right_train_path)
    x_val_right, y_val_right, _ = load_processed_data(right_val_path)
    x_train_left, y_train_left, _ = load_processed_data(left_train_path)
    x_val_left, y_val_left, _ = load_processed_data(left_val_path)

    # Merge data from both hands
    x_train = np.concatenate([x_train_right, x_train_left], axis=0)
    y_train = np.concatenate([y_train_right, y_train_left], axis=0)
    x_val = np.concatenate([x_val_right, x_val_left], axis=0)
    y_val = np.concatenate([y_val_right, y_val_left], axis=0)

    return x_train, x_val, y_train, y_val, actions


def main() -> None:
    """Main training function."""
    args = parse_args()

    log_file = "logs/training.log"
    setup_logging(level="INFO", log_file=log_file)
    logger = get_logger("handflow.training")

    # Load configuration
    config = load_config(args.config)

    # Override config with CLI arguments
    if args.architecture and not args.resume:
        config.model.architecture = args.architecture
    if args.epochs:
        config.training.epochs = args.epochs
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    # Sweep-friendly overrides (only applied if the config schema defines them)
    _maybe_setattr(config.training, "batch_size", args.batch_size)
    _maybe_setattr(config.model, "dropout", args.dropout)
    _maybe_setattr(config.model, "hidden_units", args.hidden_units)
    _maybe_setattr(config.model, "num_layers", args.num_layers)

    # Check resume path exists
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.error(f"Resume model not found: {args.resume}")
            sys.exit(1)
        logger.info(f"Will resume training from: {args.resume}")

    output_path = Path(config.model.output_dir) / "hand_action.keras"

    logger.info(f"{'='*60}")
    if args.resume:
        logger.info(f"🔄 HandFlow Training - RESUME MODE")
    else:
        logger.info(f"🖐️ HandFlow Training - Unified Model (Both Hands)")
    logger.info(f"{'='*60}")
    if args.resume:
        logger.info(f"Resume from: {args.resume}")
    else:
        logger.info(f"Architecture: {config.model.architecture.upper()}")
    logger.info(f"Epochs: {config.training.epochs}")
    logger.info(f"Batch Size: {config.training.batch_size}")
    logger.info(f"Learning Rate: {config.training.learning_rate}")
    logger.info(f"Output: {output_path}")
    logger.info(f"{'='*60}")

    logger.info("\n Loading preprocessed data from cache (both hands)...")
    x_train, x_val, y_train, y_val, actions = load_training_data(config=config)
    logger.info("✅ Data is loaded and merged from both hands")
    logger.info(f"   Training: {len(x_train)} samples")
    logger.info(f"   Validation: {len(x_val)} samples")
    logger.info(f"   Data shape: {x_train.shape}")
    logger.info(f"   ✅ Input dimension for model: {x_train.shape[-1]} features")

    # Update input dim to config
    config.model.input_dim = x_train.shape[-1]

    if args.resume:
        logger.info(f"\n🔄 Loading model from {args.resume}...")
        model = keras.models.load_model(args.resume)
        logger.info("✅ Model loaded successfully")

        # Always recompile with fresh optimizer when resuming
        # (optimizer state from saved model is tied to old variable instances)
        lr = config.training.learning_rate
        logger.info(f"   Recompiling with learning rate: {lr}")
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.summary()
    else:
        logger.info(f"\n🔨 Building {config.model.architecture.upper()} model...")
        model = build_model(config)
        model.summary()

    experiment_name = "handflow-unified"
    trainer = Trainer(
        config=config,
        model=model,
        experiment_name=experiment_name,
        use_augmentation=config.augmentation.enabled,
    )

    # Train
    logger.info(f"\nStarting training...")
    history = trainer.train(x_train, y_train, x_val, y_val)

    # Evaluate
    logger.info("\n Final Evaluation:")
    metrics = trainer.evaluate(x_val, y_val)
    for name, value in metrics.items():
        logger.info(f"   {name}: {value:.4f}")

    # Save model
    logger.info(f"\n💾 Saving model to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(output_path)

    logger.info("\n✅ Training complete!")
    logger.info(f"   Best validation accuracy: {max(history.history['val_accuracy']):.4f}")

    # --- AkashTrainer integration ----------------------------------------------
    # When this script is run inside the AkashTrainer container (or whenever
    # --output-dir is set), write a results.json + copy the trained model into
    # that directory. The container packages anything in /output/ and pushes it
    # to a `trained-output/<timestamp>` GitHub branch; AkashTrainer's monitor
    # then reads results.json and populates the sweep leaderboard.
    output_dir = args.output_dir or os.environ.get("AKASH_OUTPUT_DIR") or "/output"
    try:
        _write_sweep_results(output_dir, args, config, history, metrics, output_path, logger)
    except Exception as e:
        # Never fail the training because of result-reporting issues.
        logger.warning(f"Could not write sweep results to {output_dir}: {e}")


def _write_sweep_results(output_dir, args, config, history, metrics, model_path, logger):
    """Write /output/results.json and copy the trained model so AkashTrainer can pick it up."""
    import json
    import os
    import shutil

    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError:
            logger.info(f"Skipping sweep results — {output_dir} is not writable")
            return

    h = history.history if hasattr(history, "history") else {}

    def _last(key, default=None):
        seq = h.get(key) or []
        return float(seq[-1]) if seq else default

    def _arr(key):
        return [float(x) for x in (h.get(key) or [])]

    results = {
        # Scalars — these show up as sortable columns on the sweep leaderboard.
        "val_accuracy": _last("val_accuracy"),
        "val_loss": _last("val_loss"),
        "train_accuracy": _last("accuracy"),
        "train_loss": _last("loss"),
        "best_val_accuracy": float(max(h["val_accuracy"])) if h.get("val_accuracy") else None,
        "epochs_trained": len(h.get("loss", [])),
        # Arrays — render as overlaid curves on the run detail page.
        "val_acc_curve": _arr("val_accuracy"),
        "val_loss_curve": _arr("val_loss"),
        "train_acc_curve": _arr("accuracy"),
        "train_loss_curve": _arr("loss"),
        # Echo hyperparams so they're stored alongside metrics (useful for the
        # parallel-coordinates plot if you sweep over them).
        "hyperparams": {
            "architecture": getattr(config.model, "architecture", None),
            "epochs": getattr(config.training, "epochs", None),
            "batch_size": getattr(config.training, "batch_size", None),
            "learning_rate": getattr(config.training, "learning_rate", None),
            "dropout": getattr(config.model, "dropout", None),
            "hidden_units": getattr(config.model, "hidden_units", None),
            "num_layers": getattr(config.model, "num_layers", None),
        },
        # Echo any final eval metrics the trainer returned (precision, f1, etc.)
        "final_eval": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"✅ Wrote sweep results → {results_path}")

    # Copy the trained model alongside so the output branch has both
    if model_path and os.path.exists(model_path):
        try:
            dest = os.path.join(output_dir, os.path.basename(str(model_path)))
            shutil.copy2(str(model_path), dest)
            logger.info(f"✅ Copied model → {dest}")
        except (OSError, shutil.SameFileError) as e:
            logger.warning(f"Could not copy model to {output_dir}: {e}")


if __name__ == "__main__":
    main()
