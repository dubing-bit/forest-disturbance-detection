"""Command line entry points for mapping, window training and year evaluation."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import torch

from .evaluation import evaluate_csv
from .inference import PredictConfig, predict_raster
from .model import ModelConfig, load_model
from .training import train_csv


def make_parser():
    parser = argparse.ArgumentParser(prog="forest-disturbance", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Map an annual multiband index raster")
    predict.add_argument("--input", required=True, type=Path)
    predict.add_argument("--output", required=True, type=Path)
    predict.add_argument("--checkpoint", required=True, type=Path)
    predict.add_argument("--start-year", required=True, type=int)
    predict.add_argument("--window-size", type=int, help="Must agree with checkpoint; inferred by default")
    predict.add_argument("--step", type=int, default=1)
    predict.add_argument("--batch-size", type=int, default=4096)
    predict.add_argument("--block-size", type=int, default=128)
    predict.add_argument("--scale-factor", type=float, help="Checkpoint metadata, otherwise 10000")
    predict.add_argument("--min-slope", type=float, default=-0.05)
    predict.add_argument("--min-intercept", type=float, default=0.2)
    predict.add_argument("--min-drop", type=float, default=-0.1)
    predict.add_argument("--padding", choices=["legacy", "edge", "linear"], default="legacy")
    predict.add_argument("--mask", type=Path)
    predict.add_argument("--overwrite", action="store_true")

    train = commands.add_parser("train", help="Train a binary classifier from CSV windows")
    train.add_argument("--csv", required=True, type=Path)
    train.add_argument("--checkpoint", required=True, type=Path)
    train.add_argument("--window-size", type=int, choices=[7, 9, 11], default=9)
    train.add_argument("--feature-columns", nargs="+")
    train.add_argument("--label-column", default="label")
    train.add_argument("--scale-factor", type=float, default=1.0)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--patience", type=int, default=20)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--no-balance", action="store_true")
    train.add_argument("--overwrite", action="store_true")

    for command in (predict, train):
        command.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
        command.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))

    evaluate = commands.add_parser("evaluate", help="Evaluate reference/prediction calendar years")
    evaluate.add_argument("--csv", required=True, type=Path)
    evaluate.add_argument("--reference-column", default="reference")
    evaluate.add_argument("--prediction-column", default="prediction")
    evaluate.add_argument("--tolerance", type=int, default=0)
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            result = evaluate_csv(args.csv, reference_column=args.reference_column,
                                  prediction_column=args.prediction_column, tolerance=args.tolerance)
        else:
            if args.threads < 1:
                raise ValueError("threads must be positive")
            torch.set_num_threads(args.threads)
            device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
            if args.command == "predict":
                model = load_model(args.checkpoint, device=device, window_size=args.window_size)
                scale = args.scale_factor
                if scale is None:
                    scale = model.metadata.get("scale_factor", 10000.0)
                config = PredictConfig(
                    window_size=model.config.window_size, step=args.step,
                    batch_size=args.batch_size, block_size=args.block_size,
                    start_year=args.start_year, scale_factor=scale,
                    min_slope=args.min_slope, min_intercept=args.min_intercept,
                    min_drop=args.min_drop, padding=args.padding, device=device,
                )

                def progress(done, total):
                    if done == 1 or done == total or done % max(1, total // 20) == 0:
                        print(f"Tiles {done}/{total}", file=sys.stderr)

                result = predict_raster(args.input, args.output, model, config,
                                        mask_path=args.mask, overwrite=args.overwrite, progress=progress)
                result["configuration"] = asdict(config)
            else:
                if args.csv.resolve() == args.checkpoint.resolve():
                    raise ValueError("checkpoint must differ from training CSV")
                if args.checkpoint.exists() and not args.overwrite:
                    raise FileExistsError("checkpoint exists; use --overwrite to replace it")
                trained = train_csv(
                    args.csv, args.checkpoint, config=ModelConfig(window_size=args.window_size),
                    feature_columns=args.feature_columns, label_column=args.label_column,
                    scale_factor=args.scale_factor, epochs=args.epochs, batch_size=args.batch_size,
                    learning_rate=args.learning_rate, validation_fraction=args.validation_fraction,
                    patience=args.patience, seed=args.seed, device=device,
                    balance_classes=not args.no_balance,
                )
                result = asdict(trained)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str, allow_nan=False))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
