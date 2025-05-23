#!/usr/bin/env python3
import os
import json
import torch
import gc

from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.checkpoint as cu

from test_avgaze_net import test
from train_avgaze_net import train
from inference import benchmark_main


def main():
    print("Starting cross-validation training and testing...")

    args = parse_args()
    cfg = load_config(args)

    num_folds = 5
    for fold_idx in range(4, num_folds):
        print(f"\n=== Fold {fold_idx} ===")

        # Update paths for this fold
        train_csv_path = f"data/folds_output_aria/train_{fold_idx}.csv" #adjust as needed if ego4D
        test_csv_path = f"data/folds_output_aria/test_{fold_idx}.csv"
        output_dir_fold = os.path.join(cfg.OUTPUT_DIR, str(fold_idx))

        # ----- Training -----
        cfg_fold = cfg.clone()
        cfg_fold.OUTPUT_DIR = output_dir_fold
        cfg_fold.TRAIN.ENABLE = True
        cfg_fold.TEST.ENABLE = False
        cfg_fold.NUM_GPUS = 2
        cfg_fold.DATA.TRAIN_CSV = train_csv_path
        cfg_fold.DATA.TEST_CSV = test_csv_path
        cfg_fold = assert_and_infer_cfg(cfg_fold)

        cu.make_checkpoint_dir(cfg_fold.OUTPUT_DIR)
        print("Training config:")
        print(cfg_fold)
        launch_job(cfg=cfg_fold, init_method=args.init_method, func=train)

        # ----- Testing -----
        print("Starting testing...")
        cfg_fold.TEST.ENABLE = True
        cfg_fold.TRAIN.ENABLE = False
        cfg_fold.TEST.BATCH_SIZE = 24
        cfg_fold.NUM_GPUS = 1
        cfg_fold.CUDA_VISIBLE_DEVICES = 0
        cfg_fold = assert_and_infer_cfg(cfg_fold)

        # Load best checkpoint for testing
        best_epoch_path = os.path.join(cfg_fold.OUTPUT_DIR, "best_epoch.json")
        with open(best_epoch_path, "r") as f:
            best_epoch = json.load(f)["best_epoch"]
        checkpoint_filename = f"checkpoint_epoch_{best_epoch + 1:05d}.pyth"
        cfg_fold.TEST.CHECKPOINT_FILE_PATH = os.path.join(cfg_fold.OUTPUT_DIR, "checkpoints", checkpoint_filename)

        # Update output dir for test
        cfg_fold.OUTPUT_DIR = os.path.join(cfg_fold.OUTPUT_DIR, "test")
        cu.make_checkpoint_dir(cfg_fold.OUTPUT_DIR)

        launch_job(cfg=cfg_fold, init_method=args.init_method, func=test)
        # Optionally: benchmark_main(cfg=cfg_fold, init_method=args.init_method)

        # Free memory
        torch.cuda.empty_cache()
        gc.collect()

    print("\nAll folds completed.")


if __name__ == "__main__":
    main()
