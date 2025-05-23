#!/usr/bin/env python3

import numpy as np
import os
import torch
import torchvision.transforms as transforms

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.utils.metrics as metrics
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import TestGazeMeter
from slowfast.utils.utils import frame_softmax
# from slowfast.visualization.visualization import vis_inference, vis_video_forecasting, vis_av_st_fusion

logger = logging.get_logger(__name__)

import torch

def corrupt_random_audio_frames(audio_frames, corruption_type="gaussian", severity=1, corruption_prob=0.5):
    """
    Apply corruption to randomly selected audio frames.

    Args:
        audio_frames (Tensor): Tensor of shape [B, C, T, H, W] (usually C=1 for audio spectrograms).
        corruption_type (str): Type of corruption to apply ("gaussian", "zeros", "invert", "dropout", "amplitude").
        severity (int or float): Severity level of corruption (1 to 5 recommended).
        corruption_prob (float): Probability of corrupting each frame (between 0 and 1).

    Returns:
        Tensor: Corrupted audio frames of the same shape as input.
    """
    B, C, T, H, W = audio_frames.shape
    corrupted = audio_frames.clone()

    # Create a boolean mask for which frames to corrupt (shape: T,)
    frames_to_corrupt = torch.rand(T) < corruption_prob

    # Ensure at least one frame is corrupted if corruption_prob > 0
    if corruption_prob > 0 and not frames_to_corrupt.any():
        random_frame = torch.randint(0, T, (1,))
        frames_to_corrupt[random_frame] = True

    for t in range(T):
        if frames_to_corrupt[t]:
            if corruption_type == "gaussian":
                noise = torch.randn(B, C, H, W, device=audio_frames.device) * (severity * 0.1)
                corrupted[:, :, t, :, :] += noise
                corrupted[:, :, t, :, :] = torch.clamp(corrupted[:, :, t, :, :], 0, 1)

            elif corruption_type == "zeros":
                corrupted[:, :, t, :, :] = 0

            elif corruption_type == "invert":
                corrupted[:, :, t, :, :] = 1.0 - corrupted[:, :, t, :, :]

            elif corruption_type == "dropout":
                dropout_mask = (torch.rand(B, C, H, W, device=audio_frames.device) > severity * 0.1).float()
                corrupted[:, :, t, :, :] *= dropout_mask

            elif corruption_type == "amplitude":
                scale = max(0.0, 1.0 - (severity * 0.15))
                corrupted[:, :, t, :, :] *= scale

            else:
                raise ValueError(f"Unsupported corruption_type: {corruption_type}")

    return corrupted

def corrupt_random_frames(frames, corruption_type="gaussian", severity=1, corruption_prob=0.5):
    """
    Apply corruption to randomly selected video frames
    frames: tensor of shape [C, T, H, W]
    corruption_prob: probability of corrupting each frame
    """
    corrupted = frames.clone()
    
    # Create a mask for which frames to corrupt
    T = corrupted.size(1)
    frames_to_corrupt = torch.rand(T) < corruption_prob
    
    for t in range(T):
        if frames_to_corrupt[t]:
            if corruption_type == "gaussian":
                # Add Gaussian noise
                noise = torch.randn_like(corrupted[:, t]) * (severity * 0.1)
                corrupted[:, t] = corrupted[:, t] + noise
                corrupted[:, t] = torch.clamp(corrupted[:, t], 0, 1)
            
            elif corruption_type == "blur":
                # Apply Gaussian blur
                from torchvision.transforms import GaussianBlur
                blur = GaussianBlur(kernel_size=2*severity+1)
                corrupted[:, t] = blur(corrupted[:, t])
            
            elif corruption_type == "dropout":
                # Random pixel dropout
                mask = torch.rand_like(corrupted[:, t]) > (severity * 0.1)
                corrupted[:, t] = corrupted[:, t] * mask
    
    return corrupted

@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestGazeMeter): testing meters to log and ensemble the testing results.
        cfg (CfgNode): configs. Details can be found in slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, audio_frames, labels, labels_hm, video_idx, meta) in enumerate(test_loader):
        # severity = 5
        # inputs=[corrupt_random_frames(pathway, "gaussian", severity, 0.3) for pathway in inputs]

        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            audio_frames = audio_frames.cuda(non_blocking=True)

            # audio_frames = np.array(audio_frames).cuda()
            labels = labels.cuda()
            labels_hm = labels_hm.cuda()
            video_idx = video_idx.cuda()
        test_meter.data_toc()

        # Perform the forward pass.
        preds = model(inputs, audio_frames)

        preds = frame_softmax(preds, temperature=2)  # KLDiv

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, labels_hm, video_idx = du.all_gather([preds, labels, labels_hm, video_idx])

        # Compute the metrics.
        if cfg.NUM_GPUS:  # compute on cpu
            preds = preds.cpu()
            labels = labels.cpu()
            labels_hm = labels_hm.cpu()
            video_idx = video_idx.cpu()

        preds_rescale = preds.detach().view(preds.size()[:-2] + (preds.size(-1) * preds.size(-2),))
        preds_rescale = (preds_rescale - preds_rescale.min(dim=-1, keepdim=True)[0]) / (preds_rescale.max(dim=-1, keepdim=True)[0] - preds_rescale.min(dim=-1, keepdim=True)[0] + 1e-6)
        preds_rescale = preds_rescale.view(preds.size())
        f1, recall, precision, threshold = metrics.adaptive_f1(preds_rescale, labels_hm, labels, dataset=cfg.TEST.DATASET)

        test_meter.iter_toc()

        # Update and log stats.
        test_meter.update_stats(f1, recall, precision, preds=preds_rescale, labels_hm=labels_hm, labels=labels)  # If running  on CPU (cfg.NUM_GPUS == 0), use 1 to represent 1 CPU.
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    test_meter.finalize_metrics()
    return test_meter


def test(cfg):
    """
    Perform testing on the video model.
    Args:
        cfg (CfgNode): configs. Details can be found in slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert (test_loader.dataset.num_videos % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS) == 0)
    # Create meters for multi-view testing.
    test_meter = TestGazeMeter(
        num_videos=test_loader.dataset.num_videos // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        num_clips=cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        num_cls=cfg.MODEL.NUM_CLASSES,
        overall_iters=len(test_loader),
        dataset=cfg.TEST.DATASET
    )

    writer = None  # Forbid use tensorboard for test

    # Perform testing on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)

    logger.info("Testing finished!")
