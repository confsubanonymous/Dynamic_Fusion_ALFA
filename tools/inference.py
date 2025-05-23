import os
import torch
import time
import numpy as np
# import logging
import argparse
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

logger = logging.get_logger(__name__)

@torch.no_grad()
def benchmark_model(cfg, num_iters=100, warmup_iters=10):
    """
    Benchmark inference time and memory usage for the model.
    
    Args:
        cfg (CfgNode): Config node
        num_iters (int): Number of iterations to measure
        warmup_iters (int): Number of warmup iterations
    """
    # Set up environment
    du.init_distributed_training(cfg)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    
    # Setup logging
    logging.setup_logging(cfg.OUTPUT_DIR)
    # logger = logging.getLogger(__name__)
    #  logging.setup_logging(cfg.OUTPUT_DIR)
    
    logger.info("Benchmarking model inference...")
    
    # Build model
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    model.eval()
    model.cuda()

    # Add this after model initialization in your benchmark_model function
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Total parameters: {total_params:,}")

    
    
    # Get a sample batch for benchmarking
    test_loader = loader.construct_loader(cfg, "test")
    inputs, audio_frames = next(iter(test_loader))[0:2]
    
    # Move inputs to GPU
    if isinstance(inputs, (list,)):
        for i in range(len(inputs)):
            inputs[i] = inputs[i].cuda(non_blocking=True)
    else:
        inputs = inputs.cuda(non_blocking=True)
    
    audio_frames = audio_frames.cuda()
    
    # Record initial GPU memory allocation
    initial_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
    logger.info(f"Initial GPU memory allocation: {initial_memory:.2f} MB")
    
    # Warmup runs
    logger.info(f"Running {warmup_iters} warmup iterations...")
    for _ in range(warmup_iters):
        _ = model(inputs, audio_frames)
    torch.cuda.synchronize()
    
    # Reset memory stats after warmup
    torch.cuda.reset_peak_memory_stats()
    
    # Benchmark inference time
    logger.info(f"Benchmarking inference time over {num_iters} iterations...")
    
    # Per-iteration times
    iteration_times = []
    
    # Time the entire set of iterations
    total_start_time = time.time()
    
    for i in range(num_iters):
        # Time each individual iteration
        torch.cuda.synchronize()
        start_time = time.time()
        
        _ = model(inputs, audio_frames)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        iter_time = (end_time - start_time) * 1000  # ms
        iteration_times.append(iter_time)
        
        if i % 10 == 0:
            logger.info(f"Iteration {i}: {iter_time:.2f} ms")
    
    torch.cuda.synchronize()
    total_end_time = time.time()
    
    # Calculate metrics
    total_time = (total_end_time - total_start_time) * 1000  # ms
    avg_time = total_time / num_iters
    min_time = min(iteration_times)
    max_time = max(iteration_times)
    std_dev = np.std(iteration_times)
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    memory_increase = peak_memory - initial_memory
    
    # Log results
    logger.info("\n" + "="*50)
    logger.info("BENCHMARK RESULTS")
    logger.info("="*50)
    logger.info(f"Total time for {num_iters} iterations: {total_time:.2f} ms")
    logger.info(f"Average inference time: {avg_time:.2f} ms")
    logger.info(f"Min inference time: {min_time:.2f} ms")
    logger.info(f"Max inference time: {max_time:.2f} ms")
    logger.info(f"Standard deviation: {std_dev:.2f} ms")
    logger.info(f"Throughput: {1000 / avg_time:.2f} inferences/second")
    logger.info(f"Peak GPU memory usage: {peak_memory:.2f} MB")
    logger.info(f"Memory increase during inference: {memory_increase:.2f} MB")

    # Add these to your benchmark_results dictionary
    # benchmark_results["trainable_parameters"] = trainable_params
    # benchmark_results["total_parameters"] = total_params
    
    # Save results to file
    benchmark_results = {
        "total_time_ms": float(total_time),
        "avg_time_ms": float(avg_time),
        "min_time_ms": float(min_time),
        "max_time_ms": float(max_time),
        "std_dev_ms": float(std_dev),
        "throughput_fps": float(1000 / avg_time),
        "peak_memory_mb": float(peak_memory),
        "memory_increase_mb": float(memory_increase),
        "batch_size": inputs[0].shape[0] if isinstance(inputs, list) else inputs.shape[0],
        "model_name": cfg.MODEL.MODEL_NAME if hasattr(cfg.MODEL, "MODEL_NAME") else "unknown",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trainable_parameters": trainable_params,
        "total_parameters": total_params
    }
    
    # Save as CSV for easy import to Excel/plotting
    import csv
    csv_file = os.path.join(cfg.OUTPUT_DIR, 'benchmark_results.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(benchmark_results.keys())
        writer.writerow(benchmark_results.values())
    
    logger.info(f"Results saved to {csv_file}")
    
    # Optionally track GPU utilization if nvidia-smi is available
    try:
        import subprocess
        nvidia_smi = "nvidia-smi --query-gpu=utilization.gpu --format=csv"
        gpu_util = subprocess.check_output(nvidia_smi.split())
        logger.info(f"GPU Utilization:\n{gpu_util.decode('utf-8')}")
    except:
        logger.info("Could not get GPU utilization from nvidia-smi")
    
    return benchmark_results


def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(description='Model Inference Benchmark')
    parser.add_argument('--cfg', dest='cfg_file', help='Path to config file', required=True, type=str)
    parser.add_argument('--iters', help='Number of iterations to benchmark', type=int, default=100)
    parser.add_argument('--warmup', help='Number of warmup iterations', type=int, default=10)
    parser.add_argument('opts', help='See slowfast/config/defaults.py for all options', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def benchmark_main(cfg):
    
    # Run benchmark
    benchmark_model(cfg, num_iters=100, warmup_iters=20)