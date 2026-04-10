import time
import tracemalloc
import torch
import pandas as pd



'''
Measures the training time and memory usage of a given training function.
'''
import time
import torch
import psutil
import os


def training_benchmark(train_fn):

    process = psutil.Process(os.getpid())

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    peak_ram = 0

    start_time = time.time()

    # --- RUN TRAINING ---
    result = train_fn()

    # --- FINAL RAM CHECK ---
    peak_ram = process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()

    peak_gpu = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        if torch.cuda.is_available() else 0
    )

    stats = {
        "training_time_s": round(end_time - start_time, 2),
        "peak_ram_mb": round(peak_ram, 2),
        "peak_gpu_mb": round(peak_gpu, 2),
    }

    return result, stats


'''
Measure inference time
'''
def inference_benchmark(inference_fn, X_test):
    start_time = time.time()

    preds = inference_fn(X_test)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()

    if hasattr(X_test, 'shape'):
        n_samples = X_test.shape[0]
    else:
        n_samples = len(X_test)

    total = round(end_time - start_time, 4)
    per_sample = round((end_time - start_time) / n_samples * 1000, 4)  # Time per sample in milliseconds

    stats = {
        "inference_time_s": total,
        "per_sample_ms": per_sample
    }

    return preds, stats



'''Counts the number of trainable parameters in a PyTorch model.
'''
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable



def build_comparison_table(results: dict):
    """
    results = {
        "SVM":    {"training_time_s": 2.1, "peak_ram_mb": 45, ...},
        "BERT":   {"training_time_s": 320, "peak_ram_mb": 980, ...},
        ...
    }
    """
    df = pd.DataFrame(results).T
    return df



