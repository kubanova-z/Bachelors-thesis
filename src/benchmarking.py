import time
import tracemalloc
from tokenizers import Tokenizer
import torch
import pandas as pd
from memory_profiler import memory_usage



'''
Measures the training time and memory usage of a given training function.
'''
import time
import torch
import psutil
import os


def predict_fn(model, texts, tokenizer, device):
    model.eval()

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    return preds


def training_benchmark(train_fn):

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    # --- RUN TRAINING WITH MEMORY TRACKING ---
    mem_usage, result = memory_usage(
        (train_fn, ),
        retval=True,
        interval=0.1
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()


    peak_ram = max(mem_usage) / 1024 # GB

    peak_gpu = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        if torch.cuda.is_available() else 0
    )

    stats = {
        "training_time_s": round(end_time - start_time, 2),
        "peak_ram_gb": round(peak_ram, 2),
        "peak_gpu_gb": round(peak_gpu, 2),
    }

    return result, stats


'''
Measure inference time
'''
def inference_benchmark(inference_fn, X_test, n_latency_runs=50):
    _ = inference_fn(X_test[:10] if hasattr(X_test, 'shape') else X_test[:10])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    preds = inference_fn(X_test)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    if hasattr(X_test, 'shape'):
        n_samples = X_test.shape[0]
    else:
        n_samples = len(X_test)

    total = round(end_time - start_time, 4)
    per_sample = round((end_time - start_time) / n_samples * 1000, 4)  # Time per sample in milliseconds

    sample = X_test[0:1] if hasattr(X_test, 'shape') else [X_test[0]]

    times = []
    for _ in range(n_latency_runs):
        start = time.perf_counter()

        _ = inference_fn(sample)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)

    latency_ms = (sum(times) / len(times)) * 1000

    stats = {
        "inference_time_s": total,
        "per_sample_ms": per_sample,
        "latency_ms": round(latency_ms, 4)
    }

    return preds, stats


def inference_latency_benchmark(inference_fn, X_test, n_runs=100):

    # Take one sample
    sample = X_test[0:1] if hasattr(X_test, 'shape') else [X_test[0]]

    times = []

    for _ in range(n_runs):
        start = time.perf_counter()

        _ = inference_fn(sample)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)

    avg_latency_ms = sum(times) / len(times) * 1000

    return {
        "latency_ms": round(avg_latency_ms, 4)
    }


'''Counts the number of trainable parameters in a PyTorch model.
'''
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters()) # total parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) # trainable parameters - updated during trining
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



