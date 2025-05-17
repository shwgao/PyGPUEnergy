import os
from pathlib import Path
from typing import List, Optional

def get_available_gpus() -> List[int]:
    """
    Get list of available GPU device IDs.
    
    Returns:
        List of available GPU device IDs
    """
    try:
        import torch
        return list(range(torch.cuda.device_count()))
    except ImportError:
        # Fallback to nvidia-smi if PyTorch is not available
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True
            )
            return [int(line.strip()) for line in result.stdout.splitlines()]
        except Exception:
            return []

def ensure_directory(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_latest_log_file(log_dir: str) -> Optional[Path]:
    """
    Get the most recent log file from the specified directory.
    
    Args:
        log_dir: Directory containing log files
        
    Returns:
        Path to the most recent log file, or None if no files found
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return None
        
    log_files = list(log_path.glob("gpu_metrics_*.csv"))
    if not log_files:
        return None
        
    return max(log_files, key=lambda x: x.stat().st_mtime) 