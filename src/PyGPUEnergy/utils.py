import pandas as pd
from pathlib import Path
from typing import List, Optional, Any, Dict
import numpy as np

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

def calculate_energy_consumption(
    df: pd.DataFrame,
    records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Calculate energy consumption for each function execution period using trapezoidal integration.
    
    Args:
        df: DataFrame containing power and timestamp data
        records: List of function execution records with start and end times
        
    Returns:
        List of records with added energy consumption information
    """
    energy_records = []
    
    for record in records:
        # Get data points within the function execution period
        mask = (df['rel_time_ms'] >= record['start_offset']) & (df['rel_time_ms'] <= record['end_offset'])
        period_data = df[mask]
        
        if len(period_data) < 2:
            continue
            
        # Convert time to seconds for integration
        time_seconds = period_data['rel_time_ms'].values / 1000.0
        power_watts = period_data['power_draw[W]'].values
        
        # Calculate energy consumption using trapezoidal integration
        energy_consumption = np.trapezoid(power_watts, time_seconds)  # in Joules
        
        # Create energy record
        energy_record = {
            'name': record['name'],
            'type': record['type'],
            'start_time': record['start_offset'],
            'end_time': record['end_offset'],
            'duration_ms': record['end_offset'] - record['start_offset'],
            'energy_joules': energy_consumption,
            'avg_power_watts': energy_consumption / (record['end_offset'] - record['start_offset']) * 1000  # Convert to watts
        }
        
        energy_records.append(energy_record)
    
    return energy_records