import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import numpy as np
from datetime import datetime
import os

def plot_gpu_metrics(
    log_file: Union[str, Path],
    records_file: Union[str, Path] = "gpu_records.json",
    t0_file: Union[str, Path] = "gpu_logs/t0.txt",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> None:
    """
    Plot GPU metrics including utilization, temperature, power, and clock speeds.
    Add background highlighting for different code regions and function calls.
    The x-axis is relative time (ms) from the start of the log (t0).
    """
    # Check if log file exists and has content
    if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
        print(f"Warning: Log file {log_file} is empty or does not exist. Skipping visualization.")
        return
    try:
        df = pd.read_csv(log_file)
        if df.empty:
            print(f"Warning: No data found in log file {log_file}. Skipping visualization.")
            return
        df.columns = ['timestamp', 'utilization_gpu[%]', 'pstate', 'temperature_gpu[C]', 
                     'clocks_current_sm[MHz]', 'power_draw[W]', 'power_draw_instant[W]']
        
        print("Original timestamps:")
        print(df['timestamp'].head())
        
        # Convert timestamps to milliseconds since epoch
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].astype(np.int64) // 10**6  # Convert to milliseconds
        
        print("Timestamps in milliseconds:")
        print(df['timestamp'].head())
        
        # Use first timestamp as t0
        t0_ms = df['timestamp'].iloc[0]
        print(f"Using first timestamp as t0: {t0_ms}")
            
        # Adjust time relative to t0 and ensure proper numeric format
        df['rel_time_ms'] = (df['timestamp'] - t0_ms).astype(np.int64)
        
        print("Relative timestamps:")
        print(df['rel_time_ms'].head())
        
        # Only filter out negative timestamps
        df = df[df['rel_time_ms'] >= 0]
        
        if df.empty:
            print("Warning: No valid data points after timestamp adjustment. Skipping visualization.")
            return
            
    except Exception as e:
        print(f"Error reading log file {log_file}: {str(e)}")
        return
    # Check if records file exists and has content
    if not os.path.exists(records_file) or os.path.getsize(records_file) == 0:
        print(f"Warning: Records file {records_file} is empty or does not exist. Plotting without region highlighting.")
        records = []
    else:
        try:
            with open(records_file, 'r') as f:
                records = json.load(f)
                # convert timestamp to milliseconds since epoch
                for record in records:
                    record['start_offset'] *= 1000
                    record['end_offset'] *= 1000
        except Exception as e:
            print(f"Error reading records file {records_file}: {str(e)}")
            records = []
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('GPU Metrics with Code Regions (Relative Time)')
    if records:
        colors = plt.cm.tab20(np.linspace(0, 1, len(records)))
        color_map = {record['name']: color for record, color in zip(records, colors)}
    else:
        color_map = {}
    # Plot GPU utilization with region backgrounds
    plot_metric_with_regions(ax1, df, 'utilization_gpu[%]', 'GPU Utilization (%)', 
                           records, color_map)
    plot_metric_with_regions(ax2, df, 'temperature_gpu[C]', 'Temperature (°C)', 
                           records, color_map)
    plot_metric_with_regions(ax3, df, ['power_draw[W]', 'power_draw_instant[W]'], 
                           'Power (W)', records, color_map,
                           labels=['Average Power', 'Instant Power'])
    plot_metric_with_regions(ax4, df, 'clocks_current_sm[MHz]', 'Clock Speed (MHz)', 
                           records, color_map)
    if records:
        handles = [plt.Rectangle((0,0), 1, 1, color=color, alpha=0.3) 
                  for color in color_map.values()]
        labels = [f"{r['name']} ({'function' if r['type'] == 'function' else 'region'})" for r in records]
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5))
    plt.tight_layout()
    if save_path:
        try:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {str(e)}")
    if show:
        plt.show()
    else:
        plt.close()

def plot_metric_with_regions(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: Union[str, List[str]],
    ylabel: str,
    records: List[Dict[str, Any]],
    color_map: Dict[str, np.ndarray],
    labels: Optional[List[str]] = None
) -> None:
    if isinstance(metric, list):
        for m, label in zip(metric, labels or metric):
            ax.plot(df['rel_time_ms'], df[m], label=label)
    else:
        ax.plot(df['rel_time_ms'], df[metric])
    for r in records:
        color = color_map.get(r['name'], None)
        if color is not None:
            ax.axvspan(r['start_offset'], r['end_offset'], alpha=0.3, color=color)
    ax.set_xlabel('Relative Time (ms)')
    ax.set_ylabel(ylabel)
    ax.grid(True)
    if isinstance(metric, list):
        ax.legend() 