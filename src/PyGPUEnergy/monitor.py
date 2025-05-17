import os
import subprocess
import time
import signal
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd

class GPUMonitor:
    def __init__(self, gpu_id: int = 0, sampling_period_ms: int = 50):
        """
        Initialize GPU monitor.
        
        Args:
            gpu_id: GPU device ID to monitor
            sampling_period_ms: Sampling period in milliseconds
        """
        self.gpu_id = gpu_id
        self.sampling_period_ms = sampling_period_ms
        self.nvidia_pid: Optional[int] = None
        self.log_file: Optional[Path] = None
        self.is_recording = False
        
    def start_recording(self, log_dir: str = "gpu_logs") -> None:
        """Start recording GPU metrics."""
        if self.is_recording:
            return
            
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = log_path / f"gpu_metrics_{timestamp}.csv"
        
        # Create empty log file first
        self.log_file.touch()
        
        # Construct nvidia-smi command
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_id}",
            "--query-gpu=timestamp,utilization.gpu,pstate,temperature.gpu,clocks.current.sm,power.draw,power.draw.instant",
            "--format=csv,nounits",
            "-f", str(self.log_file),
            "-lms", str(self.sampling_period_ms)
        ]
        
        # Start nvidia-smi process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        
        self.nvidia_pid = process.pid
        self.is_recording = True
        
    def stop_recording(self) -> None:
        """Stop recording GPU metrics."""
        if not self.is_recording or self.nvidia_pid is None:
            return
            
        try:
            os.killpg(os.getpgid(self.nvidia_pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
            
        self.nvidia_pid = None
        self.is_recording = False
        
    def get_metrics(self) -> pd.DataFrame:
        """Get recorded metrics as a DataFrame."""
        if not self.log_file or not self.log_file.exists():
            raise FileNotFoundError("No log file found. Start recording first.")
            
        df = pd.read_csv(self.log_file)
        df.columns = ['timestamp', 'utilization_gpu[%]', 'pstate', 'temperature_gpu[C]', 'clocks_current_sm[MHz]', 'power_draw[W]', 'power_draw_instant[W]']
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df 