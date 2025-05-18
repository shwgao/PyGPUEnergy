import os
import subprocess
import time
import signal
from pathlib import Path
from typing import Optional, Dict, List, Any
import pandas as pd
from threading import Lock
import json
from datetime import datetime
import atexit

class GPUMonitor:
    _instance = None
    _lock = Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GPUMonitor, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, gpu_id: int = 0, sampling_period_ms: int = 50):
        """
        Initialize GPU monitor.
        
        Args:
            gpu_id: GPU device ID to monitor
            sampling_period_ms: Sampling period in milliseconds
        """
        if self._initialized:
            return
            
        self.gpu_id = gpu_id
        self.sampling_period_ms = sampling_period_ms
        self.nvidia_pid: Optional[int] = None
        self.log_file: Optional[Path] = None
        self.is_recording = False
        self.records: List[Dict[str, Any]] = []
        self.t0: Optional[float] = None
        self.t0_file: Optional[Path] = None
        self._initialized = True
        
        # Register cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        print(f"Received signal {signum}, stopping GPU monitor...")
        self.stop_monitoring()
        # Re-raise the signal to allow default handler to terminate the process
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def _cleanup(self):
        if self.is_recording:
            print("Cleaning up GPU monitor...")
            self.stop_monitoring()
        
    def start_monitoring(self, log_dir: str = "gpu_logs") -> None:
        """Start continuous GPU monitoring in the background."""
        if self.is_recording:
            return
            
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = log_path / f"gpu_metrics_{timestamp}.csv"
        self.t0_file = log_path / "t0.txt"
        
        # Create empty log file with headers
        with open(self.log_file, 'w') as f:
            f.write("timestamp,utilization.gpu,pstate,temperature.gpu,clocks.current.sm,power.draw,power.draw.instant\n")
        
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
        
        # Wait a bit to ensure the process starts and begins writing
        time.sleep(0.5)
        
        # Check if process is still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"Failed to start nvidia-smi process. Error: {stderr.decode()}")
        
        self.nvidia_pid = process.pid
        self.is_recording = True
        
        # Wait for the first log entry and parse t0
        t0_found = False
        for _ in range(20):
            time.sleep(0.1)
            try:
                df = pd.read_csv(self.log_file)
                if len(df) > 0:
                    t0_str = df.iloc[0, 0]
                    t0_dt = pd.to_datetime(t0_str, utc=True)
                    self.t0 = t0_dt.timestamp()
                    with open(self.t0_file, 'w') as f:
                        f.write(str(self.t0))
                    t0_found = True
                    break
            except Exception:
                continue
        if not t0_found:
            print("Warning: Could not determine t0 from log file.")
            self.t0 = time.time()
            with open(self.t0_file, 'w') as f:
                f.write(str(self.t0))
        
    def stop_monitoring(self) -> None:
        """Stop continuous GPU monitoring."""
        if not self.is_recording or self.nvidia_pid is None:
            return
            
        try:
            # Give nvidia-smi time to finish writing
            time.sleep(1)
            os.killpg(os.getpgid(self.nvidia_pid), signal.SIGTERM)
            
            # Wait for process to terminate
            time.sleep(0.5)
            
            # Verify log file has content
            if self.log_file and self.log_file.exists():
                if self.log_file.stat().st_size == 0:
                    print(f"Warning: Log file {self.log_file} is empty")
        except ProcessLookupError:
            pass
        except Exception as e:
            print(f"Error stopping monitoring: {str(e)}")
            
        self.nvidia_pid = None
        self.is_recording = False
        
    def record_function_call(self, func_name: str, start_time: float, end_time: float) -> None:
        """Record a function call with its start and end times."""
        if self.t0 is None:
            print("Warning: t0 is not set. Cannot record offsets correctly.")
            return
        self.records.append({
            'type': 'function',
            'name': func_name,
            'start_offset': start_time - self.t0,
            'end_offset': end_time - self.t0,
            'duration': end_time - start_time
        })
        
    def record_code_region(self, region_name: str, start_time: float, end_time: float) -> None:
        """Record a code region with its start and end times."""
        if self.t0 is None:
            print("Warning: t0 is not set. Cannot record offsets correctly.")
            return
        self.records.append({
            'type': 'region',
            'name': region_name,
            'start_offset': start_time - self.t0,
            'end_offset': end_time - self.t0,
            'duration': end_time - start_time
        })
        
    def get_metrics(self) -> pd.DataFrame:
        """Get recorded metrics as a DataFrame."""
        if not self.log_file or not self.log_file.exists():
            raise FileNotFoundError("No log file found. Start monitoring first.")
            
        if self.log_file.stat().st_size == 0:
            raise ValueError("Log file is empty. No metrics recorded.")
            
        df = pd.read_csv(self.log_file)
        df.columns = ['timestamp', 'utilization_gpu[%]', 'pstate', 'temperature_gpu[C]', 
                     'clocks_current_sm[MHz]', 'power_draw[W]', 'power_draw_instant[W]']
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        return df
        
    def save_records(self, output_file: str = "gpu_records.json") -> None:
        """Save all recorded function calls and code regions to a JSON file."""
        with open(output_file, 'w') as f:
            json.dump(self.records, f, indent=2)
            
    def get_metrics_for_period(self, start_offset: float, end_offset: float) -> pd.DataFrame:
        """Get GPU metrics for a specific time period."""
        if self.t0 is None:
            raise ValueError("t0 is not set.")
        df = self.get_metrics()
        df['rel_time'] = (df['timestamp'].astype('int64') / 1e9) - self.t0
        mask = (df['rel_time'] >= start_offset) & (df['rel_time'] <= end_offset)
        return df[mask].copy() 