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
from PyGPUEnergy.utils import calculate_energy_consumption
import numpy as np

class GPUMonitor:
    _instance = None
    _lock = Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GPUMonitor, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def _validate_machine_info(self):
        """Validate GPU and CUDA installation."""
        def is_number(s):
            try:    float(s)
            except ValueError: return False
            return True

        # Validate GPU ID
        if not isinstance(self.gpu_id, int) or self.gpu_id < 0:
            raise ValueError(f"Invalid GPU ID: {self.gpu_id}. Must be a non-negative integer.")

        # Check if nvidia-smi is available and get GPU info
        try:
            result = subprocess.run(['nvidia-smi', f'--id={self.gpu_id}', '--query-gpu=timestamp,name,serial,uuid,driver_version', '--format=csv,noheader'], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to query GPU information: {e.stderr.decode()}")
        except FileNotFoundError:
            raise RuntimeError("nvidia-smi not found. Please ensure NVIDIA drivers are installed.")

        output = result.stdout.decode().split('\n')[0].split(', ')
        if len(output) != 5:
            raise RuntimeError(f"Unexpected output format from nvidia-smi: {output}")
            
        self.nvsmi_time, self.gpu_name, self.gpu_serial, self.gpu_uuid, self.driver_version = output
        self.gpu_name = self.gpu_name.replace(' ', '_')

        # Validate CUDA installation - exit if not found
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True)
            output = result.stdout
            if not output:
                raise RuntimeError("nvcc --version returned empty output")
            nvcc_version = output.split('\n')[3].split(',')[1].strip()
            self.nvcc_version = nvcc_version
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            error_msg = "CUDA toolkit not found. This is a required dependency for GPU monitoring."
            if isinstance(e, subprocess.CalledProcessError):
                error_msg += f"\nError details: {e.stderr.decode()}"
            raise RuntimeError(error_msg)

        # Check supported power draw query options
        try:
            output = subprocess.run(['nvidia-smi', '--help-query-gpu'], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            output = output.stdout.decode()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to query GPU power options: {e.stderr.decode()}")

        # Initialize power draw options
        self.pwr_draw_options = {
            'utilization.gpu': False,
            'pstate': False,
            'temperature.gpu': False,
            'clocks.current.sm': False,
            'power.draw': False,
            'power.draw.instant': False
        }

        query_options = '--query-gpu='
        for key in self.pwr_draw_options.keys():
            if output.find(key) != -1:
                query_options += key + ','
                self.pwr_draw_options[key] = True

        if query_options == '--query-gpu=':
            raise RuntimeError("No supported power draw query options found")

        try:
            output = subprocess.run(['nvidia-smi', f'--id={self.gpu_id}', query_options, '--format=csv,noheader,nounits'], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            output = output.stdout.decode()[:-1].split(', ')
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to query GPU power metrics: {e.stderr.decode()}")

        if len(output) != len([v for v in self.pwr_draw_options.values() if v]):
            raise RuntimeError(f"Mismatch in power draw metrics count. Expected {len([v for v in self.pwr_draw_options.values() if v])}, got {len(output)}")

        for i, (key, value) in enumerate(self.pwr_draw_options.items()):
            if value:
                self.pwr_draw_options[key] = is_number(output[i])
                if not self.pwr_draw_options[key]:
                    print(f"Warning: Invalid power draw value for {key}: {output[i]}")

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
        
        # Validate machine info before proceeding
        self._validate_machine_info()
        
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
            
        self.save_records(f"gpu_logs/gpu_records_{self.gpu_id}.json")
            
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
        
        # Convert timestamps to milliseconds since epoch
        df['timestamp'] = df['timestamp'].apply(lambda x: int(datetime.strptime(x, '%Y/%m/%d %H:%M:%S.%f').timestamp() * 1000))
        
        # Convert t0 to milliseconds
        t0_ms = int(self.t0 * 1000)
        
        # Add relative time column
        df['rel_time_ms'] = (df['timestamp'] - t0_ms).astype(np.int64)
        
        # Calculate energy consumption for each function
        energy_records = calculate_energy_consumption(df, self.records)
        
        # Add energy consumption information to the DataFrame
        df['energy_joules'] = np.nan
        df['avg_power_watts'] = np.nan
        
        for record in energy_records:
            mask = (df['rel_time_ms'] >= record['start_offset']) & (df['rel_time_ms'] <= record['end_offset'])
            df.loc[mask, 'energy_joules'] = record['energy_joules']
            df.loc[mask, 'avg_power_watts'] = record['avg_power_watts']
            df.loc[mask, 'function_name'] = record['name']
            df.loc[mask, 'function_type'] = record['type']
        
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