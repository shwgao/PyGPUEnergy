import time
from typing import Optional
from .monitor import GPUMonitor

class GPUMonitorContext:
    def __init__(self, region_name: str, gpu_id: int = 0, sampling_period_ms: int = 100):
        """
        Context manager for monitoring GPU metrics in a code region.
        
        Args:
            region_name: Name of the code region being monitored
            gpu_id: GPU device ID to monitor
            sampling_period_ms: Sampling period in milliseconds
        """
        self.region_name = region_name
        self.monitor = GPUMonitor(gpu_id=gpu_id, sampling_period_ms=sampling_period_ms)
        self.start_time: Optional[float] = None
        
    def __enter__(self):
        # Ensure monitoring is started
        if not self.monitor.is_recording:
            self.monitor.start_monitoring()
        
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            end_time = time.time()
            self.monitor.record_code_region(self.region_name, self.start_time, end_time) 