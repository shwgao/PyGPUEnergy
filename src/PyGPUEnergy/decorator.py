import functools
import time
from typing import Callable, Any
from .monitor import GPUMonitor

def monitor_gpu(gpu_id: int = 0, sampling_period_ms: int = 100):
    """
    Decorator to monitor GPU power and energy consumption during function execution.
    Uses a singleton monitor that runs continuously in the background.
    
    Args:
        gpu_id: GPU device ID to monitor
        sampling_period_ms: Sampling period in milliseconds
    
    Returns:
        Decorated function that records GPU metrics during execution
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get the singleton monitor instance
            monitor = GPUMonitor(gpu_id=gpu_id, sampling_period_ms=sampling_period_ms)
            
            # Ensure monitoring is started
            if not monitor.is_recording:
                monitor.start_monitoring()
            
            # Record function execution
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                monitor.record_function_call(func.__name__, start_time, end_time)
                
        return wrapper
    return decorator 