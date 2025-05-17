import functools
from typing import Callable, Any
from .monitor import GPUMonitor

def monitor_gpu(gpu_id: int = 0, sampling_period_ms: int = 100):
    """
    Decorator to monitor GPU power and energy consumption during function execution.
    
    Args:
        gpu_id: GPU device ID to monitor
        sampling_period_ms: Sampling period in milliseconds
    
    Returns:
        Decorated function that records GPU metrics during execution
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            monitor = GPUMonitor(gpu_id=gpu_id, sampling_period_ms=sampling_period_ms)
            
            try:
                monitor.start_recording()
                result = func(*args, **kwargs)
                return result
            finally:
                monitor.stop_recording()
                
        return wrapper
    return decorator 