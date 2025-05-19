import time
import torch
from PyGPUEnergy.decorator import monitor_gpu
from PyGPUEnergy.context import GPUMonitorContext
from PyGPUEnergy.monitor import GPUMonitor
from PyGPUEnergy.visualize import plot_gpu_metrics

# Example 1: Using the decorator
@monitor_gpu(gpu_id=0, sampling_period_ms=100)
def train_model():
    # Simulate some GPU-intensive work
    model = torch.nn.Linear(1000, 1000).cuda()
    x = torch.randn(1000, 1000).cuda()
    
    for _ in range(1000):
        y = model(x)
        loss = y.sum()
        loss.backward()
        # time.sleep(0.1)  # Simulate computation time

# Example 2: Using context manager for code regions
def manual_monitoring_example():
    # Create a model
    model = torch.nn.Linear(1000, 1000).cuda()
    x = torch.randn(1000, 1000).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Monitor training loop
    with GPUMonitorContext("training_loop", gpu_id=0):
        for _ in range(1000):
            optimizer.zero_grad()
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            # time.sleep(0.01)
    
    # Monitor evaluation
    with GPUMonitorContext("evaluation", gpu_id=0):
        with torch.no_grad():
            y = model(x)
            loss = y.sum()
            time.sleep(0.01)

# Example 3: Nested monitoring
@monitor_gpu(gpu_id=0)
def outer_function():
    print("Outer function start")
    with GPUMonitorContext("inner_region"):
        print("Inner region start")
        time.sleep(0.01)
        print("Inner region end")
    print("Outer function end")

if __name__ == "__main__":
    # Get the singleton monitor instance
    monitor = GPUMonitor(gpu_id=0)
    
    print("Running decorated function example...")
    train_model()
    
    print("\nRunning manual monitoring example...")
    manual_monitoring_example()
    
    print("\nRunning nested monitoring example...")
    outer_function()
    
    monitor.stop_monitoring()
     
    # Plot the latest log file
    from PyGPUEnergy.utils import get_latest_log_file
    latest_log = get_latest_log_file("gpu_logs")
    if latest_log:
        plot_gpu_metrics(latest_log, save_path="gpu_logs/gpu_metrics_plot.png")
        
    # Stop monitoring