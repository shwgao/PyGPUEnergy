import time
import torch
from PyGPUEnergy.decorator import monitor_gpu
from PyGPUEnergy.visualize import plot_gpu_metrics

# Example 1: Using the decorator
@monitor_gpu(gpu_id=0, sampling_period_ms=100)
def train_model():
    # Simulate some GPU-intensive work
    model = torch.nn.Linear(1000, 1000).cuda()
    x = torch.randn(1000, 1000).cuda()
    
    for _ in range(100):
        y = model(x)
        loss = y.sum()
        loss.backward()
        time.sleep(0.01)  # Simulate computation time

# Example 2: Manual monitoring
from PyGPUEnergy.monitor import GPUMonitor

def manual_monitoring_example():
    monitor = GPUMonitor(gpu_id=0, sampling_period_ms=100)
    
    # Start recording
    monitor.start_recording()
    
    # Do some GPU work
    model = torch.nn.Linear(1000, 1000).cuda()
    x = torch.randn(1000, 1000).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for _ in range(50):
        optimizer.zero_grad()
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()  # Ensure GPU operations are complete
        time.sleep(0.01)
    
    # Stop recording
    monitor.stop_recording()
    
    # Add a small delay to ensure log file is written
    time.sleep(0.5)
    
    # Get and plot metrics
    metrics = monitor.get_metrics()
    print(f"Average GPU utilization: {metrics['utilization_gpu[%]'].mean():.2f}%")
    print(f"Average temperature: {metrics['temperature_gpu[C]'].mean():.2f}°C")
    print(f"Average power draw: {metrics['power_draw[W]'].mean():.2f} W")
    print(f"Average instant power draw: {metrics['power_draw_instant[W]'].mean():.2f} W")
    print(f"Max power draw: {metrics['power_draw[W]'].max():.2f} W")

if __name__ == "__main__":
    print("Running decorated function example...")
    train_model()
    
    print("\nRunning manual monitoring example...")
    manual_monitoring_example()
    
    # Plot the latest log file
    from PyGPUEnergy.utils import get_latest_log_file
    latest_log = get_latest_log_file("gpu_logs")
    if latest_log:
        plot_gpu_metrics(latest_log, save_path="gpu_metrics_plot.png") 