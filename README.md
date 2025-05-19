# PyGPUEnergy

A Python package for monitoring GPU power consumption and energy usage during code execution. This package provides both a decorator-based approach and a manual monitoring interface for tracking GPU metrics.

## How it works

The package uses the `nvidia-smi` command to monitor GPU metrics. It runs a background process that periodically calls `nvidia-smi` to get the GPU metrics. The metrics are then saved to a CSV file. 

During the execution of the code, the GPU monitor will record the start and end time of the user code region.

The energy consumption is calculated by equation:

$$E = \int P(t) dt$$

where $P(t)$ is the power consumption of the GPU at time $t$.



## Installation

```bash
pip install PyGPUEnergy
```

## Usage

### Using the Decorator

```python
from PyGPUEnergy.decorator import monitor_gpu

@monitor_gpu(gpu_id=0, sampling_period_ms=100)
def your_gpu_function():
    # Your GPU-intensive code here
    pass
```

The decorator supports the following parameters:
- `gpu_id`: GPU device ID to monitor (default: 0)
- `sampling_period_ms`: Sampling period in milliseconds (default: 100)

### Manual Monitoring

```python
from PyGPUEnergy.monitor import GPUMonitor

# Create monitor instance
monitor = GPUMonitor(gpu_id=0, sampling_period_ms=100)

# Start monitoring
monitor.start_monitoring()

# Your GPU-intensive code here

# Stop monitoring
monitor.stop_monitoring()

# Get metrics with energy consumption information
metrics_df = monitor.get_metrics()
```

### Visualization

```python
from PyGPUEnergy.visualize import plot_gpu_metrics

# Plot metrics from a log file
plot_gpu_metrics("path/to/log_file.csv", save_path="metrics_plot.png")

# Plot with custom settings
plot_gpu_metrics(
    "gpu_logs/log_file.csv",
    records_file="gpu_logs/gpu_records_0.json",  # Function execution records
    t0_file="gpu_logs/t0.txt",                   # Start time reference
    save_path="metrics_plot.png",
    show=True
)
```

## Example

See the `examples/usage_example.py` file for a complete example of using the package. Here's a quick example:

```python
from PyGPUEnergy.decorator import monitor_gpu
import torch

@monitor_gpu(gpu_id=0, sampling_period_ms=100)
def train_model():
    model = torch.nn.Linear(1000, 1000).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    
    for _ in range(100):
        x = torch.randn(1000, 1000).cuda()
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
```

## License

MIT License

## Contributing
This project is a very basic implementation and can be improved in many ways.
Contributions are welcome! Please feel free to submit a Pull Request.
