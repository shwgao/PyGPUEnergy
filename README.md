# PyGPUEnergy

A Python package for monitoring GPU power consumption and energy usage during code execution. This package provides both a decorator-based approach and a manual monitoring interface for tracking GPU metrics.

## Features

- Monitor GPU power draw and energy consumption in real-time
- Decorator-based monitoring for easy integration with existing code
- Manual monitoring interface for more control
- Visualization tools for analyzing power and energy metrics
- Support for multiple GPUs
- Automatic log file management

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

### Manual Monitoring

```python
from PyGPUEnergy.monitor import GPUMonitor

# Create monitor instance
monitor = GPUMonitor(gpu_id=0, sampling_period_ms=100)

# Start recording
monitor.start_recording()

# Your GPU-intensive code here

# Stop recording
monitor.stop_recording()

# Get metrics
metrics = monitor.get_metrics()
```

### Visualization

```python
from PyGPUEnergy.visualize import plot_gpu_metrics

# Plot metrics from a log file
plot_gpu_metrics("path/to/log_file.csv", save_path="metrics_plot.png")
```

## Example

See the `examples/usage_example.py` file for a complete example of using the package.

## Requirements

- Python >= 3.7
- NVIDIA GPU with nvidia-smi support
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- torch >= 1.9.0 (optional, for GPU detection)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.