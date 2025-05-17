import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Optional, Union

def plot_gpu_metrics(
    log_file: Union[str, Path],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> None:
    """
    Plot GPU metrics including utilization, temperature, power, and clock speeds.
    
    Args:
        log_file: Path to the CSV log file
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    # Read metrics
    df = pd.read_csv(log_file)
    df.columns = ['timestamp', 'utilization_gpu[%]', 'pstate', 'temperature_gpu[C]', 
                 'clocks_current_sm[MHz]', 'power_draw[W]', 'power_draw_instant[W]']
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('GPU Metrics')
    
    # Plot GPU utilization
    ax1.plot(df['timestamp'], df['utilization_gpu[%]'], 'b-', label='GPU Utilization')
    ax1.set_ylabel('Utilization (%)')
    ax1.set_title('GPU Utilization')
    ax1.grid(True)
    ax1.legend()
    
    # Plot temperature
    ax2.plot(df['timestamp'], df['temperature_gpu[C]'], 'r-', label='Temperature')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title('GPU Temperature')
    ax2.grid(True)
    ax2.legend()
    
    # Plot power draw
    ax3.plot(df['timestamp'], df['power_draw[W]'], 'g-', label='Average Power')
    ax3.plot(df['timestamp'], df['power_draw_instant[W]'], 'y--', label='Instant Power')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Power (W)')
    ax3.set_title('GPU Power Consumption')
    ax3.grid(True)
    ax3.legend()
    
    # Plot clock speed
    ax4.plot(df['timestamp'], df['clocks_current_sm[MHz]'], 'm-', label='SM Clock')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Clock Speed (MHz)')
    ax4.set_title('GPU Clock Speed')
    ax4.grid(True)
    ax4.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path)
    
    # Show plot if requested
    if show:
        plt.show()
    else:
        plt.close() 