#!/usr/bin/env python3

import time
import torch
import torchvision.models as models

from PyGPUEnergy.context import GPUMonitorContext
from PyGPUEnergy.visualize import plot_gpu_metrics

def main():
    BATCH_SIZE = 1024

    torch.hub.set_dir('.')

    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your setup.")
        exit()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    model = model.to(device)

    bs = BATCH_SIZE
    input_tensor = torch.randn(bs, 3, 224, 224).to(device)

    # Warmup
    with torch.no_grad():  model(input_tensor)
    torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(4):
            with GPUMonitorContext("resnet50", gpu_id=0):
                for _ in range(8):
                    model(input_tensor)
                torch.cuda.synchronize()
                # sleep for 25 miliseconds
                time.sleep(0.025)


if __name__ == '__main__':
    main()
    
    # Plot the latest log file
    from PyGPUEnergy.utils import get_latest_log_file
    latest_log = get_latest_log_file("gpu_logs")
    if latest_log:
        plot_gpu_metrics(latest_log, save_path="gpu_logs/gpu_metrics_plot.png")
        