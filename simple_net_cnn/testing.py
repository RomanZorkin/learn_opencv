from pathlib import Path

import torch
from torch.utils.data import DataLoader

from simple_net_cnn.cnn import Net


def test_net(net_path: Path, dataset: DataLoader) -> tuple[int, int]:
    net = Net()
    net.load_state_dict(torch.load(str(net_path)))
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in dataset:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct, total
