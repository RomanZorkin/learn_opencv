import logging
from multiprocessing import dummy
from pathlib import Path

import torch
from torch.autograd import Variable

from simple_net_cnn import config
from simple_net_cnn.cnn import Net

logger = logging.getLogger(__name__)


def change_type(net_path: Path) -> None:
    net = Net()
    net.load_state_dict(torch.load(str(net_path)))
    net.eval()
    dir_name = net_path.parent
    new_model = Path(dir_name, config.MODEL_NAME)
    logger.debug(new_model)
    dummy_input = Variable(torch.randn(1, 3, 32, 32))
    torch.onnx.export(net, dummy_input, new_model)
