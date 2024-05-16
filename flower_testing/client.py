import argparse
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import flwr as fl

import sys
sys.path.append('..')
from federated_har.datasets import *
from federated_har.model import *

net = create_model(3).to('cpu')
trainloader, testloader = create_dataloaders('../kin6-mini/train', '../kin6-mini/test', clip_length=32, bs=1)
optim = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
  def get_parameters(self, config):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

  def set_parameters(self, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

  def fit(self, parameters, config):
    self.set_parameters(parameters)
    train(net, trainloader, optim, epochs=1)
    return self.get_parameters(config={}), len(trainloader.dataset), {}

  def evaluate(self, parameters, config):
    self.set_parameters(parameters)
    loss, accuracy = test(net, testloader)
    return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())