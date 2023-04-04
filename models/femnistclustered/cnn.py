
from ..model import Model
import numpy as np

IMAGE_SIZE = 28

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ClientModel(Model):
    def __init__(self, seed, lr, num_classes):
        super(ClientModel, self).__init__(seed, lr)
        self.seed = seed
        self.lr = lr
        self.num_classes = num_classes

        self.loss_fn = nn.CrossEntropyLoss()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 2048)
        self.fc2 = nn.Linear(2048, self.num_classes)

        self.batch_seed = 12 + seed


    def create_model(self):
        features = nn.Parameter(torch.zeros((None, IMAGE_SIZE * IMAGE_SIZE), dtype=torch.float32), requires_grad=False)
        labels = nn.Parameter(torch.zeros(None, dtype=torch.long), requires_grad=False)
        train_op = nn.Parameter(torch.zeros([], dtype=torch.float32), requires_grad=False)
        eval_metric_ops = nn.Parameter(torch.zeros([], dtype=torch.float32), requires_grad=False)
        conf_matrix = nn.Parameter(torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64), requires_grad=False)
        loss = nn.Parameter(torch.zeros([], dtype=torch.float32), requires_grad=False)
        
        super().initialize_graph()
        
        return features, labels, train_op, eval_metric_ops, conf_matrix, loss

    def forward(self, x):
        x = x.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 7 * 7 * 64)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)

