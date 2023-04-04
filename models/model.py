"""Interfaces for ClientModel and ServerModel."""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tangle.lab.dataset import batch_data
from tangle.models.baseline_constants import ACCURACY_KEY


class Model(ABC, nn.Module):
    def __init__(self, seed, lr, optimizer=None):
        super(Model, self).__init__()
        self.seed = seed
        self.lr = lr
        self.batch_seed = 12 + seed
        self._optimizer = optimizer

        self.num_epochs = 1
        self.batch_size = 10

        self.loss_fn = nn.CrossEntropyLoss()

        self.additional_params = []

        self.graph = None
        self.sess = None
        self.saver = None
        self.size = np.sum([np.prod(p.shape) for p in self.parameters()])

        np.random.seed(self.seed)

    def initialize_graph(self):
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

    def set_params(self, model_params):
       with torch.no_grad():
            idx = 0
            for param in self.parameters():
                # set the parameter values using the provided numpy array
                param.copy_(torch.from_numpy(model_params[idx]))
                idx += 1

    def get_params(self):
        return [p.detach().numpy() for p in self.parameters()]

    @property
    def optimizer(self):
        """Optimizer to be used by the model."""
        if self._optimizer is None:
            self._optimizer = optim.SGD(self.parameters(), lr=self.lr)

        return self._optimizer

    @abstractmethod
    def create_model(self):
        """Creates the model for the task.

        Returns:
            A 4-tuple consisting of:
                features: A placeholder for the samples' features.
                labels: A placeholder for the samples' labels.
                train_op: A Tensorflow operation that, when run with the features and
                    the labels, trains the model.
                eval_metric_ops: A Tensorflow operation that, when run with features and labels,
                    returns the accuracy of the model.
                conf_matrix: A Tensorflow operation that, when run with features and labels,
                    returns the confusion matrix of the model.
                loss: A Tensorflow operation that, when run with features and labels,
                    returns the loss of the model.
        """
        pass

    def initialize_graph(self):
        self.features, self.labels, self.fc1, self.fc2 = self.create_model()
        self.optimizer  # Initialize optimizer

    def train(self, data):
        self.num_batches = len(data['x']) // self.batch_size
        for _ in range(self.num_epochs):
            self.run_epoch(data, self.batch_size, self.num_batches)
        update = self.get_params()
        return update

    def run_epoch(self, data, batch_size, num_batches):
        self.batch_seed += 1
        for batched_x, batched_y in batch_data(data, batch_size, num_batches, seed=self.batch_seed):
            input_data = self.process_x(batched_x)
            target_data = self.process_y(batched_y)

            input_tensor = torch.from_numpy(input_data).float()
            target_tensor = torch.from_numpy(target_data).long()

            self.optimizer.zero_grad()
            output = self.forward(input_tensor)
            loss = self.loss_fn(output, target_tensor)
            loss.backward()
            self.optimizer.step()

    def test(self, data):
        x_vecs = self.process_x(data['x'])
        labels = self.process_y(data['y'])

        input_tensor = torch.from_numpy(x_vecs).float()
        target_tensor = torch.from_numpy(labels).long()

        with torch.no_grad():
            output = self.forward(input_tensor)
            acc = (output.argmax(dim=1) == target_tensor).sum().item() / len(data['y'])
            conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
            for t, p in zip(target_tensor.view(-1), output.argmax(dim=1).view(-1)):
                conf_matrix[t.long(), p.long()] += 1
            loss = self.loss_fn(output, target_tensor)

        return {ACCURACY_KEY: acc, 'conf_matrix': conf_matrix, 'loss': loss.item()}

    def close(self):
        pass

    @abstractmethod
    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        pass

    @abstractmethod
    def process_y(self, raw_y_batch):
        pass
