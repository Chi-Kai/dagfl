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

        self.num_epochs = 10
        self.base_epochs = 1

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
    
    # 该函数可以重载
    @abstractmethod
    def get_params(self):
      pass
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

    def close(self):
        pass

    @abstractmethod
    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        pass

    @abstractmethod
    def process_y(self, raw_y_batch):
        pass
