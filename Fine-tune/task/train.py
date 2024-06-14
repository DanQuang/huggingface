import torch
from torch import nn, optim
import os
from data_utils.load_data import Load_Data
from models.classifier_model import Classifier_Model
from utils.utils import count_trainable_parameters


class Train_Task:
    def __init__(self, config):
        self.num_epochs = config["num_epochs"]
        self.learning_rate = config["learning_rate"]
        self.save_path = config["save_path"]
        self.patience = config["patience"]
        self.dropout = config["dropout"]

        # load data
        self.load_data = Load_Data(config)

        # Load model
        self.model = Classifier_Model(config)

        # Load loss function and optimizer
        self.loss = 