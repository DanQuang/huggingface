import torch
from torch import nn, optim
import os
from data_utils.load_data import Load_Data
from models.classifier_model import Classifier_Model
from transformers import get_scheduler
from tqdm.auto import tqdm
from utils.utils import count_trainable_parameters


class Train_Task:
    def __init__(self, config):
        self.num_epochs = config["num_epochs"]
        self.learning_rate = config["learning_rate"]
        self.save_path = config["save_path"]
        self.patience = config["patience"]
        self.dropout = config["dropout"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # load data
        self.load_data = Load_Data(config)

        # Load model
        self.model = Classifier_Model(config).to(self.device)

        # Load loss function and optimizer
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr= self.learning_rate)

        self.train_dataloader, self.dev_dataloader = self.load_data.load_train_dev()

        self.num_training_steps = self.num_epochs * len(self.train_dataloader)

        self.lr_scheduler = get_scheduler(name= "linear",
                                          optimizer= self.optimizer,
                                          num_warmup_steps= 0,
                                          num_training_steps= self.num_training_steps)
    
    def train(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        last_model = "Pretrained_last_model.pth"
        best_model = "Pretrained_best_model.pth"

        if os.path.exists(os.path.join(self.save_path, last_model)):
            checkpoint = torch.load(os.path.join(self.save_path, last_model))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optim_state_dict"])
            print("Load the last model")
            initial_epoch = checkpoint["epoch"] + 1
            print(f"Continue training from epoch {initial_epoch}")

        else:
            initial_epoch = 0
            print("First time training!!!")

        if os.path.exists(os.path.join(self.save_path, best_model)):
            checkpoint = torch.load(os.path.join(self.save_path, best_model))
            best_score = checkpoint['score']
        else:
            best_score = 0.

        threshold = 0
        print(f"The Pretrained model has {count_trainable_parameters(model= self.model)} trainable parameters")
        progress_bar = tqdm(range(self.num_training_steps))
        self.model.train()
        for epoch in range(initial_epoch, initial_epoch + self.num_epochs):
            epoch_loss = 0
            for _, item in enumerate(self.train):
                texts, labels = item["text"], item["label"]

                self.optimizer.zero_grad()

                outputs = self.model(texts)

                loss = self.loss(outputs, labels)

                self.optimizer.step()
                self.lr_scheduler.step()
                progress_bar.update(1)

                epoch_loss += loss.item()
            
            train_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch {epoch}:")
            print(f"Train loss: {train_loss:.5f}")