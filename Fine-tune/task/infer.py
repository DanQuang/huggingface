import torch
from torch import nn
import os
from data_utils.load_data import Load_Data
from models.classifier_model import Classifier_Model
from tqdm.auto import tqdm


class Infer_Task:
    def __init__(self, config):
        self.save_path = config["save_path"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # load data
        self.load_data = Load_Data(config)

        # Load model
        self.model = Classifier_Model(config).to(self.device)

        # Load loss function and optimizer
        self.loss = nn.CrossEntropyLoss()

        self.test_dataloader = self.load_data.load_test()
    
    def predict(self):
        best_model = "Pretrained_best_model.pth"

        if os.path.join(self.save_path, best_model):
            checkpoint = torch.load(os.path.join(self.save_path, best_model))
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()
        with torch.inference_mode():
                test_loss = 0
                for _, item in enumerate(tqdm(self.test_dataloader)):
                    texts, labels = item["text"], item["label"].to(self.device)

                    outputs = self.model(texts)

                    loss = self.loss(outputs, labels)
                    test_loss += loss.item()
                            
                test_loss = test_loss / len(self.test_dataloader)
                print(f"Test loss: {test_loss:.5f}")