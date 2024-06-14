from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch


class My_Dataset(Dataset):
    def __init__(self, dataset):
        super(My_Dataset, self).__init__()

        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]
    

class Load_Data:
    def __init__(self, config):
        self.train_batch = config["train_batch"]
        self.dev_batch = config["dev_batch"]
        self.test_batch = config["test_batch"]

        self.dataset = load_dataset("yelp_review_full")

        small_train = self.dataset['train'].shuffle(seed= 42).select(range(10000))
        small_test = self.dataset['test'].shuffle(seed= 42).select(range(2000))

        self.train_dataset = My_Dataset(dataset= small_train)
        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.dev_dataset = random_split(self.train_dataset, [0.8, 0.2], generator= generator)

        self.test_dataset = My_Dataset(dataset= small_test)

    def load_train_dev(self):
        train_dataloader = DataLoader(self.train_dataset,
                                      self.train_batch,
                                      shuffle= True)
        
        dev_dataloader = DataLoader(self.dev_dataset,
                                    self.dev_batch,
                                    shuffle= False)
        
        return train_dataloader, dev_dataloader
    
    def load_test(self):
        test_dataloader = DataLoader(self.test_dataset,
                                     self.test_batch,
                                     shuffle= False)
        
        return test_dataloader