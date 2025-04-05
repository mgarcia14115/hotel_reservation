import pandas as pd
import torch
from utils import data_utils as dutils
from utils import train_utils as tutils
from models import baseline

data = pd.read_csv("../../data/raw/data.csv")

train_dataset = dutils.MGDataset("../../data/raw/data.csv",True)
test_dataset  = dutils.MGDataset("../../data/raw/data.csv",False)

train_loader  = torch.utils.data.DataLoader(train_dataset,batch_size = 16)
test_loader  = torch.utils.data.DataLoader(test_dataset,batch_size = 16)

model = baseline.MGModel(28)

Trainer = tutils.MGTrainer(
                            model = model,
                            lr    = 0.01,
                            weight_decay=0.0001,
                            epochs=3,
                            loss_fn=torch.nn.CrossEntropyLoss(),
                            train_loader=train_loader,
                            test_loader=test_loader
                            )


Trainer.train()