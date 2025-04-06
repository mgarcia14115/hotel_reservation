import torch
from utils import data_utils as dutils
from utils import train_utils as tutils
from utils import console_utils as cutils
from models import baseline


train_dataset = dutils.MGDataset("../../data/processed/train.csv",True)
#test_dataset  = dutils.MGDataset("../../data/processed/test.csv",False)
val_dataset   = dutils.MGDataset("../../data/processed/val.csv",False)


train_loader  = torch.utils.data.DataLoader(train_dataset,batch_size = 16)
#test_loader  = torch.utils.data.DataLoader(test_dataset,batch_size = 16)
val_loader  = torch.utils.data.DataLoader(val_dataset,batch_size = 16)

args = cutils.get_args()

model_name = args.model_name

print(f"The model name is {model_name}")

model = baseline.MGModel(28)

Trainer = tutils.MGTrainer(
                            model = model,
                            lr    = 0.01,
                            weight_decay=0.0001,
                            epochs=3,
                            loss_fn=torch.nn.CrossEntropyLoss(),
                            train_loader=train_loader,
                            val_loader=val_loader
                            )


Trainer.train()
#Trainer.eval()