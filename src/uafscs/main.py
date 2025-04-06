import torch
from utils import data_utils     as dutils
from utils import train_utils    as tutils
from utils import console_utils  as cutils
from utils import model_utils    as mutils
from configs import default as config


train_dataset = dutils.MGDataset("../../data/processed/train.csv",True)
#test_dataset  = dutils.MGDataset("../../data/processed/test.csv",False)
val_dataset   = dutils.MGDataset("../../data/processed/val.csv",False)


train_loader  = torch.utils.data.DataLoader(train_dataset,batch_size = 16)
#test_loader  = torch.utils.data.DataLoader(test_dataset,batch_size = 16)
val_loader  = torch.utils.data.DataLoader(val_dataset,batch_size = 16)

args = cutils.get_args()

model_name   = args.model_name
weight_decay = args.weight_decay
lr           = args.lr
epochs       = args.epochs
loss         = args.loss_fn
optim        = args.optim
device       = config.DEVICE

print(f"The device you are using is {device}")

loss_fn = None

if loss == 'crossentropy':
    loss_fn = torch.nn.CrossEntropyLoss()


model = mutils.init_model(model_name)


Trainer = tutils.MGTrainer(
                            model = model,
                            lr    = lr,
                            optim = optim,
                            weight_decay=weight_decay,
                            epochs=epochs,
                            loss_fn=loss_fn,
                            train_loader=train_loader,
                            val_loader=val_loader
                            )


Trainer.train()
#Trainer.eval()