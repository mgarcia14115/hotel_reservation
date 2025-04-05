from configs import default as config
import torch
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import classification_report


class MGTrainer:

    def __init__(self,
                model               = None,
                optim               = None,
                lr                  = None,
                weight_decay        = None,
                epochs              = None,
                loss_fn             = None,
                train_loader        = None,
                test_loader         = None,
                **kwargs
                ):
        
        self.lr           = lr
        self.model        = model
        self.weight_decay = weight_decay
        self.epochs       = epochs
        self.loss_fn      = loss_fn
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.optim        = optim
        self.device       = config.DEVICE

        if optim == None:
            self.optim = torch.optim.Adam(self.model.parameters(),lr = self.lr)

        elif optim == "adam":
            
            if weight_decay == None:
                self.optim = torch.optim.Adam(self.model.parameters(),lr = self.lr)
            else:
                self.optim = torch.optim.Adam(self.model.parameters(),lr = self.lr, weight_decay = self.weight_decay)
        elif optim == "adamw":

            if weight_decay == None:
                self.optim = torch.optim.AdamW(self.model.parameters(),lr = self.lr)
            else:
                self.optim = torch.optim.AdamW(self.model.parameters(),lr = self.lr, weight_decay = self.weight_decay)

        else:

            if weight_decay == None:
                self.optim = torch.optim.SGD(self.model.parameters(),lr = self.lr)
            else:
                self.optim = torch.optim.SGD(self.model.parameters(),lr = self.lr, weight_decay = self.weight_decay)

        
    def train(self):

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        best_vloss = 10
        target_names = ['Class 0', 'Class 1']
        for epoch in range(self.epochs):

            y_true  = []
            y_pred  = []

            yv_true = []
            yv_pred = []
            loss_per_epoch = 0
            self.model.train()
            for batch in tqdm(self.train_loader):
                self.optim.zero_grad()

                inputs,labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                predictions = self.model(inputs)
                _,pred        = torch.max(predictions,1)
                loss        = self.loss_fn(predictions,labels)
                loss.backward()
                self.optim.step()
                y_true.extend(labels.detach().cpu().numpy())
                y_pred.extend(pred.detach().cpu().numpy())
                loss_per_epoch += loss.item()
            
            running_vloss = 0           
            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(self.test_loader):
                    vinputs , vlabels = batch
                    vinputs = vinputs.to(self.device)
                    vlabels = vlabels.to(self.device)

                    predictions = self.model(vinputs)
                    _,pred        = torch.max(predictions,1)
                    
                    vloss = self.loss_fn(predictions,vlabels)
                    yv_true.extend(vlabels.detach().cpu().tolist())
                    yv_pred.extend(pred.detach().cpu().tolist())
                    running_vloss += vloss.item()

                    
            print(f"Epoch: {epoch + 1}    Training Loss: {loss_per_epoch}  Validation Loss: {running_vloss}")

            print(classification_report(y_true=y_true,y_pred=y_pred,zero_division=0.0,target_names=target_names))

            print(f"##########################################################################")

            print(classification_report(y_true=yv_true,y_pred=yv_pred,zero_division=0.0,target_names=target_names))

            if running_vloss < best_vloss:
                best_vloss = running_vloss
                model_pth  = 'model_{}_{}'.format(timestamp,epoch+1)
                torch.save(self.model.state_dict(),model_pth)

                 

            
        