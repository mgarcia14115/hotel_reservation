import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

class MGDataset(torch.utils.data.Dataset):
    
    def __init__(self,data_pth):
        
        df = pd.read_csv(data_pth).drop(labels=["Booking_ID","arrival_year","arrival_date"],axis=1)
        le = LabelEncoder()      
        y = torch.tensor(le.fit_transform(df["booking_status"]))
        df = df.drop(labels=["booking_status"],axis=1)
        
        self.x = torch.tensor(np.asarray(pd.get_dummies(df,dtype=int)))
        self.y = y
    
    def __len__(self):

        return len(self.x)
    
    def __get_item__(self,idx):

        return self.x[idx],self.y[idx]
    