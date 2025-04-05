import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

class MGDataset(torch.utils.data.Dataset):
    
    def __init__(self,data_pth,fit_scaler):
        
        df = pd.read_csv(data_pth).drop(labels=["Booking_ID","arrival_year","arrival_date"],axis=1)
        le = LabelEncoder()      
        y = torch.tensor(le.fit_transform(df["booking_status"]))
   
        df = df.drop(labels=["booking_status"],axis=1)
        encoded_data = pd.get_dummies(df,dtype=int)
        cols = [col for col in encoded_data.columns if encoded_data.describe().loc["max",col] > 1]
        
        if fit_scaler == True:
            for col in cols:
                scaler = StandardScaler()        
                encoded_data[col] = scaler.fit_transform(np.asarray(encoded_data[col]).reshape(-1,1))
                joblib.dump(scaler,os.path.join("utils/scalers/",col+".pkl"))
        else:
            
            for col in cols:
                scaler = joblib.load(os.path.join("utils/scalers/",col+".pkl"))      
                encoded_data[col] = scaler.transform(np.asarray(encoded_data[col]).reshape(-1,1))
                
            

        self.x = torch.tensor(np.asarray(encoded_data))
        self.y = y
    
    def __len__(self):

        return len(self.x)
    
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    