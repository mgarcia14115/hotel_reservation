import torch


class MGModel(torch.nn.Module):

    def __init__(self , input_dims):
        super(MGModel,self).__init__()

        self.body = torch.nn.Sequential(
            torch.nn.Linear(input_dims,20),
            torch.nn.ReLU(),
            torch.nn.Linear(20,10)
        )

        self.head = torch.nn.Sequential(
            torch.nn.Linear(10,2)
        )
        
        
    def forward(self,x):

        out = self.body(x)

        out = self.head(out)

        return out
        

    