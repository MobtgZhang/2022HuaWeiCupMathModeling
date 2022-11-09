import torch
class TimeRNNModel(torch.nn.Module):
    def __init__(self,val_seq_len):
        super(TimeRNNModel,self).__init__()
        self.gru_time1 = torch.nn.GRU(val_seq_len,val_seq_len,num_layers=2)
        self.lin = torch.nn.Linear(val_seq_len,val_seq_len)
        self.gru_time2 = torch.nn.GRU(val_seq_len,val_seq_len,num_layers=2)
    def forward(self,x):
        y,_ = self.gru_time1(x)
        v = torch.sigmoid(self.lin(y))
        z = y*v+x*(1-v)
        z,_ = self.gru_time2(z)
        return z

