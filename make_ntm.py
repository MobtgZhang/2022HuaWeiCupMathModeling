import os
import dateutil
import numpy as np
import pandas as pd

import torch

from matplotlib import pyplot as plt
import matplotlib.dates as md
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 更改了字体导致显示不出负号，因此将配署文件中axes.unicode minus : True修改为False。


from ntm import NTMCell
class TimeRNNModel(torch.nn.Module):
    def __init__(self,val_seq_len):
        super(TimeRNNModel,self).__init__()
        self.gru_time1 = torch.nn.GRU(val_seq_len,val_seq_len,num_layers=2,batch_first=True,bidirectional=True)
        self.gelu = torch.nn.GELU()
        self.linp = torch.nn.Linear(val_seq_len*2,val_seq_len)
        self.lin = torch.nn.Linear(val_seq_len,val_seq_len)
        self.gru_time2 = torch.nn.GRU(val_seq_len,val_seq_len,num_layers=2,batch_first=True)
    def forward(self,x):
        y,_ = self.gru_time1(x)
        y = self.gelu(self.linp(y))
        v = torch.sigmoid(self.lin(x))
        z = y*v+x*(1-v)
        z,_ = self.gru_time2(z)
        return z
def to_var(value,device):
    return torch.tensor(value,dtype=torch.float).to(device)
def make_dataset():
    log_dir = "./log"
    result_dir = "./result"
    seq_max_len = 123
    time_seq_len = 12
    device = "cuda:0"
    model_list = ["gru-best.pt","lstm-best.pt","mdl-best.pt"]
    # dataset preparing        
    val_list = ["wet","evaporation","pressure","rain","temperate","visibility","wind"]
    all_dataset = []
    for val_name in val_list:
        load_file_name = os.path.join(result_dir,"%s.npz"%val_name)
        value = np.load(load_file_name)
        value = value["data"][:seq_max_len,:]
        all_dataset.append(value)
    all_dataset = np.hstack(all_dataset).astype(dtype=np.float64)
    s_v = all_dataset.std(axis=0)+0.01
    m_v = all_dataset.mean(axis=0)
    model_path = os.path.join(log_dir,"ntm-best.pt")
    model = torch.load(model_path).to(device)
    prev = torch.tensor(all_dataset,dtype=torch.float).to(device)
    
    
    # train
    inp_seq_len = prev.size(0)
    outp_seq_len,_ = prev.size()
    # new sequence
    model.zero_state(1)
    # feed the sequence + delimiter
    for i in range(inp_seq_len):
        model(prev[i,:].unsqueeze(0))
    # read the output (no input given)
    pred = torch.zeros(prev.size()).to(device)

    for i in range(outp_seq_len):
        pred[i,:] , _ = model(None)
        
        
        
    val_mat = pred.detach().cpu().numpy()*s_v+m_v
    val_mat = val_mat[:,:4].tolist()
    columns = ["年份","月份","10cm湿度(kg/m2)","40cm湿度(kg/m2)","100cm湿度(kg/m2)","200cm湿度(kg/m2)"]
    years = [2012+(idx+3)//time_seq_len for idx in range(pred.shape[0])]
    months = [(idx+1)%time_seq_len+1 for idx in range(pred.shape[0])]
    all_data_list = []
    for y,m,val in zip(years,months,val_mat):
        tp_list = [y,m]+val
        all_data_list.append(tp_list)
    save_file_path = os.path.join(log_dir,"ntm-predict.xlsx")
    pd.DataFrame(all_data_list,columns=columns).to_excel(save_file_path,index=None)
def make_draw():
    data_dir = "./data"
    result_dir = "/.result"
    log_dir = "./log"
    raw_file_name = os.path.join(data_dir,"附件3、土壤湿度2012—2022年.xlsx")
    lstm_file_name = os.path.join(log_dir,"lstm-predict.xlsx")
    gru_file_name = os.path.join(log_dir,"gru-predict.xlsx")
    mdl_file_name = os.path.join(log_dir,"mdl-predict.xlsx")
    val_list = ["10cm湿度(kg/m2)","40cm湿度(kg/m2)","100cm湿度(kg/m2)","200cm湿度(kg/m2)"]
    
    raw_dataset = pd.read_excel(raw_file_name)
    lstm_dataset = pd.read_excel(lstm_file_name)
    gru_dataset = pd.read_excel(gru_file_name)
    mdl_dataset = pd.read_excel(mdl_file_name)
    
    file_list = [raw_file_name,lstm_file_name,gru_file_name,mdl_file_name]
    name_list = ["Real Data","LSTM","GRU","DGGRU"]
    seq_len_months = 123
    time_seq_len = 12 
    years = [2012+(idx+3)//time_seq_len for idx in range(seq_len_months)]
    months = [(idx+1)%time_seq_len+1 for idx in range(seq_len_months)]
    
    
    # x_time_steps = [dateutil.parser.parse(s) for s in [str(year)+"-"+ str(month) if month>=10 else "0" + str(month)+"-01" for year,month in zip(years,months)]]
    x_time_steps = np.linspace(0,seq_len_months-1,seq_len_months)
    # x_time_steps = pd.TimedeltaIndex([str(year)+"-"+ str(month) if month>=10 else "0" + str(month)+"-01" for year,month in zip(years,months)])
    for val_name in val_list:
        raw_list = raw_dataset[val_name].values
        lstm_list = lstm_dataset[val_name].values
        gru_list = gru_dataset[val_name].values
        mdl_list = mdl_dataset[val_name].values
        
        
        
        #plt.subplots_adjust(bottom=0.2)
        #plt.xticks(rotation= 80)
        ax=plt.gca()
        xfmt = md.DateFormatter('%Y-%m')
        ax.xaxis.set_major_formatter(xfmt)
        plt.plot(x_time_steps,raw_list)
        plt.plot(x_time_steps,lstm_list)
        plt.plot(x_time_steps,gru_list)
        plt.plot(x_time_steps,mdl_list)
        plt.show()
def main():
    make_dataset()
if __name__ == "__main__":
    main()


