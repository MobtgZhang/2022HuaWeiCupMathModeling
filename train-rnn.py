import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    
def to_var(value,device):
    return torch.tensor(value,dtype=torch.float).to(device)
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,dataset,time_seq_len=12):
        super(MyDataset,self).__init__()
        self.time_seq_len = time_seq_len
        self.dataset = dataset
    def __getitem__(self,idx):
        prev = self.dataset[idx:idx+self.time_seq_len,:]
        prev = (prev - prev.mean(axis=0))/(prev.std(axis=0)+0.01)
        post = self.dataset[idx+self.time_seq_len:idx+2*self.time_seq_len,:]
        post = (post - post.mean(axis=0))/(post.std(axis=0)+0.01)
        return prev,post
    def __len__(self):
        return len(self.dataset) - 2*self.time_seq_len+1
def evaluate(model,datalaoder,loss_fn,device):
    loss_list = []
    for idx,(prev,post) in enumerate(datalaoder):
        prev = to_var(prev,device)
        post = to_var(post,device)
        pred,_ = model(prev)
        loss = loss_fn(pred,post)
        #loss = (pred - post).norm()
        loss_list.append(loss.item())
    return sum(loss_list)/len(loss_list)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-max-len",type=int,default=123)
    parser.add_argument("--train-times",type=int,default=1200)
    parser.add_argument("--model",type=str,default="lstm")
    parser.add_argument("--batch-size",type=int,default=10)
    parser.add_argument("--train-per",type=float,default=0.7)
    parser.add_argument("--time-seq-len",type=float,default=12)
    parser.add_argument("--rmsprop-lr",default=1e-4,type=float)
    parser.add_argument("--rmsprop-momentum",default=0.9,type=float)
    parser.add_argument("--rmsprop-alpha",default=0.95,type=float)
    parser.add_argument("--result-dir",type=str,default="./result")
    parser.add_argument("--log-dir",type=str,default="./log")
    args = parser.parse_args()
    return args
def main():
    args = get_args()
    assert args.model in ["lstm","gru"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    model_path = os.path.join(args.log_dir,"%s-best.pt"%args.model)
    save_fig_file = os.path.join(args.log_dir,"%s-loss.png"%args.model)
    save_file_path = os.path.join(args.log_dir,"%s-result.xlsx"%args.model)
    if not os.path.exists(model_path):
        val_list = ["wet","evaporation","pressure","rain","temperate","visibility","wind"]
        all_dataset = []
        for val_name in val_list:
            load_file_name = os.path.join(args.result_dir,"%s.npz"%val_name)
            value = np.load(load_file_name)
            value = value["data"][:args.seq_max_len,:]
            all_dataset.append(value)
        all_dataset = np.hstack(all_dataset).astype(dtype=np.float64)
        fea_dim = all_dataset.shape[1]
        train_len = int(args.train_per*len(all_dataset))
        train_dataset = MyDataset(all_dataset[:train_len,:],args.time_seq_len)
        test_dataset = MyDataset(all_dataset[:train_len,:],args.time_seq_len)
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True)
        if args.model == "lstm":
            model = torch.nn.LSTM(fea_dim,fea_dim,batch_first=True,num_layers=4).to(device)
        elif args.model == "gru":
            model = torch.nn.GRU(fea_dim,fea_dim,batch_first=True,num_layers=4).to(device)
        else:
            raise TypeError("The unknown model type: %s"%args.model)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.RMSprop(model.parameters(),
                                momentum=args.rmsprop_momentum,
                                alpha=args.rmsprop_alpha,
                                lr=args.rmsprop_lr)
        train_loss_list = []
        best_loss = 65535.0
        for epoch in range(args.train_times):
            for idx,(prev,post) in enumerate(train_loader):
                optimizer.zero_grad()
                prev = to_var(prev,device)
                post = to_var(post,device)
                pred,_ = model(prev)
                loss = loss_fn(pred,post)
                loss.backward()
                optimizer.step()
            eval_loss = evaluate(model,test_loader,loss_fn,device)
            print("The final result ids:%d,loss:%0.4f"%(epoch,eval_loss))
            if best_loss>eval_loss:
                best_loss = eval_loss
                torch.save(model,model_path)
            train_loss_list.append(eval_loss)
        x = np.linspace(0,len(train_loss_list)-1,len(train_loss_list))
        y = train_loss_list
        plt.plot(x,y)
        plt.savefig(save_fig_file)
        plt.show()
    
    make_xlsx(model_path,args.result_dir,save_file_path,args.seq_max_len,args.time_seq_len,device)
def make_xlsx(model_path,data_dir,save_file_path,seq_max_len,time_step_seq,device):
    model = torch.load(model_path)
    val_list = ["wet","evaporation","pressure","rain","temperate","visibility","wind"]
    all_dataset = []
    for val_name in val_list:
        load_file_name = os.path.join(data_dir,"%s.npz"%val_name)
        value = np.load(load_file_name)
        value = value["data"][:seq_max_len,:]
        all_dataset.append(value)
    all_dataset = np.hstack(all_dataset).astype(dtype=np.float64)
    prev_data = all_dataset[-time_step_seq:,:]
    m_v = prev_data.mean(axis=0)
    s_v = prev_data.std(axis=0)+0.01
    mean = prev_data.mean(axis=0)
    std = prev_data.std(axis=0)+0.01
    prev = (prev_data - mean)/std
    contin_last_date = time_step_seq-3+time_step_seq
    re_val_list = []
    for k in range(contin_last_date):
        prev = torch.tensor(prev,dtype=torch.float).to(device)
        pred ,_ = model(prev)
        val = pred[-1,:].cpu().detach().numpy()*s_v+m_v
        re_val_list.append(val[:4])
        prev = torch.concat([prev[1:],pred[-1,:].unsqueeze(0)],dim=0)
    val_mat = np.round(np.vstack(re_val_list),2).tolist()
    columns = ["年份","月份","10cm湿度(kg/m2)","40cm湿度(kg/m2)","100cm湿度(kg/m2)","200cm湿度(kg/m2)"]
    years = [2022+(idx+3)//time_step_seq for idx in range(contin_last_date)]
    months = [(idx+3)%time_step_seq+1 for idx in range(contin_last_date)]
    all_data_list = []
    for y,m,val in zip(years,months,val_mat):
        tp_list = [y,m]+val
        all_data_list.append(tp_list)
    pd.DataFrame(all_data_list,columns=columns).to_excel(save_file_path,index=None)
if __name__ == "__main__":
    main()
    
    
    
