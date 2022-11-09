import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
class TimeRNNModelCell(torch.nn.Module):
    def __init__(self,val_seq_len):
        super(TimeRNNModelCell,self).__init__()
        self.gru_time1 = torch.nn.GRUCell(val_seq_len,val_seq_len)
        self.gelu = torch.nn.ReLU()
        self.linp = torch.nn.Linear(val_seq_len,val_seq_len)
        self.lin = torch.nn.Linear(val_seq_len,val_seq_len)
        self.gru_time2 = torch.nn.GRUCell(val_seq_len,val_seq_len)
    def forward(self,x):
        y = self.gru_time1(x)
        y = self.gelu(self.linp(y))
        v = torch.sigmoid(self.lin(x))
        z = y*v+x*(1-v)
        z = self.gru_time2(z)
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
def evaluate(model,dataset,time_seq_len,device):
    mse_loss_list = []
    mae_loss_list = []
    length = len(dataset) - 2*time_seq_len+1
    for idx in range(length):
        prev = dataset[idx:idx+time_seq_len,:]
        prev = to_var(prev,device)
        post = dataset[idx+1:idx+time_seq_len+1,:]
        post = to_var(post,device)
        pred = model(prev)
        mse_loss = ((pred-post)**2).mean().cpu().detach().numpy()
        mae_loss = torch.abs(pred-post).mean().cpu().detach().numpy()
        mae_loss_list.append(mae_loss)
        mse_loss_list.append(mse_loss)
    re_dict = {
        "MSE":sum(mse_loss_list)/len(mse_loss_list),
        "MAE":sum(mae_loss_list)/len(mae_loss_list)
    }
    return re_dict
def save_txt_files(file_name,data_list):
    with open(file_name,mode="w") as wfp:
        for val in data_list:
            wfp.write(str(val)+"\n")
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir",default="./result",type=str)
    parser.add_argument("--data-dir",default="./data",type=str)
    parser.add_argument("--log-dir",default="./log",type=str)
    parser.add_argument("--time-seq-len",default=12,type=int)
    parser.add_argument("--seq-max-len",default=123,type=int)
    parser.add_argument("--batch-size",default=5,type=int)
    parser.add_argument("--rmsprop-lr",default=1e-4,type=float)
    parser.add_argument("--rmsprop-momentum",default=0.9,type=float)
    parser.add_argument("--rmsprop-alpha",default=0.95,type=float)
    parser.add_argument("--train-per",default=0.7,type=float)
    parser.add_argument("--train-times",default=1200,type=int)
    args = parser.parse_args()
    return args
def main():
    args = get_args()
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(args.log_dir,"mdl-best.pt")
    save_fig_loss_file = os.path.join(args.log_dir,"mdl-loss.png")
    save_txt_loss_file = os.path.join(args.log_dir,"mdl-loss.txt")
    save_fig_mae_file = os.path.join(args.log_dir,"mdl-mae.png")
    save_txt_mae_file = os.path.join(args.log_dir,"mdl-mae.txt")
    save_fig_mse_file = os.path.join(args.log_dir,"mdl-mse.png")
    save_txt_mse_file = os.path.join(args.log_dir,"mdl-mse.txt")
    save_file_path = os.path.join(args.log_dir,"mdl-reusult.xlsx")
    
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
        model = TimeRNNModelCell(fea_dim).to(device)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.RMSprop(model.parameters(),
                                momentum=args.rmsprop_momentum,
                                alpha=args.rmsprop_alpha,
                                lr=args.rmsprop_lr)
        train_loss_list = []
        dev_mse_list = []
        dev_mae_list = []
        best_loss = 65535.0
        length = len(all_dataset)-2*args.time_seq_len+1
        for epoch in range(args.train_times):
            all_loss = 0.0
            for idx in range(length):
                optimizer.zero_grad()
                prev = all_dataset[idx:idx+args.time_seq_len,:]
                prev = to_var(prev,device)
                post = all_dataset[idx+1:idx+args.time_seq_len+1,:]
                post = to_var(post,device)
                pred = model(prev)
                loss = loss_fn(pred,post)
                all_loss += loss.cpu().item()
                loss.backward()
                optimizer.step()
            all_loss = all_loss/length
            
            re_dict = evaluate(model,all_dataset,args.time_seq_len,device)
            print("The final result ids:%d,MAE:%0.4f,MSE:%0.4f"%(epoch,re_dict["MAE"],re_dict["MSE"]))
            if best_loss>re_dict["MSE"]:
                best_loss = re_dict["MSE"]
                torch.save(model,model_path)
            train_loss_list.append(all_loss)
            dev_mse_list.append(re_dict["MSE"])
            dev_mae_list.append(re_dict["MAE"])
        x = np.linspace(0,len(train_loss_list)-1,len(train_loss_list))
        y = train_loss_list
        plt.plot(x,y)
        plt.savefig(save_fig_loss_file)
        plt.close()
        save_txt_files(save_txt_loss_file,train_loss_list)
        x = np.linspace(0,len(dev_mse_list)-1,len(dev_mse_list))
        y = dev_mse_list
        plt.plot(x,y)
        plt.savefig(save_fig_mse_file)
        plt.close()
        save_txt_files(save_txt_mse_file,dev_mae_list)
        x = np.linspace(0,len(dev_mae_list)-1,len(dev_mae_list))
        y = dev_mae_list
        plt.plot(x,y)
        plt.savefig(save_fig_mae_file)
        plt.close()
        save_txt_files(save_txt_mae_file,dev_mae_list)
        
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
        pred  = model(prev)
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
    
    
    
