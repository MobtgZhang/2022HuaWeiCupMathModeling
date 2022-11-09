import  os
import argparse
import  numpy as np
import pandas as pd
import  torch
import matplotlib.pyplot as plt

from ntm import NTMCell

class TaskDataset(torch.utils.data.Dataset):
    def __init__(self,dataset,time_seq_len=12):
        super(TaskDataset,self).__init__()
        self.time_seq_len = time_seq_len
        self.dataset = dataset
    def __getitem__(self,idx):
        prev = self.dataset[idx:idx+self.time_seq_len,:]
        prev = (prev - prev.mean(axis=0))/(prev.std(axis=0)+0.01)
        post = self.dataset[idx+self.time_seq_len:idx+2*self.time_seq_len,:]
        post = (post - post.mean(axis=0))/(post.std(axis=0)+0.01)
        return prev,post
    def __len__(self):
        return len(self.dataset) - 2*self.time_seq_len
def to_var(value,device):
    return torch.tensor(value,dtype=torch.float).to(device)
def evalate_model(model,test_loader,loss_fn,device):
    loss_list = []
    for epoch,(x,y) in enumerate(test_loader):
        x = x.float().to(device)
        y = y.float().to(device)
        # train

        inp_seq_len = x.size(1)
        batchsz,outp_seq_len, _ = y.size()
        
        # new sequence
        model.zero_state(batchsz)

        # feed the sequence + delimiter
        for i in range(inp_seq_len):
            #print(x[i].shape)
            model(x[:,i,:])

        # read the output (no input given)
        pred = torch.zeros(y.size()).to('cuda')
        for i in range(outp_seq_len):
            pred[:,i,:], _ = model(None)

        # pred: [seq_len, b, seq_sz]
        loss = loss_fn(pred, y)
        loss_list.append(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
    return sum(loss_list)/len(loss_list)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctrlr-sz",type=int,default=50)
    parser.add_argument("--ctrlr-layers",type=int,default=1)
    parser.add_argument("--num-heads",type=int,default=1)
    parser.add_argument("--memory-N",type=int,default=10)
    parser.add_argument("--memory-M",type=int,default=20)
    parser.add_argument("--seq-max-len",type=int,default=123)
    parser.add_argument("--num-epoches",type=int,default=200)
    parser.add_argument("--batchsz",type=int,default=10)
    parser.add_argument("--train-per",type=float,default=0.7)
    parser.add_argument("--time-step-seq",type=float,default=12)
    parser.add_argument("--result-dir",type=str,default="./result")
    parser.add_argument("--log-dir",type=str,default="./log")
    args = parser.parse_args()
    return args
def train():
    args = get_args()
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)        
    model_path = os.path.join(args.log_dir,"ntm-best.pt")
    save_file_path = os.path.join(args.log_dir,"ntm-result.xlsx")
    save_fig_path = os.path.join(args.log_dir,"ntm-loss.png")
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    val_list = ["wet","evaporation","pressure","rain","temperate","visibility","wind"]
    all_dataset = []
    for val_name in val_list:
        load_file_name = os.path.join(args.result_dir,"%s.npz"%val_name)
        value = np.load(load_file_name)
        value = value["data"][:args.seq_max_len,:]
        all_dataset.append(value)
    all_dataset = np.hstack(all_dataset).astype(dtype=np.float64)
    
    seq_sz = all_dataset.shape[1]
    train_len = int(args.train_per*len(all_dataset))
    train_dataset = TaskDataset(all_dataset[:train_len,:],args.time_step_seq)
    test_dataset = TaskDataset(all_dataset[:train_len,:],args.time_step_seq)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batchsz,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batchsz,shuffle=True)
    if not os.path.exists(model_path):
        model = NTMCell(seq_sz,seq_sz, args.ctrlr_sz, args.ctrlr_layers, args.num_heads, args.memory_N, args.memory_M).to(device)
        print(model)
        loss_fn = torch.nn.MSELoss().to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), momentum=0.9, alpha=0.95, lr=1e-4)
        best_loss = float('inf')
        loss_list = []
        for idx in range(args.num_epoches):
            for epoch,(x,y) in enumerate(train_loader):
                x = x.float().to(device)
                y = y.float().to(device)
                # train
                inp_seq_len = x.size(1)
                batchsz,outp_seq_len, _ = y.size()
                # new sequence
                model.zero_state(batchsz)
                # feed the sequence + delimiter
                for i in range(inp_seq_len):
                    model(x[:,i,:])
                # read the output (no input given)
                pred = torch.zeros(y.size()).to(device)

                for i in range(outp_seq_len):
                    pred[:,i,:] , _ = model(None)
                # pred: [seq_len, b, seq_sz]
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
            loss = evalate_model(model,test_loader,loss_fn,device)
            loss_list.append(loss)
            if loss<best_loss:
                best_loss = loss
                torch.save(model,model_path)
            print("Epoch: %d, Loss: %0.4f"%(idx,loss))
        x = np.linspace(0,len(loss_list)-1,len(loss_list))
        plt.plot(x,loss_list)
        plt.savefig(save_fig_path)
        plt.show()
        plt.close()
    else:
        model = torch.load(model_path)
    make_xlsx(model_path,args.result_dir,save_file_path,args.seq_max_len,args.time_step_seq,device)
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
        prev = torch.tensor(prev,dtype=torch.float).unsqueeze(0).to(device)
        
        # (seq_len,fea_dim)
        inp_seq_len = prev.size(1)
        batchsz,outp_seq_len,_ = prev.size()
        # new sequence
        model.zero_state(batchsz)
        # feed the sequence + delimiter
        for i in range(inp_seq_len):
            model(prev[:,i,:])
        # read the output (no input given)
        pred = torch.zeros(prev.size()).to(device)

        for i in range(outp_seq_len):
            pred[:,i,:] , _ = model(None)
        pred = pred.squeeze()
        val = pred[-1,:].cpu().detach().numpy()*s_v+m_v
        re_val_list.append(val[:4])
        prev = prev.squeeze()
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
if __name__ == '__main__':
    train()
 
 
