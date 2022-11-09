import os
import argparse
import numpy as np
import pandas as pd

import torch
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir",default="./result",type=str)
    parser.add_argument("--data-dir",default="./data",type=str)
    parser.add_argument("--log-dir",default="./log",type=str)
    args = parser.parse_args()
    return args
def make_dataset(data_dir,result_dir):
    file_name_file = os.path.join(data_dir,"附件15、内蒙古自治区锡林郭勒盟典型草原轮牧放牧样地群落结构监测数据集.xlsx")
    name_list = ["年份","植物种名","放牧小区Block","株/丛数","干重(g)","鲜重(g)"]
    dataset = pd.read_excel(file_name_file,sheet_name="2016-2020物种数据库")
    tp_dataset = dataset[name_list]
    all_data_dict = {}
    for k in range(len(tp_dataset)):
        year = tp_dataset.loc[k,"年份"]
        block = tp_dataset.loc[k,"放牧小区Block"]
        num =  tp_dataset.loc[k,"株/丛数"]
        if pd.isna(num):num = 0
        nweight =  tp_dataset.loc[k,"鲜重(g)"]
        if pd.isna(nweight):nweight = 0
        mweight =  tp_dataset.loc[k,"干重(g)"]
        if pd.isna(mweight):mweight = 0
        tpkey = (year,block)
        if tpkey not in all_data_dict:
            all_data_dict[tpkey] = {
                "鲜重(g)":np.round(nweight*num,2),
                "干重(g)":np.round(mweight*num,2)
            }
        else:
            all_data_dict[tpkey]["鲜重(g)"] += np.round(nweight*num,2)
            all_data_dict[tpkey]["干重(g)"] += np.round(mweight*num,2)
    file_name_file = os.path.join(data_dir,)
    get_dict()
    header_list = ["年份","放牧小区Block","干重(g)","鲜重(g)"]
    all_dataset_list = []
    for tpkey in all_data_dict:
        tp_list = [tpkey[0],tpkey[1],all_data_dict[tpkey]["干重(g)"],all_data_dict[tpkey]["鲜重(g)"]]
        all_dataset_list.append(tp_list)
    all_dataset_list = sorted(all_dataset_list,key=lambda x:x[0])
    save_xlsx_file = os.path.join(result_dir,"biomass.xlsx")
    pd.DataFrame(all_dataset_list,columns=header_list).to_excel(save_xlsx_file,index=None)
def main():
    args = get_args()
    make_dataset(args.data_dir,args.result_dir)
if __name__ == "__main__":
    main()


