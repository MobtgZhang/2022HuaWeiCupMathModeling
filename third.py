import os

import pandas as pd
import numpy as np
#线性回归
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import dateutil
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures

def curve_fit(val_lsit,x_data):
    sum_v = 0.0
    for k,(v,x) in enumerate(zip(val_lsit,x_data)):
        sum_v += v*x**k
    return sum_v
def make_pics(result_dir,x_soc,y_soc,x_new,key,name):
    #model = PolynomialFeatures(degree=5)
    x_new = x_new[:,np.newaxis]
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=0, loss='huber')
    model.fit(x_soc,y_soc)
    y_new = model.predict(x_new)
    # 画出各个模型的回归拟合效果
    plt.figure(figsize=(10, 6))
    # 画出训练数据集（用黑色表示）
    plt.scatter(x_soc, y_soc, c="k", s=10, label="Existing data")
    # 画出adbr_1模型（最大迭代次数为1)的拟合效果（用红色表示）
    plt.plot(x_new,y_new, c="r", label="predict", linewidth=1)
    plt.xlabel("year")
    plt.ylabel("target")
    plt.title("%s area, %s method for %s value."%(key[0],key[1],name))
    plt.legend()
    save_fig_name = os.path.join(result_dir,key[0]+"-"+key[1]+"-"+name+"-pic.png")
    plt.savefig(save_fig_name)
    plt.close()
    return y_new
def make_rain(data_dir,result_dir):
    load_file_name = os.path.join(data_dir,"附件8、锡林郭勒盟气候2012-2022.xlsx")
    dataset = pd.read_excel(load_file_name,sheet_name="Sheet1")
    data_dict = {}
    for k in range(len(dataset)):
        year = dataset.loc[k,"年份"]
        rain = dataset.loc[k,"降水量(mm)"]
        if year not in data_dict:
            data_dict[year] = rain
        else:
            data_dict[year] += rain
            data_dict[year] = round(data_dict[year],2)
    x_new = [year for year in data_dict]
    y_new = [data_dict[year] for year in data_dict]
    plt.scatter(x_new, y_new, c="k", s=10,)
    plt.plot(x_new,y_new, c="r", label="precipitation", linewidth=1)
    plt.xlabel("year")
    plt.ylabel("precipitation")
    plt.title("The precipitation in this area.")
    plt.legend()
    save_fig_name = os.path.join(result_dir,"precipitation-pic.png")
    plt.savefig(save_fig_name)
    plt.close()
    save_xlsx_name = os.path.join(result_dir,"precipitation.xlsx")
    val_mat = list(zip(x_new,y_new))
    vals = np.polyfit(x_new,y_new,5)
    print(vals)
    pd.DataFrame(val_mat,columns=["年份","降水量(mm)"]).to_excel(save_xlsx_name,index=None)
def make_soc_sic_totaln(data_dir,result_dir):
    load_file_name = os.path.join(data_dir,"附件14、内蒙古自治区锡林郭勒盟典型草原不同放牧强度土壤碳氮监测数据集.xlsx")
    dataset = pd.read_excel(load_file_name,sheet_name="Sheet1")
    data_dict = {}
    length = 20
    for k in range(len(dataset)):
        year = dataset.loc[k,"year"]
        block = dataset.loc[k,"放牧小区（plot）"]
        intensity = dataset.loc[k,"放牧强度（intensity）"]
        SOC = dataset.loc[k,"SOC土壤有机碳"]
        SIC = dataset.loc[k,"SIC土壤无机碳"]
        AllN = dataset.loc[k,"全氮N"]
        key = (block,intensity)
        tp_dict = {
                "year":year,
                "SOC":SOC,
                "SIC":SIC,
                "All-N":AllN,
            }
        if key not in data_dict:
            data_dict[key] = [tp_dict]
        else:
            data_dict[key].append(tp_dict)
    all_dataset = {}
    for key in data_dict:
        x_soc_raw = [item["year"] for item in data_dict[key]]
        x_soc = []
        old_v = None
        num = 1
        for year in x_soc_raw:
            if old_v is None:
                x_soc += [year]
                old_v = year
            else:
                if old_v == year:
                    num += 1
                else:
                    x_soc += [round(old_v + k/num,2) for k in range(1,num)]
                    num = 1
                    x_soc += [year]
            old_v = year
        y_soc = [item["SOC"] for item in data_dict[key]]
        y_sic = [item["SIC"] for item in data_dict[key]]
        y_alln = [item["All-N"] for item in data_dict[key]]
        x_soc = np.array(x_soc)[:,np.newaxis]
        y_soc = np.array(y_soc)
        y_sic = np.array(y_sic)
        y_alln = np.array(y_alln)
        x_new = np.linspace(2012,2022,length)
        y_new_soc = make_pics(result_dir,x_soc,y_soc,x_new,key,"SOC")
        y_new_sic = make_pics(result_dir,x_soc,y_sic,x_new,key,"SIC")
        y_new_alln = make_pics(result_dir,x_soc,y_alln,x_new,key,"Total-N")
        all_dataset[key] = {
            "SOC":y_new_soc,
            "SIC":y_new_sic,
            "Total-N":y_new_alln,
        }
    save_soil_fiel_name = os.path.join(result_dir,"soil-SOC-SIC-TotalN.npz")
    np.savez(save_soil_fiel_name,dataset=all_dataset)
def main():
    data_dir = "./data"
    result_dir = "./result"
    #make_rain(data_dir,result_dir)
    make_creatures(data_dir,result_dir)
def make_creatures(data_dir,result_dir):
    load_soil_file_name = os.path.join(result_dir,"biomass.xlsx")
    load_precipitation_file_name = os.path.join(result_dir,"precipitation.xlsx")
    precipitation = pd.read_excel(load_precipitation_file_name)
    data_dict = {}
    for k in range(len(precipitation)):
        year = precipitation.loc[k,"年份"]
        preci = precipitation.loc[k,"降水量(mm)"]
        data_dict[year] = preci
    dataset = pd.read_excel(load_soil_file_name)
    all_data_dict = {}
    for k in range(len(dataset)):
        year = dataset.loc[k,"年份"]
        block = dataset.loc[k,"放牧小区Block"]
        mess = dataset.loc[k,"鲜重(g)"]
        if block not in all_data_dict:
            all_data_dict[block] = [(year,round(mess/data_dict[year],2))]
        else:
            all_data_dict[block].append((year,round(mess/data_dict[year],2)))
    print(all_data_dict)
    save_soil_fiel_name = os.path.join(result_dir,"precipitation-biomass.npz")
    np.savez(save_soil_fiel_name,dataset=all_data_dict)
if __name__ == "__main__":
    main()
