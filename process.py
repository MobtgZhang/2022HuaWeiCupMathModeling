import os
import numpy as np
import pandas as pd
def processA1(data_dir,result_dir):
    all_wether_file = os.path.join(data_dir,"附件3、土壤湿度2012—2022年.xlsx")
    all_dataset = pd.read_excel(all_wether_file)
    # 时间序列
    time_list = ["年份","月份"]
    time_series = list(map(lambda x:str(x[0])+"年"+str(x[1])+"月",all_dataset[time_list].values))
    # A.1. 土壤湿度
    wet_list = ["10cm湿度(kg/m2)","40cm湿度(kg/m2)","100cm湿度(kg/m2)","200cm湿度(kg/m2)"]
    wet_val = all_dataset[wet_list].values
    save_file_name = os.path.join(result_dir,"wet.npz")
    np.savez(save_file_name,time_series=time_series,data=wet_val)
    
def processA2(data_dir,result_dir):
    # A.2. 土壤蒸发数据
    all_wether_file = os.path.join(data_dir,"附件4、土壤蒸发量2012—2022年.xlsx")
    all_dataset = pd.read_excel(all_wether_file)
    # 时间序列
    time_list = ["年份","月份"]
    time_series = list(map(lambda x:str(x[0])+"年"+str(x[1])+"月",all_dataset[time_list].values))
    evaporation_list = ["土壤蒸发量(W/m2)","土壤蒸发量(mm)"]
    evaporation_val = all_dataset[evaporation_list].values
    save_file_name = os.path.join(result_dir,"evaporation.npz")
    np.savez(save_file_name,time_series=time_series,data=evaporation_val)
    
def processB(data_dir,result_dir):
    all_wether_file = os.path.join(data_dir,"附件8、锡林郭勒盟气候2012-2022.xlsx")
    all_dataset = pd.read_excel(all_wether_file)
    # 时间序列
    time_list = ["年份","月份"]
    time_series = list(map(lambda x:str(x[0])+"年"+str(x[1])+"月",all_dataset[time_list].values))
    # B.1. 气温因素
    temperate_list = ["平均气温(℃)","平均最高气温(℃)","平均最低气温(℃)","最高气温极值(℃)",
                   "最低气温极值(℃)","平均气温≥18℃的天数","平均气温≥35℃的天数","平均气温≤0℃的天数","平均露点温度(℃)"]
    temperate_val = all_dataset[temperate_list].values
    save_file_name = os.path.join(result_dir,"temperate.npz")
    np.savez(save_file_name,time_series=time_series,data=temperate_val)
    
    # B.2. 降水量因素
    rain_list = ["降水量(mm)","最大单日降水量(mm)","降水天数"]
    rain_val = all_dataset[rain_list].values
    save_file_name = os.path.join(result_dir,"rain.npz")
    np.savez(save_file_name,time_series=time_series,data=rain_val)
    
    # B.3. 气压因素
    pressure_list = ["平均海平面气压(hPa)","最低海平面气压(hPa)","平均站点气压(hPa)"]
    pressure_val = all_dataset[pressure_list].values
    save_file_name = os.path.join(result_dir,"pressure.npz")
    np.savez(save_file_name,time_series=time_series,data=pressure_val)
    
    # B.4. 能见度
    visibility_list = ["平均能见度(km)","最小能见度(km)","最大能见度(km)"]
    visibility_val = all_dataset[visibility_list].values
    save_file_name = os.path.join(result_dir,"visibility.npz")
    np.savez(save_file_name,time_series=time_series,data=visibility_val)
    
    # B.5. 风速
    wind_list = ["平均风速(knots)","平均最大持续风速(knots)","单日最大平均风速(knots)"]
    wind_val = all_dataset[wind_list].values
    save_file_name = os.path.join(result_dir,"wind.npz")
    np.savez(save_file_name,time_series=time_series,data=wind_val)
    
def main():
    """
    有关的几个基本系数
    1. 气温因素
    2. 降水量因素
    3. 气压因素
    4. 能见度
    5. 风速
    """
    result_dir = "./result"
    data_dir = "./data"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    processA1(data_dir,result_dir)
    processA2(data_dir,result_dir)
    processB(data_dir,result_dir)
if __name__ == "__main__":
    main()

