import pandas as pd
def main():
    load_file = "11.xlsx"
    dataset = pd.read_excel(load_file,header=None)
    print(dataset)
    val_mat = dataset.values
    save_file = "12.xlsx"
    mat = (val_mat - val_mat.min(axis=0))/(val_mat.max(axis=0) - val_mat.min(axis=0))
    print(mat)
    pd.DataFrame(mat).to_excel(save_file,index=None,header=None)
if __name__ == "__main__":
    main()

