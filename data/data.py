import pandas as pd
import numpy as np

def select_wavelength(df, wavelength_range):
    if wavelength_range[0] == None:
        wavelength_range[0] = df.columns.min()
    if wavelength_range[1] == None:
        wavelength_range[1] = df.columns.max()
    df = df.loc[:, (df.columns >= wavelength_range[0]) &
                (df.columns <= wavelength_range[1])]
    return df

def select_delay(df, delay_range):
    if delay_range[0] == None:
        delay_range[0] = df.index.min()
    if delay_range[1] == None:
        delay_range[1] = df.index.max()
    df = df.loc[(df.index >= delay_range[0]) &
                (df.index <= delay_range[1]), :]
    return df


def read_file(file, file_type="raw", inf_handle=False, wavelength_range=None, delay_range=None):
    print("file_type:",file_type)
    if file_type == "raw":
        df = pd.read_csv(file, index_col=0)
        try:
            df = df.iloc[:-11, :]
            df = df.T
            df.index = df.index.str.replace("0.000000.1", "0")
            df.index = pd.to_numeric(df.index)
            df.columns = pd.to_numeric(df.columns)
        except:
            print("Error processing raw file. Please check the file format.")
            return None
    elif file_type == "handle":
        df = pd.read_csv(file, index_col=0, header=0, sep="\t")
        df = df.T
        df.index = df.index.str.replace("0.000000000E+0.1", "0")#将数据0点改为0
        df.index = df.index.str.replace("0.00000E+0.1", "0")#bug暂修，后续修复
        
        #df.iloc[0,0] = 0
        df.index = pd.to_numeric(df.index)
        df.columns = pd.to_numeric(df.columns)
    if inf_handle:
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill(axis=0)
    #选择波长范围和时间范围
    df = select_wavelength(df, wavelength_range)
    df = select_delay(df, delay_range)
    #标签小数点后保留两位
    df.index = df.index.map(lambda x: round(x,2))
    df.columns = df.columns.map(lambda x: round(x,2))
    return df

if __name__ == "__main__":
    # Example usage
    file_path = "data/TAS/TA_Average-crop-TCA-tzs-tzs-tzs-chirp.csv"
    df = read_file(file_path, file_type="handle", inf_handle=True, wavelength_range=(400, 800), delay_range=(0, 10))
    print(df.head())