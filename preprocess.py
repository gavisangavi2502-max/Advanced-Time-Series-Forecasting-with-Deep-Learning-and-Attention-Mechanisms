
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(path="data.csv"):
    df = pd.read_csv(path)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return df, scaled, scaler
