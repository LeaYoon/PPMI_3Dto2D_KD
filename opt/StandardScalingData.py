import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def StandardScalingData(data, keep_dim=True, train=True, scaler=None):
    data = np.array(data).astype(np.float32)
    flattened_data = [d.flatten() for d in data]
    if train is True:
        scaler = StandardScaler()
        scaler.fit(flattened_data)
        scaled_data = scaler.transform(flattened_data)
    else :
        scaled_data = scaler.transform(flattened_data)
    print("scaled_data", scaled_data.shape)

    if keep_dim is True:
        return scaler, scaled_data.reshape(data.shape)
    else:
        return scaler, scaled_data

## MinMaxScaler def
def MinMaxScalingData(data, keep_dim=True, train=True, scaler=None):
    data = np.array(data).astype(np.float32)
    flattened_data = [d.flatten() for d in data]
    if train is True:
        scaler = MinMaxScaler()
        scaler.fit(flattened_data)
        scaled_data = scaler.transform(flattened_data)
    else :
        scaled_data = scaler.transform(flattened_data)
    print("scaled_data", scaled_data.shape)

    if keep_dim is True:
        return scaler, scaled_data.reshape(data.shape)
    else:
        return scaler, scaled_data