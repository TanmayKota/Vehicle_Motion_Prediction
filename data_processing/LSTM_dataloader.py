import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn

class LSTM_data():
    def __init__(self,x,y,dataset):
        self.x=x
        self.y=y
        self.dataset=dataset

    def convert_to_3D(self,xarray,batch_size):
        step=20
        list=np.empty((xarray.shape[0]//20,batch_size,xarray.shape[1]))
        for i in range(0,xarray.shape[0],batch_size):
            if i<xarray.shape[0]//20:
                list[i,:,:]=xarray[i:step,:]
                step=step+20

        xarray=np.array(list)
        return xarray

    def converter(self):
        x=np.hstack((self.x,self.y))
        data_size = len(x)
        train_size = int(self.dataset * data_size)
        x_train = x[:train_size]
        x_val = x[train_size:]

        data_size = len(self.y)
        train_size = int(self.dataset * data_size)
        y_train = self.y[:train_size]
        y_val = self.y[train_size:]

        x_train=self.convert_to_3D(x_train,20)
        y_train=self.convert_to_3D(y_train,20)
        x_val=self.convert_to_3D(x_val,20)
        y_val=self.convert_to_3D(y_val,20)
        
        X_train = torch.tensor(x_train).float()
        Y_train = torch.tensor(y_train).float()
        X_test = torch.tensor(x_val).float()
        y_test = torch.tensor(y_val).float()

        return X_train, Y_train, X_test, y_test

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]