

'''

Dataset classes:

1) SimpleSinDataset - most basic dataset class, a sin wave with targets up to m (currently 10) steps ahead.

2) ConstSinDataset - a sin wave with targets up to m (currently 10) steps ahead OR behind.

3) NoisySinDataset - a sin wave with targets up to m (currently 10) steps ahead OR behind AND additive measurement noise.

4) HardSinDataset - a sin waves with targets up to m (currently 10) steps ahead OR behind AND additive measurement noise AND a linear transformation applied to project the data into a higher dimension (currently 100).

5) Hard2SinDataset - 2 sin waves (additional wave has rotation of 2) with targets up to m (currently 10) steps ahead OR behind AND additive measurement noise AND a linear transformation applied to project the data into a higher dimension (currently 1000).

'''

import torch
import numpy as np
from torch.utils.data import Dataset

class SimpleSinDataset(Dataset):
    
    def __init__(self,num_data,theta_start,theta_end,r_start,r_end,obs_cov):
        self.num_data = num_data
        self.theta_start = theta_start
        self.theta_end = theta_end
        self.r_start = r_start
        self.r_end = r_end
        self.obs_cov = obs_cov
        super().__init__()
        self.true_data = self.generate_data()
        self.data = self.true_data
    
    def generate_data(self):
        thetas = np.linspace(self.theta_start,self.theta_end,self.num_data-1)
        rs = np.linspace(self.r_start,self.r_end,self.num_data-1)
        state = np.array([[1],[0]])
        states = [state]
        for r,theta in zip(rs,thetas):
            A = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            state = r*A@state
            states.append(state)
        data = torch.from_numpy(np.array(states,dtype=np.float32)).squeeze()
        return data
        
    def __len__(self):
        return self.num_data
    
    def __getitem__(self,index):
        #change this code for full version with random target index and resmaple index if is equal to len -1
        if index >= len(self) - 11:
            index = index - 12
        target_ind = index + np.random.randint(1,10)
        inp = self.data[index]
        outp = self.data[target_ind]
        delta_t = target_ind - index
        return [inp,delta_t], outp
    
    
    
class ConstSinDataset(Dataset):
    
    def __init__(self,num_data,theta_start,theta_end,r_start,r_end,obs_cov):
        self.num_data = num_data
        self.theta_start = theta_start
        self.theta_end = theta_end
        self.r_start = r_start
        self.r_end = r_end
        self.obs_cov = obs_cov
        super().__init__()
        self.true_data = self.generate_data()
        self.data = self.true_data
    
    def generate_data(self):
        thetas = np.linspace(self.theta_start,self.theta_end,self.num_data-1)
        rs = np.linspace(self.r_start,self.r_end,self.num_data-1)
        state = np.array([[1],[0]])
        states = [state]
        for r,theta in zip(rs,thetas):
            A = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            state = r*A@state
            states.append(state)
        data = torch.from_numpy(np.array(states,dtype=np.float32)).squeeze()
        return data
        
    def __len__(self):
        return self.num_data
    
    def __getitem__(self,index):
        #change this code for full version with random target index and resmaple index if is equal to len -1
        if index >= len(self) - 11:
            index = index - 12
        if index <= 11:
            index = 12
        target_ind = index + np.random.choice([np.random.randint(-10,-1),np.random.randint(1,10)])
        inp = self.data[index]
        outp = self.data[target_ind]
        delta_t = target_ind - index
        return [inp,delta_t], outp


    
class NoisySinDataset(Dataset):
    
    def __init__(self,num_data,theta_start,theta_end,r_start,r_end,obs_cov):
        self.num_data = num_data
        self.theta_start = theta_start
        self.theta_end = theta_end
        self.r_start = r_start
        self.r_end = r_end
        self.obs_cov = obs_cov
        super().__init__()
        self.true_data = self.generate_data()
        noise = np.random.multivariate_normal([0]*2,obs_cov*np.identity(2),num_data)
        #add to add noise
        self.data = self.true_data + torch.from_numpy(noise.astype(np.float32))    
    
    def generate_data(self):
        thetas = np.linspace(self.theta_start,self.theta_end,self.num_data-1)
        rs = np.linspace(self.r_start,self.r_end,self.num_data-1)
        state = np.array([[1],[0]])
        states = [state]
        for r,theta in zip(rs,thetas):
            A = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            state = r*A@state
            states.append(state)
        data = torch.from_numpy(np.array(states,dtype=np.float32)).squeeze()
        return data
        
    def __len__(self):
        return self.num_data
    
    def __getitem__(self,index):
        #change this code for full version with random target index and resmaple index if is equal to len -1
        if index >= len(self) - 11:
            index = index - 12
        if index <= 11:
            index = 12
        target_ind = index + np.random.choice([np.random.randint(-10,-1),np.random.randint(1,10)])
        inp = self.data[index]
        outp = self.data[target_ind]
        delta_t = target_ind - index
        return [inp,delta_t], outp
    
    

class HardSinDataset(Dataset):
    
    def __init__(self,num_data,theta_start,theta_end,r_start,r_end,obs_cov):
        self.num_data = num_data
        self.theta_start = theta_start
        self.theta_end = theta_end
        self.r_start = r_start
        self.r_end = r_end
        self.obs_cov = obs_cov
        super().__init__()
        self.true_data = self.generate_data()
        #generate noise
        noise = np.random.multivariate_normal([0]*100,obs_cov*np.identity(100),num_data)
        #add to have the autoencoder have to find the linear transform back to rotation coordinates
        self.measure_op = torch.rand([2,100])
        self.true_data = torch.matmul(self.true_data,self.measure_op)
        #add to add noise
        self.data = self.true_data + torch.from_numpy(noise.astype(np.float32))    
    
    def generate_data(self):
        thetas = np.linspace(self.theta_start,self.theta_end,self.num_data-1)
        rs = np.linspace(self.r_start,self.r_end,self.num_data-1)
        state = np.array([[1],[0]])
        states = [state]
        for r,theta in zip(rs,thetas):
            A = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            state = r*A@state
            states.append(state)
        data = torch.from_numpy(np.array(states,dtype=np.float32)).squeeze()
        return data
        
    def __len__(self):
        return self.num_data
    
    def __getitem__(self,index):
        #change this code for full version with random target index and resmaple index if is equal to len -1
        if index >= len(self) - 11:
            index = index - 12
        if index <= 11:
            index = 12
        target_ind = index + np.random.choice([np.random.randint(-10,-1),np.random.randint(1,10)])
        inp = self.data[index]
        outp = self.data[target_ind]
        delta_t = target_ind - index
        return [inp,delta_t], outp
    
    
    
class Hard2SinDataset(Dataset):
    
    def __init__(self,num_data,theta_start,theta_end,r_start,r_end,obs_cov):
        self.num_data = num_data
        self.theta_start = theta_start
        self.theta_end = theta_end
        self.r_start = r_start
        self.r_end = r_end
        self.obs_cov = obs_cov
        super().__init__()
        self.true_data = self.generate_data()
        self.theta_start = 2
        self.theta_end = 2
        extra_data = self.generate_data()
        #add to add another frequency
        self.true_data = torch.hstack([self.true_data,extra_data])
        noise = np.random.multivariate_normal([0]*1000,obs_cov*np.identity(1000),num_data)
        #add to have the autoencoder have to find the linear transform back to rotation coordinates
        self.measure_op = torch.rand([4,1000])
        self.true_data = torch.matmul(self.true_data,self.measure_op)
        #add to add noise
        self.data = self.true_data + torch.from_numpy(noise.astype(np.float32))    
    
    def generate_data(self):
        thetas = np.linspace(self.theta_start,self.theta_end,self.num_data-1)
        rs = np.linspace(self.r_start,self.r_end,self.num_data-1)
        state = np.array([[1],[0]])
        states = [state]
        for r,theta in zip(rs,thetas):
            A = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            state = r*A@state
            states.append(state)
        data = torch.from_numpy(np.array(states,dtype=np.float32)).squeeze()
        return data
        
    def __len__(self):
        return self.num_data
    
    def __getitem__(self,index):
        #change this code for full version with random target index and resmaple index if is equal to len -1
        if index >= len(self) - 11:
            index = index - 12
        if index <= 11:
            index = 12
        target_ind = index + np.random.choice([np.random.randint(-10,-1),np.random.randint(1,10)])
        inp = self.data[index]
        outp = self.data[target_ind]
        delta_t = target_ind - index
        return [inp,delta_t], outp