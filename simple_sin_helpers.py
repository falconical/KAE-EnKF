from torch.utils.data import Dataset
import numpy as np
import torch
from libraries.DMDEnKF import DMDEnKF, TDMD, EnKF
import pydmd
import cmath
from libraries.KAE import gen_jordan_rotation_matrix

class SimpleSinDataset(Dataset):
    
    def __init__(self,num_data,num_spinup,theta_start,theta_end,r_start,r_end,obs_cov):
        self.num_data = num_data
        self.num_spinup = num_spinup
        self.theta_start = theta_start
        self.theta_end = theta_end
        self.r_start = r_start
        self.r_end = r_end
        self.obs_cov = obs_cov
        super().__init__()
        self.true_data, self.thetas = self.generate_data()
        noise = np.random.multivariate_normal([0]*2,obs_cov*np.identity(2),num_data)
        #add to add noise
        self.data = self.true_data + torch.from_numpy(noise.astype(np.float32))
        self.data = self.data - torch.mean(self.data,0)
        self.spinup_data = self.data[:self.num_spinup]
        self.filter_data = self.data[self.num_spinup:]
    
    def generate_data(self):
        thetas = np.linspace(self.theta_start,self.theta_end,self.num_data-1)
        rs = np.linspace(self.r_start,self.r_end,self.num_data-1)
        state = np.array([[1],[0]])
        states = [state]
        for r,theta in zip(rs,thetas):
            A = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            state = r*A@state
            states.append(state)
        states = np.hstack(states).T
        states = np.power(states,3)
        states = states - np.mean(states,axis=0)
        data = torch.from_numpy(np.array(states,dtype=np.float32))
        return data, thetas
        
    def __len__(self):
        return self.num_spinup
    
    def __getitem__(self,index):
        #select random datapoint as long as it's not the last one
        if index >= len(self) - 2:
            index = np.random.randint(0,len(self) - 2)
        #then select random target, up to 10 steps ahead but not over spinup size
        target_ind = np.min([index + np.random.randint(1,10),len(self) - 1])
        inp = self.data[index]
        outp = self.data[target_ind]
        delta_t = target_ind - index
        #return the input state, time difference and output state
        return [inp,delta_t], outp
    

    
def apply_dmdenkf(data,num_for_spin_up,rank,system_cov_const,obs_cov_const,eig_cov_const,ensemble_size = 100):
    """
    Initialises the DMDEnKF, and fits it to the provided data

    Parameters
    ----------
    data : numpy.array
        Measurements to fit the DMDEnKF to
    num_for_spin_up : int
        Number of measurements to use training the DMD model in the spin-up phase
    system_cov_const : float
        Float that governs the size of the system state uncertainty covariance matrix (used in filtering step)
    obs_cov_const : float
        Float that governs the size of the measurement uncertainty covariance matrix (used in filtering step)
    eig_cov_const : float
        Float that governs the size of the system eigenvalue uncertainty covariance matrix (used in filtering step)
    ensemble_size : int
        Number of ensemble members to use in the EnKF (default 100)
    
    Returns
    -------
    dmdenkf : DMDEnKF.DMDEnKF
        A fitted DMDEnKF object, with all relevant info to make reconstructions/predictions stored attributes
    """
    
    #Sets up the DMDEnKF wiht relevant matrices, fits and returns full filter
    #DMD Joint EnKF
    f = TDMD()
    f.fit(data[:,:num_for_spin_up],r=rank)

    #Usual dmdenkf setup of inputs
    x_len = f.data.shape[0]
    e_len = f.E.shape[0]
    observation_operator = np.hstack((np.identity(x_len),np.zeros((x_len,e_len))))
    system_cov = np.diag([system_cov_const]*x_len + [eig_cov_const]*e_len)
    observation_cov = obs_cov_const * np.identity(x_len)
    #P0 = np.diag([system_cov_const]*x_len + [eig_cov_const]*e_len)
    P0 = np.real(np.cov(f.Y-(f.DMD_modes@np.diag(f.E)@np.linalg.pinv(f.DMD_modes)@f.X)))
    P0 = np.block([[P0,np.zeros([x_len,e_len])],[np.zeros([e_len,x_len]),np.diag([eig_cov_const]*e_len)]])
    Y = data[:,num_for_spin_up:]
    #Fit DMDEnKF
    dmdenkf = DMDEnKF(observation_operator=observation_operator, system_cov=system_cov,
                          observation_cov=observation_cov,P0=P0,DMD=f,ensemble_size=ensemble_size,Y=None)
    dmdenkf.fit(Y=Y)
    #return dmdenkf
    return dmdenkf



def iterate_streaming_tdmd(data, rank):
    """
    Apply streaming total DMD over each data point received

    Parameters
    ----------
    data : list
        List of measurements to perform Streaming TDMD over
    """
    
    #Applys TDMD over the first data points, adding a new data point each step until all data has had TDMD applied to it
    dmds = []
    for i in range(2,len(data)+1):
        pdmd = pydmd.DMD(rank,rank,tikhonov_regularization=0.01)
        pdmd.fit(data[:i].T)
        dmds.append(pdmd)
    return dmds



def windowed_tdmd(data,rank,window_size):
    """
    Applies Windowed TDMD with a sliding window over all the data received

    Parameters
    ----------
    data : list
        List of measurements to perform Windowed TDMD over
    window_size : int
        The size of the sliding windowed to use for Windowed TDMD
    """
    
    #Applies windowed TDMD with specified window size
    #also should be backwards aligned
    dmds = []
    for i in range(window_size,len(data)+1):
        pdmd = pydmd.DMD(rank,rank,tikhonov_regularization=0.01)
        pdmd.fit(data[i-window_size:i].T)
        dmds.append(pdmd)
    return dmds

def get_dom_eigs_arg(eigs):
    dom_eig = eigs[np.argmax(abs(eigs))]
    dom_arg = cmath.polar(dom_eig)[1]
    return dom_arg

def get_dom_eigs_mod(eigs):
    dom_eig = eigs[np.argmax(abs(eigs))]
    dom_arg = cmath.polar(dom_eig)[0]
    return dom_arg


class HighDimSinDataset(Dataset):
    
    def __init__(self,num_data,num_spinup,theta_start,theta_end,r_start,r_end,obs_cov,data_dim,power):
        self.num_data = num_data
        self.num_spinup = num_spinup
        self.theta_start = theta_start
        self.theta_end = theta_end
        self.r_start = r_start
        self.r_end = r_end
        self.obs_cov = obs_cov
        self.power = power
        super().__init__()
        self.true_data, self.thetas = self.generate_data()
        #apply linear to transform to raise nonlinear system to a high dimension
        self.highdim_transform = torch.rand(2,data_dim)
        self.true_highdim_data = torch.matmul(self.true_data, self.highdim_transform)
        self.true_highdim_data = self.true_highdim_data - torch.mean(self.true_highdim_data,0)
        noise = np.random.multivariate_normal([0]*data_dim,obs_cov*np.identity(data_dim),num_data)
        #add to add noise
        self.data = self.true_highdim_data + torch.from_numpy(noise.astype(np.float32))
        self.spinup_data = self.data[:self.num_spinup]
        self.filter_data = self.data[self.num_spinup:]
    
    def generate_data(self):
        thetas = np.linspace(self.theta_start,self.theta_end,self.num_data-1)
        rs = np.linspace(self.r_start,self.r_end,self.num_data-1)
        state = np.array([[1],[0]])
        states = [state]
        for r,theta in zip(rs,thetas):
            A = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            state = r*A@state
            states.append(state)
        states = np.hstack(states).T
        states = np.power(states,self.power)
        states = states - np.mean(states,axis=0)
        data = torch.from_numpy(np.array(states,dtype=np.float32))
        return data, thetas
        
    def __len__(self):
        return self.num_spinup
    
    def __getitem__(self,index):
        #select random datapoint as long as it's not the last one
        if index >= len(self) - 2:
            index = np.random.randint(0,len(self) - 2)
        #then select random target, up to 10 steps ahead but not over spinup size
        target_ind = np.min([index + np.random.randint(1,10),len(self) - 1])
        inp = self.data[index]
        outp = self.data[target_ind]
        delta_t = target_ind - index
        #return the input state, time difference and output state
        return [inp,delta_t], outp

    
    
def hankelify(data, hankel_dim):       
    #stacks data so that matrix structure is col1: [x1,...,xhankel], col2: [x2,...,xhankel+1], etc
    hankel_list = [data[:,i:-hankel_dim + i + 1] if i+1 != hankel_dim else data[:,i:] for i in range(hankel_dim)]
    #reverses the matrix sections, so newest data is on TOP
    hankel_data = np.vstack(list(reversed(hankel_list)))
    return hankel_data
    
    
#Neaten this code up in similar style to that of DMDEnKF class
def apply_kae_enkf_filter(kae,filter_data,sys_cov_const,param_cov_consts,obs_cov_const,init_cov_const,ensemble_size = 100):
    
    #set x0 and data to filter in latent space with numpy format
    Y = kae.encoder(filter_data).detach().numpy()
    x0 = np.hstack([Y[0],
                    kae.linear_koopman_layer.amplitudes.detach().numpy(),
                    kae.linear_koopman_layer.frequencies.detach().numpy()])
    Y = Y[1:]
    
    #record the size of the state and parameter sections of the filter state
    state_size = Y.shape[-1]
    param_size = len(kae.linear_koopman_layer.frequencies)*2

    #define dynamics and how they act over the state/param combo of each ensemble member
    def system_dynamics(ensemble_member):
        #create ensemble dynamics
        amp = torch.tensor(ensemble_member[-param_size:-int(param_size/2)])
        freq = torch.tensor(ensemble_member[-int(param_size/2):])
        A = gen_jordan_rotation_matrix(amp, freq, 1, kae.device)
        #apply dynamics to ensemble state
        new_ensemble_state = A@ensemble_member[:state_size]
        #reassemble state_param
        new_ensemble_state = torch.hstack([new_ensemble_state,amp,freq])
        return new_ensemble_state
    
    #use covariance constant to define generally diagonal matrices of the appropriate dimension
    param_cov = np.diag(np.hstack([[p]*int(param_size/2) for p in param_cov_consts]))
    observation_operator = np.hstack([np.identity(state_size),np.zeros([state_size,param_size])])
    system_cov = np.vstack([np.hstack([np.identity(state_size)*sys_cov_const,np.zeros([state_size,param_size])])
                            ,np.hstack([np.zeros([param_size,state_size]),param_cov])])
    obs_cov = np.identity(state_size) * obs_cov_const
    P0 = np.vstack([np.hstack([np.identity(state_size)*init_cov_const,np.zeros([state_size,param_size])])
                            ,np.hstack([np.zeros([param_size,state_size]),param_cov])])

    enkf = EnKF(system_dynamics=system_dynamics, observation_operator=observation_operator, system_cov=system_cov,
                observation_cov=obs_cov, x0=x0, P0=P0, ensemble_size=ensemble_size)
    enkf.fit(Y.T)
    
    return enkf



#Neaten this code up in similar style to that of DMDEnKF class
def apply_fullstate_kae_enkf_filter(kae,filter_data,sys_cov_const,param_cov_consts,obs_cov_const,init_cov_const,ensemble_size = 100):
    
    #set x0 and data to filter in latent space with numpy format
    Y = filter_data.detach().numpy()
    x0 = np.hstack([Y[0],
                    kae.linear_koopman_layer.amplitudes.detach().numpy(),
                    kae.linear_koopman_layer.frequencies.detach().numpy()])
    Y = Y[1:]
    
    #record the size of the state and parameter sections of the filter state
    state_size = Y.shape[-1]
    param_size = len(kae.linear_koopman_layer.frequencies)*2

    #define dynamics and how they act over the state/param combo of each ensemble member
    def system_dynamics(ensemble_member):
        #create ensemble dynamics
        amp = torch.tensor(ensemble_member[-param_size:-int(param_size/2)])
        freq = torch.tensor(ensemble_member[-int(param_size/2):])
        A = gen_jordan_rotation_matrix(amp, freq, 1, kae.device)
        #apply dynamics to ensemble state
        with torch.no_grad():
            new_ensemble_state = kae.decoder(A@kae.encoder(torch.tensor(ensemble_member[:state_size].astype(np.float32))))
        #reassemble state_param
        new_ensemble_state = torch.hstack([new_ensemble_state,amp,freq])
        return new_ensemble_state
    
    #use covariance constant to define generally diagonal matrices of the appropriate dimension
    param_cov = np.diag(np.hstack([[p]*int(param_size/2) for p in param_cov_consts]))
    observation_operator = np.hstack([np.identity(state_size),np.zeros([state_size,param_size])])
    system_cov = np.vstack([np.hstack([np.identity(state_size)*sys_cov_const,np.zeros([state_size,param_size])])
                            ,np.hstack([np.zeros([param_size,state_size]),param_cov])])
    obs_cov = np.identity(state_size) * obs_cov_const
    P0 = np.vstack([np.hstack([np.identity(state_size)*init_cov_const,np.zeros([state_size,param_size])])
                            ,np.hstack([np.zeros([param_size,state_size]),param_cov])])

    enkf = EnKF(system_dynamics=system_dynamics, observation_operator=observation_operator, system_cov=system_cov,
                observation_cov=obs_cov, x0=x0, P0=P0, ensemble_size=ensemble_size)
    enkf.fit(Y.T)
    
    return enkf





def p_step_ahead_pred(current_state,p_step,dmd_modes,dmd_eigs,inv_dmd_modes=None):    
    #Basic forward prediction for DMD methods
    if inv_dmd_modes is None:
        inv_dmd_modes = np.linalg.pinv(dmd_modes)
    diag_eigs = np.linalg.matrix_power(np.diag(dmd_eigs),p_step)
    pred = dmd_modes@diag_eigs@inv_dmd_modes@current_state
    return pred

def edmd_list_pred(data,stdmds,p_step):
    preds = []
    data = data[-len(stdmds):]
    for dp,dmd in zip(data,stdmds):
        preds.append(p_step_ahead_pred(dp,p_step,dmd.modes,dmd.eigs))
    return np.real(np.vstack(preds))

def dmdenkf_ensemble_member_p_step_pred(dmdenkf,ensemble_member,p_step):
    state = ensemble_member[:dmdenkf.state_state_size]
    eigs = dmdenkf.kf_param_state_to_eigs(ensemble_member[-dmdenkf.param_state_size:])
    ensemble_member_pred = p_step_ahead_pred(state,10,dmdenkf.DMD.DMD_modes,eigs,inv_dmd_modes=dmdenkf.DMD.inv_DMD_modes)
    return ensemble_member_pred

def dmdenkf_ensembles_p_step_pred(dmdenkf,ensemble,p_step):
    ensemble_pred = np.vstack([dmdenkf_ensemble_member_p_step_pred(dmdenkf,em,p_step) for em in ensemble])
    ensemble_pred = np.mean(ensemble_pred,axis=0)
    return np.real(ensemble_pred)

def kaeenkf_ensemble_member_p_step_pred(kae,ensemble_member,p_step):
    amp = torch.tensor(ensemble_member[-kae.num_frequencies*2:-int(kae.num_frequencies)])
    freq = torch.tensor(ensemble_member[-kae.num_frequencies:])
    A = gen_jordan_rotation_matrix(amp, freq, p_step, kae.device)
    #apply dynamics to ensemble state
    new_ensemble_state = A@ensemble_member[:-kae.num_frequencies*2]
    with torch.no_grad():
        ensemble_member_pred = kae.decoder(new_ensemble_state.float())
    return np.array(ensemble_member_pred)

def kaeenkf_ensembles_p_step_pred(kae,ensemble,p_step):
    ensemble_pred = np.vstack([kaeenkf_ensemble_member_p_step_pred(kae,em,p_step) for em in ensemble])
    ensemble_pred = np.mean(ensemble_pred,axis=0)
    return ensemble_pred

def kaeenkf_fullstate_ensemble_member_p_step_pred(kae,ensemble_member,p_step):
    amp = torch.tensor(ensemble_member[-kae.num_frequencies*2:-int(kae.num_frequencies)])
    freq = torch.tensor(ensemble_member[-kae.num_frequencies:])
    A = gen_jordan_rotation_matrix(amp, freq, p_step, kae.device)
    #apply dynamics to ensemble state
    with torch.no_grad():
        new_ensemble_state = A@kae.encoder(torch.tensor(ensemble_member[:-kae.num_frequencies*2].astype(np.float32)))
        ensemble_member_pred = kae.decoder(new_ensemble_state.float())
    return np.array(ensemble_member_pred)

def kaeenkf_fullstate_ensembles_p_step_pred(kae,ensemble,p_step):
    ensemble_pred = np.vstack([kaeenkf_fullstate_ensemble_member_p_step_pred(kae,em,p_step) for em in ensemble])
    ensemble_pred = np.mean(ensemble_pred,axis=0)
    return ensemble_pred




class HighDimMultiFreqSinDataset(Dataset):
    '''
    same as high dim sin, but takes in lists for theta and r start and end points,
    then sums data generated for each frequency
    '''
    
    def __init__(self,num_data,num_spinup,theta_start,theta_end,r_start,r_end,obs_cov,data_dim,power):
        self.num_data = num_data
        self.num_spinup = num_spinup
        self.theta_starts = theta_start
        self.theta_ends = theta_end
        self.r_starts = r_start
        self.r_ends = r_end
        self.obs_cov = obs_cov
        self.power = power
        super().__init__()
        
        self.true_data = []
        self.thetas = []
        for theta_start, theta_end, r_start, r_end in zip(self.theta_starts, self.theta_ends, self.r_starts, self.r_ends):
            self.theta_start = theta_start
            self.theta_end = theta_end
            self.r_start = r_start
            self.r_end = r_end
            self.onefreq_true_data, self.onefreq_thetas = self.generate_data()
            self.true_data.append(self.onefreq_true_data)
            self.thetas.append(self.onefreq_thetas)
        self.true_data = torch.hstack(self.true_data)
        
        #apply linear to transform to raise nonlinear system to a high dimension
        self.highdim_transform = torch.rand(2*len(self.theta_starts),data_dim)
        self.true_highdim_data = torch.matmul(self.true_data, self.highdim_transform)
        self.true_highdim_data = self.true_highdim_data - torch.mean(self.true_highdim_data,0)
        noise = np.random.multivariate_normal([0]*data_dim,obs_cov*np.identity(data_dim),num_data)
        #add to add noise
        self.data = self.true_highdim_data + torch.from_numpy(noise.astype(np.float32))
        self.spinup_data = self.data[:self.num_spinup]
        self.filter_data = self.data[self.num_spinup:]
    
    def generate_data(self):
        thetas = np.linspace(self.theta_start,self.theta_end,self.num_data-1)
        rs = np.linspace(self.r_start,self.r_end,self.num_data-1)
        state = np.array([[1],[0]])
        states = [state]
        for r,theta in zip(rs,thetas):
            A = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            state = r*A@state
            states.append(state)
        states = np.hstack(states).T
        states = np.power(states,self.power)
        states = states - np.mean(states,axis=0)
        data = torch.from_numpy(np.array(states,dtype=np.float32))
        return data, thetas
        
    def __len__(self):
        return self.num_spinup
    
    def __getitem__(self,index):
        #select random datapoint as long as it's not the last one
        if index >= len(self) - 2:
            index = np.random.randint(0,len(self) - 2)
        #then select random target, up to 10 steps ahead but not over spinup size
        target_ind = np.min([index + np.random.randint(1,10),len(self) - 1])
        inp = self.data[index]
        outp = self.data[target_ind]
        delta_t = target_ind - index
        #return the input state, time difference and output state
        return [inp,delta_t], outp