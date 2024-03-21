

'''

Koopman Autoencoder Classes:

1) KEDcoder - represents encoder/decoders, essentially nn.sequential, but with more flexibility to add stuff in the future

2) LinearKoopmanLayer - Middle layer that applies frequencies and amplitudes of eigenvalues as parameters, with global fourier frequency finding method available to be called during the training process.

3) KoopmanAE - Encoder -> LinearKoopmanLayer -> Decoder structure, all encompassing class with methods to train, validate and global fourier frequency find built in.

'''

import numpy as np
import torch
from torch import nn
from torch import optim
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from numba import njit


'''
3 ex-methods for the LinearKoopmanLayer Class:

These are helper functions, called from within optimised torchscript staticmethods.
These JIT compiled staticmethods cannot access the classes namespace, hence the helper functions are uglily dumped here.

'''

def insert_rotation_block(jordan_rotation_matrix, amplitude, frequency, top_left_diag : int):
    #inplace adds a rotation block to the provided matrix
    jordan_rotation_matrix[top_left_diag,top_left_diag] = amplitude*torch.cos(frequency)
    jordan_rotation_matrix[top_left_diag,top_left_diag+1] = -amplitude*torch.sin(frequency)
    jordan_rotation_matrix[top_left_diag+1,top_left_diag] = amplitude*torch.sin(frequency)
    jordan_rotation_matrix[top_left_diag+1,top_left_diag+1] = amplitude*torch.cos(frequency)

    
def gen_jordan_rotation_matrix(amplitudes, frequencies, delta_t, device: str):
    #generate full jordan rotation matrix from frequencies
    jordan_rotation_matrix = torch.zeros([2*len(frequencies),2*len(frequencies)]).to(device)
    [insert_rotation_block(jordan_rotation_matrix,amp_freq[0]**delta_t,amp_freq[1]*delta_t,i*2) for i,amp_freq in enumerate(zip(amplitudes,frequencies))]
    return jordan_rotation_matrix


def single_frequency_forward(amplitudes,frequencies,frequency,frequency_index: int,dt,inp,device: str):
    #helper function to calc one frequency estimate from the sample provided
    #make clone to avoid causing issues with shared tensor memory
    #frequencies = frequencies.detach().clone()
    #replace the relevant frequency with the one we are testing
    frequencies[frequency_index] = frequency
    #generate the prediction using the rotation matrix with given frequency and delta_t on x and return
    res = torch.matmul(gen_jordan_rotation_matrix(amplitudes,frequencies,dt,device),inp)
    return res


#Loss func taken not to be an attribute as this was not being removed from memory properly
def LuschLoss(self, full_pred, enc_pred, enc_x, x_and_delta_t, y):
    #loss function used in Lusch Nature paper (each essentially equivalent to MSE)
    standard_loss = torch.square(torch.linalg.vector_norm(full_pred - y,2))/torch.numel(full_pred)
    linear_loss = torch.square(torch.linalg.vector_norm(enc_pred - self.encoder(y),2))/torch.numel(enc_pred)
    reconstruction_loss = torch.square(torch.linalg.vector_norm(self.decoder(enc_x)- x_and_delta_t[0],2))/torch.numel(x_and_delta_t[0])

    #regularisation loss, trying l1 and l2 however Lusch uses l2 only
    l1e = sum([torch.linalg.vector_norm(p,1) for p in self.encoder.parameters()])/sum([torch.numel(p) for p in self.encoder.parameters()])
    l1d = sum([torch.linalg.vector_norm(p,1) for p in self.decoder.parameters()])/sum([torch.numel(p) for p in self.decoder.parameters()])
    l1_autoencoder_regularisation = l1e + l1d
    l2e = sum([torch.square(torch.linalg.vector_norm(p,2)) for p in self.encoder.parameters()])/sum([torch.numel(p) for p in self.encoder.parameters()])
    l2d = sum([torch.square(torch.linalg.vector_norm(p,2)) for p in self.decoder.parameters()])/sum([torch.numel(p) for p in self.decoder.parameters()])
    l2_autoencoder_regularisation = l2e + l2d
    regularisation_loss = l1_autoencoder_regularisation + l2_autoencoder_regularisation

    #extra stability loss (which is a Steve special)
    stability_loss = nn.L1Loss()(self.linear_koopman_layer.amplitudes,torch.ones(self.num_frequencies).to(self.device))
    return self.loss_hyperparameters[0]*standard_loss + self.loss_hyperparameters[1]*linear_loss + self.loss_hyperparameters[2]*reconstruction_loss + self.loss_hyperparameters[3]*stability_loss + self.loss_hyperparameters[4]*regularisation_loss



class KEDcoder(nn.Module):

    #Takes in a module list in order that the encoder layer will follow
    def __init__(self,module_list):
        super().__init__()
        self.module_list = module_list
    
    #define one forward pass, this includes the reshaping and deshaping of the model
    def forward(self,x):
        for module in self.module_list:
            x = module(x)
        return x

    

class LinearKoopmanLayer(nn.Module):
    
    '''
    Middle layer where the frequency stuff happens, inbetween the encoding and decoding
    '''
    
    def __init__(self,num_frequencies):
        #initialise trainable parameters
        super().__init__()
        self.num_frequencies = num_frequencies
        #set frequency params as nn.params to allow optimising, then initialise
        self.frequencies = nn.Parameter(torch.zeros(self.num_frequencies))
        self.amplitudes = nn.Parameter(torch.zeros(self.num_frequencies))
        nn.init.uniform_(self.frequencies,0,2*np.pi)
        #also init the frequencies at some point
        '''
        REMINDER:
        When you inevitably forget why the initial frequency is always pi and init amp is 1, look here to find out why!!!
        '''
        #self.frequencies.data = torch.tensor([np.pi]*self.num_frequencies)
        self.amplitudes.data = torch.tensor([1.0]*self.num_frequencies)
        
        self.frequency_tracker = [self.frequencies.detach().clone()]
        self.amplitude_tracker = [self.frequencies.detach().clone()]
        

        
    ''' PRE JIT IMPLEMENTATION
    
    def forward(self, x, delta_t):
        #perform froward pass through the network
        #minibatch with different delta_t for each x, hence each needs it's own jordan rotation matrix
        res = [torch.matmul(self.gen_jordan_rotation_matrix(self.amplitudes,self.frequencies,dt),inp) for inp,dt in zip(x,delta_t)]
        return torch.stack(res)

    
    @classmethod
    def gen_jordan_rotation_matrix(cls, amplitudes, frequencies, delta_t: int):
        #generate full jordan rotation matrix from frequencies
        jordan_rotation_matrix = torch.zeros([2*len(frequencies),2*len(frequencies)])
        [cls.insert_rotation_block(jordan_rotation_matrix,amp_freq[0]**delta_t,amp_freq[1]*delta_t,i*2) for i,amp_freq in enumerate(zip(amplitudes,frequencies))]
        return jordan_rotation_matrix

    @staticmethod
    def insert_rotation_block(jordan_rotation_matrix, amplitude, frequency, top_left_diag : int):
        #inplace adds a rotation block to the provided matrix
        jordan_rotation_matrix[top_left_diag,top_left_diag] = amplitude*torch.cos(frequency)
        jordan_rotation_matrix[top_left_diag,top_left_diag+1] = -amplitude*torch.sin(frequency)
        jordan_rotation_matrix[top_left_diag+1,top_left_diag] = amplitude*torch.sin(frequency)
        jordan_rotation_matrix[top_left_diag+1,top_left_diag+1] = amplitude*torch.cos(frequency)'''
    
    
    #ugly but faster jit implementation    
    def forward(self, x, delta_t):
        #perform froward pass through the network
        #minibatch with different delta_t for each x, hence each needs it's own jordan rotation matrix
        res = self._lkl_jit_forward(self.amplitudes, self.frequencies, x, delta_t,self.device)
        return res
    
    
    @staticmethod
    @torch.jit.script
    def _lkl_jit_forward(amplitudes,frequencies,x,delta_t,device: str):
        res = [torch.matmul(gen_jordan_rotation_matrix(amplitudes,frequencies,dt,device),inp) for inp,dt in zip(x,delta_t)]
        return torch.stack(res)

    
    ''' PRE JIT IMPLEMENTATION
    
    def global_frequency_forward(self,x,delta_t,frequency_index,sample_num):
            #generate the frequency intervals to evaluate based on the number of samples (evenly spaced points to calc the error)
            #use abs(dt), as dt could be negative if trying to do something like consistent KAE
            frequency_samples = [np.linspace(0,2*np.pi/abs(dt),sample_num) for dt in delta_t]
            #init list to store each x's losses for each frequency in
            results = []
            #for each x, delta_t pair calculate forward pass over all sample frequencies
            #this bit looks like a prime NUMBA candidate
            for freq_samps, inp, dt in zip(frequency_samples,x,delta_t):
                res = torch.stack([self.single_frequency_forward(freq,frequency_index,dt,inp) for freq in freq_samps])
                results.append(res)
            #stack into a tensor and return
            results = torch.stack(results)
            return results
        
            
    def single_frequency_forward(self,frequency,frequency_index,dt,inp):
        #helper function to calc one frequency estimate from the sample provided
        #make clone to avoid causing issues with shared tensor memory
        frequencies = self.frequencies.detach().clone()
        #replace the relevant frequency with the one we are testing
        frequencies[frequency_index] = frequency
        #generate the prediction using the rotation matrix with given frequency and delta_t on x and return
        res = torch.matmul(self.gen_jordan_rotation_matrix(self.amplitudes,frequencies,dt),inp)
        return res'''
    
    def global_frequency_forward(self,x,delta_t,frequency_index,sample_num):
        results = self._lkl_jit_global_frequency_forward(self.amplitudes.detach().clone(), self.frequencies.detach().clone(),x,delta_t,frequency_index,sample_num,self.device)
        return results
    
    
    @staticmethod
    @torch.jit.script
    def _lkl_jit_global_frequency_forward(amplitudes,frequencies,x,delta_t,frequency_index:int,sample_num: int,device: str):
        #generate the frequency intervals to evaluate based on the number of samples (evenly spaced points to calc the error)
        #use abs(dt), as dt could be negative if trying to do something like consistent KAE
        frequency_samples = [torch.linspace(0,2*torch.pi/abs(dt),sample_num) for dt in delta_t]
        #init list to store each x's losses for each frequency in
        results = []
        #for each x, delta_t pair calculate forward pass over all sample frequencies
        for freq_samps, inp, dt in zip(frequency_samples,x,delta_t):
            res = torch.stack([single_frequency_forward(amplitudes,frequencies,freq,frequency_index,dt,inp,device) for freq in freq_samps])
            results.append(res)
        #stack into a tensor and return
        results = torch.stack(results)
        return results
    
    
    #Numba could optimise but orders of magnitudes less than current speed bottlenecks so maybe in future
    def fourier_resampling(self,losses,delta_t,sample_num):
        #helper function to resample global frequency losses efficiently
        #take fft of all loss intervals
        losses_ft = np.fft.fft(losses)
        #find max delta_t to know how large scaling is required (abs used as dt could be negative for backwards operator)
        delta_t = delta_t.cpu().detach().numpy()
        max_delta_t = np.max(abs(delta_t))
        #create array to store global loss fft in
        E_ft = np.zeros(max_delta_t*sample_num, dtype=np.complex64)
        #for each loss interval, rescale frequency in line with delta_t's value
        for lft,dt in zip(losses_ft,delta_t):
            E_ft[np.arange(int(np.ceil(sample_num/2)))*abs(dt)] += lft[:(int(np.ceil(sample_num/2)))]
        #add flipped and conjugated back half to ensure fft has correct properties (with 0 frequency value removed)
        E_ft = np.concatenate([E_ft, np.conj(np.flip(E_ft, -1))])[:-1]
        #inverse fft and make real to eliminate imag rounding errors to produce final global loss surface
        E = np.real(np.fft.ifft(E_ft))
        #Use Henning heuristic due to unkown phase problem
        E = -np.abs(E-np.median(E))
        return E
    
    
    
class KoopmanAE(nn.Module):
    #Huge class that contains all Koopman AutoEncoder functionality appropriately modularlised
    
    def __init__(self,input_size,num_frequencies,encoder_module_list,decoder_module_list,multi_freq_tol=0):
        
        '''
        Inputs input size and number of frequencies are as expected
        module_list should contain a list of modules in the order you intend them to be applied in the encoder phase
        They will then be applied in the reverse in the networks decoder phase
        The first and last dimensions in the module list will automatically be updated to match the input size
        and 2*num_frequencies respectively
        '''
        
        #call nn.Module initialisation as required to register parameters etc
        super().__init__()
        
        #Initialise input variables as attributes where appropriate
        self.input_size = input_size
        self.num_frequencies = num_frequencies
        self.multi_freq_tol = multi_freq_tol    #tolerance when training of how close frequencies can be (abs value)
        
        #create encoder and decoders (with corrected feature sizes), and make torchscript (fast) modules
        #very possible source of error in the fututre
        encoder_module_list[0] = encoder_module_list[0].__class__(input_size,encoder_module_list[0].out_features)
        encoder_module_list[-1] = encoder_module_list[-1].__class__(encoder_module_list[-1].in_features,num_frequencies*2)
        decoder_module_list[0] = decoder_module_list[0].__class__(num_frequencies*2,decoder_module_list[0].out_features)
        decoder_module_list[-1] = decoder_module_list[-1].__class__(decoder_module_list[-1].in_features,input_size)
        self.encoder = KEDcoder(encoder_module_list)
        self.decoder = KEDcoder(decoder_module_list)
        self.encoder = torch.jit.script(self.encoder)
        self.decoder = torch.jit.script(self.decoder)
        #create linear middle layer where the frequencies will be applied
        self.linear_koopman_layer = LinearKoopmanLayer(self.num_frequencies)


    def forward(self,x_and_delta_t):
        #define one forward pass, this includes the reshaping and deshaping of the model
        #split x and delta_t up
        x = x_and_delta_t[0]
        delta_t = x_and_delta_t[1]
        
        #apply model for 1 forward pass and return output
        enc_x = self.encoder(x)
        enc_pred = self.linear_koopman_layer(enc_x,delta_t)
        full_pred = self.decoder(enc_pred)
        return full_pred, enc_pred, enc_x
    
    
    def set_training_attributes(self,opt,lossfunc,loss_hyperparameters,train_dataloader,val_dataloader=None,svd_init=None):
        #attaches an optimiser and loss function to the KAE class for convenience when training, and inits
        self.opt = opt
        self.lossfunc = lossfunc
        self.loss_hyperparameters = loss_hyperparameters
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        #init the encoder/decoder as Ur^T and Ur of the svd respectively (best linear dimension reduction)
        self.svd_init=None
        if svd_init is not None:
            if type(svd_init) is torch.Tensor:
                self.svd_init = svd_init.clone().detach()
            else:
                self.svd_init = torch.tensor(svd_init)
            U_r = self.svd_init[:,:self.encoder.module_list[0].out_features]
            state_dict = self.encoder.state_dict()
            for param_name in state_dict.keys():
                if param_name[-7:] == '.weight':
                    break
            state_dict[param_name] = U_r.T 
            #+ state_dict[param_name] #(tests currently say just do svd no noise as default noise is too large)
            self.encoder.load_state_dict(state_dict)
            state_dict = self.decoder.state_dict()
            for param_name in reversed(state_dict.keys()):
                if param_name[-7:] == '.weight':
                    break
            state_dict[param_name] = U_r 
            #+ state_dict[param_name] #(tests currently say just do svd no noise as default noise is too large)
            self.decoder.load_state_dict(state_dict)
            
            
        '''very experimental setting decoder weights to be transpose of encoders'''
        #state_dict = self.encoder.state_dict()
        #dec_sd = {}
        #dec_sd['module_list.0.weight'] = state_dict['module_list.2.weight'].T
        #dec_sd['module_list.2.weight'] = state_dict['module_list.0.weight'].T
        #self.decoder.load_state_dict(dec_sd)

        
    def run_training_loop(self,epochs,global_fourier_interval=None,print_interval=1):
        self.print_interval = print_interval
        self.global_fourier_interval = global_fourier_interval
        next_fourier = 0
        #if not set already, create a list to store losses
        if not hasattr(self, 'train_losses'):
            self.train_losses = []
        if not hasattr(self, 'val_losses'):
            self.val_losses = []
        for i in range(epochs):
            #print stuff to keep an eye on what's happening
            if i%print_interval == 0:
                print(i)
                print(self.linear_koopman_layer.amplitudes.data)
                print(self.linear_koopman_layer.frequencies.data)
            #if global fourier mode method is enabled
            if not self.global_fourier_interval is None:
                #if it is the nth epoch
                if i == next_fourier:
                    #gather full dataset via dataloader
                    '''Try optimising the freq/amps using mini batch only
                    full_dataset = [[x_and_delta_t,y] for x_and_delta_t,y in self.train_dataloader]
                    #break down into constituent parts
                    x = torch.vstack([f[0][0] for f in  full_dataset])
                    dt = torch.hstack([f[0][1] for f in  full_dataset])
                    y = torch.vstack([f[1] for f in  full_dataset])
                    x_and_delta_t = [x,dt]'''
                    #now just use mini batch for global fourier instead
                    x_and_delta_t, y = next(iter(self.train_dataloader))
                    x_and_delta_t = [i.to(self.device) for i in x_and_delta_t]
                    y = y.to(self.device)
                    
                    #apply the global fourier finding technique
                    self.global_frequency_finder(x_and_delta_t,y)
                    #increase global fourier interval value by 10% to allow for more stable later training
                    self.global_fourier_interval = int(np.ceil(1.1 * self.global_fourier_interval))
                    #set next timestep to perform fourier as such
                    next_fourier += self.global_fourier_interval
                    
            #records frequencies in linear koopman layer
            self.linear_koopman_layer.frequency_tracker.append(self.linear_koopman_layer.frequencies.detach().clone())
            
            mb_train_loss = 0
            #perform standard mini batch gradient descent and record losses
            for x_and_delta_t, y in self.train_dataloader:
                x_and_delta_t = [i.to(self.device) for i in x_and_delta_t]
                y = y.to(self.device)
                mb_train_loss += self.train_mini_batch(x_and_delta_t,y)
            self.train_losses.append(mb_train_loss)
                
            #perform validation loss calculation and record
            mb_val_loss = []
            if self.val_dataloader:
                for x_and_delta_t, y in self.val_dataloader:
                    x_and_delta_t = [i.to(self.device) for i in x_and_delta_t]
                    y = y.to(self.device)
                    mb_val_loss.append(self.mini_batch_validation_loss(x_and_delta_t,y))
                mean_mb_val_loss = np.mean(mb_val_loss)
                self.val_losses.append(mean_mb_val_loss)
            if i%print_interval == 0:
                print(f'Train Loss: {mb_train_loss}')
                if self.val_dataloader:
                    print(f'Val Loss: {mean_mb_val_loss}')

            
    def train_mini_batch(self,x_and_delta_t,y):
        #trains 1 minibatch with gradient descent
        self.opt.zero_grad()
        full_pred, enc_pred, enc_x = self(x_and_delta_t)
        loss = self.lossfunc(self,full_pred, enc_pred, enc_x, x_and_delta_t,y)
        loss.backward()
        self.opt.step()
        return float(loss)
    
    
    def mini_batch_validation_loss(self,x_and_delta_t,y):
        with torch.no_grad():
            full_pred, enc_pred, enc_x = self(x_and_delta_t)
            #Use MSE for validation loss for now
            loss = torch.square(torch.linalg.vector_norm(full_pred - y,2))/torch.numel(full_pred)
            return float(loss)
        

    def global_frequency_finder(self,x_and_delta_t,y,sample_num=100):
        #at some point loop needed to go through all frequency indexs
        #turn off gradients, as we are optimising gloablly not using SGD
        with torch.no_grad():
            #split x and delta_t up
            x = x_and_delta_t[0]
            delta_t = x_and_delta_t[1]
            for frequency_index in range(len(self.linear_koopman_layer.frequencies)):
                #loop through all frequency indexs, encoding then decoding as required
                enc_x = self.encoder(x)
                enc_pred = self.linear_koopman_layer.global_frequency_forward(enc_x,delta_t,frequency_index,sample_num)
                #Previous, heavy memory implementation requires mem for:(number of states * state size * sample_num)
                '''full_pred = self.decoder(enc_pred)
                losses = self._frequency_big_loss_loop(full_pred.detach().numpy(),y.detach().numpy())'''
                #New implementation only requires memory for: (number of states * state size)
                #Create loss array at each sampled frequency
                #for now we just use MSE to keep things simple although may edit this going forward
                losses = []
                for enc_pred_sample in torch.split(enc_pred,split_size_or_sections=1,dim=1):
                    full_pred_sample = self.decoder(enc_pred_sample)
                    loss_sample = self._frequency_big_loss_loop(full_pred_sample.cpu().detach().numpy(),y.cpu().detach().numpy())
                    losses.append(loss_sample)
                losses = np.hstack(losses)
                
                #resample and combine losses efficiently using fft
                global_loss_surface = self.linear_koopman_layer.fourier_resampling(losses,delta_t,sample_num)

                #determine argmin frequency for global loss
                ''' Previously just chose argmin, now implement checking for duplicate frequencies
                minimising_freq = (np.argmin(global_loss_surface)/(global_loss_surface.shape[0]-1))*2*np.pi
                print(minimising_freq)
                #set this as the new frequency at the required index
                self.linear_koopman_layer.frequencies[frequency_index] = minimising_freq'''
                #now we make sure they are not within fractional distance from each other
                current_frequencies = self.linear_koopman_layer.frequencies.data.cpu().detach().numpy()
                current_frequencies[frequency_index] = -1
                current_frequencies = np.hstack([current_frequencies,2*np.pi-current_frequencies])
                sorted_new_frequencies = (np.argsort(global_loss_surface)/(global_loss_surface.shape[0]-1))*2*np.pi
                print(f'First Choice: {sorted_new_frequencies[0]}')
                for minimising_freq in sorted_new_frequencies:
                    if minimising_freq != 0:
                        if np.all(np.abs(current_frequencies - minimising_freq)>self.multi_freq_tol):
                        #if np.all(np.abs(2*np.pi/(current_frequencies) - 2*np.pi/(minimising_freq))>0.1):
                            self.linear_koopman_layer.frequencies[frequency_index] = minimising_freq
                            break
                print(f'New Frequency: {minimising_freq}')
                
                
    @staticmethod
    @njit
    def _frequency_big_loss_loop(full_pred,y):
        losses = np.array([np.linalg.norm(out-y[i])**2/len(out) for i,row in enumerate(full_pred) for out in row]).reshape(full_pred.shape[:-1])
        return losses
        
    
    def hyperparameter_optimiser(self,space):
        #this assumes an instance of KAE has already been init with the following attributes set:
        #train dataloader, val dataloader, lossfunc
        print('move old to cpu')
        self.to('cpu')
        #gather variables not passed in the space to be tweaked from current KAE object
        '''if type(self.svd_init) is torch.Tensor:
            svd_init = self.svd_init.clone().detach()
        else:
            svd_init = self.svd_init'''
        train_dataloader = self.train_dataloader
        val_dataloader = self.val_dataloader 
        lossfunc = self.lossfunc

        #initialise Koopman AE, loss function and optimiser and attatch them to the new KAE class
        kae = self.__class__(space['input_size'],space['num_frequencies'],
                             space['encoder_module_list'],space['decoder_module_list'])
        opt = optim.AdamW(kae.parameters(),lr=space['lr'])
        #lossfunc = getattr(kae,self.lossfunc.__name__)
        kae.set_training_attributes(opt,lossfunc,space['loss_hyperparameters'],train_dataloader,val_dataloader=val_dataloader, svd_init=self.svd_init.clone().detach())
        print('move new to gpu')
        #Move to GPU if required
        kae.to(self.device)
        kae.device = self.device
        kae.linear_koopman_layer.device = self.device
        
        #run training loop
        kae.run_training_loop(space['epochs'],global_fourier_interval=space['global_fourier_interval'],print_interval=self.print_interval)

        #collect relevant metrics like minimum loss attained etc and return in the reccommended format for hyperopt fmin
        output_dict = {'loss': np.min(kae.val_losses), 'status': STATUS_OK,
                   'training losses': kae.train_losses, 'validation losses': kae.val_losses}
        return output_dict