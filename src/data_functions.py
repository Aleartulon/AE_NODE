import numpy as np
import torch as tc
import yaml
import json
import numpy as np
import shutil
import csv
import gc
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time

def normalize_field_known_values_field(F, ma, mi):
    """this function applies the max-min normalization to the input F (solution field), based on the maximum ma and the minimum mi.

    Args:
        F (torch.tensor()): takes as input the solution field over time to be normalized (if the solution field has multiple channel each one is normalized independently)
        ma (torch.tensor()): list of maxima (per channel)
        mi (torch.tensor()): list of minima (per channel)

    Returns:
        torch.Tensor: returns a normalized F tensor
    """    
    F_new = F.clone()
    for count in range(ma.size()[0]):
        F_new[:,:,count,...] = (F_new[:,:,count,...]-mi[count])/(ma[count]-mi[count])
    return F_new

def normalize_field_known_values_param(F, ma,mi):
    """this function applies the max-min normalization to the input F (vector of parameters), based on the maximum ma and the minimum mi.

    Args:
        F (torch.tensor()): takes as input the vector of parameters to be normalized (each parameter is normalized independently)
        ma (torch.tensor()): list of maxima (per channel)
        mi (torch.tensor()): list of minima (per channel)

    Returns:
        torch.Tensor: returns a normalized F tensor
    """    
    return (F-mi)/(ma-mi)

def inverse_normalization_field(F,ma,mi, spatial_dimensions): #here one less dimension because the dimension 1 is squeezed in that part of the code
    """takes a normalized input field and applies the inverse normalization to go back to the initial scale

    Args:
        F (torch.tensor()): takes as input the solution field over time to be normalized (if the solution field has multiple channel each one is normalized independently)
        ma (torch.tensor()): list of maxima (per channel)
        mi (torch.tensor()): list of minima (per channel)
        spatial_dimensions (int): number of spatial dimensions of the field

    Returns:
        torch.tensor(): returns the un-normalized input field
    """    
    F_new = F.clone()
    if spatial_dimensions == 1:
        for count in range(ma.size()[0]):
            F_new[...,count,:] = F_new[...,count,:]*(ma[count]-mi[count]) + mi[count]
        return F_new
    elif spatial_dimensions == 2:
        for count in range(ma.size()[0]):
            F_new[...,count,:,:] = F_new[...,count,:,:]*(ma[count]-mi[count]) + mi[count]
        return F_new
    
def save_checkpoint(enco, f , dec, optimizer, scheduler, epoch, loss, loss_coeff_2, start_backprop,full_training_count,filepath):
    """saves the checkpoint of the system

    Args:
        enco (class 'src.architecture.Convolutional_Encoder): Encoder
        f (src.architecture.F_Latent): function f of the ODE of the latent dynamics
        dec (src.architecture.Convolutional_Decoder): Decoder
        optimizer (torch.optim): optimizer
        scheduler (torch.optim.lr_scheduler): scheduler to decrease the learning rate per epoch during the training
        epoch (int): number of epoch when checkpoint saved
        loss (float): minimum reached by the validation function
        loss_coeff_2 (float): coefficient that multiplies L_2^A (when increased dynamically during epochs)
        start_backprop (list): list with values that determine up to which time-step in the past backpropagate the gradients. important to save as start_backprop[1] could be increased dynamically during the training
        full_training_count (int): effective number of epochs (when all loss functions used)
        filepath (str): where to save the checkpoint
    """    
    checkpoint = {
            'enco':enco.state_dict(),
            'f':f.state_dict(),
            'dec':dec.state_dict(),
            'optim':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),
            'epoch' : epoch,
            'loss' : loss,
            'loss_coeff_2': loss_coeff_2,
            'start_backprop': start_backprop,
            'full_training_count' : full_training_count
        }
    tc.save(checkpoint, filepath)


def load_checkpoint(enco, f , dec, optim, scheduler, filepath, device):
    """loads from the saved chackpoint the necessary object to start the training from the same point

    Args:
        enco (class 'src.architecture.Convolutional_Encoder): Encoder
        f (src.architecture.F_Latent): function f of the ODE of the latent dynamics
        dec (src.architecture.Convolutional_Decoder): Decoder
        optim (torch.optim): optimizer
        scheduler (torch.optim.lr_scheduler): scheduler to decrease the learning rate per epoch during the training
        filepath (str): where to save the checkpoint
        device (torch.device): device where the training and validation are done

    Returns:
        src.architecture.Convolutional_Encoder, src.architecture.F_Latent, src.architecture.Convolutional_Decoder, torch.optim.lr_scheduler, int, float, int, list, int :see save_checkpoint
    """    
    checkpoint = tc.load(filepath, map_location=device)
    enco.load_state_dict(checkpoint['enco'])
    f.load_state_dict(checkpoint['f'])
    dec.load_state_dict(checkpoint['dec'])
    optim.load_state_dict(checkpoint['optim'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    loss_coeff_2 = checkpoint['loss_coeff_2']
    start_backprop = checkpoint['start_backprop']
    full_training_count = checkpoint['full_training_count']
        
    return enco, f , dec, optim, scheduler , epoch, loss, loss_coeff_2, start_backprop, full_training_count

def get_max_and_min(dataset, param_size, dim_input, normalization_field_ma, normalization_field_mi, normalization_parameters_ma, normalization_parameters_mi):
    """gets the maxima and the minima for the input fields and for the vector of parameters. In initial_information.yaml, under normalization_field_ma, normalization_field_mi, normalization_parameters_ma
    and normalization_parameters_mi one can decide to turn on or off the normalization (off simply by putting 1.0 as max and 0.0 as min)

    Args:
        dataset (torch.utils.data.dataloader.DataLoader): training dataloader
        param_size (int): number of parameters
        dim_input (list): first dimension is the channels of the solution field, second is the number of spatial dimensions
        normalization_field_ma (list):  if first dimension true, maxima of each dimension of the solution fields are found. If not, dimension 1,2, etc are the maxima of dimension 1,2, etc of the solution field (only one if it is a scalar field)
        normalization_field_mi (list): if first dimension true, minima of each dimension of the solution fields are found. If not, dimension 1,2, etc are the minima of dimension 1,2, etc of the solution field (only one if it is a scalar field)
        normalization_parameters_ma (list): if first dimension true, maxima of each dimension of the vector of parameters are found. If not, dimension 1,2, etc are the maxima of dimension 1,2, etc of the parameter vector
        normalization_parameters_mi (list): if first dimension true, minima of each dimension of the vector of parameters are found. If not, dimension 1,2, etc are the minima of dimension 1,2, etc of the parameter vector

    Returns:
        [tc.tensor(), tc.tensor(), tc.tensor(), tc.tensor()]: [tensor of maxima of fields, tensor of minima of fields, tensor of maxima of parameters, tensor of minima of parameters]
    """    
    # get max and min of field
    num_channels_input = dim_input[0]
    spatial_dim = dim_input[1]

    ma_field = tc.ones( num_channels_input) * (-1e10)
    mi_field = tc.ones( num_channels_input) * (1e10)

    if param_size > 0:
        ma_param = -1e10 * tc.ones(param_size)
        mi_param = 1e10 * tc.ones(param_size)
    else:
        ma_param = tc.tensor([1.0])
        mi_param = tc.tensor([0.0])

    for field, dt, param in dataset:
        for j in range(num_channels_input):
            if spatial_dim == 1:
                check_ma_field = tc.max(field[:,:,j,:])
                check_mi_field = tc.min(field[:,:,j,:])
                
            elif spatial_dim == 2:
                check_ma_field = tc.max(field[:,:,j,:,:])
                check_mi_field = tc.min(field[:,:,j,:,:])     

            if check_ma_field > ma_field[j]:
                ma_field[j] = check_ma_field

            if check_mi_field < mi_field[j]:
                mi_field[j] = check_mi_field

        if param_size > 0:
            for i in range(param_size):
                check_ma_param = tc.max(param.reshape(-1, param_size)[:,i])
                check_mi_param = tc.min(param.reshape(-1, param_size)[:,i])	
                if check_ma_param > ma_param[i]:
                    ma_param[i] = check_ma_param

                if check_mi_param < mi_param[i]:
                    mi_param[i] = check_mi_param

    if not normalization_field_ma[0]:
        for j in range(num_channels_input):
            ma_field[j] = normalization_field_ma[j+1]
            mi_field[j] = normalization_field_mi[j+1]

    if not normalization_parameters_ma[0]:
        for j in range(param_size):
            ma_param[j] = normalization_parameters_ma[j+1]
            mi_param[j] = normalization_parameters_mi[j+1]

    return [ma_field.clone().detach(), mi_field.clone().detach(), ma_param.clone().detach(), mi_param.clone().detach()]

class CustomStarDataset(Dataset):
    """builds the dataset by loading the whole dataset on the gpu. Usefull only if small dataset

    Args:
        Dataset (type):
    """    

    # This loads the data and converts it, make data rdy
    def __init__(self,file_path_field,file_path_parameter):
        """initialization path data

        Args:
            file_path_field (str): path to field data
            file_path_parameter (str): path to parameter data
        """        
        self.fields = tc.tensor(np.load(file_path_field))
        self.params = tc.tensor(np.load(file_path_parameter))
        
    
    # This returns the total amount of samples in your Dataset
    def __len__(self):
        """ returns length of the dataset

        Returns:
            int: length of the dataset
        """        
        return len(self.fields)
    
    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):

        """gets the batch at index idx

        Args:
            idx (int): index of batch

        Returns:
            torch.tensor(), torch.tensor(): returns fields and params of batch at index idx
        """        
        
        return self.fields[idx],self.params[idx]

class CustomStarDataset_Big_Dataset(Dataset):
    """builds the dataset without the whole dataset on the gpu. Usefull when whole dataset does not fit in gpu memory

    Args:
        Dataset (type): 

    """    
    # This loads the data and converts it, make data rdy
    def __init__(self,file_path_field,file_path_parameter, dim_param, time_dependence_in_f):
        """initialization path data, dimension of parameters vector and boolean on time dependence in f

        Args:
            file_path_field (str): path to field data
            file_path_parameter (str): path to parameter data
            dim_param (int): dimension of latent space
            time_dependence_in_f (bool): if true, f also depends on time
        """        
        self.fields = tc.tensor(np.load(file_path_field,mmap_mode='r'))
        self.params = tc.tensor(np.load(file_path_parameter,mmap_mode='r'))
        self.dim_param = dim_param
        self.size = self.fields.size()
        self.time_dependence_in_f = time_dependence_in_f
    
    # This returns the total amount of samples in your Dataset
    def __len__(self):
        """ returns length of the dataset

        Returns:
            int: length of the dataset
        """         
        return self.size[0]
    
    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        """gets the batch at index idx

        Args:
            idx (int): index of batch

        Returns:
            torch.tensor(), torch.tensor(): returns fields and params of batch at index idx
        """        
        if self.dim_param > 0:
            if self.time_dependence_in_f:
                params = self.params[idx][0:self.dim_param-1]*tc.ones(self.size[1]).unsqueeze(-1) #-1 if time is considered as parameter
            else:
                params = self.params[idx][0:self.dim_param]*tc.ones(self.size[1]).unsqueeze(-1)
            if self.time_dependence_in_f:
                time = tc.arange(0,2.05,0.05).unsqueeze(-1)
                params = tc.cat((params,time), dim=-1) 
        else:
            params = tc.zeros(self.size[1]).unsqueeze(-1)
        dt = tc.ones(self.size[1]-1)*self.params[idx][-1]
        return self.fields[idx] ,dt , params
