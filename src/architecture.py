import numpy as np
import torch as tc
import torch.nn.functional as F
from torch import nn

class Convolutional_Encoder(nn.Module):
    def __init__(self, dim_input, kernel, filters, stride, input_dfnn, output_dfnn, last_activation):
        """Initialization parameters of the Encoder

        Args:
            dim_input (list): first dimension is the channels of the solution field, second is the number of spatial dimensions
            kernel (list): list of the kernel sizes of the encoder per layer
            filters (list): list of the number of filters of the encoder per layer
            stride (list): list of the stride of the encoder per layer
            input_dfnn (int): dimension of the (flattened) vector which is output of the last convolutional layer and input to the linear layer which maps it into the reduced representation
            output_dfnn (int): dimension of the latent space
            last_activation (bool) : if true, after the final linear layer an activation function is used
        """        
        super().__init__()
        
        self.channels = np.concatenate(([dim_input[0]], filters))
        self.size_kernel = [(k - 1) // 2 for k in kernel]  # Adjust padding based on kernel size

        # Activation function (use only one for now)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.activation = self.gelu
        self.tanh = nn.Tanh()

        # Convolutional layers and BatchNorm layers
        self.convolutionals = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.len_kernel = len(kernel)
        self.last_activation = last_activation

        if dim_input[1] == 1:
            for i in range(self.len_kernel):
                if i  == 0 or i == 2:
                    self.convolutionals.append(nn.Conv1d(self.channels[i], filters[i], kernel[i], stride=stride[i], padding=self.size_kernel[i], padding_mode='replicate', bias=True))
                    #self.norm_layers.append(nn.BatchNorm1d(filters[i]))
                else:
                    self.convolutionals.append(nn.Conv1d(self.channels[i], filters[i], kernel[i], stride=stride[i], padding=self.size_kernel[i], padding_mode='replicate', bias=True))

                nn.init.kaiming_uniform_(self.convolutionals[i].weight)  # Xavier initialization

        elif dim_input[1] == 2:
            for i in range(self.len_kernel):
                if i == 0 or i ==1:
                    self.convolutionals.append(nn.Conv2d(self.channels[i], filters[i], kernel[i], stride=stride[i], padding=self.size_kernel[i], padding_mode='replicate', bias=True))
                    #self.norm_layers.append(nn.BatchNorm2d(filters[i]))
                else:
                    self.convolutionals.append(nn.Conv2d(self.channels[i], filters[i], kernel[i], stride=stride[i], padding=self.size_kernel[i], padding_mode='replicate', bias=True))
                    
                nn.init.kaiming_uniform_(self.convolutionals[i].weight)  # Xavier initialization

        # Dense fully-connected layer
        self.dfnn = nn.Linear(input_dfnn, output_dfnn, bias=True)
        nn.init.kaiming_uniform_(self.dfnn.weight)  # Xavier initialization

    def forward(self, x):
        """Forward pass of the Encoder

        Args:
            x (torch.Tensor): input of the Encoder, dimension is [B * T, C, dim_x_1, dim_x_2, ...], where B is batch_size, T is the length of the full time series, C is the number of channels of the 
            solution field, dim_x_1, dim_x_2, ... are the dimensions of the first spatial dimension, second spatial dimension, etc
        Returns:
            torch.Tensor: returns a tensor of size [B, output_dfnn], where B is the batch_size and output_dfnn is the dimension of the latent space
        """        
        for i, conv_layer in enumerate(self.convolutionals):
            x = conv_layer(x)
            x = self.activation(x)
        x = tc.flatten(x, 1)  # Flatten across the batch dimension
        x = self.dfnn(x)
        if self.last_activation:
            x = self.activation(x)
        return x


class F_Latent(nn.Module): 
    def __init__(self, parameter_information, param_dim, latent_dim, n_neurons, n_layers, n_FiLM_conditioning):
        """Initialization parameters of the function f of the latent ODE

        Args:
            parameter_information (str): if 'concatenation', the parameters are just concatenated to the latent vector. if 'FiLM', a FiLM layer is used to give the parameter information to f
            param_dim (int): dimension of parameter vector (without time)
            latent_dim (int): dimension of the latent space
            n_neurons (int): number of neurons per Dense hidden layer of f
            n_layers (int): number of Dense hidden layers that approximate f
            n_FiLM_conditioning (int): number of FiLM layers (if FiLM is chosen instead of concatenation). FiLM = 1 only applies to the latent vector

        Raises:
            ValueError: if parameter_information is neither 'concatenation' nor 'FiLM' an error is raised
        """        
        super().__init__()

        self.param_dim = param_dim
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()
        self.leaky = nn.LeakyReLU()
        self.activation = self.gelu
        self.n_layers = n_layers
        self.parameter_information = parameter_information
        self.n_FiLM_conditioning = n_FiLM_conditioning

        self.norm_layers = nn.ModuleList()
        self.batch_norm = nn.BatchNorm1d(latent_dim)
        self.dropout = nn.Dropout(p=0.5)

        if parameter_information == 'concatenation':
            if n_layers !=1:

                if self.param_dim > 0:
                    self.linears = nn.ModuleList([nn.Linear(latent_dim + param_dim, n_neurons, bias = True)])
                else:
                    self.linears = nn.ModuleList([nn.Linear(latent_dim, n_neurons, bias = True)])

                self.linears.extend([nn.Linear(n_neurons, n_neurons, bias = True) for i in range(n_layers)])
                self.linears.append(nn.Linear(n_neurons, latent_dim, bias = True))

                for i in self.linears:
                    nn.init.kaiming_uniform_(i.weight)
            else:
                if self.param_dim > 0:
                    self.dfnn = nn.Linear(latent_dim + param_dim, latent_dim, bias = True)
                    nn.init.kaiming_uniform_(self.dfnn.weight)
                else:
                    self.dfnn = nn.Linear(latent_dim, latent_dim, bias = True)
                    nn.init.kaiming_uniform_(self.dfnn.weight)


        elif parameter_information == 'FiLM':

            if self.param_dim > 0:
                self.param_FiLM_gamma = nn.ModuleList([nn.Linear(param_dim, latent_dim, bias = True)])
                self.param_FiLM_beta = nn.ModuleList([nn.Linear(param_dim, latent_dim, bias = True)])
                self.param_FiLM_gamma.extend([nn.Linear(param_dim, n_neurons, bias = True) for i in range(n_FiLM_conditioning)])
                self.param_FiLM_beta.extend([nn.Linear(param_dim, n_neurons, bias = True) for i in range(n_FiLM_conditioning)])

            self.linears = nn.ModuleList([nn.Linear(latent_dim, n_neurons, bias = True)])
            self.linears.extend([nn.Linear(n_neurons, n_neurons, bias = True) for i in range(n_layers)])
            self.linears.append(nn.Linear(n_neurons, latent_dim, bias = True))        

            for i in self.linears:
                nn.init.kaiming_uniform_(i.weight)
        else:
            raise ValueError("Wrong name of the type of parameter information")

    def forward(self, x, parameter):
        """forward pass of the f function, which takes as input latent vectors x and parameters parameter. It takes (T-1) snapshots, all of them besides the last one to predict the next one

        Args:
            x (torch.tensor): a tensor of latent vectors of dimension [B*(T-1), latent_dim], where B is batch size, T is the len of the time series and latent dim is the dimension of the latent space.
            parameter (torch.tensor): a tensor of dimension [B*(T-1), dim_param], where B is batch size, T is the len of the time series and dim_param is the number of parameters.

        Returns:
            torch.tensor: a tensor of latent vectors of dimension [B*T, latent_dim], where B is batch size, T is the len of the time series and latent dim is the dimension of the latent space.
            It is the output of the function f.
        """        
        
        if self.parameter_information == 'concatenation':
            if self.param_dim > 0:
                x = tc.cat((x, parameter), dim=1)
            
            for count, i in enumerate(self.linears[0:-1]):
                x = i(x)
                x = self.activation(x) 
            x = self.linears[-1](x)
            return x

        elif self.parameter_information == 'FiLM':
            if self.param_dim > 0:
                parameter_vector_gamma = self.param_FiLM_gamma[0](parameter)
                parameter_vector_beta = self.param_FiLM_beta[0](parameter)
                x = parameter_vector_gamma * x + parameter_vector_beta

            for count, i in enumerate(self.linears[0:-1]):
                x = i(x)
                if (count+1) < self.n_FiLM_conditioning:
                    parameter_vector_gamma = self.param_FiLM_gamma[count+1](parameter)
                    parameter_vector_beta = self.param_FiLM_beta[count+1](parameter)
                    x = parameter_vector_gamma * x + parameter_vector_beta
                x = self.activation(x) 
                
            x = self.linears[-1](x)
            return x

class Convolutional_Decoder(nn.Module):
    def __init__(self , dim_input, kernel, filters, stride, input_dfnn, output_dfnn, final_reduction, initial_activation):
        """Parameters for the initialization of the Decoder

        Args:
            dim_input (list): first dimension is the channels of the solution field, second is the number of spatial dimensions
            kernel (list): list of the kernel sizes of the decoder per layer
            filters (list): list of the number of filters of the decoder per layer
            stride (list): list of the stride of the decoder per layer
            input_dfnn (int): dimension of the vector which is input of the initial linear layer of the decoder, i.e., the latent dimension. 
            output_dfnn (int): dimension of the (flattened) vector which is output of the last convolutional layer of the encoder. It is chosen to be the same to make the decoder symmetric to the encoder
            final_reduction (int): final length of each spatial dimension after repeated reduction (halving) operated by the encoder. Used here to make the decoder symmetric to the encoder
            initial_activation (bool) : if true, after the initial linear layer an activation function is used
        """        
        super().__init__()
        self.filters = filters
        self.channels = np.concatenate((filters[1:],[dim_input[0]]))
        self.transposed_convolutionals = nn.ModuleList([])
        self.size_kernel = int((kernel[0]-1)/2)
        self.dim_input = dim_input[1]
        self.final_reduction = final_reduction

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()
        self.activation = self.gelu
        self.initial_activation = initial_activation

        self.dfnn = nn.Linear(input_dfnn, output_dfnn)
        nn.init.kaiming_uniform_(self.dfnn.weight)

        if self.dim_input == 1:
            for i in range(len(kernel)):
                self.transposed_convolutionals.extend([nn.ConvTranspose1d(filters[i],self.channels[i],kernel[i],stride[i],padding=self.size_kernel, output_padding = 0)])
                nn.init.kaiming_uniform_(self.transposed_convolutionals[i].weight)

        elif self.dim_input == 2:
            for i in range(len(kernel)):
                self.transposed_convolutionals.extend([nn.ConvTranspose2d(filters[i],self.channels[i],kernel[i],stride[i],padding=self.size_kernel, output_padding = 0)])
                nn.init.kaiming_uniform_(self.transposed_convolutionals[i].weight)

    def forward(self,x):
        
        """Forward pass of the decoder

        Args:
            x (torch.tensor): a tensor of latent vectors of dimension [B*T, latent_dim], where B is batch size, T is the len of the time series and latent dim is the dimension of the latent space

        Returns:
            torch.tensor: a tensor of size [B * T, C, dim_x_1, dim_x_2, ...], where B is batch_size, T is the length of the full time series, C is the number of channels of the 
            predicted solution field, dim_x_1, dim_x_2, ... are the dimensions of the first spatial dimension, second spatial dimension, etc
        """        

        x = self.dfnn(x)
        if self.initial_activation:
            x = self.activation(x) #do not use this if relu in encoder and decoder !!!!!!!!

        if self.dim_input == 1:
            x = x.view(x.size()[0], self.filters[0], self.final_reduction)
        elif self.dim_input == 2:
            x = x.view(x.size()[0], self.filters[0], self.final_reduction, self.final_reduction)
    
        length = len(self.transposed_convolutionals)

        for i in range(length-1):
            x = self.transposed_convolutionals[i](x)
            x = self.activation(x)
        x = self.transposed_convolutionals[-1](x)
        
        return x




