import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.architecture import *
from src.training_validation_functions import nn_training
from src.data_functions import *

#set type of tensors
tc.set_default_dtype(tc.float32)

def load_config(config_path: str):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
 
    initial_information = load_config('configs/initial_information.yaml')
    model_information = load_config('configs/model_information.yaml')

    n_halving = len(np.where(np.array(model_information['stride_enc'])==2)[0]) 
    final_reduction = int(initial_information['side_size'] /(2**n_halving)) #final

    if initial_information['dim_input'][1] == 1:
        model_information['input_output_dfnn'] = int(final_reduction * model_information['filters_enc'][-1] ) #dimension of the vector the final linear layer of the encoder receives
    elif initial_information['dim_input'][1] == 2:
        model_information['input_output_dfnn'] = int(final_reduction**2 * model_information['filters_enc'][-1] ) #dimension of the vector the final linear layer of the encoder receives

    #define directories and get data

    initial_information['PATH'] = initial_information['physics_model'] +'/Models/' + initial_information['description']

    #encode information in txt files and delete path where losses are saved to avoid having data from different runs
    os.makedirs(initial_information['PATH']+'/scripts/bin',exist_ok=True)
    os.makedirs(initial_information['PATH']+'/scripts/src',exist_ok=True)
    os.makedirs(initial_information['PATH']+'/scripts/configs',exist_ok=True)
    
    if os.path.exists(initial_information['PATH'] +'/runs'):
        shutil.rmtree(initial_information['PATH']+'/runs') 

    #copy scripts
    shutil.copy('bin/main.py', initial_information['PATH']+'/scripts/bin/')
    shutil.copy('src/architecture.py', initial_information['PATH']+'/scripts/src')
    shutil.copy('src/training_validation_functions.py', initial_information['PATH']+'/scripts/src')
    shutil.copy('src/data_functions.py', initial_information['PATH']+'/scripts/src')
    shutil.copy('src/method_functions.py', initial_information['PATH']+'/scripts/src')
    shutil.copy('configs/initial_information.yaml', initial_information['PATH']+'/scripts/configs')
    shutil.copy('configs/model_information.yaml', initial_information['PATH']+'/scripts/configs')

    # go to gpu if possible
    device = tc.device(initial_information['which_device']) if tc.cuda.is_available() else tc.device("cpu")
    print(f'Selected device: {device}')

    #define the ENCODER, the function f of the latent dynamics and the Decoder 
    conv_encoder = Convolutional_Encoder(initial_information['dim_input'], model_information['kernel_enc'], model_information['filters_enc'], model_information['stride_enc'], model_information['input_output_dfnn'], model_information['latent_dim'], model_information['final_and_initial_activation'])
    conv_decoder = Convolutional_Decoder(initial_information['dim_input'], model_information['kernel_deco'], model_information['filters_deco'], model_information['stride_dec'], model_information['latent_dim'], model_information['input_output_dfnn'], final_reduction, model_information['final_and_initial_activation'], model_information['number_channels_input_cnns_deco'])
    f = F_Latent(model_information['parameter_information'], initial_information['dim_parameter'], model_information['latent_dim'], model_information['n_neurons_f'], model_information['n_layers_f'], model_information['n_FiLM_conditioning'])

    #depending on whether the system is coupled, define f and load conv_encoder and conv_decoder
    if not initial_information['is_coupled'][0] and initial_information['is_coupled'][1] == 'NODE':
        checkpoint = tc.load(initial_information['path_trained_AE']+'/checkpoint/check.pt', map_location=device, weights_only=False)

        conv_encoder.load_state_dict(checkpoint['enco'])
        conv_decoder.load_state_dict(checkpoint['dec'])

        for param in conv_encoder.parameters():
            param.requires_grad = False
        for param in conv_decoder.parameters():
            param.requires_grad = False

        params_to_optimize = [
        {'params': f.parameters(), 'weight_decay': model_information['weight_decay']['dfnn']}
    ]
        
    elif not initial_information['is_coupled'][0] and initial_information['is_coupled'][1] == 'AE':
        for param in f.parameters():
            param.requires_grad = False
            
        params_to_optimize = [
        {'params': conv_encoder.parameters(), 'weight_decay': model_information['weight_decay']['encoder']},
        {'params': conv_decoder.parameters(), 'weight_decay': model_information['weight_decay']['decoder']}
    ]

    elif initial_information['is_coupled'][0]:
        params_to_optimize = [
        {'params': conv_encoder.parameters(), 'weight_decay': model_information['weight_decay']['encoder']},
        {'params': f.parameters(), 'weight_decay': model_information['weight_decay']['dfnn']},
        {'params': conv_decoder.parameters(), 'weight_decay': model_information['weight_decay']['decoder']}
    ]
    #move the models to the device
    conv_encoder.to(device)
    f.to(device)
    conv_decoder.to(device)

    #define optimizer, the pre scheduler for the warmup of the model and the scheduler
    optim = tc.optim.Adam(params_to_optimize, lr=initial_information['learning_rate'])
    lambda1 = lambda i : i / initial_information['time_of_AE']
    pre_scheduler = tc.optim.lr_scheduler.LambdaLR(optim,lambda1)
    scheduler = tc.optim.lr_scheduler.ExponentialLR(optim, initial_information['gamma_lr'])

    #print information

    print('---------- INITIAL INFORMATION ----------')
    for key, value in initial_information.items():
        print(key, ' : ', value)
    print(" ")
    print('---------- MODEL INFORMATION ----------')
    for key, value in model_information.items():
        print(key, ' : ', value)
    print(" ")


    # prepare dataloaders for training and validation
    data_path = initial_information['data_path'] 
    dataset_training = CustomStarDataset_Big_Dataset(data_path + '/'+initial_information['name_training_field'],data_path + '/'+initial_information['name_training_parameter'],initial_information['dim_parameter'], initial_information['time_dependence_in_f'])
    dataset_validation = CustomStarDataset_Big_Dataset(data_path + '/'+initial_information['name_validation_field'],data_path + '/'+initial_information['name_validation_parameter'], initial_information['dim_parameter'] ,initial_information['time_dependence_in_f'])

    training = DataLoader(dataset_training,batch_size=initial_information['batch_size'], num_workers=8, shuffle=True,drop_last=True,pin_memory=True)
    validation = DataLoader(dataset_validation,batch_size=initial_information['batch_size'], num_workers=8, shuffle=True,drop_last=True,pin_memory=True)

    # get max and min for training from the training dataset
    ma_mi = get_max_and_min(training, initial_information['dim_parameter'],initial_information['dim_input'], initial_information['normalization_field_ma'],initial_information['normalization_field_mi'] ,initial_information['normalization_parameters_ma'],initial_information['normalization_parameters_mi']) 

    norm = {'Field':{'Max': ma_mi[0],'Min': ma_mi[1]},
                'Param':{'Max': ma_mi[2],'Min': ma_mi[3]},
                }
    with open(initial_information['PATH'] + '/Normalization.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(norm.keys())  # Write header
        writer.writerow(norm.values())  # Write data
    RK = {
            '1' : tc.tensor([[0,0],[0,1]]),
            '2' : tc.tensor([[0,0,0],[1,1,0],[0, 1/2,1/2]]),
            '3' : tc.tensor([[0,0,0,0],[1/2,1/2,0,0],[1,-1,2,0],[0,1/6,2/3,1/6]]),
            '4' : tc.tensor([[0,0,0,0,0],[1/2,1/2,0,0,0],[1/2,0,1/2,0,0],[1,0,0,1,0],[0,1/6,1/3,1/3,1/6]])
            }
    
    #training starts
    if not initial_information['is_coupled'][0]:
        initial_information['loss_coeff_TF_AR_together'] = initial_information['loss_coeff_not_coupled']

    nn_training(conv_encoder ,f, conv_decoder,training, validation, ma_mi, device, optim, initial_information['epochs'], 
        initial_information['PATH'], initial_information['loss_coeff_TF_AR_together'], initial_information['AR_strength'], initial_information['time_only_TF'], pre_scheduler,scheduler, RK, initial_information['k'], initial_information['start_backprop'], 
        initial_information['checkpoint'], initial_information['time_of_AE'], initial_information['dim_input'], 
        model_information['lambda_regularization'], initial_information['time_dependence_in_f'], 
        initial_information['TBPP_dynamic'],initial_information['clipping'], initial_information['is_coupled'])
    
if __name__ == '__main__':
    main()