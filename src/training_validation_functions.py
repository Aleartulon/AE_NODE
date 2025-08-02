import os
from src.data_functions import *
from src.method_functions import *

def train_epoch(conv_encoder, f, conv_decoder, ma_mi, device, optim, training_data,loss_coeff,
 RK, k,start_backprop, dim_input, lambda_regularization, time_dependence_in_f, clipping, is_coupled):
    """ This function performs the training cycle for one epoch, i.e., it cycles on the training batches, 
    calls .backward() and optimizes the model weights accordingly.

    Args:
        conv_encoder (class 'src.architecture.Convolutional_Encoder): Encoder
        f (src.architecture.F_Latent): function f of the ODE of the latent dynamics
        conv_decoder (src.architecture.Convolutional_Decoder): Decoder
        ma_mi (list): list of lists of maximum and minima of fields and parameters
        device (torch.device): device where the training and validation are done
        optim (torch.optim): optimizer
        training_data (torch.utils.data.dataloader.DataLoader): data_loader for training dataset
        loss_coeff (list): list of importance weights of the loss function terms
        RK (dict): dictionary with Butcher tablue for Runge-Kutta algorithms
        k (int): stage of Runge-Kutta algorithm
        start_backprop (list): list with values that determine up to which time-step in the past backpropagate the gradients
        dim_input (list): first dimension is the channels of the solution field, second is the number of spatial dimensions
        lambda_regularization (float): coefficients that multiplies the regularization term of the latent vector
        time_dependence_in_f (bool): if true, the function f depends on time as well.
        clipping (list): first dimension is a boolean. if true, clipping is applied to the gradients of the function f with max norm given by second dimension

    Returns:
       [float, float, float, float, float] : The output are the mean values of the 5 training losses: [L_1, L_2^T, L_2^A, L_3, sum_of_previous]
    """
    l1_loss = 0
    l2_TF_loss = 0
    l2_AR_loss = 0
    l3_loss = 0
    loss = 0
    count = 0
    regularization_loss = 0
    conv_encoder.train()
    f.train()
    conv_decoder.train()

    for field, dt, param in training_data:
        
        optim.zero_grad()

        l1,l2_TF,l2_AR,l3, _,regularization_latent  = loss_sup_mixed(conv_encoder, f, conv_decoder, field, dt, param ,ma_mi, device,loss_coeff, RK, k, start_backprop, dim_input,lambda_regularization,True, time_dependence_in_f, is_coupled)

        (l1+l2_TF+l2_AR+l3+regularization_latent).backward()
        if clipping[0]:
            tc.nn.utils.clip_grad_norm_(f.parameters(), max_norm=clipping[1])
        optim.step()
        loss += (l1+l2_TF+l2_AR+l3).detach().cpu().item()
        
        l1_loss += l1.detach().cpu().item()
        l2_TF_loss += l2_TF.detach().cpu().item()
        l2_AR_loss += l2_AR.detach().cpu().item()
        l3_loss += l3.detach().cpu().item()
        regularization_loss += regularization_latent.detach().cpu().item()
        count += 1
    return l1_loss/count, l2_TF_loss/count, l2_AR_loss/count ,l3_loss/count, regularization_loss/count, loss/count


def valid_epoch(conv_encoder, f, conv_decoder, ma_mi, device, validation_data,loss_coeff, RK, k, start_backprop, dim_input, lambda_regularization, time_dependence_in_f, is_coupled):
    """ This function performs the validation cycle for one epoch, i.e., it cycles on the validation batches and gets the corresponding validation metrics. 
    Those metrics are used to verify whether the model is overfitting by using early-stop.
    Args:
        conv_encoder (class 'src.architecture.Convolutional_Encoder): Encoder
        f (src.architecture.F_Latent): function f of the ODE of the latent dynamics
        conv_decoder (src.architecture.Convolutional_Decoder): Decoder
        ma_mi (list): list of lists of maximum and minima of fields and parameters
        device (torch.device): device where the training and validation are done
        validation_data (torch.utils.data.dataloader.DataLoader): data_loader for validation dataset
        loss_coeff (list): list of importance weights of the loss function terms
        RK (dict): dictionary with Butcher tablue for Runge-Kutta algorithms
        k (int): stage of Runge-Kutta algorithm
        start_backprop (list): list with values that determine up to which time-step in the past backpropagate the gradients
        dim_input (list): first dimension is the channels of the solution field, second is the number of spatial dimensions
        lambda_regularization (float): coefficients that multiplies the regularization term of the latent vector
        time_dependence_in_f (bool): if true, the function f depends on time as well.
    Returns:
       [float, float, float, float, float, float, float] : The output are the mean values of the 7 training losses: [L_1, L_1 unnormalized, L_2^T, L_2^A, L_3, sum_of_previous, loss when predicting the full solution autoregressively]
    """
    conv_encoder.eval()
    f.eval()
    conv_decoder.eval()

    l1_loss = 0
    l1_loss_unnorm = 0
    l2_TF_loss = 0
    l2_AR_loss = 0
    l3_loss = 0
    loss = 0
    regularization_loss = 0 
    l_real = 0
    count = 0
    with tc.no_grad():
        for field, dt, param in validation_data:

            l1,l2_TF,l2_AR,l3, l_final, regularization_latent = loss_sup_mixed(conv_encoder, f, conv_decoder, field, dt, param ,ma_mi, device,loss_coeff, RK, k, start_backprop, dim_input, lambda_regularization, False, time_dependence_in_f, is_coupled)

            loss += (l1[0]+l2_TF+l2_AR+l3+regularization_latent+l_final).detach().item()
            l_real +=  l_final.detach().item()
            l1_loss += l1[0].detach().cpu().item()
            l1_loss_unnorm += l1[1].detach().cpu().item()
            l2_TF_loss += l2_TF.detach().cpu().item()
            l2_AR_loss += l2_AR.detach().cpu().item()
            l3_loss += l3.detach().cpu().item()
            regularization_loss += regularization_latent.detach().cpu().item()
            count += 1

    return l1_loss/count, l1_loss_unnorm/count, l2_TF_loss/count, l2_AR_loss/count , l3_loss/count, l_real/count, regularization_loss/count , loss/count


def nn_training(conv_encoder ,f, conv_decoder, training_data, validation_data, ma_mi, device, optim, 
     epochs, PATH, loss_coeff, AR_strength, time_only_TF, pre_scheduler, scheduler, RK, k, start_backprop, checkpoint, 
     time_of_AE, dim_input, lambda_regularization, time_dependence_in_f, TBPP_dynamic, clipping, is_coupled):
    """This functions starts the training and cycle for a given number of epochs on the training and valiation datasets.
    It gathers the traiing and validation metrics, prints them and saves them and saves the NNs weights if the validation function has hit a new minimum.
    It is possible to restart the training from a previous checkpoint by setting checkpoint = true
    Args:
        conv_encoder (class 'src.architecture.Convolutional_Encoder): Encoder
        f (src.architecture.F_Latent): function f of the ODE of the latent dynamics
        conv_decoder (src.architecture.Convolutional_Decoder): Decoder
        training_data (torch.utils.data.dataloader.DataLoader): data_loader for training dataset
        validation_data (torch.utils.data.dataloader.DataLoader): data_loader for validation dataset
        ma_mi (list): list of lists of maximum and minima of fields and parameters
        device (torch.device): device where the training and validation are done
        optim (torch.optim): optimizer
        epochs (int): maximum number of training epochs
        PATH (str): path specified in initial_information.yaml by combining physics_model+description: it is where all the outputs of a training are saved
        loss_coeff (list): list of importance weights of the loss function terms
        AR_strength (float): if L_2^A is used, it can start with a coefficient weight 'AR_strength' < 1 for traiing stabilities. It is then increased linearly each epoch until final value is reached.
        time_only_TF (int): after warmup period, it is possible to have 'time_only_TF' epochs where L_2 is multiplied by 0
        pre_scheduler (torch.optim.lr_scheduler): scheduler to implement the warming up of the learning rate in the initial stages of the training
        scheduler (torch.optim.lr_scheduler): scheduler to decrease the learning rate per epoch during the training
        RK (dict): dictionary with Butcher tablue for Runge-Kutta algorithms
        k (int): stage of Runge-Kutta algorithm
        start_backprop (list): list with values that determine up to which time-step in the past backpropagate the gradients
        checkpoint (bool): if true, the training starts from the last checkpoint. if false, a new training is started
        time_of_AE (int): number of initial epochs where only the AutoEncoder is trained, i.e., loss_coeff_TF_AR_together = [1, 0, 0, 0]. In parallel of this a linear warm-up of the learning rate is performed
        dim_input (list): first dimension is the channels of the solution field, second is the number of spatial dimensions
        lambda_regularization (float): coefficients that multiplies the regularization term of the latent vector
        time_dependence_in_f (bool): if true, the function f depends on time as well.
        TBPP_dynamic (list): if first dimension is true, when computing L_2_A, the depth in time in the past to which the backprop algorithm is applied increase dinamically during the training
        clipping (list): first dimension is a boolean. if true, clipping is applied to the gradients of the function f with max norm given by second dimension
        is_coupled (list): #if true the AutoEncoder (AE) is trained coupled with the NODE. if false, if second dimension is 'AE' the AE is trained, if second dimension is 'NODE' the NODE is trained.
    """    

    if not checkpoint:
        # create losses file
        os.makedirs(PATH+'/losses/',exist_ok=True)
        os.makedirs(PATH+'/checkpoint/',exist_ok=True)

        loss_coeff_2 = -10
        #start the training

        #check for coupled_system
        if not is_coupled[0] and is_coupled[1] == 'NODE':
            time_of_AE = 0

        print("------------------TRAINING STARTS------------------")
        loss_value = 100
        early_stopping = 0 
        full_training_count = 1

        train_l1 = np.zeros(epochs)
        train_l2_TF = np.zeros(epochs)
        train_l2_AR = np.zeros(epochs)
        train_l3 = np.zeros(epochs)
        train_regularization = np.zeros(epochs)
        train_loss_tot = np.zeros(epochs)

        valid_l1 = np.zeros(epochs)
        valid_l1_unnorm = np.zeros(epochs)
        valid_l2_TF = np.zeros(epochs)
        valid_l2_AR = np.zeros(epochs)
        valid_l3 = np.zeros(epochs)
        valid_real = np.zeros(epochs)
        valid_regularization = np.zeros(epochs)
        valid_loss_tot = np.zeros(epochs)

        if TBPP_dynamic[0]:
            start_backprop = [start_backprop[0],1]

        for i in range(epochs):

            early_stopping += 1
            if early_stopping == 200:
                print('Training stopped due to early stopping')
                #writer.close()
                break
            time1 = time.time()
            if i < time_of_AE: #use only AR
                train_l1_data, train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = train_epoch(conv_encoder, f, conv_decoder, ma_mi, device, optim, training_data, [loss_coeff[0],0,0,0], RK, k, start_backprop, dim_input, lambda_regularization, time_dependence_in_f, clipping, is_coupled)
                valid_l1_data, valid_l1_unnorm_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_regularization_data, valid_loss_data = valid_epoch(conv_encoder, f, conv_decoder, ma_mi, device, validation_data,[1,1,1,1], RK, k, start_backprop, dim_input, lambda_regularization, time_dependence_in_f, is_coupled)
                if is_coupled[0]:
                    valid_loss_data = 100.0
            elif i >=time_of_AE and i < time_only_TF: #use only TF
                train_l1_data, train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = train_epoch(conv_encoder, f, conv_decoder, ma_mi, device, optim, training_data,[loss_coeff[0],loss_coeff[1],0,loss_coeff[3]], RK, k, start_backprop, dim_input,lambda_regularization, time_dependence_in_f, clipping, is_coupled)
                valid_l1_data, valid_l1_unnorm_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_regularization_data, valid_loss_data = valid_epoch(conv_encoder, f, conv_decoder, ma_mi, device, validation_data,[1,1,1,1], RK, k, start_backprop, dim_input, lambda_regularization, time_dependence_in_f, is_coupled)
            else:
                loss_coeff_2 = loss_coeff[2] * AR_strength * full_training_count
                full_training_count +=1
                if loss_coeff_2 >= loss_coeff[2]:
                    loss_coeff_2 = loss_coeff[2]

                if TBPP_dynamic[0]  and start_backprop[1] < TBPP_dynamic[2] and full_training_count%TBPP_dynamic[1] == 0:
                    start_backprop[1] += 1
                    
                train_l1_data, train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = train_epoch(conv_encoder, f, conv_decoder, ma_mi, device, optim, training_data,[loss_coeff[0],loss_coeff[1],loss_coeff_2,loss_coeff[3]], RK, k, start_backprop, dim_input,lambda_regularization, time_dependence_in_f, clipping, is_coupled)
                valid_l1_data, valid_l1_unnorm_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_regularization_data, valid_loss_data = valid_epoch(conv_encoder, f, conv_decoder, ma_mi, device, validation_data,[1,1,1,1], RK, k,start_backprop, dim_input, lambda_regularization, time_dependence_in_f, is_coupled)

            time2 = time.time()

            if i > time_of_AE:
                scheduler.step()
            else:
                pre_scheduler.step()
            train_l1[i] = train_l1_data
            train_l2_TF[i] = train_l2_TF_data
            train_l2_AR[i] = train_l2_AR_data
            train_l3[i] = train_l3_data
            train_regularization[i] = train_regularization_data
            train_loss_tot[i] = train_loss_data

            valid_l1[i] = valid_l1_data
            valid_l1_unnorm[i] = valid_l1_unnorm_data
            valid_l2_TF[i] = valid_l2_TF_data
            valid_l2_AR[i] = valid_l2_AR_data
            valid_l3[i] = valid_l3_data
            valid_real[i] = valid_real_data
            valid_regularization[i] = valid_regularization_data
            valid_loss_tot[i] = valid_loss_data

            np.save(PATH + "/losses/train_l1.npy", train_l1)
            np.save(PATH + "/losses/train_l2_TF.npy", train_l2_TF)
            np.save(PATH + "/losses/train_l2_AR.npy", train_l2_AR)
            np.save(PATH + "/losses/train_l3.npy", train_l3)
            np.save(PATH + "/losses/train_regularization.npy", train_regularization)
            np.save(PATH + "/losses/train_loss_tot.npy", train_loss_tot)

            np.save(PATH + "/losses/valid_l1.npy", valid_l1)
            np.save(PATH + "/losses/valid_l1_unnorm.npy", valid_l1_unnorm)
            np.save(PATH + "/losses/valid_l2_TF.npy", valid_l2_TF)
            np.save(PATH + "/losses/valid_l2_AR.npy", valid_l2_AR)
            np.save(PATH + "/losses/valid_l3.npy", valid_l3)
            np.save(PATH + "/losses/valid_real.npy", valid_real)
            np.save(PATH + "/losses/valid_regularization.npy", valid_regularization)
            np.save(PATH + "/losses/valid_loss_tot.npy", valid_loss_tot)


            print("Epoch: " +str(i)+', ' + str(time2-time1)+ ' s')
            if TBPP_dynamic[0]:
                print("Start_backprop for TBPP: " +str(start_backprop[1]))
            print("Loss coefficient 2: " +str(loss_coeff_2))
            print('Train_loss_data = ' + str(train_loss_data) + ', l1 train loss = ' +str(train_l1_data) + ', l2 TF train loss = ' + str(train_l2_TF_data)+ ', l2 AR train loss = ' + str(train_l2_AR_data)+ ', l3 train loss = ' + str(train_l3_data)+ ', regularization loss = ' + str(train_regularization_data))
            print('Valid_loss_data = ' + str(valid_loss_data)+ ', valid Real loss = ' +str(valid_real_data)  + ', l1 valid loss = ' +str(valid_l1_data) +  ', l1 valid unnorm loss = ' +str(valid_l1_unnorm_data) + ', l2 valid TF loss = ' + str(valid_l2_TF_data)+ ', l2 valid AR loss = ' + str(valid_l2_AR_data)+ ', l3 valid loss = ' + str(valid_l3_data)+ ', valid regularization = ' + str(valid_regularization_data))
            print('The validation loss has not decreased for ' + str(early_stopping) + ' epochs!')
            
            print('------------------------------------------------------')

            #check if training a noncoupled system and adjust accordingly the validatin losses to be checked for early stopping
            if not is_coupled[0] and is_coupled[1] == 'AE' and i >=time_of_AE:
                valid_loss_data = valid_l1_data + valid_regularization_data
            elif not is_coupled[0] and is_coupled[1] == 'NODE':
                valid_loss_data = valid_real_data + valid_l2_TF_data + valid_l2_AR_data + valid_l3_data

            if valid_loss_data < loss_value: #careful valid loss tot!!
                loss_value = valid_loss_data
                print('Models saved!')
                save_checkpoint(conv_encoder, f , conv_decoder, optim, scheduler, i, loss_value, loss_coeff_2 , start_backprop, full_training_count,PATH+'/checkpoint/check.pt')
                early_stopping = 0
    
    else:

        conv_encoder, f, conv_decoder, optimizer, scheduler, start_epoch, loss, loss_coeff_2, start_backprop, full_training_count = load_checkpoint(conv_encoder, f , conv_decoder, optim, scheduler, PATH+'/checkpoint/check.pt', device)
        conv_encoder.to(device)
        f.to(device)
        conv_decoder.to(device) 

        #start the training
        print("------------------TRAINING STARTS------------------")
        loss_value = loss
        early_stopping = 0 

        train_l1 = np.load(PATH + "/losses/train_l1.npy", allow_pickle=True)
        train_l2_TF = np.load(PATH + "/losses/train_l2_TF.npy",allow_pickle=True)
        train_l2_AR = np.load(PATH + "/losses/train_l2_AR.npy",allow_pickle=True)
        train_l3 = np.load(PATH + "/losses/train_l3.npy",allow_pickle=True)
        train_regularization = np.load(PATH + "/losses/train_regularization.npy",allow_pickle=True)
        train_loss_tot = np.load(PATH + "/losses/train_loss_tot.npy",allow_pickle=True)
        valid_l1 = np.load(PATH + "/losses/valid_l1.npy",allow_pickle=True)
        valid_l1_unnorm = np.load(PATH + "/losses/valid_l1_unnorm.npy",allow_pickle=True)
        valid_l2_TF = np.load(PATH + "/losses/valid_l2_TF.npy",allow_pickle=True)
        valid_l2_AR = np.load(PATH + "/losses/valid_l2_AR.npy",allow_pickle=True)
        valid_l3 = np.load(PATH + "/losses/valid_l3.npy",allow_pickle=True)
        valid_regularization = np.load(PATH + "/losses/valid_regularization.npy",allow_pickle=True)
        valid_loss_tot = np.load(PATH + "/losses/valid_loss_tot.npy",allow_pickle=True)
        valid_real = np.load(PATH + "/losses/valid_real.npy",allow_pickle=True)

        for i in np.arange(start_epoch+1, epochs+1, 1):

            early_stopping += 1
            if early_stopping == 200:
                print('Training stopped due to early stopping')
                #writer.close()
                break
            time1 = time.time()
            if i < time_of_AE: #use only AR
                train_l1_data, train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = train_epoch(conv_encoder, f, conv_decoder, ma_mi, device, optim, training_data, [loss_coeff[0],0,0,0], RK, k, start_backprop, dim_input, lambda_regularization, time_dependence_in_f, clipping, is_coupled)
                valid_l1_data, valid_l1_unnorm_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_regularization_data, valid_loss_data = valid_epoch(conv_encoder, f, conv_decoder, ma_mi, device, validation_data,[1,1,1,1], RK, k, start_backprop, dim_input, lambda_regularization, time_dependence_in_f, is_coupled)
                if is_coupled[0]:
                    valid_loss_data = 100.0
            elif i >=time_of_AE and i < time_only_TF: #use only TF
                train_l1_data, train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = train_epoch(conv_encoder, f, conv_decoder, ma_mi, device, optim, training_data,[loss_coeff[0],loss_coeff[1],0,loss_coeff[3]], RK, k, start_backprop, dim_input,lambda_regularization, time_dependence_in_f, clipping, is_coupled)
                valid_l1_data, valid_l1_unnorm_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_regularization_data, valid_loss_data = valid_epoch(conv_encoder, f, conv_decoder, ma_mi, device, validation_data,[1,1,1,1], RK, k, start_backprop, dim_input, lambda_regularization, time_dependence_in_f, is_coupled)
            else:
                loss_coeff_2 = loss_coeff[2] * AR_strength * full_training_count
                full_training_count +=1
                if loss_coeff_2 >= loss_coeff[2]:
                    loss_coeff_2 = loss_coeff[2]

                if TBPP_dynamic[0]  and start_backprop[1] < TBPP_dynamic[2] and full_training_count%TBPP_dynamic[1] == 0:
                    start_backprop[1] += 1
                    
                train_l1_data, train_l2_TF_data, train_l2_AR_data, train_l3_data, train_regularization_data, train_loss_data = train_epoch(conv_encoder, f, conv_decoder, ma_mi, device, optim, training_data,[loss_coeff[0],loss_coeff[1],loss_coeff_2,loss_coeff[3]], RK, k, start_backprop, dim_input,lambda_regularization, time_dependence_in_f, clipping, is_coupled)
                valid_l1_data, valid_l1_unnorm_data, valid_l2_TF_data, valid_l2_AR_data, valid_l3_data, valid_real_data, valid_regularization_data, valid_loss_data = valid_epoch(conv_encoder, f, conv_decoder, ma_mi, device, validation_data,[1,1,1,1], RK, k,start_backprop, dim_input, lambda_regularization, time_dependence_in_f, is_coupled)

            time2 = time.time()

            if i > time_of_AE:
                scheduler.step()
            else:
                pre_scheduler.step()
            train_l1[i] = train_l1_data
            train_l2_TF[i] = train_l2_TF_data
            train_l2_AR[i] = train_l2_AR_data
            train_l3[i] = train_l3_data
            train_regularization[i] = train_regularization_data
            train_loss_tot[i] = train_loss_data

            valid_l1[i] = valid_l1_data
            valid_l1_unnorm[i] = valid_l1_unnorm_data
            valid_l2_TF[i] = valid_l2_TF_data
            valid_l2_AR[i] = valid_l2_AR_data
            valid_l3[i] = valid_l3_data
            valid_real[i] = valid_real_data
            valid_regularization[i] = valid_regularization_data
            valid_loss_tot[i] = valid_loss_data

            np.save(PATH + "/losses/train_l1.npy", train_l1)
            np.save(PATH + "/losses/train_l2_TF.npy", train_l2_TF)
            np.save(PATH + "/losses/train_l2_AR.npy", train_l2_AR)
            np.save(PATH + "/losses/train_l3.npy", train_l3)
            np.save(PATH + "/losses/train_regularization.npy", train_regularization)
            np.save(PATH + "/losses/train_loss_tot.npy", train_loss_tot)

            np.save(PATH + "/losses/valid_l1.npy", valid_l1)
            np.save(PATH + "/losses/valid_l1_unnorm.npy", valid_l1_unnorm)
            np.save(PATH + "/losses/valid_l2_TF.npy", valid_l2_TF)
            np.save(PATH + "/losses/valid_l2_AR.npy", valid_l2_AR)
            np.save(PATH + "/losses/valid_l3.npy", valid_l3)
            np.save(PATH + "/losses/valid_real.npy", valid_real)
            np.save(PATH + "/losses/valid_regularization.npy", valid_regularization)
            np.save(PATH + "/losses/valid_loss_tot.npy", valid_loss_tot)


            print("Epoch: " +str(i)+', ' + str(time2-time1)+ ' s')
            if TBPP_dynamic[0]:
                print("Start_backprop for TBPP: " +str(start_backprop[1]))
            print("Loss coefficient 2: " +str(loss_coeff_2))
            print('Train_loss_data = ' + str(train_loss_data) + ', l1 train loss = ' +str(train_l1_data) + ', l2 TF train loss = ' + str(train_l2_TF_data)+ ', l2 AR train loss = ' + str(train_l2_AR_data)+ ', l3 train loss = ' + str(train_l3_data)+ ', regularization loss = ' + str(train_regularization_data))
            print('Valid_loss_data = ' + str(valid_loss_data)+ ', valid Real loss = ' +str(valid_real_data)  + ', l1 valid loss = ' +str(valid_l1_data) +  ', l1 valid unnorm loss = ' +str(valid_l1_unnorm_data) + ', l2 valid TF loss = ' + str(valid_l2_TF_data)+ ', l2 valid AR loss = ' + str(valid_l2_AR_data)+ ', l3 valid loss = ' + str(valid_l3_data)+ ', valid regularization = ' + str(valid_regularization_data))
            print('The validation loss has not decreased for ' + str(early_stopping) + ' epochs!')
            
            print('------------------------------------------------------')

            #check if training a noncoupled system and adjust accordingly the validatin losses to be checked for early stopping
            if not is_coupled[0] and is_coupled[1] == 'AE' and i >=time_of_AE:
                valid_loss_data = valid_l1_data + valid_regularization_data
            elif not is_coupled[0] and is_coupled[1] == 'NODE':
                valid_loss_data = valid_real_data + valid_l2_TF_data + valid_l2_AR_data + valid_l3_data

            if valid_loss_data < loss_value: #careful valid loss tot!!
                loss_value = valid_loss_data
                print('Models saved!')
                save_checkpoint(conv_encoder, f , conv_decoder, optim, scheduler, i, loss_value, loss_coeff_2 , start_backprop, full_training_count,PATH+'/checkpoint/check.pt')
                early_stopping = 0

