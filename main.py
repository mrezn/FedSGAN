from getdata import GetDataSet
from clients import Client, ClientsGroup
import torch.nn.functional as F
import torch.nn as nn

from model.Net import Net
import torch
import argparse
from tqdm import tqdm
import argparse
import os
from torch import optim
import numpy as np
from copy import deepcopy
from model import gen_model


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')

parser.add_argument('-nc', '--num_of_clients', type=int, default=10, help='numer of the clients')

parser.add_argument('-cf', '--cfraction', type=float, default=0.1,
                    help='C fraction, 0 means 1 client, 1 means total clients')

parser.add_argument('-E', '--epoch', type=int, default=20, help='local train epoch')

parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')

parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')

parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-dataset', "--dataset", type=str, default="MNIST", help='CIFAR10 or MNIST')

parser.add_argument('-vf', "--val_freq", type=int, default=3, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')

parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')

parser.add_argument('-m', '--W_size', type=int, default=5, help='the W space size')

parser.add_argument('-sigma', '--sigma_value', type=int, default=0.1, help='the value of sigma for optimatization')

def calculate_M(net):
    shap=0
    for name, param in net.named_parameters():
        w=param.data.view(-1)
        shap+=w.shape[0]
        # print(w.shape)
    return shap

def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__


    test_txt = open("test_accuracy.txt", mode="a")

    test_mkdir(args['save_path'])

    dev = torch.device("cpu")

    # m=args['W_size']
    sigma=args['sigma_value']

    net = None
    generator=None
    
    min_coumm_round= 0
    gen_iters= 100
    gen_lr= 0.01
    n_cls= 10
    if args['model_name'] == 'mnist_cnn':
        net = Net()
        generator=gen_model.CGeneratorA()
    else:
        pass

    # generator.to("cpu")
    loss_func = F.cross_entropy
    criterion = F.cross_entropy#nn.MSELoss()
    # opti = optim.SGD(net.parameters(), lr=args['learning_rate'])
    dataset=args['dataset']



    num_in_comm = int(max(args['num_of_clients'], 1))#int(max(args['num_of_clients'] * args['cfraction'], 1))

    global_parameters = {}

    for key, var in net.state_dict().items():
        # print(str(var.shape))
        # print(str(var.size()))
        global_parameters[key] = var.clone()

    m=calculate_M(net)
#     state_dict_ = net.state_dict()
#     weights = {key: value for key, value in state_dict_.items() if 'weight' in key}

# # Print the weights
#     for key, weight in weights.items():
#         print(f"Layer: {key}, Weights: {weight.shape}")
#     # print(global_parameters)
#     art=input()

    lambda_k =torch.zeros(2*m*(args['num_of_clients']-1))# {name: torch.zeros_like(param) for name, param in net.named_parameters()}
    # weights={name: param.data.clone() for name, param in net.named_parameters()}
    
    
    myClients = ClientsGroup(dataset, args['num_of_clients'], dev,net,global_parameters,criterion,loss_func,args['epoch'],args['batchsize'],m,args['learning_rate'],sigma,
                             generator,min_coumm_round,gen_iters,gen_lr,n_cls)
    # testDataLoader = myClients.testDataLoader



    
    for i in range(args['num_comm']):
        print("communicate round {}".format(i + 1))

        order = np.random.permutation(args['num_of_clients'])
        # print("order:")
        # print(len(order))
        # print(order)
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        print(str(clients_in_comm))
        # print(type(clients_in_comm))  

        # sum_parameters = None
        y_k_plus_1_list = []
        lambda_k_plus_1_list = []
        ACC_list=[]
        ACC_gan_list=[]

        client_gradients  = []
        for client in tqdm(clients_in_comm):
            
            if i==0:
                y_k = myClients.init_y_k[client]
                average_gradients = []
            # local_parameters = myClients.clients_set[client].localModelUpdate(lambda_k,y_k)
            y_k_plus_1, lambda_k_plus_1 , ACC , gradient ,loss_generator =myClients.clients_set[client].localModelUpdate(lambda_k,y_k,average_gradients,i)
            y_k_plus_1_list.append(y_k_plus_1)
            lambda_k_plus_1_list.append(lambda_k_plus_1)
            ACC_list.append(ACC)
            ACC_gan_list.append(np.abs(1-loss_generator))
            client_gradients.append(gradient)
            # print(gradient)
            # if sum_parameters is None:
            #     sum_parameters = {}
            #     for key, var in local_parameters.items():
            #         sum_parameters[key] = var.clone()
            # else:
            #     for var in sum_parameters:
            #         sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        average_gradients = []
        for grads in zip(*client_gradients):
            avg_grad = sum(grads) / num_in_comm
            average_gradients.append(avg_grad)
        # print(f'average_gradients : {average_gradients}')
        # art=input("input")
        avg_y_k_plus_1 = deepcopy(y_k_plus_1_list[0])
        avg_lambda_k_plus_1 = deepcopy(lambda_k_plus_1_list[0])
        avg_acc= deepcopy(ACC_list[0])
        avg_gan_acc= deepcopy(ACC_gan_list[0])
        n_models=len(lambda_k_plus_1_list)#args['num_of_clients']
        for i in range(1, n_models):
            avg_acc += ACC_list[i]
            avg_gan_acc += ACC_gan_list[i]
            for key in range(len(avg_y_k_plus_1)):
                avg_y_k_plus_1[key] += y_k_plus_1_list[i][key]
                avg_lambda_k_plus_1[key] += lambda_k_plus_1_list[i][key]
                
        for key in range(len(avg_y_k_plus_1)):
            avg_y_k_plus_1[key] /= n_models
            avg_lambda_k_plus_1[key] /= n_models
        avg_acc /=n_models
        avg_gan_acc /=n_models
        lambda_k =  avg_lambda_k_plus_1
        y_k =  avg_y_k_plus_1

        # for var in global_parameters:
        #     global_parameters[var] = (sum_parameters[var] / num_in_comm)
        ###############################################################################################################
        # grad_f = {name: param.grad.clone() for name, param in net.named_parameters()}
        # weights_ = {name: param.data.clone() for name, param in net.named_parameters()}
        # u_k = {name: grad_f[name] + avg_lambda_k_plus_1[name] for name in grad_f}
        # w_k_plus_1 = {name: weights_[name] - args['learning_rate'] * u_k[name] for name in weights_}
        # with torch.no_grad():
        #     for name, param in net.named_parameters():
        #         param.copy_(w_k_plus_1[name])



        # # net.load_state_dict(global_parameters, strict=True)
        # sum_accu = 0
        # num = 0
        # for data, label in testDataLoader:
        #     data, label = data.to(dev), label.to(dev)
        #     preds = net(data)
        #     preds = torch.argmax(preds, dim=1)
        #     sum_accu += (preds == label).float().mean()
        #     num += 1
        # print("\n" + 'accuracy: {}'.format(sum_accu / num))

        # test_txt.write("communicate round " + str(i + 1) + "  ")
        print('average accuracy: ' + str(float(avg_acc)) + "\n")
        print('average generator accuracy: ' + str(float(avg_gan_acc)) + "\n")

        # if (i + 1) % args['save_freq'] == 0:
        #     torch.save(net, os.path.join(args['save_path'],
        #                                  '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
        #                                                                                         i, args['epoch'],
        #                                                                                         args['batchsize'],
        #                                                                                         args['learning_rate'],
        #                                                                                         args['num_of_clients'],
        #                                                                                         args['cfraction'])))

    test_txt.close()

