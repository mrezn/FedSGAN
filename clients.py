import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getdata import GetDataSet
import numpy as np
from copy import deepcopy



class Client():
    def __init__(self, trainDataSet,testDataLoader, device,model,globalPara,criterion,lossFun,localEpoch,localBatchSize,m,alpha,sigma,index_list,clientsNum,
                 ID,client_G_weight_k, g_model,gen_lr, min_coumm_round,gen_iters,labels_all_train_gen,labels_all_train_model,n_cls):
        self.trainDataSet = trainDataSet
        self.testDataLoader=testDataLoader
        self.device = device
        self.trainDataLoader = None
        self.localPara = None
        self.model=model
        self.model.load_state_dict(globalPara, strict=True)
        self.lossFun=lossFun
        self.criterion=criterion
        self.index_list=index_list
        self.m=m
        self.alpha=alpha
        self.g_model=g_model
        self.opti=CustomOptimizer(self.model.parameters(),self.alpha ,self.index_list,self.m)
        # self.optimizer_G=genOptimizer(self.g_model.parameters(), gen_lr)
        self.optimizer_G = torch.optim.Adam(self.g_model.parameters(), lr=gen_lr)
        self.localEpoch=localEpoch
        self.localBatchSize=localBatchSize
        self.sigma=sigma
        self.clientsNum=clientsNum
        self.client_ID=ID
        self.client_G_weight_k=client_G_weight_k
        self.min_coumm_round=min_coumm_round
        self.gen_iters=gen_iters
        self.gen_lr=gen_lr
        self.labels_all_train_gen=labels_all_train_gen
        self.labels_all_train_model=labels_all_train_model
        self.n_cls=n_cls

    def creat_lamda_weight_shape(self,lamda):
        a=[]
        for  param in self.model.parameters():
            w=deepcopy(param.data)
            a.append(w)
        shp=0
        for f in a:
            cons=0
            for i in self.index_list:
                if np.sign(i) == 0:
                    cons=cons+lamda[np.abs(i)*self.m+shp].item()
                else :
                    cons=cons+np.sign(i)*lamda[np.abs(i)*self.m+shp].item()
            if len(f.shape) ==2:
                for j in range((f.data).shape[0]):
                    for k  in range((f.data).shape[1]):
                        f.data[j][k] = cons
                        shp+=1
            else:
                for j in range((f.data).shape[0]):                      
                        f.data[j] =cons
                        shp+=1 
        return a
    
    def check_new_lamda(self,weight,lamda):
        sha=0
        a=[]
        for i in weight : 
            if len(i.shape) == 2 :
                for j in range(i.shape[0]):
                    for k in range(i.shape[1]):
                        if i[j][k] == lamda[sha]:
                            a.append(True)
                        else :
                             a.append(False)
            else:
                for j in range(i.shape[0]):
                        if i[j] == lamda[sha]:
                            a.append(True)
                        else :
                             a.append(False)   
            if False in a:
                return False
            else:
                return True 
            
    # def compute_generator_gradients(model, loss):
    #     model.zero_grad()
    #     loss.backward()
    #     gradients = []
    #     for param in model.parameters():
    #         gradients.append(param.grad.clone())
    #     return gradients
    
    def update_generator(self,mean_gradient,lr):

        with torch.no_grad():
            for param, avg_grad in zip(self.g_model.parameters(), mean_gradient):
                param -= lr * avg_grad

    def train_generator(self,batch_size):
        
        # self.model.eval()
        # self.g_model.train()
        # self.g_model.trian()
        # self.g_model.zero_grad()
        # gradients = []
        loss=0
        for i in range(self.gen_iters):
            labels = self.labels_all_train_gen[i*batch_size:(i*batch_size+batch_size)]
            onehot = np.zeros((batch_size, self.n_cls))
            onehot[np.arange(batch_size), labels] = 1
            y_onehot = torch.Tensor(onehot).cpu()
            z = torch.randn((batch_size, 100, 1, 1)).cpu()
            fake = self.g_model(z, y_onehot)#.detach()
            # with torch.no_grad():
            #     pred_g = self.model((self.g_model(z, y_onehot)))
            pred_g = self.model(fake)
            loss_g = self.lossFun(pred_g, torch.from_numpy(labels))
            # self.optimizer_G.zero_grad()
            loss_g.backward()
            # print("*"*80)
            # for param in self.model.parameters():
            #     print(param.grad)
            #     print("*"*80)
            #     art=input("**")
            #     gradients.append(param.grad.clone())
            # print({name: param.grad for name, param in self.g_model.named_parameters() if param.grad is not None})

            # self.optimizer_G.step()
            loss += loss_g.item()
            # print(loss)
            # art=input()
            # gradients=self.optimizer_G.step(gradients)
        gradients = []
        for param in self.g_model.parameters():
            # print(param.grad)
            gradients.append(param.grad.clone())
        return gradients , loss
    
    def localModelUpdate(self,lambda_k,y_k,mean_gradient,comm_round):
        # self.model.to("cpu")
        # self.g_model.to("cpu")
        batch_size=self.localBatchSize
        self.trainDataLoader = DataLoader(self.trainDataSet, batch_size=self.localBatchSize, shuffle=True)
        lambda_k_weight = self.creat_lamda_weight_shape(lambda_k)
        


        if comm_round >0 :
            self.update_generator(mean_gradient,self.gen_lr)
        # total_loss=0   

        # self.model.train()
        # self.g_model.eval()
        
        for i in range(self.localEpoch):
            self.model.train()
            for X, y in self.trainDataLoader:
                self.opti.zero_grad()
                # self.g_model.zero_grad()
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.lossFun(pred, y)
                # print(pred.shape)
                # print(pred)
                # print(y.shape)
                # print(y)
                # print(f'loss without gen: {loss}')
                if comm_round > self.min_coumm_round :
                    labels = self.labels_all_train_model[i*batch_size:(i*batch_size+batch_size)]
                    onehot = np.zeros((batch_size, self.n_cls))
                    onehot[np.arange(batch_size), labels] = 1
                    y_onehot = torch.Tensor(onehot).cpu()
                    z = torch.randn((batch_size, 100, 1, 1)).cpu()
                    fake = self.g_model(z, y_onehot).detach()
                    pred_g = self.model(fake)
                    # print(pred_g.shape)
                    # print(pred_g)
                    # print(torch.from_numpy(labels).shape)
                    # print(torch.from_numpy(labels))
                    loss_g = self.lossFun(pred_g, torch.from_numpy(labels))
                    # print(f'loss with gen: {loss_g}')
                    # art=input()
                    loss += loss_g
                loss.backward()
                self.opti.step(lambda_k_weight)
        
        gradient , loss_generator =self.train_generator(batch_size)
        # print(f'gradient : {gradient}')
        # art=input()
        v_k = y_k - (self.sigma*self.alpha*lambda_k) # {name: y_k[name] - self.sigma * self.alpha * lambda_k[name] for name in y_k}
        

        lambda_k_plus_1 = lambda_k + self.alpha*v_k#{name: lambda_k[name] + self.alpha * v_k[name] for name in lambda_k}

        client_weight_vector_k_plus_1 = creat_weight_vector(self.model,self.m)
        client_G_weight_k_plus_1, _ = creat_g_matrix(i, self.clientsNum,self.m,client_weight_vector_k_plus_1)

        y_k_plus_1 =y_k + self.m*(client_G_weight_k_plus_1 - self.client_G_weight_k) #{name: y_k[name] + self.m * (w_k_plus_1[name] - weights[name]) for name in y_k}
        sum_accu = 0
        num = 0
        
        for data, label in self.testDataLoader:
            data, label = data.to(self.device), label.to(self.device)
            preds = self.model(data)
            preds = torch.argmax(preds, dim=1)
            sum_accu += (preds == label).float().mean()
            num += 1

        ACC_client=sum_accu / num
        print("\n")
        print(f'client ID : {self.client_ID}  accuracy: {ACC_client}')


        self.client_G_weight_k=client_G_weight_k_plus_1
        return y_k_plus_1, lambda_k_plus_1 , ACC_client ,gradient ,loss_generator




class genOptimizer:
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.lr = lr
 
    def step(self,gradients):
        for p in self.parameters:
            if p.grad is not None:
                print(p.grad.data())
                print("#"*80)
                gradients.append(p.grad.data())
    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

class CustomOptimizer:
    def __init__(self, parameters, lr,index_list, m):
        self.parameters = list(parameters)
        self.lr = lr
        self.m= m
        self.index_list=index_list
    
    def step(self,lamda):
        shp=0
        for p in self.parameters:
            if p.grad is not None:
                p.data -= self.lr * (p.grad.data + lamda[shp])
                shp += 1

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()



def creat_weight_vector(net,m):
        shap=0
        wght=torch.zeros(m)
        for name, param in net.named_parameters():
            w=param.data.view(-1)

            for j in w:
                wght[shap]=j.item()
                shap+=1
        return wght
    
def creat_y_k(n,m,g_matrix):
    y_k=torch.zeros(2*m*(n-1))
    for i in range(2*m*(n-1)):
        #    index=int(i%m)
        y_k[i]= m*g_matrix[i].item()
    return y_k

def creat_g_matrix(i,n,m,wi):
    g=[]

    zero_matrix=torch.zeros([m])
    index_list=[]
    if i==0:
        g.append(wi)
        g.append(wi*(-1))
        index_list.append(0)
        index_list.append(-1)
        for j in range(2,2*(n-1)):
            g.append(zero_matrix)
    elif i==(n-1):
        for j in range(2*(n-2)):
            g.append(zero_matrix)
        g.append(wi*(-1))
        g.append(wi)
        index_list.append((-1)*2*(n-2))
        index_list.append(2*(n-2)+1)
    else:
        index=2*(i-1)
        m_flag=0
        for j in range(4):
            index_list.append(index+j)
        for j in range(2*(n-1)):
            if j in index_list:
                a=[-1 if m_flag%4 ==0 or m_flag%4==3 else 1 ]
                m_flag+=1
                g.append(wi*((a[0])))
            else:
                g.append(zero_matrix)
        for j in range(4):
            a=[-1 if j%4 ==0 or j%4==3 else 1 ]
            index_list[j]=index_list[j]*a[0]
    g_matrix=torch.zeros(2*m*(n-1))
    shp=0
    for j in range(2*(n-1)):
        for k in range(m):
            g_matrix[shp]=g[j][k].item()
            shp+=1
    return g_matrix , index_list

class ClientsGroup():

    def __init__(self, dataSetName, clientsNum, device,model,globalPara,criterion,lossFun,localEpoch,localBatchSize,m,alpha,sigma
                 ,g_model,min_coumm_round,gen_iters,gen_lr,n_cls):

        self.dataSetName = dataSetName
        self.clientsNum = clientsNum
        self.device = device
        self.clients_set = {}
        self.client_weight_vector={}
        self.client_G_vector={}
        self.init_y_k={}
        self.index_list=[]
        self.testDataLoader = None
        self.model=model
        self.globalPara=globalPara
        self.lossFun=lossFun
        self.localEpoch=localEpoch
        self.localBatchSize=localBatchSize
        self.m=m
        self.alpha=alpha
        self.sigma=sigma
        self.g_model=g_model
        self.gen_lr=gen_lr
        self.min_coumm_round =min_coumm_round
        self.gen_iters = gen_iters
        self.n_cls=n_cls
        self.criterion=criterion
        self.dataSetAllocation()

   
    def get_class_number(self,localLabel,n_cls):
        clnt_cls_num = np.zeros((n_cls))
        for j in localLabel:
            clnt_cls_num[j.item()] += 1
        return clnt_cls_num  
    
    def generate_labels(self,number, cls_num):
        labels = np.arange(number)
        proportions = cls_num / cls_num.sum()
        proportions = (np.cumsum(proportions) * number).astype(int)[:-1]
        labels_split = np.split(labels, proportions)
        for i in range(len(labels_split)):
            labels_split[i].fill(i)
        labels = np.concatenate(labels_split)
        np.random.shuffle(labels)
        return labels.astype(int)
    
    def dataSetAllocation(self):

        dataSet = GetDataSet(self.dataSetName, )
        testData = dataSet.testData
        testLabel = dataSet.testLabel

        self.testDataLoader = DataLoader(TensorDataset(testData, testLabel), batch_size=100, shuffle=False)

        trainData = dataSet.trainData
        trainLabel = dataSet.trainLabel


        subClientDataSize = dataSet.trainDataSize // self.clientsNum # z

        for i in range(self.clientsNum):

            localData, localLabel = trainData[i*subClientDataSize:(i+1)*subClientDataSize], trainLabel[i*subClientDataSize:(i+1)*subClientDataSize]
            weight_vector=creat_weight_vector(self.model,self.m)

            G_vector, self.index_list =creat_g_matrix(i, self.clientsNum,self.m,weight_vector)

            # art=input("G_vector")
            init_y=creat_y_k(self.clientsNum,self.m,G_vector)

            self.init_y_k['client{}'.format(i)]=deepcopy(init_y)
            self.client_weight_vector['client{}'.format(i)]=deepcopy(weight_vector)
            self.client_G_vector['client{}'.format(i)]= deepcopy(G_vector)
            clnt_cls_num=self.get_class_number(localLabel,self.n_cls)
            labels_all_train_gen = self.generate_labels(self.gen_iters * self.localBatchSize, clnt_cls_num)
            labels_all_train_model=self.generate_labels(self.localEpoch* self.localBatchSize, clnt_cls_num)
            local = Client(TensorDataset(torch.tensor(localData), torch.tensor(localLabel)),self.testDataLoader, self.device,self.model,self.globalPara,self.criterion,self.lossFun,
            self.localEpoch,self.localBatchSize,self.m,self.alpha,self.sigma,self.index_list,self.clientsNum,i,self.client_G_vector['client{}'.format(i)]
                                                                        , self.g_model,self.gen_lr, self.min_coumm_round,self.gen_iters,labels_all_train_gen,labels_all_train_model,self.n_cls)

            self.clients_set['client{}'.format(i)] = local
            




