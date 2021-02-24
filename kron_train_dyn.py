import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import Parameter
import torch.optim as optim
import torch.nn.functional as F

import pickle
import numpy as np
import time
import math # to check nan
from sklearn import metrics

HYP={
    'node_num':128,
    'miss_percent':0.1,
    'seed':2051,
    'batch_size':1024,
    'lr_net':0.001,
    'lr_net_stru':0.001,
    'lr_net_dyn':0.0001,
    # 'lr_net_dyn':0.000003,
    'lr_dyn':0.0001,
    'epoch':30000
}
print(HYP)
del_num = int(HYP['node_num']*HYP['miss_percent'])
torch.manual_seed(HYP['seed'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## here is a dynamics learner, and ites optimizer
class IO_B(nn.Module):
    """docstring for IO_B"""
    def __init__(self,dim,hid):
        super(IO_B, self).__init__()
        self.dim = dim
        self.hid = hid
        self.n2e = nn.Linear(2*dim,hid)
        self.e2e = nn.Linear(hid,hid)
        self.e2n = nn.Linear(hid,hid)
        self.n2n = nn.Linear(hid,hid)
        self.output = nn.Linear(dim+hid,dim)
    def forward(self, x, adj_col, i):
        # x : features of all nodes at time t,[b*n*d]
        # adj_col : i th column of adj mat,[n]
        # i : just i
        starter = x # 128,10,4
        ender = x[:,i,:] # 128,4
        ender = ender.unsqueeze(1) #128,1,4

        ender = ender.expand(starter.size(0),starter.size(1),starter.size(2)) #128,10,4
        x = torch.cat((starter,ender),2) #128,10,8
        x = F.relu(self.n2e(x))#128,10,256

        x = F.relu(self.e2e(x))#128,10,256
        adj_col = adj_col.unsqueeze(1)#10,1
        adj_col = adj_col.unsqueeze(0).repeat(x.shape[0],1,x.shape[2]) #128,10,256
        
        x = x * adj_col#128,10,256

        x = torch.sum(x,1)#128,256
        x = F.relu(self.e2n(x))#128,256
        x = F.relu(self.n2n(x))#128,256

        x = torch.cat((starter[:,i,:],x),dim=-1)#128,256+4
        x = self.output(x)#128,4

        # skip connection
        # x = starter[:,i,:]+x # dont want in CML
        return x

# kronecker product for 2 matrix
def kronecker(A,B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

## network generator, and ites optimizer
class Gumbel_union_kron(nn.Module):
    def __init__(self,sz,k):
        super(Gumbel_union_kron,self).__init__()
        self.p = Parameter(torch.Tensor(sz,sz).uniform_(0,0.5))
#         self.p = Parameter(torch.tensor([[0.92,0.4],[0.5,0.6]]))
        self.k = k
    
    # generate adj from original kernel
    def generate_adj(self):
        p0 = torch.relu(self.p-0.001)+0.001 - torch.relu(self.p-0.999)
        adj = torch.relu(self.p-0.001)+0.001 - torch.relu(self.p-0.999)
        for i in range(self.k-1):
            adj = kronecker(adj,p0)
        return adj
    
    def sample_all(self):
        p1 = self.generate_adj() # prob to get 1
        p0 = 1-p1 # prob to get 0
        logits = torch.cat([p0.unsqueeze(2),p1.unsqueeze(2)],dim=2)#[node,node,2]
        # log it to suit for gumbel softmax
        logits = torch.log(logits)#[node,node,2]
        # gumbel softmax
        sampled_adj = torch.nn.functional.gumbel_softmax(logits,hard=False,tau=1)[:,:,1]
        return sampled_adj
    # generate p0 to init former gumbel generator
    def generate_p0_for_init(self):
        p1 = self.generate_adj() # prob to get 1
        p0 = 1-p1 # prob to get 0
        logits = torch.cat([p0.unsqueeze(2),p1.unsqueeze(2)],dim=2)#[node,node,2]
        # log it to suit for gumbel softmax
        logits = torch.log(logits)#[node,node,2]
        return logits


# normal network generator
class Gumbel_Generator(nn.Module):
    def __init__(self,sz):
        super(Gumbel_Generator,self).__init__()
        self.p = Parameter(torch.randn(sz,sz,2))
    def generate_adj(self):
        p0 = torch.nn.functional.softmax(self.p,dim=2)[:,:,1]
        symp0 = torch.triu(p0,diagonal=0)+torch.triu(p0,diagonal=1).transpose(0,1) # make it symmetric
        return symp0
        
    def sample_all(self):
        sampled_adj = torch.nn.functional.gumbel_softmax(self.p,hard=False,tau=1)[:,:,1]
        sym_adj = torch.triu(sampled_adj,diagonal=0)+torch.triu(sampled_adj,diagonal=1).transpose(0,1)# make it symmetric
        return sym_adj

def load_cml_ggn(data_path,batch_size = 128,node=10,seed=2050):

    with open(data_path, 'rb') as f:
        object_matrix, train_data, val_data, test_data = pickle.load(f) # (samples, nodes, timesteps, 1)
    
    print('\nMatrix dimension: %s \nTrain data size: %s \nVal data size: %s \nTest data size: %s'
          % (object_matrix.shape, train_data.shape, val_data.shape, test_data.shape))    

    print('use 2000 instead of 10000')
    # train_data = train_data[:2000]

    train_loader = DataLoader(train_data[:], batch_size=batch_size, shuffle=False)#
    val_loader = DataLoader(val_data[:], batch_size=batch_size, shuffle=False) # 记得改回来
    test_loader = DataLoader(test_data[:], batch_size=batch_size, shuffle=False) # 记得改回来
    return train_loader,val_loader,test_loader,object_matrix


## generate initial state for unobserved nodes
class Unknown_X_Generator(nn.Module):
    def __init__(self,batch_num,batch_sz,unobserved_node):
        super(Unknown_X_Generator,self).__init__()
        self.para = Parameter(torch.Tensor(batch_num,batch_sz,unobserved_node).uniform_(0,1))
    # get x
    def get_x(self,batch_idx):
        return self.para[batch_idx,:,:].unsqueeze(2) #[batchsz,unobserved_node,1]

def auc_unknown(p0,object_matrix,del_num):
    yscore = []
    y = []
    for i in range(p0.shape[0]):
        if i < p0.shape[0]-del_num:
            yscore.extend(p0[i,-del_num:])
            y.extend(object_matrix[i,-del_num:])
        else:
            yscore.extend(p0[i])
            y.extend(object_matrix[i])
    yscore = np.array(yscore)
    y_true = np.array(y)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_true, yscore,drop_intermediate=False)
    area = metrics.auc(fpr, tpr)
    return area

# load data
data_path = './data/2000cmlkron-ws128-10000-00.pickle'
# data_path = './data/2000cmlkron-10-100-10000-00.pickle'
# a = torch.load(data_path,map_location=torch.device('cpu'))
train_loader, val_loader, test_loader, object_matrix = load_cml_ggn(data_path,batch_size=HYP['batch_size'],node=HYP['node_num'])
observed_adj = object_matrix[:-del_num,:-del_num]

generator = Gumbel_union_kron(2,7).to(device)
# generator = Gumbel_Generator(128).to(device)
op_net = optim.Adam(generator.parameters(), lr=HYP['lr_net_stru'])
# op_net_dyn = optim.Adam(generator.parameters(), lr=HYP['lr_net_dyn'])

dyn = IO_B(1,128).to(device)
op_dyn = optim.Adam(dyn.parameters(), lr=HYP['lr_dyn'])

train_uxg = Unknown_X_Generator(len(train_loader),HYP['batch_size'],del_num).to(device)
op_uxg = optim.Adam(train_uxg.parameters(), lr=HYP['lr_net'])

loss_f = nn.MSELoss()

def train_dyn_gen():
    losses = []
    for b_idx,data in enumerate(train_loader):
        data = data.to(device)
        # data:[batch,node,2,1]
        x = data[:,:-del_num,0,:] # [batchsz,observed_node,1]
        y = data[:,:-del_num,1,:] # same size
        
        # pred state for each node which can be observed
        for j in range(HYP['node_num']-del_num):
            op_init.zero_grad()
            # op_net_dyn.zero_grad()
            op_uxg.zero_grad()
            op_dyn.zero_grad()
            
            
            thisbatchsz = x.shape[0]
            x_un_paed = train_uxg.get_x(b_idx)[:thisbatchsz] #[batchsz,unobserved_node,1]
            x_hypo = torch.cat([x,x_un_paed],dim=1) #[batchsz,node,1]
            
            # hypo_adj = generator.sample_all()
            hypo_adj = g_init.sample_all()
            
            # replace the observed part
            t0 = torch.ones(hypo_adj.shape).to(device)
            t0[:-del_num,:-del_num] = 0
            t1 = torch.zeros(hypo_adj.shape).to(device)
            t1[:-del_num,:-del_num] = observed_adj
            hypo_adj = hypo_adj*t0+t1
            adj_col = hypo_adj[:,j]
            
            yhat = dyn(x_hypo,adj_col,j)
            loss = loss_f(yhat,y[:,j,:])
            loss.backward()
            
            op_init.step()
            # op_net_dyn.step()
            op_dyn.step()
            op_uxg.step()
            losses.append(loss.item())
    return np.mean(losses)

# train by observed structure
def train_stru():
    op_net.zero_grad()
    hypo_adj = generator.generate_adj()
    o_part = hypo_adj[:-del_num,:-del_num]
    # print(o_part)
    # print(observed_adj)
    loss = loss_f(o_part,observed_adj.to(device))
    loss.backward()
    op_net.step()
    return loss.item()



areas = []
losses = []
loss_strus = []
times = []

# train by structure
for i in range(500):
    train_stru()
#     print(generator.p)
    p0 = generator.generate_adj().detach().cpu().numpy()
    area = auc_unknown(p0,object_matrix,del_num)
    print('auc:',area)

# normal network generator
class Gumbel_Generator_Init(nn.Module):
    def __init__(self,init_p0):
        super(Gumbel_Generator_Init,self).__init__()
#         self.p = Parameter(torch.randn(sz,sz,2))
        self.p = Parameter(init_p0)
    def generate_adj(self):
        p0 = torch.nn.functional.softmax(self.p,dim=2)[:,:,1]
        symp0 = torch.triu(p0,diagonal=0)+torch.triu(p0,diagonal=1).transpose(0,1) # make it symmetric
        return symp0
    def sample_all(self):
        sampled_adj = torch.nn.functional.gumbel_softmax(self.p,hard=False,tau=1)[:,:,1]
        sym_adj = torch.triu(sampled_adj,diagonal=0)+torch.triu(sampled_adj,diagonal=1).transpose(0,1)# make it symmetric
        return sym_adj

init_p0 = generator.generate_p0_for_init().detach()
g_init = Gumbel_Generator_Init(init_p0)
op_init = optim.Adam(g_init.parameters(), lr=HYP['lr_net'])

for i in range(HYP['epoch']):
    # train by dynamics info
    print('epoch:',i+500)

    loss = train_dyn_gen()
    losses.append(loss)
    print('train loss:',loss)

    # train by observed stru info
    # lstru = train_stru()
    # loss_strus.append(lstru)
    # print('train structural loss:',lstru)

    # check auc
    p0 = g_init.generate_adj().cpu().detach().numpy()
    area = auc_unknown(p0,object_matrix,del_num)
    print('auc:',area)
    areas.append(area)

    # estimate end time
    times.append(time.time())
    if i > 2:
        ls = times[-1]
        fi = times[0]
        end = (ls-fi)/i*HYP['epoch']+fi
        tt = time.localtime(end) #time tuple
        print('estimated end time:',tt.tm_year,'/',tt.tm_mon,'/',tt.tm_mday,'  ',tt.tm_hour,':',tt.tm_min)