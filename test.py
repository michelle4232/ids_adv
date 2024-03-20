import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import pandas as pd
import torch as th
from torch import nn
from torch.autograd import Variable as V


ids_model = pickle.load(open('surrogate_model/ml_model/dt_model_from_dtdata.pickle', 'rb')) #surrogate_model/ml_model/lr_model_from_dtdata.pickle

 # feature_names of the CICIDS 2017 dataset
feature_names = ['FlowDuration', 'TotFwdPkts', 'TotBwdPkts', 'TotLenFwdPkts',
    'TotLenBwdPkts', 'FwdPktLenMin', 'FwdPktLenStd', 'BwdPktLenMax',
    'BwdPktLenMean', 'BwdPktLenStd', 'FlowByts/s', 'FlowPkts/s',
    'FlowIATStd', 'FwdIATTot', 'FwdIATMean', 'FwdIATMax', 'BwdIATMean',
    'BwdIATStd', 'BwdIATMax', 'BwdIATMin', 'BwdPSHFlags', 'FwdHeaderLen',
    'BwdHeaderLen', 'FwdPkts/s', 'BwdPkts/s', 'PktLenMax', 'PktLenStd',
    'FINFlagCnt', 'SYNFlagCnt', 'ACKFlagCnt', 'Down/UpRatio',
    'BwdSegSizeAvg', 'FwdHeaderLen.1', 'SubflowFwdPkts', 'SubflowFwdByts',
    'IdleStd', 'SubflowBwdPkts', 'SubflowBwdByts', 'InitBwdWinByts',
    'FwdActDataPkts', 'ActiveStd', 'ActiveMax']

class Generator(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(Generator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 21), #input_dim//2
            nn.ReLU(True),
            nn.Linear(21, 21),
            nn.ReLU(True),
            nn.Linear(21, 21),
            nn.ReLU(True),
            # nn.Linear(21, 21),
            # nn.ReLU(True),
            nn.Linear(21,output_dim),
        )
    def forward(self,x):
        x = self.layer(x)
        return th.clamp(x,0.,1.)

class Discriminator(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim * 2, input_dim),
            nn.LeakyReLU(True),
            #nn.Linear(input_dim*2 , input_dim*2),
            #nn.LeakyReLU(True),
            nn.Linear(input_dim,input_dim//2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim//2,output_dim),
        )

    def forward(self,x):
        return self.layer(x)



def Preprocess_GAN(train):
        
    # min max standardization
    numeric_columns = list(train.select_dtypes(include=['int', "float"]).columns) # select all columns that are numeric
    numeric_columns.remove("BwdPSHFlags")
    for c in numeric_columns:
        max_ = train[c].max()
        min_ = train[c].min()
        if max_ == 0:
            max = 0.1
        train[c] = train[c].map(lambda x: (x - min_) / (max_ - min_))


    #  1: annomaly; 0: normaly
    # train["DT_Predicted"] = train["DT_Predicted"].map(lambda x: 1 if x == "anomaly" else 0)
    # get all rows of malicious traffic, and all columns except the last one
    raw_attack = np.array(train[train["Label"] == 1])[:, :-1]
    # get all rows of benign traffic, and all columns except the last one
    normal = np.array(train[train["Label"] == 0])[:, :-1]
    
    # get the true label of the train set
    true_label = train["Label"]

    del train["Label"]

    return train, raw_attack, normal, true_label

train_dataset = pd.read_csv("datasets/surrogate_model/CICIDS2017/data_for_gan/train_dt_predicted.csv")
val_dataset = pd.read_csv("datasets/surrogate_model/CICIDS2017/data_for_gan/val_dt_predicted.csv")
test_dataset = pd.read_csv("datasets/surrogate_model/CICIDS2017/data_for_gan/test_dt_predicted.csv")
train_data,raw_attack,normal,true_label = Preprocess_GAN(train_dataset)

BATCH_SIZE = 64 # Batch size
CRITIC_ITERS = 15 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10     # Gradient penalty lambda hyperparameter
MAX_EPOCH = 50 # How many generator iterations to train for
D_G_INPUT_DIM = len(train_data.columns) # 9 features
G_OUTPUT_DIM = len(train_data.columns) # 9 features
D_OUTPUT_DIM = 1
CLAMP = 0.01
LEARNING_RATE_G=0.0001
LEARNING_RATE_D=0.000001


generator = Generator(D_G_INPUT_DIM,G_OUTPUT_DIM)
print(100*'=')
print(generator)

discriminator = Discriminator(D_G_INPUT_DIM,D_OUTPUT_DIM)
print(100*'=')
print(discriminator)


#Optimization. Similar to Gradient Descent. https://viblo.asia/p/thuat-toan-toi-uu-adam-aWj53k8Q56m
optimizer_G = optim.RMSprop(generator.parameters(), LEARNING_RATE_G)
optimizer_D = optim.RMSprop(discriminator.parameters(), LEARNING_RATE_D)

# 由於不可能放入整個資料集，因此資料集會分批輸出（更小、相等的部分）。
#batch_attack = CreateBatch_GAN(raw_attack,BATCH_SIZE)
d_losses,g_losses = [],[] #loss status
#ids_model.eval()

generator.train()
discriminator.train()

cnt = -5
print("IDSGAN start training")
print("-"*100)


# ids_input = normal[:2, :]
# print(ids_input.shape)
adversarial_traffic_first_batch = pd.read_csv("datasets/surrogate_model/CICIDS2017/data_for_gan/adversarial_traffic_first_batch.csv")
print(adversarial_traffic_first_batch.shape)
print(adversarial_traffic_first_batch   )
pd.set_option('display.max_columns', None)
adversarial_traffic_first_batch.head()
# to numpy
adversarial_traffic_first_batch = np.array(adversarial_traffic_first_batch)
ids_input = V(th.Tensor(adversarial_traffic_first_batch))
ids_input_df = pd.DataFrame(ids_input.detach().numpy(), columns=feature_names)
ids_pred_label = ids_model.predict(ids_input_df)   
print("ids_pred_label: ", ids_pred_label) 

pred_normal = ids_input.numpy()[ids_pred_label==0]
pred_attack = ids_input.numpy()[ids_pred_label==1]


# if len(pred_attack) == 0: #!!!!!why!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#     cnt += 1
#     break

D_normal = discriminator(V(th.Tensor(pred_normal)))
D_attack= discriminator(V(th.Tensor(pred_attack)))
# print("D_normal: ", D_normal)
# print("D_attack: ", D_attack)
loss_normal = th.mean(D_normal)
# print(th.isnan(D_normal).any())
# print(th.isinf(D_normal).any())
# print("loss_normal: ", loss_normal)
loss_attack = th.mean(D_attack)
#gradient_penalty = compute_gradient_penalty(discriminator, normal_b.data, adversarial_traffic.data)
d_loss =  loss_attack - loss_normal #+ LAMBDA * gradient_penalty
d_loss.backward()
optimizer_D.step()
print(d_loss)
# epoch_d_loss += d_loss.item()


print("IDSGAN finish training")