import matplotlib as mpl
mpl.use('TkAgg')
import numpy as np
import pandas as pd
import torch as th
from torch.autograd import Variable as V
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from preprocessing import Preprocess_GAN,CreateBatch_GAN
# from model.model_class import Blackbox_IDS,Generator,Discriminator
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import torch as th
from torch import nn
from torch.autograd import Variable as V
import torch
import torch.nn as nn
import torch.optim as optim

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

def Preprocess_GAN(train):
        
    # RobustScaler
    scaler = RobustScaler()
    # 選擇所有數值型列，並移除"BwdPSHFlags"
    numeric_columns = list(train.select_dtypes(include=['int', "float"]).columns)
    numeric_columns.remove("BwdPSHFlags")
    numeric_columns.remove("Label")
    
    # 對每一個數值型列進行縮放
    for c in numeric_columns:
        train[c] = scaler.fit_transform(train[[c]]) # return a dataframe

    raw_attack = np.array(train[train["Label"] == 1])[:, :-1]
    # get all rows of benign traffic, and all columns except the last one
    normal = np.array(train[train["Label"] == 0])[:, :-1]
    
    # get the true label of the train set
    true_label = train["Label"]

    del train["Label"]

    return train, raw_attack, normal, true_label


def CreateBatch_GAN(x, batch_size):
    a = list(range(len(x)))
    np.random.shuffle(a)
    x = x[a]
    batch_x = [x[batch_size * i: (i + 1) * batch_size, :] for i in range(len(x) // batch_size)]
    return batch_x


class Generator(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(Generator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim//2, 256),
            nn.ReLU(True),
            nn.Linear(256, input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim // 2, input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim // 2,input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim//2,output_dim),
            # nn.Sigmoid()
        )
    def forward(self,x):
        x = self.layer(x)
        # return x
        return th.clamp(x,-1.,1.)

class Discriminator(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim*2, input_dim*2),
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
    

class DNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DNN, self).__init__()
        # Our first linear layer take input_size, in this case 784 nodes to 50
        # and our second linear layer takes 50 to the num_classes we have, in
        # this case 10.
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=0.6)
        
        self.relu = nn.ReLU()
        self.act3 = nn.Sigmoid()
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.batchnorm3 = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.act3(x)
        return x


# read the training dataset and preprocess it
train_dataset = pd.read_csv("surrogate_model/data_for_training/0322/df_dt_predicted_0322.csv")

# get the last column name
last_col = train_dataset.columns[-1]
train_dataset.rename(columns={last_col: 'Label'}, inplace=True)

train_data,raw_attack,normal,true_label = Preprocess_GAN(train_dataset)

# check if raw_traffic is nan
print(np.isnan(raw_attack).any())
print(raw_attack.shape)
print(type(raw_attack))
np.argwhere(np.isnan(raw_attack))

#DEFINE
BATCH_SIZE = 128 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10     # Gradient penalty lambda hyperparameter
MAX_EPOCH = 50 # How many generator iterations to train for
D_G_INPUT_DIM = len(train_data.columns) # 42 features
G_OUTPUT_DIM = len(train_data.columns) # 42 features
D_OUTPUT_DIM = 1
CLAMP = 0.00001 # WGAN clip weights
LEARNING_RATE_G=0.001
LEARNING_RATE_D=0.0001


# Load BlackBox IDS model 
# ids_model = Blackbox_IDS(D_G_INPUT_DIM,2)
ids_model = DNN(input_size=D_G_INPUT_DIM, num_classes=2)
# param = th.load('datasets/KDD_dataset/IDS.pth')
param = th.load('surrogate_model/surrogate_DNN_model/surrogateDNN_model_fromDT_20240326_221707_37')#.pth
ids_model.load_state_dict(param)
# ids_model = pickle.load(open('surrogate_model/ml_model/lr_model_from_dnndata.pickle', 'rb')) #surrogate_model/ml_model/lr_model_from_dtdata.pickle


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
batch_attack = CreateBatch_GAN(raw_attack,BATCH_SIZE)
d_losses,g_losses = [],[] #loss status
ids_model.eval()

generator.train()
discriminator.train()

cnt = -5
print("IDSGAN start training")
print("-"*100)
for epoch in range(MAX_EPOCH):
    # train one batch per epoch
    normal_batch = CreateBatch_GAN(normal,BATCH_SIZE)
    epoch_g_loss = 0.
    epoch_d_loss = 0.
    c=0
    for nb in normal_batch:
        normal_b = th.Tensor(nb)
        #  Train Generator
        for p in discriminator.parameters():
            p.requires_grad = False

        optimizer_G.zero_grad()
        
        # 將 raw_attack 中的隨機 n=BATCH_SIZE 個元素提取為 random_traffic
        random_attack_traffic = raw_attack[np.random.randint(0,len(raw_attack),BATCH_SIZE)]
        # 從 random_traffic 中提取，並添加來自0到1之間的隨機噪音值。
        ###!! random_traffic_noised - random_traffic_noised 的值可能大於 1
        random_traffic_noised = random_attack_traffic + np.random.uniform(0,1,(BATCH_SIZE,D_G_INPUT_DIM))

        z = V(th.Tensor(random_traffic_noised))
        adversarial_traffic = generator(z) #generate attack traffic

        D_pred= discriminator(adversarial_traffic) #discriminator generated output

        g_loss = -1 * discriminator(generator(z)).mean()
        g_loss.backward()
        optimizer_G.step()

        epoch_g_loss += g_loss.item()
        # Train Discriminator
        for p in discriminator.parameters():
            p.requires_grad = True

        for c in range(CRITIC_ITERS): # update discriminator parameter per loop
            optimizer_D.zero_grad() # zero_grad() clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).
            for p in discriminator.parameters(): #weighting clipping
                p.data.clamp_(-CLAMP, CLAMP)
                
                
            # generate adversarial traffic
            temp_data = raw_attack[np.random.randint(0,len(raw_attack),BATCH_SIZE)] + np.random.uniform(0, 1,(BATCH_SIZE,D_G_INPUT_DIM))
            z = V(th.Tensor(temp_data))
            adversarial_traffic = generator(z).detach()

            # print("adversarial_traffic: ", adversarial_traffic)
            ids_input = th.cat((adversarial_traffic,normal_b))

                
            l = list(range(len(ids_input)))
            np.random.shuffle(l)
            ids_input = V(th.Tensor(ids_input[l]))

            ids_pred = ids_model(ids_input)
            ids_pred_label = th.argmax(ids_pred,dim = 1).detach().numpy()
                        
            # # 將 ids_input 轉換為 DataFrame，並設置特徵名稱
            # ids_input_df = pd.DataFrame(ids_input.detach().numpy(), columns=feature_names)
            # ids_pred_label = ids_model.predict(ids_input_df)   
            print("ids_pred_label: ", ids_pred_label) 

            pred_normal = ids_input.numpy()[ids_pred_label==0]
            pred_attack = ids_input.numpy()[ids_pred_label==1]
            # print("ids_pred_label: ", ids_pred_label)

            if len(pred_attack) == 0: 
                cnt += 1
                break

            D_normal = discriminator(V(th.Tensor(pred_normal)))
            D_attack= discriminator(V(th.Tensor(pred_attack)))

            loss_normal = th.mean(D_normal)
            loss_attack = th.mean(D_attack)  
            d_loss =  loss_attack - loss_normal 
            d_loss.backward()
            optimizer_D.step()
            epoch_d_loss += d_loss.item()

    d_losses.append(epoch_d_loss/CRITIC_ITERS)
    g_losses.append(epoch_g_loss)
    print(f"{epoch} : {epoch_g_loss} \t {epoch_d_loss/CRITIC_ITERS}")


print("IDSGAN finish training")

th.save(generator.state_dict(), 'GAN_materials/testGAN/gan_model/from_dnn_surrogate_model/0326/generator_dnn_model_from_dtdata_0326_1721.pth') # GAN_materials\testGAN\gan_model\from_dt_surrogate_model\discriminator_dt_model_from_dtdata_0319_2338.pth
th.save(discriminator.state_dict(), 'GAN_materials/testGAN/gan_model/from_dnn_surrogate_model/0326/discriminator_dnn_model_from_dtdata_0326_1721.pth')

plt.plot(d_losses,label = "D_loss")
plt.plot(g_losses, label = "G_loss")
plt.legend()
plt.show()




