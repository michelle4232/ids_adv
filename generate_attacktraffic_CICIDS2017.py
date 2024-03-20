from sklearn.metrics import confusion_matrix, recall_score
import pandas as pd
import numpy as np
import torch as th
from torch import nn
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')



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
# Preprocess the dataset
def Preprocess_GAN(test):
    # select the last column
    testLabel = test.columns[-1] 
    test.rename(columns={testLabel: 'Label'}, inplace=True)
    # # min max standardization
    # numeric_columns = list(test.select_dtypes(include=['int', "float"]).columns) # select all columns that are numeric
    # numeric_columns.remove("BwdPSHFlags")
    # for c in numeric_columns:
    #     max_ = test[c].max()
    #     min_ = test[c].min()
    #     if max_ == 0:
    #         max = 0.1
    #     test[c] = test[c].map(lambda x: (x - min_) / (max_ - min_))

    # 創建一個MinMaxScaler對象
    scaler = MinMaxScaler()

    # 選擇所有數值型列，並移除"BwdPSHFlags"
    numeric_columns = list(test.select_dtypes(include=['int', "float"]).columns)
    numeric_columns.remove("BwdPSHFlags")

    # 對每一個數值型列進行縮放
    for c in numeric_columns:
        test[c] = scaler.fit_transform(test[[c]])

    #  1: annomaly; 0: normaly
    # train["DT_Predicted"] = train["DT_Predicted"].map(lambda x: 1 if x == "anomaly" else 0)
    # get all rows of malicious traffic, and all columns except the last one
    raw_attack = np.array(test[test["Label"] == 1])[:, :-1]
    # get all rows of benign traffic, and all columns except the last one
    normal = np.array(test[test["Label"] == 0])[:, :-1]
    
    # get the true label of the train set
    true_label = test["Label"]

    del test["Label"]

    return test, raw_attack, normal, true_label

class Generator(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(Generator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim // 2, input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim // 2, input_dim//2),
            nn.ReLU(True),
            # nn.Linear(input_dim // 2,input_dim//2),
            # nn.ReLU(True),
            nn.Linear(input_dim//2,output_dim),
        )
    def forward(self,x):
        x = self.layer(x)
        return th.clamp(x,0.,1.)


def CreateBatch_GAN(x, batch_size):
    
    # shuffle the index of the data
    a = list(range(len(x)))
    np.random.shuffle(a)
    x = x[a]
    
    # transfer data to batch with batch size = 256
    # There are 17930 rows in the dataset, so the batch size is 256, there will be 77 batches
    batch_x = [x[batch_size * i: (i + 1) * batch_size, :] for i in range(len(x) // batch_size)]

    return batch_x



# test_dataset = pd.read_csv("datasets/KDD_dataset/KDDTest+.csv")
# datasets\target_model\CICIDS2017\target_model_CICIDS2017data.csv
# test_dataset = pd.read_csv("datasets/surrogate_model/CICIDS2017/data_for_gan/test_dt_predicted.csv")
test_dataset = pd.read_csv("datasets/surrogate_model/CICIDS2017/data_for_gan/test_dt_predicted.csv")
test, test_raw_attack, test_normal, true_label = Preprocess_GAN(test_dataset)

BATCH_SIZE = 256 # Batch size
D_G_INPUT_DIM = test_normal.shape[1]
G_OUTPUT_DIM =test_normal.shape[1] 
D_OUTPUT_DIM = 1
NUM_OF_TESTS=100

#read model
random_g = Generator(D_G_INPUT_DIM,G_OUTPUT_DIM)
learned_g = Generator(D_G_INPUT_DIM,G_OUTPUT_DIM)

# ids_model = Blackbox_IDS(D_G_INPUT_DIM,2)
# ids_param= th.load('datasets/KDD_dataset/IDS.pth',map_location=lambda x,y:x)
# ids_model.load_state_dict(ids_param)
ids_model = pickle.load(open('surrogate_model/ml_model/dt_model_from_dtdata.pickle', 'rb')) #surrogate_model/ml_model/lr_model_from_dtdata.pickle
# ids_model = pickle.load(open('target_model/ml_model/CICIDS2018target_lr.pickle', 'rb')) #surrogate_model/ml_model/lr_model_from_dtdata.pickle
g_param = th.load('GAN_materials/testGAN/gan_model/from_dt_surrogate_model/generator_dt_model_from_dtdata_0319_2338.pth',map_location=lambda x,y:x)
learned_g.load_state_dict(g_param)

# learned_g = parameters from the trained model
# model_g = {"no_learn":random_g,"learned":learned_g}
model_g = {"learned":learned_g}

# print("model_g: ", model_g)
print("adversarial traffic evaluating")
print("-"*100)
odr_tests, adr_tests, eir_tests = [], [], []
for NUM in range (NUM_OF_TESTS): #100
    test_batch_normal = CreateBatch_GAN(test_normal,BATCH_SIZE) #benign traffic
    # print("test_batch_normal: ", test_batch_normal)
    # print("test_batch_normal shape: ", len(test_batch_normal))
    for n,g in model_g.items():
        o_dr,a_dr,eir=[],[],[]
        # print("n: ", n)
        # print("g: ", g)
        g.eval()
        with th.no_grad():
            i = 0
            for bn in test_batch_normal: #70 batches, per batch size = 256 (256 normal traffic samples per batch)
                # print("type(bn): ", type(bn))
                # print("bn shape: ", bn.shape)
                # print(f"Batch {i + 1} of {len(test_batch_normal)}")
                i += 1
                normal_b = th.Tensor(bn)
                # print("type(normal_b): ", type(normal_b))
                # print("normal_b shape: ", normal_b.shape)
                # print("type(test_raw_attack): ", type(test_raw_attack))
                # print("test_raw_attack shape: ", test_raw_attack.shape) #(6155, 42)
                
                #select 256 random attack samples in test_raw_attack(each batch size = 256, 42 features per sample)
                # transform the numpy array to tensor
                batch_a= th.Tensor(test_raw_attack[np.random.randint(0,len(test_raw_attack),BATCH_SIZE)])
                #test_raw_attack with random noise
                z = batch_a + th.Tensor(np.random.uniform(0,1,(BATCH_SIZE,D_G_INPUT_DIM))) 
                
                # generate adversarial attack from generator
                # shape of adversarial_attack: (256, 42)
                adversarial_attack = g(z)
                # print("adversarial_attack: ", adversarial_attack)
                # print("adversarial_attack shape: ", adversarial_attack.shape)
                # adversarial_attack[:,33:] = th.Tensor(np.where(adversarial_attack[:,33:].detach().cpu().numpy()>= 0.5 , 1,0))
                # print("adversarial_attack: ", len(adversarial_attack[:,33:]))
                # print("adversarial_attack shape: ", adversarial_attack[:,34:].shape)
                ori_input = th.cat((batch_a,normal_b)) #combine normal and attack traffic
                #adv_input = th.cat((adversarial_attack,normal_b))#combine normal and adversarial attack traffic
                adv_input = th.cat((normal_b, adversarial_attack))
                l = list(range(len(ori_input)))
                np.random.shuffle(l)
                # print("ori_input: ", ori_input) 
                # print("ori_input shape: ", ori_input.shape) # (512, 42)
                # print("adv_input ", adv_input)
                # print("adv_input shape: ", adv_input.shape) # (512, 42)
                # (print("l: ", l))
                # print("l shape: ", len(l)) # 512
                
                # print adverarial attack
                # print("adversarial attack: ", adversarial_attack)
                # print("adversarial attack shape: ", adversarial_attack.shape)
        
                adv_input_l = adv_input[l] # raw attack traffic
                ori_input_l = ori_input[l] # adversarial attack traffic
                # ids_pred_adv = ids_model(adv_input)
                # ids_pred_ori = ids_model(ori_input)
                # adv_input = adv_input.values
                # ori_input = ori_input.values

                adv_input_df = pd.DataFrame(adv_input_l.detach().numpy(), columns=feature_names)
                ori_input_df = pd.DataFrame(ori_input_l.detach().numpy(), columns=feature_names) 
                ids_pred_adv = ids_model.predict(adv_input_df)
                ids_pred_ori = ids_model.predict(ori_input_df)
                # print("ids_pred_adv: ", ids_pred_adv)
                ids_true_label = np.r_[np.ones(BATCH_SIZE),np.zeros(BATCH_SIZE)][l].astype(int)
                # ids_true_label = np.r_[np.ones(BATCH_SIZE),np.zeros(BATCH_SIZE)]
                # pred_label_adv = th.argmax(nn.Sigmoid()(ids_pred_adv),dim = 1).cpu().numpy()
                # pred_label_ori = th.argmax(nn.Sigmoid()(ids_pred_ori),dim = 1).cpu().numpy()
                #print("ids_pred_adv: ", ids_pred_adv)
                pred_label_adv = ids_pred_adv
                pred_label_ori = ids_pred_ori
                # print("pred_label_adv: ", pred_label_adv)
                # print("pred_label_ori: ", pred_label_ori)
                # print("ids_true_label: ", ids_true_label)
                # print("ids_true_label shape: ", ids_true_label.shape)
                # print("pred_label_adv shape: ", pred_label_adv.shape)
                # print("pred_label_ori shape: ", pred_label_ori.shape)
                # cm = confusion_matrix(ids_true_label,pred_label_adv)
                # print("confusion matrix: ", cm)
                # tn1, fp1, fn1, tp1 = confusion_matrix(ids_true_label,pred_label_adv).ravel()
                # tn2, fp2, fn2, tp2 = confusion_matrix(ids_true_label,pred_label_ori).ravel()
                # print("tn1: ", tn1)
                # print("fp1: ", fp1)
                # print("fn1: ", fn1)
                # print("tp1: ", tp1)
                # print("recall: ", recall_score(ids_true_label,pred_label_adv))
                # o_dr.append(tp2/(tp2 + fp2))
                # a_dr.append(tp1/(tp1 + fp1))
                # eir.append(1 - (tp1/(tp1 + fp1))/(tp2/(tp2 + fp2)))
               
                o_dr.append(recall_score(ids_true_label, pred_label_ori))
                a_dr.append(recall_score(ids_true_label, pred_label_adv))
                eir.append(1 - recall_score(ids_true_label, pred_label_adv)/recall_score(ids_true_label, pred_label_ori))
        avg_odr=np.mean(o_dr)
        avg_adr=np.mean(a_dr)
        avg_eir=np.mean(eir)

        print(f"{NUM + 1}: {n} => origin_DR : {avg_odr} \t adversarial_DR : {avg_adr} \t EIR : {avg_eir}") 
        if (n=="learned"):
            odr_tests.append(avg_odr)
            adr_tests.append(avg_adr)
            eir_tests.append(avg_eir)
    print()  

plt.plot(odr_tests, label="Original detection rate")
plt.plot(adr_tests, label="Adversarial detection rate")
#plt.plot(eir_tests, label="Evasion increase rate")
# set y axis from 0 to 1
plt.ylim(0, 1)
plt.legend()
timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"GAN_materials/result/graph/Test_{timestamp}.png")
plt.show()