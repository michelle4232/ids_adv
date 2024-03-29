from sklearn.metrics import confusion_matrix, recall_score
import pandas as pd
import numpy as np
import torch as th
from torch import nn
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
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
    # min max standardization
    # 創建一個MinMaxScaler對象
    # scaler = MinMaxScaler()
    # scaler = StandardScaler()
    scaler = RobustScaler()
 
    # 選擇所有數值型列，並移除"BwdPSHFlags"
    numeric_columns = list(test.select_dtypes(include=['int', "float"]).columns)
    numeric_columns.remove("BwdPSHFlags")
    numeric_columns.remove("Label")
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
test_dataset = pd.read_csv("datasets/surrogate_model/CICIDS2017/split_dataset/test_CICIDS2017.csv")
# test_dataset = pd.read_csv("surrogate_model/data_for_training/0322/df_lr_predicted_0322.csv")
# test_dataset = test_dataset.drop(columns=['Label'], axis=1)
# # get the last column name
# last_col = test_dataset.columns[-1]
# test_dataset.rename(columns={last_col: 'Label'}, inplace=True)
test, test_raw_attack, test_normal, true_label = Preprocess_GAN(test_dataset)
print("test_normal shape: ", test_normal.shape)
print("test_raw_attack shape: ", test_raw_attack.shape)
print("test shape: ", test.shape)



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
ids_model = pickle.load(open('surrogate_model/ml_model/xgb_model_from_lrdata.pickle', 'rb')) #surrogate_model/ml_model/lr_model_from_dtdata.pickle
# ids_model = pickle.load(open('target_model/ml_model/CICIDS2018target_xgb.pickle', 'rb')) #surrogate_model/ml_model/lr_model_from_dtdata.pickle
g_param = th.load('GAN_materials/testGAN/gan_model/from_xgb_surrogate_model/0328/generator_xgb_model_from_lrdata_0329_1501.pth',map_location=lambda x,y:x)
learned_g.load_state_dict(g_param)






# model predict
result = ids_model.predict(test)
print(len(result), len(true_label))
print(result)
print(true_label)
print("classification report: ", classification_report(true_label, result))

# learned_g = parameters from the trained model
# model_g = {"no_learn":random_g,"learned":learned_g}
model_g = {"learned":learned_g}
adv_attack_data = pd.DataFrame(columns=feature_names)
adv_attack_raw_data = pd.DataFrame(columns=feature_names)
tmp = pd.DataFrame(columns=feature_names)
print("model_g: ", model_g)
print("adversarial traffic evaluating")
print("-"*100)
odr_tests, adr_tests, eir_tests, o_asr_tests, a_asr_tests = [], [], [], [], []
for NUM in range (NUM_OF_TESTS): #100
    test_batch_normal = CreateBatch_GAN(test_normal,BATCH_SIZE) #benign traffic
    # print("test_batch_normal: ", test_batch_normal)
    # print("test_batch_normal shape: ", len(test_batch_normal))
    for n,g in model_g.items():
        o_dr,a_dr,eir,o_asr,a_asr =[],[],[],[],[]
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

                # print("test_raw_attack shape: ", test_raw_attack.shape) #(6155, 42)
                
                #select 256 random attack samples in test_raw_attack(each batch size = 256, 42 features per sample)
                # transform the numpy array to tensor
                normal_b = th.Tensor(bn)
                batch_a= th.Tensor(test_raw_attack[np.random.randint(0,len(test_raw_attack),BATCH_SIZE)])
                #test_raw_attack with random noise
                z = batch_a + th.Tensor(np.random.uniform(0,1,(BATCH_SIZE,D_G_INPUT_DIM))) 
                # generate adversarial attack from generator
                # shape of adversarial_attack: (256, 42)
                adversarial_attack = g(z)
                # print("batch_a[0]: ", batch_a[0])   
                # print("z[0]: ", z[0])
                # print("adversarial_attack[0]: ", adversarial_attack[0]) 
                # print("adversarial_attack shape: ", adversarial_attack.shape)
                # adversarial_attack[:,33:] = th.Tensor(np.where(adversarial_attack[:,33:].detach().cpu().numpy()>= 0.5 , 1,0))
                # print("adversarial_attack: ", len(adversarial_attack[:,33:]))
                # print("adversarial_attack shape: ", adversarial_attack[:,34:].shape)
                # print("batch_a", batch_a)
                # print("adversarial_attack", adversarial_attack)
                ori_input = th.cat((batch_a,normal_b)) #combine normal and attack traffic
                adv_input = th.cat((adversarial_attack,normal_b))#combine normal and adversarial attack traffic
                # adv_input = th.cat((normal_b, adversarial_attack))
                l = list(range(len(ori_input)))
                np.random.shuffle(l)
                
                # print("ori_input: ", ori_input) 
                # print("ori_input shape: ", ori_input.shape) # (512, 42)
                # print("adv_input ", adv_input)
                # print("adv_input shape: ", adv_input.shape) # (512, 42)
                # print("l shape: ", len(l)) # 512
                
                # print adverarial attack
                # print("adversarial attack: ", adversarial_attack)
                # print("adversarial attack shape: ", adversarial_attack.shape)
        
                adv_input_l = adv_input[l] # raw attack traffic
                ori_input_l = ori_input[l] # adversarial attack traffic
                # adv_input_l = adv_input
                # ori_input_l = ori_input
                
                # ids_pred_adv = ids_model(adv_input)
                # ids_pred_ori = ids_model(ori_input)

                adv_input_df = pd.DataFrame(adv_input_l.detach().numpy(), columns=feature_names)
                ori_input_df = pd.DataFrame(ori_input_l.detach().numpy(), columns=feature_names) 
                ids_pred_adv = ids_model.predict(adv_input_df)
                ids_pred_ori = ids_model.predict(ori_input_df)
                # print("ids_pred_adv: ", ids_pred_adv)
                # print("adv_input_df shape: ", adv_input_df.shape)
                # print("adv_input_df: ", adv_input_df)
                # print("ids_pred_adv : ", ids_pred_adv)
                ids_true_label = np.r_[np.ones(BATCH_SIZE),np.zeros(BATCH_SIZE)][l].astype(int)
                # print("ids_true_label: ", ids_true_label)
                # ids_true_label = np.r_[np.ones(BATCH_SIZE),np.zeros(BATCH_SIZE)].astype(int)
                # combine adv_input_df and ids_pred_adv
                # adv_input_df["IDS_Predicted"] = ids_pred_adv
                # adv_input_df["True_Label"] = ids_true_label
                # adv_attack_data = pd.concat([adv_attack_data, adv_input_df], ignore_index=True)
                # tmp = pd.DataFrame(batch_a, columns=feature_names)
                # tmp["IDS_Predicted"] = 1
                # tmp["True_Label"] = np.ones(BATCH_SIZE).astype(int)
                # adv_attack_raw_data = pd.concat([adv_attack_raw_data, tmp], ignore_index=True)
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
                # malicious_count = [x for x in ids_true_label if x == 1]
                # print("malicious count: ", len(malicious_count))
                # ori_mali_count = [x for x in pred_label_ori if x == 1]
                # print("ori malicious count: ", len(ori_mali_count))
                # adv_mali_count = [x for x in pred_label_adv if x == 1]
                # or_count = []
                # for i in range(len(pred_label_ori)):
                #     if pred_label_adv[i] == 1 and ids_true_label[i] == 1:
                #         or_count.append(i)
                # print("or count: ", len(or_count))  
                # print("adv malicious count: ", len(adv_mali_count))
                # print("tn2: ", tn2)
                # print("fp2: ", fp2)
                # print("fn2: ", fn2)
                # print("tp2: ", tp2)
                # print("tn1: ", tn1)
                # print("fp1: ", fp1)
                # print("fn1: ", fn1)
                # print("tp1: ", tp1)
                # print("tp1/(tp1 + fn1): ", tp1/(tp1 + fn1))
                # print("recall: ", recall_score(ids_true_label,pred_label_adv))
                # print("recall: ", recall_score(ids_true_label,pred_label_adv))
                # o_dr.append(tp2/(tp2 + fp2))
                # a_dr.append(tp1/(tp1 + fp1))
                # eir.append(1 - (tp1/(tp1 + fp1))/(tp2/(tp2 + fp2)))
                # print()
                #實際yes代表惡意 1 ，因此 recall = tp/(tp+fn) = tp/(實際惡意的數量)
                #recall_score 已測試過，是對的 = tp/(tp+fn) = tp/(實際惡意的數量)
                o_dr.append(recall_score(ids_true_label, pred_label_ori))# malicious traffic detection rate = malicious traffic detected / total malicious traffic
                a_dr.append(recall_score(ids_true_label, pred_label_adv))
                eir.append(1 - recall_score(ids_true_label, pred_label_adv)/recall_score(ids_true_label, pred_label_ori))
                o_asr.append(1 - recall_score(ids_true_label, pred_label_ori)) 
                a_asr.append(1 - recall_score(ids_true_label, pred_label_adv))     
        avg_odr=np.mean(o_dr)
        avg_adr=np.mean(a_dr)
        avg_eir=np.mean(eir)
        avg_o_asr = np.mean(o_asr)
        avg_a_asr = np.mean(a_asr)

        print(f"{NUM + 1}: {n} => origin_DR : {avg_odr} \t adversarial_DR : {avg_adr} \t EIR : {avg_eir}") 
        if (n=="learned"):
            odr_tests.append(avg_odr)
            adr_tests.append(avg_adr)
            eir_tests.append(avg_eir)
            o_asr_tests.append(o_asr)
            a_asr_tests.append(a_asr)
    print()  

# # save adversarial attack data
# adv_attack_data.to_csv("evaluation/adversarial_attack_data/0323/adv_data_0323_1446.csv", index=False)
# adv_attack_raw_data.to_csv("evaluation/adversarial_attack_data/0323/adv_raw_data_0323_1446.csv", index=False)

# average detection rate
print("Average detection rate")
print("-"*100)
print(f"Original detection rate: {np.mean(odr_tests)}")
print(f"Adversarial detection rate: {np.mean(adr_tests)}")
print(f"Evasion increase rate: {np.mean(eir_tests)}")
print(f"origin_ASR: {np.mean(o_asr_tests)}\t adversarial_ASR: {np.mean(a_asr_tests)}")



plt.plot(odr_tests, label="Original detection rate")
plt.plot(adr_tests, label="Adversarial detection rate")
#plt.plot(eir_tests, label="Evasion increase rate")
# set y axis from 0 to 1
plt.ylim(0, 1)
plt.legend()
timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"GAN_materials/result/graph/Test_{timestamp}.png")
plt.show()