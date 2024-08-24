import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  scipy as scp
import random, sys
from sklearn.ensemble import RandomForestRegressor
import sys, time
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import distance
    
# Custom function for RMSE and R2 calculations
def rmse(a,b):
    return np.sqrt(np.sum((a-b)**2)/len(a))

def r2(a,b):
    return 1-np.sum((a-b)**2)/np.sum((a-np.mean(a))**2)
    
cell_data_names = {'BR:MCF7':'MCF7', 'BR:MDA-MB-231':'MDA-MB-231_ATCC', 'BR:HS 578T':'HS_578T', 
                   'BR:BT-549':'BT-549', 'BR:T-47D':'T-47D', 'CNS:SF-268':'SF-268',
                   'CNS:SF-295':'SF-295', 'CNS:SF-539':'SF-539', 'CNS:SNB-19':'SNB-19', 
                   'CNS:SNB-75':'SNB-75', 'CNS:U251':'U251', 'CO:COLO 205':'COLO_205', 
                   'CO:HCC-2998':'HCC-2998', 'CO:HCT-116':'HCT-116', 'CO:HCT-15':'HCT-15',
                   'CO:HT29':'HT29', 'CO:KM12':'KM12', 'CO:SW-620':'SW-620', 
                   'LE:CCRF-CEM':'CCRF-CEM', 'LE:HL-60(TB)':'HL-60(TB)', 'LE:K-562':'K-562', 
                   'LE:MOLT-4':'MOLT-4', 'LE:RPMI-8226':'RPMI-8226', 'LE:SR':'SR', 
                   'ME:LOX IMVI':'LOX_IMVI','ME:MALME-3M':'MALME-3M', 'ME:M14':'M14', 
                   'ME:SK-MEL-2':'SK-MEL-2', 'ME:SK-MEL-28':'SK-MEL-28', 'ME:SK-MEL-5':'SK-MEL-5',
                   'ME:UACC-257':'UACC-257', 'ME:UACC-62':'UACC-62', 'ME:MDA-MB-435':'MDA-MB-435', 
                   'ME:MDA-N':'MDA-N', 'LC:A549/ATCC':'A549_ATCC', 'LC:EKVX':'EKVX', 
                   'LC:HOP-62':'HOP-62', 'LC:HOP-92':'HOP-92', 'LC:NCI-H226':'NCI-H226',
                   'LC:NCI-H23':'NCI-H23', 'LC:NCI-H322M':'NCI-H322M', 'LC:NCI-H460':'NCI-H460', 
                   'LC:NCI-H522':'NCI-H522', 'OV:IGROV1':'IGROV1', 'OV:OVCAR-3':'OVCAR-3', 
                   'OV:OVCAR-4':'OVCAR-4', 'OV:OVCAR-5':'OVCAR-5', 'OV:OVCAR-8':'OVCAR-8', 
                   'OV:SK-OV-3':'SK-OV-3', 'OV:NCI/ADR-RES':'NCI_ADR-RES', 'PR:PC-3':'PC-3', 
                   'PR:DU-145':'DU-145', 'RE:786-0':'786-0', 'RE:A498':'A498',
                   'RE:ACHN':'ACHN', 'RE:CAKI-1':'CAKI-1', 'RE:RXF 393':'RXF_393', 
                   'RE:SN12C':'SN12C', 'RE:TK-10':'TK-10', 'RE:UO-31':'UO-31'}

a = pd.read_excel('Profile_data/Methy/DNA__Illumina_450K_methylation_Gene_average.xls', skiprows=10, header=0, na_values='-')
a.dropna(subset=a.columns[9:], axis=0, inplace=True)# only in methy case
gexp = np.array(a.groupby("Gene name d", as_index=False).mean().loc[:,'BR:MCF7':].T, dtype=np.float32)
cells_in_gexp = [cell_data_names[i] for i in a.columns[9:]]


spearman_kernel = distance.pdist(gexp, metric=lambda x,y:spearmanr(x,y)[0])
spearman_kernel = distance.squareform(spearman_kernel)
spearman_kernel += np.eye(spearman_kernel.shape[0])

best_indices = list(range(len(gexp[0])))

start = time.time()

x_train = np.loadtxt('/home/vishwakarma/Documents/GI50/NEW_DATASET/cell_line/'+cells_in_gexp[0]+'.train.csv', delimiter=',', dtype=np.float32)
y_train = np.loadtxt('/home/vishwakarma/Documents/GI50/NEW_DATASET/cell_line/'+cells_in_gexp[0]+'.train.act', dtype=np.float32)
x_train = np.hstack([x_train, np.tile(spearman_kernel[0], (len(x_train),1))])

for c in cells_in_gexp[1:]:
    t = np.loadtxt('/home/vishwakarma/Documents/GI50/NEW_DATASET/cell_line/'+c+'.train.csv', delimiter=',', dtype=np.float32)
    t = np.hstack([t, np.tile(spearman_kernel[cells_in_gexp.index(c)], (len(t),1))])
    x_train = np.vstack([x_train, t])
    y_train = np.concatenate([y_train, np.loadtxt('/home/vishwakarma/Documents/GI50/NEW_DATASET/cell_line/'+c+'.train.act')])
print(x_train.shape)

mymodel = RandomForestRegressor(n_estimators=500, max_features=0.3, n_jobs = -1)
mymodel.fit(x_train, y_train)


for c in cells_in_gexp:
    t = np.loadtxt('/home/vishwakarma/Documents/GI50/NEW_DATASET/cell_line/'+c+'.test.csv', delimiter=',', dtype=np.float32)
    x_test = np.hstack([t, np.tile(spearman_kernel[cells_in_gexp.index(c)], (len(t),1))])
    y_test = np.loadtxt('/home/vishwakarma/Documents/GI50/NEW_DATASET/cell_line/'+c+'.test.act', dtype=np.float32)
    
    preds = mymodel.predict(x_test)
    
    with open('result_new/Methy/'+c+'.pred', 'w') as output:
        for j in range(len(y_test)):
            output.write(str(y_test[j])+'\t'+str(preds[j])+'\n')
            
print('Model building took', int(time.time() - start)//60, 'min.' )
