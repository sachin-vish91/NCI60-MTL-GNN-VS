# Shut up the annoying warnings
import glob, argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  scipy as scp
import random, sys
import sys, time, os
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import distance
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras import optimizers, regularizers

# Keras DNN model initialization meta-function
def build_model(layers, lr, dropout=0, decay=0, bn=False, input_dim=526):
    model = Sequential()
    model.add(Dense(layers[0], input_dim=input_dim, kernel_regularizer=regularizers.l2(decay)))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    if dropout:
        model.add(Dropout(dropout))
    for i in layers[1:]:
        model.add(Dense(i, kernel_regularizer=regularizers.l2(decay)))
        if bn:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        if dropout:
            model.add(Dropout(dropout))
    model.add((Dense(1)))
    model.add(Activation('linear'))
    
    optimizer = optimizers.Adam(lr)

    model.compile(loss='mean_squared_error',
                 optimizer=optimizer)
    return model

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

# Argument parser parses the command line for cell line, drugs and method
parser = argparse.ArgumentParser(description='Predicting GI50 from NCI-60 dataset')
parser.add_argument('--descriptors_folder', type=str, default='', help='path to the folder containing Fingerprints files')
parser.add_argument('--profile_folder', type=str, help='path to the folder containing molecular profiles data')
parser.add_argument('-p', type=str, default='GeneExp', choices=['GeneExp', 'CNA', 'Methy', 'EXSNP', 'Protein'], help='desired molecular profile data type')
parser.add_argument('-o', type=str, default='output', help='output folder for predictions')
parser.add_argument('--preprocessing', type=str, default='none', choices=['none','normalize','kernel', 'gex1000'], help='molecular profile data preprocessing')
parser.add_argument('-alg', type=str, default='RF', choices=['RF', 'XGB', 'DNN'], help='desired machine learning algorithm')
args = parser.parse_args()

try:
    os.mkdir(args.o +'/'+args.p)
except OSError:
    pass
    
print('Selected Profile: ',args.p)
print('Selected ML algorithm: ',args.alg)
print('Selected Pre-processing method: ',args.preprocessing)
f = glob.glob(args.profile_folder+'/'+args.p+'/*.xls')
a = pd.read_excel(f[0], skiprows=10, header=0, na_values='-')

if args.p != 'GeneExp' and args.p != 'EXSNP':
    a.dropna(subset=a.columns[9:], axis=0, inplace=True)

tmp = np.array(a.groupby('Gene name d', as_index=False).mean().loc[:,'BR:MCF7':].T, dtype=np.float32)
cells_from_profile = [cell_data_names[i] for i in a.columns[9:]]


if args.p == 'EXSNP':
    print('empty cell replaced with zeros')
    tmp[np.where(np.isnan(tmp))] = 0

if args.p == 'GeneExp':
	tmp = np.delete(tmp,33,0)
	cells_from_profile.remove('MDA-N')

if args.preprocessing == 'none':
    print('Using data as-is, no preprocessing is applied')
    profile = tmp

elif args.preprocessing == 'normalize':
    print('Features are normalized')
    scaler = MinMaxScaler()
    profile = scaler.fit_transform(tmp)

elif args.preprocessing == 'kernel':
    print('Spearman correlation kernel is applied')
    profile = distance.pdist(tmp, metric=lambda x,y:spearmanr(x,y)[0])
    profile = distance.squareform(profile)
    profile += np.eye(profile.shape[0])
    
best_indices = list(range(len(tmp[0])))

#Multi Task Learning Model
x_train = np.loadtxt(args.descriptors_folder +'/'+ cells_from_profile[0] + '.train.csv', delimiter=',', dtype=np.float32)
y_train = np.loadtxt(args.descriptors_folder +'/'+ cells_from_profile[0] + '.train.act', dtype=np.float32)
x_train = np.hstack([x_train, np.tile(profile[0], (len(x_train),1))])

for cell in cells_from_profile[1:]:
    t = np.loadtxt(args.descriptors_folder +'/'+ cell + '.train.csv', delimiter=',', dtype=np.float32)
    t = np.hstack([t, np.tile(profile[cells_from_profile.index(cell)], (len(t),1))])
    x_train = np.vstack([x_train, t])
    y_train = np.concatenate([y_train, np.loadtxt(args.descriptors_folder +'/'+ cell + '.train.act')])
print(x_train.shape)

print('Training model...')
start = time.time()
if args.alg == 'RF':
	mymodel = RandomForestRegressor(n_estimators=500, max_features=0.3, n_jobs = -1)
	mymodel.fit(x_train, y_train)

elif args.alg == 'XGB':
	mymodel=xgb.XGBRegressor(max_depth=7, learning_rate=0.05, n_estimators=745, colsample_bytree=0.56, n_jobs=-1)
	mymodel.fit(x_train, y_train, eval_metric='rmse')
	 	
elif args.alg == 'DNN':
	mymodel = build_model((4096,2048,1024), 0.01, dropout=0.6, decay=0, bn=True, input_dim=len(X_train[0]))
	mymodel.fit(X_train, y_train, epochs=200, batch_size=len(X_train[0]), verbose=False)

print(args.alg, ' model building took', int(time.time() - start)//60, 'min.' )



print('Predicting...')	
for cell in cells_from_profile:
    t = np.loadtxt(args.descriptors_folder +'/'+ cell +'.test.csv', delimiter=',', dtype=np.float32)
    x_test = np.hstack([t, np.tile(profile[cells_from_profile.index(cell)], (len(t),1))])
    y_test = np.loadtxt(args.descriptors_folder +'/'+ cell + '.test.act', dtype=np.float32)
    
    if args.alg == 'RF':
        preds = mymodel.predict(x_test)
    elif args.alg == 'XGB':
        preds = mymodel.predict(x_test)
    elif args.alg == 'DNN':
        preds = mymodel.predict(x_test)[:,0]
    
    with open(args.o+'/'+args.p +'/'+ cell + '.pred', 'w') as output:
        for j in range(len(y_test)):
            output.write(str(y_test[j])+'\t'+str(preds[j])+'\n')
            
print('Predictions saved to %s/%s' % (args.o, args.p))         
#python MTL_model.py --descriptors_folder /home/vishwakarma/Documents/GI50/NEW_DATASET/cell_line --profile_folder /home/vishwakarma/Documents/GI50/RUN/Random_Selection/Kernel/Profile_data -p GeneExp --preprocessing kernel -alg DNN -o /home/vishwakarma/Documents/Multiprocessing/MTL/result_NEW