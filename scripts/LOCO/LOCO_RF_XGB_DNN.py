
#for c in MCF7 MDA-MB-231_ATCC HS_578T BT-549 T-47D SF-268 SF-295 SF-539 SNB-19 SNB-75 U251 COLO_205 HCC-2998 HCT-116 HCT-15 HT29 KM12 SW-620 CCRF-CEM 'HL-60(TB)' K-562 MOLT-4 RPMI-8226 SR LOX_IMVI MALME-3M M14 SK-MEL-2 SK-MEL-28 SK-MEL-5 UACC-257 UACC-62 MDA-MB-435 A549_ATCC EKVX HOP-62 HOP-92 NCI-H226 NCI-H23 NCI-H322M NCI-H460 NCI-H522 IGROV1 OVCAR-3 OVCAR-4 OVCAR-5 OVCAR-8 SK-OV-3 NCI_ADR-RES PC-3 DU-145 786-0 A498 ACHN CAKI-1 RXF_393 SN12C TK-10 UO-31; do python LOCO_XGB.py --descriptors_folder /home/nguyenl/Documents/CLUSTER/LOCO -p CNA -cell $c -alg XGB -o /home/nguyenl/Documents/CLUSTER/LOCO/Results/XGB --profile_folder /home/nguyenl/Documents/CLUSTER/LOCO/Profile_data; done

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
import keras
import tarfile
import pickle
import joblib
from itertools import chain

# Total 60 cell lines included to build model
cells_from_profile = ['MCF7',  'MDA-MB-231_ATCC', 'HS_578T', 'BT-549', 'T-47D', 'SF-268', 
					'SF-295', 'SF-539', 'SNB-19', 'SNB-75','U251', 'COLO_205', 'HCC-2998',
					'HCT-116', 'HCT-15', 'HT29', 'KM12', 'SW-620', 'CCRF-CEM', 'HL-60(TB)', 
					'K-562', 'MOLT-4', 'RPMI-8226', 'SR', 'LOX_IMVI', 'MALME-3M','M14', 
					'SK-MEL-2', 'SK-MEL-28', 'SK-MEL-5', 'UACC-257', 'UACC-62', 'MDA-MB-435', 
					'MDA-N', 'A549_ATCC', 'EKVX', 'HOP-62', 'HOP-92','NCI-H226', 'NCI-H23', 
					'NCI-H322M', 'NCI-H460', 'NCI-H522', 'IGROV1', 'OVCAR-3', 'OVCAR-4', 
					'OVCAR-5', 'OVCAR-8', 'SK-OV-3', 'NCI_ADR-RES','PC-3', 'DU-145', '786-0', 
					'A498', 'ACHN', 'CAKI-1', 'RXF_393', 'SN12C', 'TK-10', 'UO-31']
    
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

# Argument parser parses the command line
parser = argparse.ArgumentParser(description='Predicting GI50 from NCI-60 dataset')
#parser.add_argument('--profile_folder', type=str, help='path to the folder containing molecular profiles data')
parser.add_argument('--descriptors_folder', type=str, default='', help='path to the folder containing Fingerprints files')
parser.add_argument('--profile_folder', type=str, help='path to the folder containing molecular profiles data')
parser.add_argument('-p', type=str, default='GeneExp', choices=['GeneExp', 'CNA', 'Methy', 'EXSNP', 'Protein'], help='desired molecular profile data type')
parser.add_argument('-alg', type=str, default='RF', choices=['RF', 'XGB', 'DNN'], help='desired machine learning algorithm')
parser.add_argument('-o', type=str, default='output', help='output folder for saving model')
parser.add_argument('-cell', type=str, default='U251',  help='choose the cell line.')
args = parser.parse_args()

#In case of GeneExp and MDA-N cell line exit the code
if args.p == 'GeneExp' and args.cell == 'MDA-N':
	print('MDA-N cell line skiped because the data is not available for GeneExp')
	exit()

#Training dataset
print('Loading dataset')
dataset = pd.read_csv(args.descriptors_folder + '/dataset_60_cell_line.csv', header=0)

print(dataset.shape)
print(len((dataset.CELL).unique()))
#Rename the cell lines
dataset["CELL"].replace({"A549/ATCC": "A549_ATCC", 
                                   "NCI/ADR-RES": "NCI_ADR-RES",
                                   'MDA-MB-231/ATCC': 'MDA-MB-231_ATCC',
                                   'HS 578T':'HS_578T',
                                   'COLO 205':'COLO_205',
                                   'RXF 393':'RXF_393',
                                   'LOX IMVI':'LOX_IMVI'}, inplace=True)
                                   
######Profile
f = glob.glob(args.profile_folder+'/'+args.p+'/*.xls')
a = pd.read_excel(f[0], skiprows=10, header=0, na_values='-')

if args.p != 'GeneExp' and args.p != 'EXSNP':
    a.dropna(subset=a.columns[9:], axis=0, inplace=True)

#Transforming the data using groupby and calculating mean of duplicate entry
tmp = np.array(a.groupby('Gene name d', as_index=False).mean().loc[:,'BR:MCF7':].T, dtype=np.float32)
cells_from_profile = [cell_data_names[i] for i in a.columns[9:]]

#Important pre-processing of where the data is not reported we are considering 0 (values are reported as percentage)
if args.p == 'EXSNP':
    print('empty cell replaced with zeros')
    tmp[np.where(np.isnan(tmp))] = 0

#Important pre-processing of GeneExp Profile because in this profile the data is not available for MDA-N cell line
if args.p == 'GeneExp':
    dataset = dataset.loc[dataset['CELL'] != 'MDA-N'] #remove one cell line from the dataset
    tmp = np.delete(tmp,33,0)
    cells_from_profile.remove('MDA-N')
    
profile = distance.pdist(tmp, metric=lambda x,y:spearmanr(x,y)[0])
profile = distance.squareform(profile)
profile += np.eye(profile.shape[0])

df = pd.DataFrame(profile)
df.columns = cells_from_profile
df['CELL'] = cells_from_profile

dataset = dataset.merge(df, how='left', on='CELL') #Merge dataset from molecular profile

col_name_remove = ['NSC', 'CONCUNIT', 'LCONC', 'PANEL', 'CELL', 'PANELNBR', 'CELLNBR', 'INDN', 'TOTN', 
                   'STDDEV','SMILE', 'STD_SMILE','NLOGGI50_N']

###################--------------Training dataset--------------
print('-----Loading Training dataset--------------')
x_train = dataset.loc[dataset['CELL'] != args.cell] #remove one cell line from the dataset
print(len((x_train.CELL).unique()))
y_train = x_train.NLOGGI50_N.values.ravel()
x_train.drop(col_name_remove , axis='columns', inplace=True)
x_train = x_train.to_numpy()
print(x_train.shape)


###################--------------Test dataset--------------
print('-----Loading Test dataset--------------')
x_test = dataset[dataset.CELL == args.cell] # select left out cell line for the test set
print((x_test.CELL).unique())
y_test = x_test.NLOGGI50_N.values.ravel()
x_test.drop(col_name_remove , axis='columns', inplace=True)
x_test = x_test.to_numpy()
print(x_test.shape)


print('Training model...')
start = time.time()
if args.alg == 'RF':
    mymodel = RandomForestRegressor(n_estimators=500, max_features=0.3, n_jobs = 30)#Recommeded hyperparameters
    mymodel.fit(x_train, y_train)

    #pickle.dump(mymodel, open(args.o + '/' + args.p + '.RF.sav', 'wb'))
    #joblib.dump(mymodel, args.o+'/'+ args.p + ".RF.joblib")
    #print('Model saved to %s' % (args.o))

elif args.alg == 'XGB':
    mymodel=xgb.XGBRegressor(max_depth=9, learning_rate=0.05, n_estimators=1000, colsample_bytree=0.4, n_jobs=30)#Tuned hyperparameters
    mymodel.fit(x_train, y_train, eval_metric='rmse')

    #pickle.dump(mymodel, open(args.o+'/'+args.p + ".XGB.dat", "wb"))
    #print('Model saved to %s' % (args.o))

elif args.alg == 'DNN':
    mymodel = build_model((512,256,64), 0.01, dropout=0.6, decay=0, bn=True, input_dim=len(x_train[0]))#Tuned hyperparameters
    mymodel.fit(x_train, y_train, epochs=100, batch_size=100, verbose=False)

    #mymodel.save(args.o + '/' + args.p + '.DNN.h5')
    #print('Model saved to %s' % (args.o))	

print(args.alg, ' Model building took ', int(time.time() - start)//60, 'min.' )


print('Prediting...')
if args.alg == 'RF':
	#mymodel = pickle.load(open( 'Models/'+ args.p + ".RF.sav", "rb"))
	preds = mymodel.predict(x_test)
	
elif args.alg == 'XGB':
	#mymodel = pickle.load(open( 'Models/'+ args.p + ".XGB.dat", "rb"))
	preds = mymodel.predict(x_test)
	
elif args.alg == 'DNN':
	#mymodel = keras.models.load_model( 'Models/'+ args.p + ".DNN.h5")
	preds = mymodel.predict(x_test)[:,0]
    
with open(args.o + '/' + args.p + '/' + args.cell + '.pred', 'w') as output:
	for j in range(len(y_test)):
			output.write(str(y_test[j])+'\t'+str(preds[j])+'\n')

