import numpy as np
import pandas as pd
from molvs import Standardizer
from itertools import chain
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdmolops
import re
from molvs import Standardizer
from rdkit.Chem import SaltRemover
import concurrent.futures
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
import openbabel
########################################################################################
# 60 cell line included in this study
cells_from_profile = ['MCF7',  'MDA-MB-231_ATCC', 'HS_578T', 'BT-549', 'T-47D', 'SF-268', 
          'SF-295', 'SF-539', 'SNB-19', 'SNB-75','U251', 'COLO_205', 'HCC-2998',
          'HCT-116', 'HCT-15', 'HT29', 'KM12', 'SW-620', 'CCRF-CEM', 'HL-60(TB)', 
          'K-562', 'MOLT-4', 'RPMI-8226', 'SR', 'LOX_IMVI', 'MALME-3M','M14', 
          'SK-MEL-2', 'SK-MEL-28', 'SK-MEL-5', 'UACC-257', 'UACC-62', 'MDA-MB-435', 
          'MDA-N', 'A549_ATCC', 'EKVX', 'HOP-62', 'HOP-92','NCI-H226', 'NCI-H23', 
          'NCI-H322M', 'NCI-H460', 'NCI-H522', 'IGROV1', 'OVCAR-3', 'OVCAR-4', 
          'OVCAR-5', 'OVCAR-8', 'SK-OV-3', 'NCI_ADR-RES','PC-3', 'DU-145', '786-0', 
          'A498', 'ACHN', 'CAKI-1', 'RXF_393', 'SN12C', 'TK-10', 'UO-31']

out_dir = 'Datasets'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Convert .sdf file into .smi for extracting the SMILES notation and NSC IDs of the molecules.
conv=openbabel.OBConversion()
conv.OpenInAndOutFiles("Datasets/Chem2D_Jun2016.sdf","Datasets/Chem2D_Jun2016.smi")
conv.SetInAndOutFormats("sdf","smi")
conv.Convert()


filename = 'Datasets/CANCER60GI50.LST'
SMILE_file = 'Datasets/Chem2D_Jun2016.smi'

#Normalize the data this will return the value between 0 and 1
def Normalization(x,overall_min,overall_max):
    #overall_min = 45 # specify the global minimun from all the datasets
    #overall_max = 800 # specify the global maximum from all the datasets
    x = (x - overall_min)/(overall_max - overall_min)
    
    return x

# Function to process and load the NCI-60 dataset
def preprocesing(filename,SMILE_file):
  # Load the NCI-60 dataset
  data = pd.read_csv(filename, delimiter=',')

  # Remove the pGI50 below 4
  data_MT_4 = data[data.NLOGGI50 >= 4 ]

  data_MT_4['key'] =  data_MT_4["NSC"].map(str)+ "_" + data_MT_4['CELL']

  data_MT_4_key = data_MT_4.loc[:, ['NLOGGI50', 'key']]

  # calcute mean for the duplicate entries
  data_MT_4_mean = data_MT_4_key.groupby('key', as_index=False).mean() 
  data_MT_4_mean.columns = ['key', 'NLOGGI50_N']

  data_MT_4_merged = data_MT_4.merge(data_MT_4_mean, how='left',left_on=['key'],right_on=['key'])

  # Remove the duplicate entries
  data_MT_4_merged = data_MT_4_merged.drop_duplicates(['NSC','CELL'])

  # Drop NLOGGI50 as we have new column name NLOGGI50_N
  data_MT_4_merged = data_MT_4_merged.drop(['NLOGGI50'] , axis=1)

  ################################# Load SMILES notation file ############################
  smile = pd.read_csv(SMILE_file,sep='\t', header=None)

  smile.columns = ['SMILE', 'NSC']

  smile = smile.iloc[:,[1,0]]
  #############################################################

  data_SMILE = data_MT_4_merged.merge(smile, how = 'left')

  NCI60 = data_SMILE[data_SMILE.SMILE.notnull()] # to remove the row where is Nan

  return NCI60

df_smile = preprocesing(filename, SMILE_file)

smiles = df_smile.SMILE.unique()


# Initialize the list for storing the data
SMILES = []
STD_SMILES = []
Error = []
fingerprints = []
fingerprints_1 = []
fingerprints_2 = []
fingerprints_3 = []
fingerprints_4 = []
fingerprints_5 = []
fingerprints_6 = []
fingerprints_7 = []

# Morgan fingerprint and physicochemical descriptor generation
for m in smiles:
    try:
        m1 = m
        m = Chem.MolFromSmiles(m)
        remover = SaltRemover.SaltRemover() #remove salt
        m = remover.StripMol(m)
        s = Standardizer() # standardize molecule
        m = s.standardize(m)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=256)
        fingerprints.append(list(fp1))
        fingerprints_1.append(rdMolDescriptors.CalcTPSA(m)) #total polar surface area
        fingerprints_2.append(rdMolDescriptors.CalcExactMolWt(m)) #molecular weight 
        fingerprints_3.append(rdMolDescriptors.CalcCrippenDescriptors(m)[0]) #logP
        fingerprints_4.append(rdMolDescriptors.CalcNumAliphaticRings(m)) #number of aliphatic ring
        fingerprints_5.append(rdMolDescriptors.CalcNumAromaticRings(m)) #number of aromatic ring
        fingerprints_6.append(rdMolDescriptors.CalcNumHBA(m)) #Number of hydrongen bond acceptor
        fingerprints_7.append(rdMolDescriptors.CalcNumHBD(m)) #Number of hydrongen bond doner
        
        STD_SMILES.append(Chem.MolToSmiles(m))
        SMILES.append(m1)
    except:
        Error.append(m1)

# Transform the features into Dataset
Fingerprints = pd.DataFrame(fingerprints)
fingerprints_1 = pd.DataFrame(fingerprints_1)
fingerprints_2 = pd.DataFrame(fingerprints_2)
fingerprints_3 = pd.DataFrame(fingerprints_3)
fingerprints_4 = pd.DataFrame(fingerprints_4)
fingerprints_5 = pd.DataFrame(fingerprints_5)
fingerprints_6 = pd.DataFrame(fingerprints_6)
fingerprints_7 = pd.DataFrame(fingerprints_7)

std_smile = pd.DataFrame(STD_SMILES)
Original_SMILES = pd.DataFrame(SMILES)
smile_fingerprint = pd.concat([Original_SMILES, 
                               std_smile, 
                               Fingerprints,
                               fingerprints_1,
                               fingerprints_2,
                               fingerprints_3,
                               fingerprints_4,
                               fingerprints_5,
                               fingerprints_6,
                               fingerprints_7], axis=1)

# Rename the column names
smile = ['SMILE','STD_SMILE']
Fig = list("F_{0}".format(i) for i in range(1,257))
PH = ["MW","TPSA","LOGP","NAR","NARR","HBA","BHD"]
A = [smile,Fig,PH]
col_names = list(chain(*A))

smile_fingerprint.columns = col_names

# Normalization of physicochemical properties
smile_fingerprint.iloc[:,258] =  Normalization(x=smile_fingerprint.iloc[:,256], overall_min=0, overall_max=1288.39)
smile_fingerprint.iloc[:,259] =  Normalization(x=smile_fingerprint.iloc[:,257], overall_min=32.03, overall_max=3351.54)
smile_fingerprint.iloc[:,260] =  Normalization(x=smile_fingerprint.iloc[:,258], overall_min=-18.69, overall_max=41.84)
smile_fingerprint.iloc[:,261] =  Normalization(x=smile_fingerprint.iloc[:,259], overall_min=0, overall_max=22)
smile_fingerprint.iloc[:,262] =  Normalization(x=smile_fingerprint.iloc[:,260], overall_min=0, overall_max=19)
smile_fingerprint.iloc[:,263] =  Normalization(x=smile_fingerprint.iloc[:,261], overall_min=0, overall_max=79)
smile_fingerprint.iloc[:,264] =  Normalization(x=smile_fingerprint.iloc[:,262], overall_min=0, overall_max=47)

# Merge the compouds features with main data set.
Merged_data = df_smile.merge(smile_fingerprint, how = 'left', left_on=['SMILE'],right_on=['SMILE'])

# Remove any entry which contains NaN 
Combined_data = Merged_data[Merged_data.BHD.notnull()]

Combined_data["CELL"] = Combined_data.loc[:,"CELL"].str.strip()

# Rename some of cell line names
Combined_data["CELL"].replace({"A549/ATCC": "A549_ATCC", 
                                   "NCI/ADR-RES": "NCI_ADR-RES",
                                   'MDA-MB-231/ATCC': 'MDA-MB-231_ATCC',
                                   'HS 578T':'HS_578T',
                                   'COLO 205':'COLO_205',
                                   'RXF 393':'RXF_393',
                                   'LOX IMVI':'LOX_IMVI'}, inplace=True)

Combined_data = Combined_data[Combined_data['CELL'].isin(cells_from_profile)]
Tested_molecule = Combined_data.loc[:, ['NSC', 'CELL', 'SMILE']]

# Export tested molecules to a file
Tested_molecule.to_csv('Datasets/All_tested_molecules.csv', index=False)

pGI50_NSC = ['NLOGGI50_N','NSC','SMILE']
Fig = list("F_{0}".format(i) for i in range(1,257))
PH = ["MW","TPSA","LOGP","NAR","NARR","HBA","BHD"]
A = [pGI50_NSC,Fig,PH]
features = list(chain(*A))

out_dir = 'Features'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Export the data into file for 60 cell lines
for i in cells_from_profile:
    df_cell_subset = Combined_data[Combined_data['CELL'] == i]
    
    x_train = df_cell_subset.loc[:,features]
    x_train.to_csv('Features/'+i+'.csv', index=None, header = True)

########################################################################################
