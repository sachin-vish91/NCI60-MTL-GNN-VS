# NCI60-MTL-GNN-VS
Multi-task learning model built using NCI-60 dataset for compounds features and CellMiner for molecular profile features


##  1. Data preparation
NCI-60 data preparation.

**Requirements:**<br/>
Python      &nbsp; &nbsp; &nbsp;  3.7.3<br/>
MolVS       &nbsp; &nbsp; &nbsp;   0.1.1<br/>
standardiser &nbsp; &nbsp; &nbsp;  0.1.9<br/>
matplotlib   &nbsp; &nbsp; &nbsp;  3.3.4<br/>
numpy        &nbsp; &nbsp; &nbsp;    1.17.3<br/>
pandas        &nbsp; &nbsp; &nbsp;   1.2.1<br/>
Babel         &nbsp; &nbsp; &nbsp;   2.7.0<br/>
scikit-learn   &nbsp; &nbsp; &nbsp;  0.23.2<br/>
scipy         &nbsp; &nbsp; &nbsp;   1.4.1<br/>
tensorflow    &nbsp; &nbsp; &nbsp;   2.2.0<br/>
Keras 			  &nbsp; &nbsp; &nbsp;   2.3.0<br/>
keras-tuner   &nbsp; &nbsp; &nbsp;   1.0.3<br/>
xgboost       &nbsp; &nbsp; &nbsp;   1.1.1<br/>
hyperopt      &nbsp; &nbsp; &nbsp;   0.2.5<br/>


Steps to run the script:
1. Download the Dataprocessing.py file.
2. Create a directory named **Dataset** in you current working directory.
3. Download the NCI-60 and chemical dataset into your **Dataset** directory using link below.

NCI-60 dataset download link: https://wiki.nci.nih.gov/display/NCIDTPdata/NCI-60+Data+Download+-+Previous+Releases?preview=/147193864/374736079/NCI60_GI50_2016b.zip.

Chemical data download link: https://wiki.nci.nih.gov/display/NCIDTPdata/Chemical+Data?preview=/155844992/339380766/Chem2D_Jun2016.zip

4. Unzip both the downloaded files.
4. Once all the files are ready, run the followint script: **python Data_processing.py**
6. The output of this scipt will be automatically store into **Dataset/All_tested_molecules.csv** and **Features** directory for next step.
7. The **Features** directory will have 60 .csv files, each for the one cell line. The colums names are: [NLOGGI50, NSC, SMILE, molecular features]

## 2. Machine learning model building

**Random CV**

`TODO`

**LOCO CV**

`TODO`

**LOTO CV**

`TODO`

