"""
Description: the file runs an algorithm for multi-task data preparation and DNN learning. One needs to fill out the fields with training settings and indicate the path for the investigated data. The data should be presented in the csv file in the following format:

	| 'CompoundID'	| 'TargetID'| 'activity' | 'SMILES' 							| 'Split' |
	| 'CHEMBL10'    | 809114    | 4.00       | 'C[S+]([O-])c1ccc(cc1)c2nc(c3ccc(F)cc3)c([nH]2)c4ccncc4' 	| TRN     |
	| 'CHEMBL98123' | 688293    | 5.10       | 'Cl.COc1ccc(Cc2nccc3cc(OC)c(OC)cc23)cc1OC'          		| TRN     |
	| 'CHEMBL98350' | 809272    | 5.50       | 'O=C1C=C(Oc2c1cccc2c3ccccc3)N4CCOCC4'               		| TST     |

Return: txt file containing a dataset with columns: Pred (prediction results), True (actual values from the investigated dataset), TargetID (targets' ID from the analyzed dataset), CompoundID (compounds' ID from the analyzed dataset), Iter (epoch of learning)

Settings for data preparation:
	data 		name of investigated data (full name with path indication is allowed); by default, None
	nBits 		number of bits during the calculation of Morgan fingerprints using RDKit; by default, 1024
	data_save 	if 'yes', then the calculated descriptors will be saved; by default, 'no'
	DescrName 	name for saving a bin file with compounds' descriptors (full name with path indication is allowed); by default, 'DescrName'

Training parameters:
	task 		modelling mode in two options: 'classification', 'regression'; by default, 'regression'
	DNNlayers 	number of neurons in the layers; by default, [512,256,128]
	dropout 	dropout values for each layer; by default, [0.0,0.0,0.0]
	actF 		activation function in four options: '0' (LeakyReLU), '1' (PReLU), '2' (ELU), '3' (Hardshrink); by default, '1'
	optF 		optimizers in three options: Adam, RAdam, and Yogi; by default, RAdam 
	lr 		learning rate; by default, 0.001
	seed 		random seed; by default, 70922
	FileName 	the name for results saving (full name with path indication is
    allowed); by default, 'Results'
	savestep 	results will be saved for each iteration multiple of set number; by default,  10
	num_epochs 	number of epochs
	batch_size 	batch size for training data in DataLoader; by default, 64
	ngpu		used GPU; by default, 1
"""


import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from SI2_DNN import DNNModel
from SI2_dataprep import prepare_dataset, load_dataset

#Settings for data preparation:
data = 'data.csv'
nBits = 1024
data_save = 'no'
DescrName = 'DescrName'

#Training parameters:
task = 'regression'
DNNlayers = [512,256,128]
dropout = [0.0,0.0,0.0]
actF = '1'
optF = 'RAdam'
lr = 0.001
seed = 70922
FileName = 'Results'
savestep = 10
num_epochs = 100
batch_size = 64
ngpu = 0

(train_dataset,test_dataset,idx2target,idx2compid) = load_dataset(data,nBits,DescrName,data_save)
model=DNNModel(task=task,DNNlayers=DNNlayers,dropout=dropout,actF=actF,optF=optF,lr=lr,idx2target=idx2target,idx2compid=idx2compid,nBits=nBits,FileName=FileName,seed=seed,savestep=savestep)
trainer = pl.Trainer(gpus=ngpu,max_epochs=num_epochs)
trainer.fit(model, DataLoader(train_dataset,batch_size=batch_size),DataLoader(test_dataset,batch_size=500000))



