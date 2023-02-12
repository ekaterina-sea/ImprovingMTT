import pandas as pd
import torch
import numpy as np
import random
import rdkit
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from tqdm import tqdm
from os.path import isfile, isdir

def ecfp(mol, r=3, nBits=1024, errors_as_zeros=True):
    mol = Chem.MolFromSmiles(mol) if not isinstance(mol, rdkit.Chem.rdchem.Mol) else mol
    try:
        arr = np.zeros((1,))
        ConvertToNumpyArray(GetMorganFingerprintAsBitVect(mol, r, nBits), arr)
        return arr.astype(np.float32)
    except:
        return np.NaN if not errors_as_zeros else np.zeros((nBits,), dtype=np.float32)

class ChEMBLDataset():
    def __init__(self,R,X,idx2target,idx2compid):
        self.R = R
        self.X = X
        self.idx2target = idx2target
        self.idx2compid = idx2compid
    def __len__(self):
        return len(self.X)
    def __getitem__(self, item):
        return self.X[item,:],self.R[item,:]

def prepare_dataset(data,nBits=1024,DescrName='DescrName',data_save='no'):
    R = pd.read_csv('{}'.format(data))
    reverse = lambda d: {v: k for k, v in d.items()}
    idx2compid = dict(enumerate(R['CompoundID'].unique()))
    idx2target = dict(enumerate(R['TargetID'].unique()))
    compid2idx = reverse(idx2compid)
    target2idx = reverse(idx2target)
    trainMatrix = np.empty((len(idx2compid), len(idx2target)), dtype=np.float32)
    trainMatrix[:] = np.NaN
    descriptorsMatrix = np.zeros((len(idx2compid), nBits), dtype=np.float32)
    DataMask = np.zeros((len(idx2compid), len(idx2target)), dtype=np.bool)
    
    for _, record in tqdm(R.iterrows()):
        trainMatrix[compid2idx[record.CompoundID], target2idx[record.TargetID]] = record.activity
        descriptorsMatrix[compid2idx[record.CompoundID], :] = ecfp(record.SMILES, nBits=nBits)
        if record['Split'] == "TST":
            DataMask[compid2idx[record.CompoundID], target2idx[record.TargetID]] = True
    testMatrix = trainMatrix.copy()
    testMatrix[~DataMask] = np.NaN
    trainMatrix[DataMask] = np.NaN
    train_dataset = ChEMBLDataset(trainMatrix, descriptorsMatrix, idx2target,idx2compid)
    test_dataset = ChEMBLDataset(testMatrix, descriptorsMatrix, idx2target,idx2compid)
    if data_save == 'yes':
        torch.save((train_dataset,test_dataset,idx2target,idx2compid),'{}.bin'.format(DescrName)) 
    return (train_dataset,test_dataset,idx2target,idx2compid)

def load_dataset(data, nBits=1024,DescrName='DescrName',data_save='no'):
    if not isfile('{}.bin'.format(DescrName)):
        return prepare_dataset(data,nBits,DescrName=DescrName,data_save=data_save) 
    else:
        return torch.load('{}.bin'.format(DescrName))









