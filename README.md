## Multi-Task Data Preparation and DNN Learning Algorithm

This file contains an algorithm for multi-task data preparation and DNN learning. To use this algorithm, you need to fill out the necessary fields with your training settings and indicate the path to the investigated data in the csv format inside the `SI2_run.py` file .

## Input data

The investigated data should be presented in a csv file with the following format (See Table).

An example of imput data is:

| CompoundID  | TargetID | activity | SMILES                                                 | Split |
|-------------|----------|----------|--------------------------------------------------------|-------|
| CHEMBL10    | 809114   | 4.00     | C[S+]([O-])c1ccc(cc1)c2nc(c3ccc(F)cc3)c([nH]2)c4ccncc4 | TRN   |
| CHEMBL98123 | 688293   | 5.10     | COc1ccc(Cc2nccc3cc(OC)c(OC)cc23)cc1OC                  | TRN   |
| CHEMBL98350 | 809272   | 5.50     | O=C1C=C(Oc2c1cccc2c3ccccc3)N4CCOCC4                    | TST   |

## Results

The algorithm will return a txt file containing a dataset with columns:

* Pred (prediction results)
* True (actual values from the investigated dataset)
* TargetID (targets' ID from the analyzed dataset)
* CompoundID (compounds' ID from the analyzed dataset)
* Iter (epoch of learning)

## Settings for Data Preparation

* data: name of the investigated data (full name with path indication is allowed); default: None
* nBits: number of bits during the calculation of Morgan fingerprints using RDKit; default: 1024
* data_save: if set to 'yes', the calculated descriptors will be saved; default: 'no'
* DescrName: name for saving a bin file with compounds' descriptors (full name with path indication is allowed); default: 'DescrName'

## Training Parameters

* task: modelling mode in two options: 'classification', 'regression'; default: 'regression'
* DNNlayers: number of neurons in the layers; default: [512, 256, 128]
* dropout: dropout values for each layer; default: [0.0, 0.0, 0.0]
* actF: activation function in four options: '0' (LeakyReLU), '1' (PReLU), '2' (ELU), '3' (Hardshrink); default: '1'
* optF: optimizers in three options: Adam, RAdam, and Yogi; default: RAdam
* lr: learning rate; default: 0.001
* seed: random seed; default: 70922
* FileName: the name for results saving (full name with path indication is allowed); default: 'FileName'
* savestep: results will be saved for each iteration multiple of the set number; default: 10
* num_epochs: number of epoch

## Citation 
Sosnina, E.A., Sosnin, S. & Fedorov, M.V. Improvement of multi-task learning by data enrichment: application for drug discovery. J Comput Aided Mol Des 37, 183–200 (2023). https://doi.org/10.1007/s10822-023-00500-w
