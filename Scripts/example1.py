# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:49:44 2019

@author: hcji
"""

import time
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from tqdm import tqdm
from libmetgem import msp
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from PyCFMID.PyCFMID import cfm_predict, fraggraph_gen
from DeepFrag.utils import load_model, ms_correlation
from DeepFrag.utils import read_ms, morgan_fp, ms2vec, model_predict, plot_compare_ms
from DeepFrag.loss import pearson, loss
from DeepFrag.annotate import annotate_ms

msp_file = 'RIKEN_PlaSMA/RIKEN_PlaSMA_Pos.msp'
model = load_model('RIKEN_PlaSMA_Pos_10')
pretrain = load_model('simulated_Pos_10V')
result = pd.read_csv('Result/RIKEN_PlaSMA_Pos_10.csv')

# parser dataset
ms = []
smiles = []
energies = []
modes = []
parser = msp.read(msp_file)
for i, (params, data) in enumerate(parser):
    if 'collisionenergy' in params:
        energy = params['collisionenergy']
    else:
        energy = ''
    if 'precursortype' in params:
        ion_mode = params['precursortype']
    else:
        ion_mode = ''
    if 'smiles' in params:
        smi = params['smiles']
    else:
        smi = ''

    if energy != '10':
        continue
    data = pd.DataFrame(np.array(data))
    data.columns = ['mz', 'intensity']
    modes.append(ion_mode)
    ms.append(data)
    smiles.append(smi)
    energies.append(energy)
summary = pd.DataFrame({'smiles': smiles, 'ion_mode': modes, 'energy': energies})


# example 1
idx = 1297
smi = smiles[idx]
mol = Chem.MolFromSmiles(smi)
t1 = time.time()
ms_pred = model_predict(smi, model)
t2 = time.time()
ms_cfm = cfm_predict(smi)
t3 = time.time()
ms_pretrain = model_predict(smi, pretrain)
ms_real = ms[idx]
plot_compare_ms(ms_real, ms_pretrain)
plot_compare_ms(ms_pretrain, ms_cfm['low_energy'])
plot_compare_ms(ms_real, ms_pred)
plot_compare_ms(ms_real, ms_cfm['low_energy'])
print ('computing time of CFM is: ' + str(t3-t2) + ' s')
print ('computing time of DeepFrag is: ' + str(t2-t1) + ' s')

## precursor
Chem.MolFromSmiles('Oc1cc(O)c2c(c1)OC(c1ccc(O)c(O)c1)C([OH2+])C2')

## pretrain and CFM 
Chem.MolFromSmiles('C=C1C=CC(O)=CC1=[OH+]') # 123.044
Chem.MolFromSmiles('C=C1C(O)=CC(O)=CC1=[OH+]') # 139.038
Chem.MolFromSmiles('O=C1CCC(=C2C=CC3=C([OH2+])C=C(O)C=C3O2)C=C1O') # 273.075

## DeepFrag only
Chem.MolFromSmiles('CC(=[OH+])C=C1C=CC(=O)C(O)=C1') # 165.059
Chem.MolFromSmiles('C=C1C(=[OH+])CC(O)CC1O') # 207.070

