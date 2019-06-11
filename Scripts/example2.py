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

msp_file = 'RIKEN_PlaSMA/RIKEN_PlaSMA_Neg.msp'
model = load_model('RIKEN_PlaSMA_Neg_10')
pretrain = load_model('simulated_Neg_10V')
result = pd.read_csv('Result/RIKEN_PlaSMA_Neg_10.csv')

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


# example 2
idx = 551
smi = smiles[idx]
mol = Chem.MolFromSmiles(smi)
t1 = time.time()
ms_pred = model_predict(smi, model)
t2 = time.time()
ms_cfm = cfm_predict(smi, ionization_mode='-')
t3 = time.time()
ms_pretrain = model_predict(smi, pretrain)
ms_anno, frags = annotate_ms(ms_pred, smi)
ms_real = ms[idx]
plot_compare_ms(ms_real, ms_pretrain)
plot_compare_ms(ms_pretrain, ms_cfm['low_energy'])
plot_compare_ms(ms_real, ms_pred)
plot_compare_ms(ms_real, ms_cfm['low_energy'])
print ('computing time of CFM is: ' + str(t3-t2) + ' s')
print ('computing time of DeepFrag is: ' + str(t2-t1) + ' s')


## precursor
Chem.MolFromSmiles('OCC1OC(C(O)C(O)C1O)C1=C(O)C2=C(OC(=CC2=O)C2=CC(O)=C(O)C=C2)C=C1O')

## Suppress
Chem.MolFromSmiles('C=C1OC(C2=C(O)C3=C(CC2O)OC(C2=CC(O)C(O)CC2)=CC3=O)=C([O-])C(=O)C1=O') # 429.082
Chem.MolFromSmiles('O=C=C(C=O)OC(=C=O)C1=C([O-])C2=C(CC1O)OC(C1CCC(O)C(O)C1)=CC2=O') # 417.082

## enhance
Chem.MolFromSmiles('O=C1C=C(C2=CC(=O)C3=C(CC(O)C(C#C[O-])=C3O)O2)CCC1=O') # 327.051
Chem.MolFromSmiles('O=CC([O-])=C=C1C(=O)C2=C(CC1O)OC(C1=CC(=O)C(O)CC1)=CC2=O') # 357.061
                                  

