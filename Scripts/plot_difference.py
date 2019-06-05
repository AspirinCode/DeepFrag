# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:49:44 2019

@author: hcji
"""


import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from rdkit import Chem
from tqdm import tqdm
from libmetgem import msp
from PyCFMID.PyCFMID import cfm_predict, fraggraph_gen
from DeepFrag.utils import load_model, ms_correlation
from DeepFrag.utils import read_ms, morgan_fp, ms2vec, model_predict, plot_compare_ms
from DeepFrag.loss import pearson, loss
from DeepFrag.annotate import annotate_ms

msp_file = 'RIKEN_PlaSMA/RIKEN_PlaSMA_Pos.msp'
model = load_model('RIKEN_PlaSMA_Pos_30')
result = pd.read_csv('Result/RIKEN_PlaSMA_Pos_30.csv')

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

    if energy != '30':
        continue
    data = pd.DataFrame(np.array(data))
    data.columns = ['mz', 'intensity']
    modes.append(ion_mode)
    ms.append(data)
    smiles.append(smi)
    energies.append(energy)
summary = pd.DataFrame({'smiles': smiles, 'ion_mode': modes, 'energy': energies})

idx = 1
smi = smiles[idx]
ms_pred = model_predict(smi, model)
ms_anno = annotate_ms(ms_pred, smi)
ms_real = ms[idx]
plot_compare_ms(ms_real, ms_pred)

