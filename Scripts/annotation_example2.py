# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 10:24:50 2019

@author: hcji
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libmetgem import msp
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcMolFormula
from rdkit.Chem import AllChem
from DeepFrag.utils import load_model, ms_correlation
from DeepFrag.utils import read_ms, morgan_fp, ms2vec, model_predict, plot_compare_ms
from DeepFrag.loss import pearson, loss
from DeepFrag.annotate import annotate_ms

from pycdk.pycdk import add_formula, subtract_formula, check_formula, getFormulaExactMass
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as numpy2ri
numpy2ri.activate()
robjects.r('''source('DeepFrag/metfrag.R')''')
generateFragments = robjects.globalenv['generateFragments']


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


# example 1
idx = 551
smi = smiles[idx]
mol = Chem.MolFromSmiles(smi)
ms_pred = model_predict(smi, model)
ms_real = ms[idx]

# annotation
mzs = np.array(ms_pred['mz'])
intensities = np.array(ms_pred['intensity'])
mol = Chem.MolFromSmiles(smi)
precursor = CalcExactMolWt(mol) - 1.0032
formula = CalcMolFormula(mol)
frags = np.unique(generateFragments(smi, treeDepth=2))
frags_new = [Chem.MolFromSmiles(s) for s in frags]
frags_formula = np.unique([CalcMolFormula(f) for f in frags_new])
loss_formula = []
for f in frags_formula:
    l = subtract_formula(formula, f)
    if l == '':
        continue
    if check_formula(l):
        loss_formula.append(l)
    add_H = add_formula(l, 'H')
    de_H = subtract_formula(l, 'H')
    if check_formula(add_H):
        loss_formula.append(add_H)
    if check_formula(de_H):
        loss_formula.append(de_H)
loss_formula = np.unique(loss_formula)
loss_mass = np.array([getFormulaExactMass(f) for f in loss_formula])
ms_new = pd.DataFrame(columns=['mz', 'intensity', 'annotate_loss', 'exact_mass'])
for i, mz in enumerate(mzs):
    intensity = intensities[i]
    diff = precursor - mz
    if min(np.abs(loss_mass - diff)) < 0.5:
        match = np.where(np.abs(loss_mass - diff) < 0.5)[0]
        annotate_loss = loss_formula[match]
        accurate_mass = precursor - loss_mass[match]
    else:
        annotate_loss = ''
        accurate_mass = ''
    ms_new.loc[len(ms_new)] = [mz, intensity, annotate_loss, accurate_mass]
