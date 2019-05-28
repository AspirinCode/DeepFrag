# -*- coding: utf-8 -*-
"""
Created on Sat May 18 07:47:19 2019

@author: hcji
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
from DeepFrag.utils import ms_correlation, ms_jaccard, ms_residual, model_predict

def identification(ms, candidates, model, method='jaccard'):
    smiles = []
    scores = []
    inchis = []
    masses = []
    pred_ms = []
    if method == 'residual':
        score = ms_residual
    elif method == 'correlation':
        score = ms_correlation
    else:
        score = ms_jaccard
    if 'InChI=' in candidates[0]:
        read_candidate = Chem.MolFromInchi
    else:
        read_candidate = Chem.MolFromSmiles
    for i in candidates:
        try:
            mol = read_candidate(i)
            smi = Chem.MolToSmiles(mol)
            inchi = Chem.MolToInchi(mol)
            mass = CalcExactMolWt(mol)
        except:
            continue
        pms = model_predict(smi, model)
        scr = score(ms, pms)
        smiles.append(smi)
        inchis.append(inchi)
        scores.append(scr)
        masses.append(mass)
        pred_ms.append(pms)
    output = pd.DataFrame({'SMILES': smiles, 'InChI': inchis, 'mass': masses, 'scores': scores, 'pred_ms': pred_ms})
    output = output.sort_values('scores', ascending=False)
    return output
        