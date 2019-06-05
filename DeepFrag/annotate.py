# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:03:58 2019

@author: hcji
"""

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as numpy2ri
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

numpy2ri.activate()
robjects.r('''source('DeepFrag/metfrag.R')''')
generateFragments = robjects.globalenv['generateFragments']


def annotate_ms(ms_pred, smi, treeDepth=2):
    mzs = np.array(ms_pred['mz'])
    intensities = np.array(ms_pred['intensity'])
    frags = list(generateFragments(smi, treeDepth=treeDepth)) + [smi]
    frags_new = [Chem.MolFromSmiles(f) for f in frags]
    frags_mass = np.array([CalcExactMolWt(f) for f in frags_new])
    ms_new = pd.DataFrame(columns=['mz', 'intensity', 'smiles', 'addH'])
    for i, mz in enumerate(mzs):
        diff = np.abs(mz - frags_mass)
        if min(diff) <= 3:
            this = np.argmin(diff)
            smiles = frags[this]
            addH = round(mz - frags_mass[this])
            mass = frags_mass[this] + 1.003*addH
            intensity = intensities[i]
            ms_new.loc[len(ms_new)] = [mass, intensity, smiles, addH]
    return ms_new
        
        