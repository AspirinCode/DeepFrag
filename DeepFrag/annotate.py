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
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcMolFormula
numpy2ri.activate()
robjects.r('''source('DeepFrag/metfrag.R')''')
generateFragments = robjects.globalenv['generateFragments']


def annotate_ms(ms_pred, smi, treeDepth=2):
    mzs = np.array(ms_pred['mz'])
    intensities = np.array(ms_pred['intensity'])
    frags = list(generateFragments(smi, treeDepth=treeDepth)) + [smi]
    frags_new = [Chem.MolFromSmiles(f) for f in frags]
    frags_mass = np.array([CalcExactMolWt(f) for f in frags_new])
    frags_formula = np.array([CalcMolFormula(f) for f in frags_new])
    summary = pd.DataFrame({'formula': frags_formula, 'mass': frags_mass})
    summary = summary.drop_duplicates()
    summary = summary.reset_index(drop=True)
    ms_new = pd.DataFrame(columns=['mz', 'intensity', 'annotation', 'exact_mass'])
    for i, mz in enumerate(mzs):
        diff = np.abs(mz - summary['mass'])
        if min(diff) <= 3:
            intensity = intensities[i]
            this = np.where(diff<=3)[0]
            addH = np.array(np.round(mz - summary['mass'][this]))
            annotation = [str(this[s]) + '.' + str(int(addH[s])) + 'H' for s in range(len(this))]
            exact_mass = list(summary['mass'][this] + addH)
            ms_new.loc[len(ms_new)] = [mz, intensity, annotation, exact_mass]
    return ms_new, summary


