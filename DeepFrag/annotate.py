# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:03:58 2019

@author: hcji
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcMolFormula
from pycdk.pycdk import add_formula, subtract_formula, check_formula, getFormulaExactMass

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as numpy2ri
numpy2ri.activate()
robjects.r('''source('DeepFrag/metfrag.R')''')
generateFragments = robjects.globalenv['generateFragments']

def annotate_ms(ms_pred, smi, treeDepth=2):
    mzs = np.array(ms_pred['mz'])
    intensities = np.array(ms_pred['intensity'])
    mol = Chem.MolFromSmiles(smi)
    precursor = CalcExactMolWt(mol) + 1.0032
    formula = CalcMolFormula(mol)
    frags = np.unique(generateFragments(smi, treeDepth=2))
    frags_new = np.array([Chem.MolFromSmiles(s) for s in frags])
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
    loss_mass = [getFormulaExactMass(f) for f in loss_formula]
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
    return ms_new


