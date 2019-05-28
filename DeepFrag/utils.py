# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:52:11 2019

@author: hcji
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
from keras.models import Model, model_from_json

def read_ms(csv, precursor = None, norm = True):
    '''
    Task: 
        Read spectrum from csv file
    Parameters:
        csv: str, file path of spectrum
        precursor: folat, m/z of precursor
        norm: logic, whether to normalizate the spectrum or not
    '''
    spec = pd.read_csv(csv)
    spec = spec.iloc[:,range(2)]
    spec.columns = ['mz', 'intensity']
    if norm:
        spec['intensity'] = spec['intensity'] / max(spec['intensity'])
    if precursor is not None:
        keep = spec['mz'] < precursor + 1
        spec = spec.loc[keep]
    return spec


def ms2vec(smi, peakindex, peakintensity, direction='forward', maxmz=1500):
    mass = round(CalcExactMolWt(Chem.MolFromSmiles(smi))) + 2
    output = np.zeros(maxmz)
    for i, j in enumerate(peakindex):
        if round(j) >= maxmz:
            continue
        else:
            if direction == 'forward':
                output[int(round(j))] = float(peakintensity[i])
            else:
                if mass - round(j) < 0 or mass - round(j) > maxmz:
                    continue
                output[mass - int(round(j))] = float(peakintensity[i])
    if max(output) == 0:
        pass
    output = output / (max(output) + 10 ** -6)
    return output


def vec2ms(smi, vec, direction='forward', maxmz=1500, norm = True):
    if direction == 'reverse':
        mass = round(CalcExactMolWt(Chem.MolFromSmiles(smi))) + 2
    peakindex = np.where(vec > 0.05*max(vec))[0]
    peakintensity = vec[peakindex]
    peakintensity[np.where(peakintensity < 0)[0]] = 0
    if direction == 'reverse':
        peakindex = mass - peakindex
    if norm:
        peakintensity = peakintensity / (max(peakintensity) + 10 ** -6)
    output = pd.DataFrame({'mz': peakindex, 'intensity':peakintensity})
    return output

 
def morgan_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096)
    return np.array(ecfp)    


def model_predict(smi, model):
    mass = CalcExactMolWt(Chem.MolFromSmiles(smi)) + 2
    input_data = morgan_fp(smi)
    input_data = np.array([input_data])
    pred_spec_forward, pred_spec_reverse = model.predict(input_data)
    pred_spec_forward = vec2ms(smi, pred_spec_forward[0], norm=False, direction='forward')
    pred_spec_reverse = vec2ms(smi, pred_spec_reverse[0], norm=False, direction='reverse')
    pred_spec_forward = pred_spec_forward[pred_spec_forward.mz <= 0.5*mass]
    pred_spec_reverse = pred_spec_reverse[pred_spec_reverse.mz > 0.5*mass]
    output = pd.concat([pred_spec_forward, pred_spec_reverse])
    output = output.sort_values('mz')
    output = output.reset_index(drop=True)
    output['intensity'] = output['intensity'] / max(output['intensity'])
    return output


def plot_ms(spectrum):
    plt.figure(figsize=(6, 4))
    plt.vlines(spectrum['mz'], np.zeros(spectrum.shape[0]), np.array(spectrum['intensity']), 'red') 
    plt.axhline(0, color='black')
    plt.xlabel('m/z')
    plt.ylabel('Relative Intensity')
    plt.show()


def plot_compare_ms(spectrum1, spectrum2, tol=0.5):
    spectrum1 = pd.DataFrame(spectrum1)
    spectrum2 = pd.DataFrame(spectrum2)
    spectrum1.columns = ['mz', 'intensity']
    spectrum2.columns = ['mz', 'intensity']
    spectrum1['intensity'] /= np.max(spectrum1['intensity'])
    spectrum2['intensity'] /= np.max(spectrum2['intensity'])
    c_mz = []
    c_int = []
    for i in spectrum1.index:
        diffs = abs(spectrum2['mz'] - spectrum1['mz'][i])
        if min(diffs) < tol:
            c_mz.append(spectrum1['mz'][i])
            c_mz.append(spectrum2['mz'][np.argmin(diffs)])
            c_int.append(spectrum1['intensity'][i])
            c_int.append(-spectrum2['intensity'][np.argmin(diffs)])
    c_spec = pd.DataFrame({'mz':c_mz, 'intensity':c_int}) 
    plt.figure(figsize=(6, 6))
    plt.vlines(spectrum1['mz'], np.zeros(spectrum1.shape[0]), np.array(spectrum1['intensity']), 'gray')
    plt.axhline(0, color='black')
    plt.vlines(spectrum2['mz'], np.zeros(spectrum2.shape[0]), -np.array(spectrum2['intensity']), 'gray')
    plt.vlines(c_spec['mz'], np.zeros(c_spec.shape[0]), c_spec['intensity'], 'red')
    plt.xlabel('m/z')
    plt.ylabel('Relative Intensity')
    plt.show()
    
    
def load_model(model_name, model_path='Model'):
    json_file = open(model_path + '/' + model_name + '.json', 'r') 
    loaded_model_json = json_file.read() 
    json_file.close()  
    model = model_from_json(loaded_model_json)
    model.load_weights(model_path + '/' + model_name + '.h5')
    return model


def ms_residual(spectrum1, spectrum2, tol=0.5):
    c_mz = np.append(spectrum1['mz'], spectrum2['mz'])
    c_int = np.append(spectrum1['intensity'], -spectrum2['intensity'])
    residual = 0
    res_inds = range(len(c_mz))
    for i, m in enumerate(c_mz):
        if i not in res_inds:
            continue
        inds = np.where(abs(c_mz - m) < tol)[0]
        residual += abs(sum(c_int[inds]))
        res_inds = np.setdiff1d(res_inds, inds)
    return 1 - residual/sum(np.abs(c_int))
  

def ms_correlation(spectrum1, spectrum2):
    spectrum1 = pd.DataFrame(spectrum1)
    spectrum2 = pd.DataFrame(spectrum2)
    v1 = ms2vec('', np.round(spectrum1.iloc[:,0]), spectrum1.iloc[:,1], 'forward')
    v2 = ms2vec('', np.round(spectrum2.iloc[:,0]), spectrum2.iloc[:,1], 'forward')
    return pearsonr(v1, v2)[0]


def ms_jaccard(spectrum1, spectrum2):
    v1 = np.where(spectrum1.iloc[:,1] >= 0.01 * np.max(spectrum1.iloc[:,1]))[0]
    v2 = np.where(spectrum2.iloc[:,1] >= 0.01 * np.max(spectrum2.iloc[:,1]))[0]
    v1 = np.round(spectrum1['mz'][v1])
    v2 = np.round(spectrum2['mz'][v2])
    intersection_cardinality = len(set.intersection(*[set(v1), set(v2)]))
    union_cardinality = len(set.union(*[set(v1), set(v2)]))
    return intersection_cardinality/float(union_cardinality)
