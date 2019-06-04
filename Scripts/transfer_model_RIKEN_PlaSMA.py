# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:32:19 2019

@author: hcji
"""

import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from rdkit import Chem
from tqdm import tqdm
from libmetgem import msp
from PyCFMID.PyCFMID import cfm_predict
from DeepFrag.utils import load_model, ms_correlation
from DeepFrag.utils import read_ms, morgan_fp, ms2vec, model_predict, plot_compare_ms
from DeepFrag.loss import pearson, loss

pretrain_model = 'simulated_Pos_10V'
model = load_model(pretrain_model)
msp_file = 'RIKEN_PlaSMA/RIKEN_PlaSMA_Pos.msp'

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

    modes.append(ion_mode)
    ms.append(data)
    smiles.append(smi)
    energies.append(energy)
summary = pd.DataFrame({'smiles': smiles, 'ion_mode': modes, 'energy': energies})

# split dataset
test_index = np.random.choice(range(len(ms)), int(0.05 * len(ms)), replace=False)
train_index = [i for i in range(len(ms)) if i not in test_index]
np.save('Temp/RIKEN_train_index', train_index)
np.save('Temp/RIKEN_test_index', test_index)
'''
train_index = np.load('Temp/RIKEN_train_index.npy')
test_index = np.load('Temp/RIKEN_test_index.npy')
'''

# train model
input_data = []
output_forward = []
output_reverse = []
for i in train_index:
    smi = smiles[i]
    fps = morgan_fp(smi)
    data = np.array(ms[i])
    vec_forward = ms2vec(smi, np.round(data[:, 0]), data[:, 1], 'forward')
    vec_reverse = ms2vec(smi, np.round(data[:, 0]), data[:, 1], 'reverse')
    input_data.append(fps)
    output_forward.append(vec_forward)
    output_reverse.append(vec_reverse)

input_data = np.array(input_data)
output_forward = np.array(output_forward)
output_reverse = np.array(output_reverse)

opt = getattr(keras.optimizers, 'adam')
opt = opt(lr=0.0005)
model.compile(optimizer=opt, loss=loss, metrics=[pearson])
history = model.fit(input_data, [output_forward, output_reverse], epochs=10, batch_size=1024, validation_split=0.1)

plt.plot(history.history['regr_forward_pearson'])
plt.plot(history.history['regr_reverse_pearson'])
plt.plot(history.history['val_regr_forward_pearson'])
plt.plot(history.history['val_regr_reverse_pearson'])
plt.ylabel('pearson correlation')
plt.xlabel('epoch')
plt.legend(['train forward', 'train reverse', 'test forward', 'test reverse'], loc='upper left')
plt.show()

model_json = model.to_json()
save_path = 'Model/RIKEN_PlaSMA_Pos_10'
with open(save_path + '.json', "w") as json_file:
    json_file.write(model_json)
model.save_weights(save_path + '.h5')

result = pd.DataFrame(columns = ['idx', 'smiles', 'DeepFrag', 'CFM_10', 'CFM_20', 'CFM_40'])
for i in tqdm(test_index):
    try:
        smi = smiles[i]
        ms_cfm = cfm_predict(smi)
        ms_real = ms[i]
        ms_pred = model_predict(smi, model)
        trans = ms_correlation(ms_real, ms_pred)
        cfm_10 = ms_correlation(ms_real, ms_cfm['low_energy'])
        cfm_20 = ms_correlation(ms_real, ms_cfm['medium_energy'])
        cfm_40 = ms_correlation(ms_real, ms_cfm['high_energy'])
    except:
        continue
    '''
    plot_compare_ms(ms_real, ms_cfm['medium_energy'])
    plot_compare_ms(ms_real, ms_pred)
    '''
    result.loc[len(result)] = [i, smi, trans, cfm_10, cfm_20, cfm_40]
result.to_csv('Result/RIKEN_PlaSMA_Pos_10.csv')