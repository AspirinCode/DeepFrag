# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:16:44 2019

@author: hcji
"""

import h5py
import numpy as np
from libmetgem import msp
from DeepFrag.utils import read_ms, ms2vec, morgan_fp
from DeepFrag.loss import pearson, loss

# transform msp to dataset
mine_1 = msp.read('D:/Data/MINE_KEGG_Positive_CFM_Spectra.msp/Positive_CFM_Spectra.msp')
mine_2 = msp.read('D:/Data/MINE_Ec_Positive_CFM_Spectra.msp/Positive_CFM_Spectra.msp')
mine_3 = msp.read('D:/Data/MINE_YMDB_Positive_CFM_Spectra.msp/Positive_CFM_Spectra.msp')

def parser_msp(ms, output, energy):
    input_data = []
    output_forward = []
    output_reverse = []
    for i, (params, data) in enumerate(ms):
        if params['energy'] != energy:
            continue
        smi = params['smiles']
        try:
            fps = morgan_fp(smi)
            vec_forward = ms2vec(smi, np.round(data[:,0]), data[:,1], 'forward')
            vec_reverse = ms2vec(smi, np.round(data[:,0]), data[:,1], 'reverse')
        except:
            continue
        input_data.append(fps)
        output_forward.append(vec_forward)
        output_reverse.append(vec_reverse)         
        if i % 1000 == 999:
            print ('finish ' + str(i) + ' compounds')
    input_data = np.array(input_data)
    output_forward = np.array(output_forward)
    output_reverse = np.array(output_reverse) 
    with h5py.File(output, 'w') as hf:
        hf.create_dataset("input_data",  data=input_data)
        hf.create_dataset("output_forward",  data=output_forward)
        hf.create_dataset("output_reverse",  data=output_reverse)

parser_msp(ms=mine_1, output='Data/MINE_KEGG.h5', energy='40 V')
parser_msp(ms=mine_2, output='Data/MINE_EC.h5', energy='40 V')
parser_msp(ms=mine_3, output='Data/MINE_YMDB.h5', energy='40 V')

# import dataset
with h5py.File('Data/MINE_KEGG.h5', 'r') as hf:
    input_data = hf['input_data'][:]
    output_forward = hf['output_forward'][:]
    output_reverse = hf['output_reverse'][:]
with h5py.File('Data/MINE_EC.h5', 'r') as hf:
    d1 = hf['input_data'][:]
    d2 = hf['output_forward'][:]
    d3 = hf['output_reverse'][:]
    input_data = np.concatenate([input_data, d1])
    output_forward = np.concatenate([output_forward, d2])
    output_reverse = np.concatenate([output_reverse, d3])
with h5py.File('Data/MINE_YMDB.h5', 'r') as hf:
    d1 = hf['input_data'][:]
    d2 = hf['output_forward'][:]
    d3 = hf['output_reverse'][:]
    input_data = np.concatenate([input_data, d1])
    output_forward = np.concatenate([output_forward, d2])
    output_reverse = np.concatenate([output_reverse, d3])

# train model
import keras
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
from DeepFrag.utils import model_predict, plot_compare_ms

layer_in = Input(shape=(input_data.shape[1],), name="smile")
layer_forward = layer_in
layer_reverse = layer_in
for i in range(5):
    layer_forward = Dense(1024, activation="relu")(layer_forward)
    layer_reverse = Dense(1024, activation="relu")(layer_reverse)
layer_forward = Dense(output_forward.shape[1], activation="linear", name="regr_forward")(layer_forward)
layer_reverse = Dense(output_forward.shape[1], activation="linear", name="regr_reverse")(layer_reverse)
opt = getattr(keras.optimizers, 'adam')
opt = opt(lr=0.001)
model = Model(input=layer_in, outputs=[layer_forward, layer_reverse])
model.compile(optimizer=opt, loss=loss, metrics=[pearson])
history = model.fit(input_data, [output_forward, output_reverse], epochs=50, batch_size=1024, validation_split=0.1)

plt.plot(history.history['regr_forward_pearson'])
plt.plot(history.history['regr_reverse_pearson'])
plt.plot(history.history['val_regr_forward_pearson'])
plt.plot(history.history['val_regr_reverse_pearson'])
plt.title('pearson correlation')
plt.ylabel('pearson correlation')
plt.xlabel('epoch')
plt.legend(['train forward', 'train reverse', 'test forward', 'test reverse'], loc='upper left')
plt.show()

model_json = model.to_json()
save_path = 'Model/simulated_40V'
with open(save_path + '.json', "w") as json_file:  
    json_file.write(model_json)
model.save_weights(save_path + '.h5')

test_smi = 'C1=C(NC=N1)C[C@@H](C(=O)O)N'
real_ms = read_ms('E:/project/DeepMASS/data/spectra/simulated_spectra/40V/C00135.csv')
pred_ms = model_predict(test_smi, model)
plot_compare_ms(real_ms, pred_ms)