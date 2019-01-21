#-*- coding: UTF-8 -*-
import json
import numpy as np
import sys
import time as time_mod
import datetime as dt_mod

from os import path
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from time import time

# 用存出的Infer Model重讀進來使用

version_path = '02'
modelsFolder_path = './' if __name__ == '__main__' else path.join(path.dirname(__file__))+'/'
modelsFolder_path += version_path+'/'
print('In model')

h5name = 'CFR_MODEL_LSTM[LrDsDsDsDsDr_f11hn512bs3dp0.001]_ModelInf_R0_aefghijklm_v2_LTB_spc1.h5'
sel_ftr = [0] + list(range(4, 13))
ffx = ""
for v in sel_ftr:
    ffx += chr(v+97) # convert 0~12 to a~m
print(ffx)

model_lan =  load_model(
    modelsFolder_path + h5name, custom_objects={
            'swish': lambda x: x * K.sigmoid(x),
            'binary': lambda x: K.switch(x < 0, x*0, x/x),
            'bround': lambda x: K.switch(x < 0, x*0, K.round(x))})

graph = tf.get_default_graph()

def predict(ftr_data, ts_prd):
    # global CFR_models_dict
    # global graph
    debug = False

    s = time()
    with graph.as_default():

        hidden_neurons = 512

        s_scaler = joblib.load(modelsFolder_path + 's_scaler.pkl')

        dt = dt_mod.datetime(year=int(ftr_data[0]), month=int(ftr_data[1]), day=1)

        ftr_a = time_mod.mktime(dt.timetuple())

        ftr_all = [ftr_a] + ftr_data[:2] + [-1.] + ftr_data[2:]

        ftr_data_norm = s_scaler.transform(np.array(ftr_all).reshape(1,13))

        f_0 = ftr_data_norm[:, sel_ftr].reshape(1,1,len(sel_ftr))
        x_0 = np.concatenate((np.zeros((1, 1, 1)), f_0), axis = -1)
        a_0 = np.zeros((1, hidden_neurons))
        c_0 = np.zeros((1, hidden_neurons))

        x_j = x_0
        a_j = a_0
        c_j = c_0

        L = []
        for j in range(1, ts_prd + 1):
            pred = model_lan.predict([x_j, a_j, c_j])

            x_j = np.concatenate((pred[0].reshape(1,1,1), f_0), axis = -1)
            a_j = pred[1]
            c_j = pred[2]

            L.append(float(pred[0]))


        cfr_norm = np.array(L).reshape(-1)
        cfr_norm_str =  ','.join(map(str, cfr_norm))

        c_scaler = joblib.load(modelsFolder_path + 'c_scaler.pkl')

        cfr_orig = c_scaler.inverse_transform(cfr_norm.reshape(-1,1)).reshape(-1)
        cfr_orig_str = ','.join(map(str, cfr_orig))

        print("\n\n\nprediction:\n{0}\n\n\n".format(cfr_norm_str)) if debug else False
        print("\n\n\nprediction (inversed):\n{0}\n\n\n".format(cfr_orig_str)) if debug else False


        e = time()
        print("%.5f sec"%(e-s)) if debug else False
        return(
            {
              'cfr': cfr_orig_str,
              'cfr_norm': cfr_norm_str
            }
        )


if __name__ == '__main__':

    """
          |0_Ts |1_Yr |2_Mt |3_N  |4_IL |5_NG |6_NL |7_AG |8_AL |9_GG |10GL |11Pn |12ED |
    INPUT |     |    V|    V|     |    V|    V|    V|    V|    V|    V|    V|    V|    V|
          |    a|    b|    c|    d|    e|    f|    g|    h|    i|    j|    k|    l|    m|
    E-VL  |     | 2011| 1.00| 0.00| 1.00| 4.00| 1.00| 6.00| 1.00|15.00| 6.00| 7.50| 8.00|
     
    evl = [      2011.,   1.,         1.,   4.,   1.,   6.,   1.,  15.,    6,  7.5,   8.]
    
    """

    ts_prd = 36
    evl = [      2011.,   1.,         1.,   4.,   1.,   6.,   1.,  15.,    6,  7.5,   8.]
    print(predict(evl, ts_prd))
