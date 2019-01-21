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

# modelsFolder_path = './' if __name__ == '__main__' else path.join(path.dirname(__file__))+'/'
# print('In model')
# print(modelsFolder_path)
#
# h5name = 'CFR_MODEL_LSTM[LrDsDsDsDsDr_f11hn512bs3dp0.001]_ModelInf_R0_aefghijklm_v2_LTB_spc1.h5'
# sel_ftr = [0] + list(range(4, 13))
# ffx = ""
# for v in sel_ftr:
#     ffx += chr(v+97) # convert 0~12 to a~m
# print(ffx)

CFR_models_dict = {
    # 'LAN': load_model(
    # modelsFolder_path + h5name, custom_objects={
    #         'swish': lambda x: x * K.sigmoid(x),
    #         'binary': lambda x: K.switch(x < 0, x*0, x/x),
    #         'bround': lambda x: K.switch(x < 0, x*0, K.round(x))})
}


from LAN.inference import predict


def run(component, ftr_data, ts_prd):
    # global CFR_models_dict

    component = component.upper()

    routes = {
        'LAN': lambda x: predict(x, ts_prd),
        'AUDIO': lambda x: AUDIO(x,CFR_models_dict[component]),
        'SIO': lambda x: SIO(x,CFR_models_dict[component]),
        'PCH': lambda x: PCH(x,CFR_models_dict[component]),
        'POWER': lambda x: POWER(x,CFR_models_dict[component]),
    }
    return routes[component](ftr_data)


if __name__ == '__main__':
    component = 'LAN'
    
    """
          |0_Ts |1_Yr |2_Mt |3_N  |4_IL |5_NG |6_NL |7_AG |8_AL |9_GG |10GL |11Pn |12ED |
    INPUT |     |    V|    V|     |    V|    V|    V|    V|    V|    V|    V|    V|    V|
          |    a|    b|    c|    d|    e|    f|    g|    h|    i|    j|    k|    l|    m|
    E-VL  |     | 2011| 1.00| 0.00| 1.00| 4.00| 1.00| 6.00| 1.00|15.00| 6.00| 7.50| 8.00|
     
    evl = [      2011.,   1.,         1.,   4.,   1.,   6.,   1.,  15.,    6,  7.5,   8.]
    
    """

    ts_prd = 36
    evl = [      2011.,   1.,         1.,   4.,   1.,   6.,   1.,  15.,    6,  7.5,   8.]
    print(run(component, evl, ts_prd))
