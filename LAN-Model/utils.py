import pandas as pd
import numpy as np
import os
import re

"""
read spec. table for LANs, and read cfr files
"""

def read_csv(datapath):

    spcname = 'lan_spc_1.csv'
    spc_csv = pd.DataFrame()
    spc_csv = pd.read_csv(datapath + spcname)
    spc_csv = spc_csv.loc[(spc_csv['LAN'] != '88E8059') & (spc_csv['LAN'] != 'QCA8171')]

    NAME_IN_CSV = set(spc_csv['LAN'])
    LAN_CSV_DCT = dict()

    for dirname, dirnames, filenames in os.walk(datapath):
        # print path to all subdirectories first
        for subdirname in dirnames:
            print(os.path.join(dirname, subdirname))

        print("Check all {0} files ...".format(len(filenames)))

        # go through all entries
        for filename in filenames:
            filepath = os.path.join(dirname, filename)
            #print (filepath)
            m = re.search('LAN_(.+?)_CFR', filename)
            if m:
                name = m.group(1)
                if name in NAME_IN_CSV:
                    CSV = pd.read_csv(filepath)
                    LAN_CSV_DCT[name] = CSV

    print("Got {0} cfr files".format(len(LAN_CSV_DCT.keys())))

    LAN_NAME_ARR = np.array(spc_csv['LAN'])
    LAN_NAME_LST = list(LAN_NAME_ARR)
    print (LAN_NAME_ARR)

    """
    Setup Maximum and Minimum
    """
    s = pd.Series({'LAN': 'MIN',
                   'Year': 2005.,
                   'Month': 1.,
                   'New': 0.,
                   'IC Surge L2L': 0.,  
                   'MB Surge Normal L2G': 0., 
                   'MB Surge Normal L2L': 0., 
                   'MB Surge Air Discharge L2G': 0., 
                   'MB Surge Air Discharge L2L': 0., 
                   'MB Surge Guard L2G': 0., 
                   'MB Surge Guard L2L': 0., 
                   'Power Pin': 0.,
                   'ESD': 0.})

    spc_csv = spc_csv.append(s, ignore_index=True)

    s = pd.Series({'LAN': 'MAX',
                   'Year': 2020.,
                   'Month': 12.,
                   'New': 5.,
                   'IC Surge L2L': 20.,  
                   'MB Surge Normal L2G': 20., 
                   'MB Surge Normal L2L': 20., 
                   'MB Surge Air Discharge L2G': 20., 
                   'MB Surge Air Discharge L2L': 20., 
                   'MB Surge Guard L2G': 20., 
                   'MB Surge Guard L2L': 20., 
                   'Power Pin': 20.,
                   'ESD': 20.})
    spc_csv = spc_csv.append(s, ignore_index=True)
    return LAN_CSV_DCT, spc_csv


from sklearn.preprocessing import MinMaxScaler

# scaling data, data must be 2d array
def scale(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return scaler, data_scaled

import time
import datetime


def create_S(raw_spc):

    """
    Combine Year and Month into a timestamp
    """
    TP = [];
    for i in range(len(raw_spc)):
        if i < 10:
            fp = ["{:0.2f}".format(v) for v in raw_spc[i]]
            #print (fp)
        dt = datetime.datetime(year=int(raw_spc[i][0]), month=int(raw_spc[i][1]), day=1)
        TP.append(time.mktime(dt.timetuple()))

    S_ALL = np.concatenate((np.array(TP).reshape(-1,1), raw_spc[:,:]), axis=1)
    return S_ALL


"""
create (beg, end) tuples
""" 
LAN_TUP_DCT = dict()
# avg 440000
LAN_TUP_DCT[  'RTL8111E-VL'] = ('B5','E4'); LAN_TUP_DCT[  'RTL8111F-VB'] = ('C3', 'F2'); # avg 610000
# avg 450000
LAN_TUP_DCT[  'RTL8111G-CG'] = ('D6','G5'); LAN_TUP_DCT[ 'RTL8111GR-CG'] = ('D6', 'G5'); # avg 420000
# avg 420000
LAN_TUP_DCT[  'RTL8111H-CG'] = ('F4','HB')
"""
other lan (AR, RTL, WG, ...)
"""
# avg 70000
LAN_TUP_DCT[ 'AR8131L-AL1E'] = ('B1','DC'); LAN_TUP_DCT[       'AR8151'] = ('B1','DC'); # avg 70000
# avg 30000
LAN_TUP_DCT['AR8161-AL3A-R'] = ('C4','F3'); LAN_TUP_DCT['AR8161-BL3A-R'] = ('C9','F8'); # avg 100000
# avg 200
LAN_TUP_DCT[      'RTL8112'] = ('B5','E4'); LAN_TUP_DCT[     'RTL8112L'] = ('B5','E4'); # avg 70000
# avg 300
LAN_TUP_DCT[      'RTL8131'] = ('B5','E4'); LAN_TUP_DCT[ 'RTL8111DP-VC'] = ('D3','G2'); # avg 400
# avg 110000
LAN_TUP_DCT[  'RTL8111E-VB'] = ('B5','E4'); LAN_TUP_DCT[ 'RTL8111EP-CG'] = ('E7','H6'); # avg 140
# avg 1500
LAN_TUP_DCT[     'WG82567V'] = ('B1','DC'); LAN_TUP_DCT[    'WG82578DM'] = ('B1','DC'); # avg 800
# avg 57000
LAN_TUP_DCT[     'WG82579V'] = ('B1','DC'); LAN_TUP_DCT[    'WG82579LM'] = ('B5','E4'); # avg 2500
# avg 2600
LAN_TUP_DCT[     'WG82583V'] = ('B1','DC'); LAN_TUP_DCT[ 'WGI211AT(A2)'] = ('E8','H7'); # avg 8000
# avg 9000
LAN_TUP_DCT[     'WGI217LM'] = ('D6','G5'); LAN_TUP_DCT[      'WGI217V'] = ('D6','G5'); # avg 30000
# avg 70000
LAN_TUP_DCT[      'WGI218V'] = ('E5','H4'); LAN_TUP_DCT[     'WGI218LM'] = ('E5','H4'); # avg 800

"""
below should be skipped since there's any shippings or data
"""
LAN_TUP_DCT[      '82567LF'] = ('B1','DC'); LAN_TUP_DCT[      '82567LM'] = ('B1','DC');
LAN_TUP_DCT[    'RTL8103EL'] = ('B5','E4');

"""
TARGET (id set) will determine the maximum and minimum
will return normalized cfr array and original cfr array
"""

def create_C(LAN_CSV_DCT, LAN_NAME_ARR, TARGET, lan_tup_dct = LAN_TUP_DCT, debug = False):

    print ("Target ids: {0}".format(TARGET))
    
    A = []
    for name in LAN_NAME_ARR[TARGET]:
        lan_csv = LAN_CSV_DCT[name]
        snt = lan_tup_dct[name]
        bid = int(lan_csv.loc[lan_csv['SN']==snt[0]].index.values)
        eid = int(lan_csv.loc[lan_csv['SN']==snt[1]].index.values) + 1
        cfr_arr = np.array(lan_csv.iloc[bid:eid, 4:]) # exclude name, year, month, shipping
        L = [row for row in cfr_arr.T]
        cnan = lambda x: str(-1) if pd.isnull(x) else str(x)
        C = []
        for l in L:
            nl = [float(cnan(e).replace('%', '')) for e in l]
            cfr = 0.0
            if -1.0 in nl:
                cfr = np.mean(nl[0:nl.index(-1.0)])
            else:
                cfr = np.mean(nl)
            C.append(cfr)
        A.append(C)

    X = []
    for a in A:
        X += a
    # CFR
    c_scaler, _ = scale(np.array(X).reshape(-1, 1))

    CFR_ARR = np.array(A) # 2D

    def scale_lst_to_arr(scaler, lst):
        return np.array([scaler.transform(v) for v in lst]).reshape(len(lst))

    L = []
    [L.append(scale_lst_to_arr(c_scaler, arr)) for arr in CFR_ARR]
    
    c_norm = np.array(L).reshape(CFR_ARR.shape[0], CFR_ARR.shape[1], 1)
    
    print ("c_norm.shape: {0}".format(c_norm.shape))
    if debug:
        print ("c_norm[0]:")
        print (c_norm[0].T)
    return c_norm, CFR_ARR.reshape(CFR_ARR.shape[0], CFR_ARR.shape[1], 1), c_scaler


def show_spc(S_ALL_NORM, LAN_NAME_ARR, printout = True):
    L = ['Tmst','Year','Mont','New','IC2L','Nm2G','Nm2L','AD2G','AD2L','Gd2G','Gd2L','PwPn','ESD']
    ftr_dct = {}
    for i in range(len(L)):
        ftr_dct[i] = L[i]
    S = "";
    for v in L:
        S += "{:5s}|".format(v)
    print("|{0:13s}|{1}".format("", S)) if printout else False
    S = "";
    N = ['a_Ts','b_Yr','c_Mt','d_N','e_IL','f_NG','g_NL','h_AG','i_AL','j_GG','k_GL','l_Pn','m_ED']
    for v in N:
        S += "{:5s}|".format(v)
    print("|{0:13s}|{1}".format("LAN", S)) if printout else False
    S = "";
    for v in ["-"]*13:
        S += "{:5s}|".format(v)
    print("|{0:13s}|{1}".format("-", S)) if printout else False

    for i in range(len(S_ALL_NORM)):
        row = S_ALL_NORM[i]
        S = "";
        for j in range(len(row)):
            S += " {:0.2f}|".format(row[j])

        if i < len(S_ALL_NORM) - 2:
            print("|{0:13s}|{1}".format(LAN_NAME_ARR[i], S)) if printout else False
        else:
            print("|{0:13s}|{1}".format("", S)) if printout else False

    return ftr_dct


"""
x_v1: the X array with only cloumns in feature set
x_v2: the X array with cfr value + col. in feature set
x_v3: the X array with a distance + cfr value + col. in feature set
"""

def get_xy_from_sAc(s_all, c_all, ts, sel):    

    def expand_s(s_arr, timesteps):
        S = []
        for i in range(timesteps):
            S.append(s_arr)
        return np.array(S).reshape(timesteps, len(s_arr))

    i_scaler, _ = scale(np.array([0.,48.]).reshape(-1, 1)) # 4 years
    
    A = []
    [A.append(expand_s(s_arr, ts)) for s_arr in s_all[sel]]
    x_v1 = np.array(A).reshape(len(sel), ts, s_all.shape[1])

    A = []
    [A.append(c_arr) for c_arr in c_all[sel]]
    y = np.array(A).reshape(len(sel), ts, 1)

    # shift cfr, change the cfr of the first timestep by broadcasting
    A = np.roll(y[:,:,0].reshape(len(sel), ts, 1), 1, axis=1)
    A[:,0,:] = 0
    
    x_v2 = np.concatenate((A, x_v1[:,:,:]), axis=-1)
    
    # create distance array
    B = np.array(list(range(ts))).reshape(-1, 1)
    L = list(i_scaler.transform(B).reshape(-1))
    C = np.array(L * len(sel)).reshape(len(sel), ts, 1)
    
    x_v3 = np.concatenate((C, A, x_v1[:,:,:]), axis=-1)
    
    return x_v1, x_v2, x_v3, y


"""
plot function for cfr curves and training history
"""

import matplotlib.pyplot as plt


DEF_CLR = ['red','b','g','orange', 'saddlebrown', 'r', 'm', 'y', 'k', 'c', 'royalblue', 'coral',
           'mediumpurple', 'greenyellow','mediumseagreen', 'tomato', 'gold', 'fuchsia', 'tan', 'teal',
           'turquoise', 'violet', 'silver', 'sienna', 'maroon', 'plum', 'salmon', 'magenta', 'lime',
           'red','b','g','orange', 'saddlebrown', 'r', 'm', 'y', 'k', 'c', 'royalblue', 'coral',
           'mediumpurple', 'greenyellow','mediumseagreen', 'tomato', 'gold', 'fuchsia', 'tan', 'teal',
           'turquoise', 'violet', 'silver', 'sienna', 'maroon', 'plum', 'salmon', 'magenta', 'lime']

def plot_results(title, data, styles, colors, labels, label_flags,
                 timesteps, pred_timesteps, imgpath='./', save_flag = False):
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.title(title)    
    axes = plt.gca()
    #axes.set_xlim([-1,36])
    #axes.set_ylim([0,1.1]) # set y lim

    got_handle = False
    for i in range(len(data)):
        if label_flags[i]:
            got_handle = True
            plt.plot(data[i], styles[i], color=colors[i], label=labels[i])
        else:
            plt.plot(data[i], styles[i], color=colors[i])

    def annot_line(x,y,plt):
        plt.plot([x, x], [0, 1.25], 'k:', lw=1)

    if pred_timesteps > timesteps:
        annot_line(timesteps-1, 0, plt)

    if got_handle:
        plt.legend()
    plt.tight_layout()
    if save_flag:
        plt.savefig(imgpath+title+".png", dpi=None, transparent=False, pad_inches=0.0, format='png')
    plt.show()
    return


def plot_training_process(title, history, imgpath, save_flag = False):
    
    print("Print history ...")
    
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.title(title)
    plt.plot(history.history['loss'], color='orange')
    plt.plot(history.history['val_loss'], color='peru')
    plt.plot(history.history['kendall'], color='green')
    plt.legend(['loss', 'val_loss', 'kendall'])

    def annot_1(px, py, text, ax=None):
        if not ax:
            ax=plt.gca()
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
        kw = dict(xycoords='data',textcoords="axes fraction",
                  arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        plt.annotate(text, xy=(px, py), xytext=(0.75,0.5), **kw)

    px = np.argmin(history.history['val_loss'])
    py = np.min(history.history['val_loss'])
    po = history.history['kendall'][px]
    annot_1(px, py, text= "epo={:.0f}, mse={:.5f}, kt={:.2f}".format(px, py, po))
    
    def annot_2(px, py, text, ax=None):
        if not ax:
            ax=plt.gca()
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
        kw = dict(xycoords='data',textcoords="axes fraction",
                  arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        plt.annotate(text, xy=(px, py), xytext=(0.55,0.9), **kw)
    px = np.argmin(history.history['kendall'])
    py = np.min(history.history['kendall'])
    po = history.history['val_loss'][px]
    annot_2(px, py, text= "epo={:.0f}, kt={:.2f}, mse={:.5f}".format(px, py, po))
    
    plt.tight_layout()
    if save_flag:
        plt.savefig(imgpath+title+".png", dpi=None, transparent=False, pad_inches=0.0, format='png')
    plt.show()

    return

def mse(y_pred, y_true, axis):
        return np.mean(np.square(y_pred - y_true), axis=axis)

def kend_cust(dct1, dct2, tolerance=0.):
    """
    return if two differences are not equal
    """
    def neq(diff1, diff2, tolerance):
        if (diff1**2 <= tolerance**2) and\
           (diff2**2 <= tolerance**2):
            return False
        return (diff1 * diff2 <= 0)
    
    n = len(dct1) 
    if len(dct2) != n:
        return 1.
    x = 0.
    keys = set(dct1.keys())
    for i in range(n):
        k = keys.pop()
        for m in keys:
            if neq(dct1[k]-dct1[m], dct2[k]-dct2[m], tolerance = tolerance):
                x += 1.
    x = x / ((n * (n - 1)) / 2)
    return x

def dcg_at_k(r, k, method = 0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

"""
Below are model creations using keras APIs
"""

import keras
from keras import backend as K
from keras.models import load_model, Model, Sequential
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, GRU,\
                         Multiply, Flatten, Embedding, SimpleRNN, RepeatVector, Dense, Activation, Lambda, Reshape,\
                         Dropout
from keras import callbacks as Kcallback
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Activation
from keras import __version__ as keras__version__

def swish(x):
    return x * K.sigmoid(x)

def binary(x):
    return keras.backend.switch(x < 0, x*0, x/x)

def bround(x):
    return keras.backend.switch(x < 0, x*0, keras.backend.round(x))

K.linear = 'linear'


ACT_DCT = {}
LYR_LST = []
lstm_activation = K.relu; ACT_DCT['lstm_activation'] = 'relu'; LYR_LST.append('lstm_activation');
densor_1_act = swish; ACT_DCT['densor_1_act'] = 'swish'; LYR_LST.append('densor_1_act');
densor_2_act = swish; ACT_DCT['densor_2_act'] = 'swish'; LYR_LST.append('densor_2_act');
densor_3_act = swish; ACT_DCT['densor_3_act'] = 'swish'; LYR_LST.append('densor_3_act');
densor_4_act = swish; ACT_DCT['densor_4_act'] = 'swish'; LYR_LST.append('densor_4_act');
densor_5_act = K.relu; ACT_DCT['densor_5_act'] = 'relu'; LYR_LST.append('densor_5_act');


def lstm_train_model(timesteps, hidden_neurons, input_features, output_features, para_drop_rate):

    # LSTM cell, no need for input_shape
    lstm_cell = LSTM(hidden_neurons, return_sequences=True, activation=lstm_activation) 
    densor_layer_1 = Dense(output_features, activation=densor_1_act)
    densor_layer_2 = Dense(hidden_neurons,  activation=densor_2_act)
    densor_layer_3 = Dense(hidden_neurons,  activation=densor_3_act)
    densor_layer_4 = Dense(hidden_neurons,  activation=densor_4_act)
    densor_layer_5 = Dense(hidden_neurons,  activation=densor_5_act)

    X = Input(shape=(timesteps, input_features))
    a = lstm_cell(X)
    a = densor_layer_5(a)
    a = Dropout(para_drop_rate)(a)
    a = densor_layer_4(a)
    a = Dropout(para_drop_rate)(a)
    a = densor_layer_3(a)
    a = Dropout(para_drop_rate)(a)
    a = densor_layer_2(a)
    a = Dropout(para_drop_rate)(a)
    o = densor_layer_1(a)

    model = Model(inputs = [X], outputs = o, name='lstm_train_model')
    return model



# inference model
def lstm_inference_model(Lstm_cell, Densor_layer, Densor_layer_2, Densor_layer_3, Densor_layer_4, Densor_layer_5,
                         input_features, output_features, hidden_neurons, ts_prd):
    
    reshapor = Reshape((-1, 1))
    concat = Concatenate(axis=-1)
    
    x0 = Input(shape=(1, input_features))
    s = Input(shape=(1, input_features-1))
    a0 = Input(shape=(hidden_neurons,), name='a0')
    c0 = Input(shape=(hidden_neurons,), name='c0')
    a = a0
    c = c0
    x = x0

    outputs = []

    for t in range(ts_prd):
        a, _, c = Lstm_cell(x, initial_state=[a,c])
        ao = a; ao = Densor_layer_5(ao);ao = Densor_layer_4(ao); ao = Densor_layer_3(ao); ao = Densor_layer_2(ao);
        out = Densor_layer(ao)
        outputs.append(out)
        x = concat([reshapor(out), s])

    inference_model_instance = Model(inputs = [x0, s, a0, c0], outputs = outputs, name='lstm_inference_model')
    return inference_model_instance


# inference model (predict only once)
def lstm_model_inf_exp(Lstm_cell, Densor_layer, Densor_layer_2, Densor_layer_3, Densor_layer_4, Densor_layer_5,
                       input_features, output_features, hidden_neurons):
    x0 = Input(shape=(1, input_features))
    a0 = Input(shape=(hidden_neurons,), name='a0')
    c0 = Input(shape=(hidden_neurons,), name='c0')
    a = a0; c = c0; x = x0

    outputs = []

    a, _, c = Lstm_cell(x, initial_state=[a,c])
    ao = a
    ao = Densor_layer_5(ao); ao = Densor_layer_4(ao); ao = Densor_layer_3(ao); ao = Densor_layer_2(ao)
    out = Densor_layer(ao)
    outputs.append(out)
    outputs.append(a)
    outputs.append(c)

    inference_model_instance = Model(inputs = [x0, a0, c0], outputs = outputs, name='lstm_model_exp')
    return inference_model_instance

"""
Sub-processes
"""


def round_part_prd(model, X, ts_prd, hn):


    all_pred = []
    
    for i in range(X.shape[0]): # (samples, timesteps, features)
        x0 = X[i,0,:]
        spec = X[i,0,1:]
        x_initializer = x0.reshape([1, 1, X.shape[2]])
        a_initializer = np.zeros((1, hidden_neurons))
        c_initializer = np.zeros((1, hidden_neurons))
        spec = spec.reshape([1, 1, X.shape[2] - 1])
        pred = model.predict([x_initializer, spec, a_initializer, c_initializer]) 
        all_pred.append(pred)
    
    all_pred = np.array(all_pred).reshape(len(all_pred), ts_prd, -1)
    
    return all_pred


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from IPython.display import Image

def round_part_inf(path, i_dim, o_dim, hn, ts_prd):
    print ("Create inference model ... ")
    
    model_tra = load_model(path, custom_objects={'swish': swish, 'binary': binary, 'bround': bround})
    i = 1
    lcell_inf  = model_tra.layers[i]; i=i+1
    dlyr_inf_5 = model_tra.layers[i]; i=i+2
    dlyr_inf_4 = model_tra.layers[i]; i=i+2
    dlyr_inf_3 = model_tra.layers[i]; i=i+2
    dlyr_inf_2 = model_tra.layers[i]; i=i+2
    dlyr_inf   = model_tra.layers[i]; i=i+2

    lcell_inf.return_state = True
    lcell_inf.return_sequences = False

    # tells inf model to predict pred_timesteps
    #model_inf = lstm_inference_model(lcell_inf, dlyr_inf, dlyr_inf_2, dlyr_inf_3, dlyr_inf_4, dlyr_inf_5,
    #                                 i_dim, o_dim, hn, ts_prd)
    model_inf = lstm_model_inf_exp(lcell_inf, dlyr_inf, dlyr_inf_2, dlyr_inf_3, dlyr_inf_4, dlyr_inf_5,
                                         i_dim, o_dim, hn)

    #SVG(model_to_dot(inference_model, show_shapes=True).create(prog='dot', format='svg'))    
    #Image(model_to_dot(model_inf, show_shapes=True).create(prog='dot', format='png'))
    print ("Done ({0})".format(path))
    return model_inf


"""
note that F contains distance and the previous cfr value
"""

def round_part_prd_exp(model, F, ts_prd, hn, version = 1, DEBUG=False):
    
    IDX = 3 - version
    
    print("\nPredict (VERSION {0}, num_ftr: {1}) ...".format(version, F[0,0,IDX:].shape[0]))

    all_pred = []
    
    for i in range(F.shape[0]): # (samples, timesteps, features)
        x0 = F[i,0,IDX:]
        
        if DEBUG:
            print("X[{0}]".format(i))
        
        x_j = x0.reshape([1, 1, x0.shape[0]])
        a_j = np.zeros((1, hn))
        c_j = np.zeros((1, hn))
        
        L = []
        for j in range(1, ts_prd + 1):
            pred = model.predict([x_j, a_j, c_j])
            
            A = F[i,0,2:].reshape(1,1,F.shape[2]-2)
            if version == 2:
                A = np.concatenate((pred[0].reshape(1,1,1), A), axis = -1)
            elif version == 3:
                distance = i_scaler.transform(float(j))
                A = np.concatenate((distance.reshape(1,1,1), pred[0].reshape(1,1,1), A), axis = -1)
            x_j = A
            

            """
            DEBUG
            """
            if DEBUG:
                if j < 4: print(["{:0.2f}".format(v) for v in x_j.reshape(-1)])
                elif j == 5:
                    print("...")
                elif j > ts_prd - 3 and j < ts_prd:
                    print(["{:0.2f}".format(v) for v in x_j.reshape(-1)])
                elif j == ts_prd:
                    print("{0}\n".format(["{:0.2f}".format(v) for v in x_j.reshape(-1)]))
            
            a_j = pred[1]
            c_j = pred[2]
            
            L.append(float(pred[0]))
        all_pred.append(L)
    
    
    all_pred = np.array(all_pred).reshape(len(all_pred), ts_prd, -1)
    print("all_pred.shape: {0}\n".format(all_pred.shape))
    
    return all_pred



def round_part_mse(all_pred, X, Y, ts_tra):
    
    def mse(y_pred, y_true, axis):
        return np.mean(np.square(y_pred - y_true), axis=axis)
    
    all_pred_c = all_pred[:,:ts_tra,:]
    
    """
    use training timesteps to compute all mse
    """
    MSE_M = mse(all_pred_c, Y[:,:ts_tra,:], axis=-1)
    mse_all = np.mean(MSE_M); mse_all_str = "{:0.3f}".format(mse_all); mas = "{:0.8f}".format(mse_all)

    
    """
    cut remaining timesteps
    """
    MSE_M_R = mse(all_pred[:, ts_tra:, :], Y[:, ts_tra:, :], axis=-1)
    
    
    mse_L = [] #mse_L = np.mean(MSE_M_R, axis=1);
    for i in range(MSE_M_R.shape[0]):
        if i == MSE_M_R.shape[0] - 1:
            mse_L.append(np.mean(MSE_M_R[i, :2]))
        else:
            mse_L.append(np.mean(MSE_M_R[i]))
    mse_L_str = ["{:0.5f}".format(e) for e in mse_L]
    mse_cut = np.mean(mse_L); mse_cut_str = "{:0.5f}".format(mse_cut); mcs = "{:0.8f}".format(mse_cut)
    #print ("mse all: {0}, mse cut: {1}".format(mas, mcs))

    return mse_all, mse_cut