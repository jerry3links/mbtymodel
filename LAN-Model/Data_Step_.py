import numpy as np
import pandas as pd
import utils

from sklearn import preprocessing
from sklearn.externals import joblib

import _CFR_Constant_Data as topicConst
import _Library._Utils_FileHandler as fh
from _Library.CFR._Utils_DataFlow import get_monthly_data, expand_input, \
    norm_transform, numpy_to_dataframe
from _Library._ProjectConstant._Constant_Data import Data

raw_data_folder = topicConst.RAW_DATA_FILE_PREFIX

excluded_sets = []
excluded_sets.append([])

exclude0 = ['82567LF', 'AR8151']
excluded_sets.append(exclude0)

"""
Read spec. & cfr value
"""

LAN_CSV_DCT, spc_csv = utils.read_csv(raw_data_folder)
LAN_NAME_ARR = np.array(spc_csv['LAN'])[:-2] # exclude MIN & MAX
LAN_NAME_LST = list(LAN_NAME_ARR)

"""
Spec Scaling
"""

raw_spc = np.array(spc_csv.iloc[:,1:]) # exclude name
s_all_orig = utils.create_S(raw_spc)
s_scaled, s_all_norm = utils.scale(s_all_orig)
print("s_all_norm.shape: {0}".format(s_all_norm.shape))
joblib.dump(s_scaled, topicConst.NORMALIZE_SCALER_OBJ_DUMP_PATH_SPEC)
print("Spec. Table:")
_ = utils.show_spc(s_all_norm, LAN_NAME_ARR)
print("\n")


"""
CFR Scaling
"""

group_a = [v for v in LAN_NAME_LST if v not in excluded_sets[1]]
group_a_ids = [LAN_NAME_LST.index(v) for v in LAN_NAME_LST if v not in excluded_sets[1]]
c_norm, c_orig, c_scaled = utils.create_C(LAN_CSV_DCT, LAN_NAME_ARR, group_a_ids)
joblib.dump(c_scaled, topicConst.NORMALIZE_SCALER_OBJ_DUMP_PATH_CFR)


"""
X-Y Output
"""
def outputA(outdest, array):
    with open(outdest, 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(array.shape))
        cnt = 0
        for data_slice in array:
            outfile.write('# {0}\n'.format(LAN_NAME_ARR[group_a_ids[cnt]]))
            np.savetxt(outfile, data_slice, delimiter=",", fmt='%s')
            cnt += 1
    print("Save array of {0} to {1}".format(array.shape, outdest))
    return

VERSION = 2
ftr_set = [0] + list(range(4, s_all_norm.shape[1]))

# Nomalized XY
s_norm = s_all_norm[group_a_ids][:, ftr_set]
_, _, F_norm, Y_norm = utils.get_xy_from_sAc(s_norm, c_norm, 36, list(range(len(c_norm))))
X_norm = F_norm[:,:,(3-VERSION):]
print("X_norm.shape: {0}, Y_norm.shape: {1}".format(X_norm.shape, Y_norm.shape))
outputA(topicConst.X_DATA_PATH, X_norm)
outputA(topicConst.Y_DATA_PATH, Y_norm)

# Original XY
s_orig = s_all_orig[group_a_ids][:, ftr_set]
_, _, F_orig, Y_orig = utils.get_xy_from_sAc(s_orig, c_orig, 36, list(range(len(c_orig))))
X_orig = F_orig[:,:,(3-VERSION):]
print("X_orig.shape: {0}, Y_orig.shape: {1}".format(X_orig.shape, Y_orig.shape))
outputA(topicConst.X_DATA_NORM_PATH, X_orig)
outputA(topicConst.Y_DATA_NORM_PATH, Y_orig)
