import time
import numpy as np
import cdflib

cdf = cdflib.CDF("image/preprocessing/im_hks_ast_20010913000043_20010913235814_cdaweb.cdf")

cdf_vars = ['Epoch', 'BS_QUATQ1', 'BS_QUATQ2', 'BS_QUATQ3', 'BS_QUATQ4', 'BS_ATTUNCX', 'BS_ATTUNCY', 'BS_ATTUNCZ', 'BS_RMSERR', 'BS_INTTIME']
data = {}
for var in cdf_vars:
    data[var] = cdf.varget(var)

data["Epoch"] = np.vectorize(lambda e: e / 1000 - 62167219200 - 3600)(data["Epoch"])

quats = np.zeros((data['BS_QUATQ1'].shape[0], 4))
for i in range(data["BS_QUATQ1"].shape[0]):
    quats[i] = np.array([
        data["BS_QUATQ1"][i], data["BS_QUATQ2"][i], data["BS_QUATQ3"][i], data["BS_QUATQ4"][i],
    ])

quats /= np.linalg.norm(quats)

ast2gci = np.zeros((data['BS_QUATQ1'].shape[0], 3, 3))
for i, q in enumerate(quats):
    ast2gci[i] = np.array([
        [2 * (q[0] ** 2 + q[3] ** 2) - 1, 2 * (q[0] * q[1] + q[2] * q[3]), 2 * (q[0] * q[2] - q[1] * q[3])],
        [2 * (q[0] * q[1] - q[2] * q[3]), 2 * (q[1] ** 2 + q[3] ** 2) - 1, 2 * (q[1] * q[2] + q[0] * q[3])],
        [2 * (q[0] * q[2] + q[1] * q[3]), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[2] ** 2 + q[3] ** 2) - 1]
    ])

ast2sc = np.array([
    [ 0.382683, -0.923880,  0],
    [-0.909844, -0.376870,  0.173648],
    [-0.160430, -0.066452, -0.984808]
])

wic_sc = np.array([
    1/np.sqrt(2), -1/np.sqrt(2), 0
])

wic_gci = np.zeros((ast2gci.shape[0], 3))
for i, val in enumerate(ast2gci):
    wic_gci[i] = np.matmul(val, np.matmul(ast2sc, wic_sc))
    #print(wic_gci[i], time.ctime(data["Epoch"][i]))

cdf = cdflib.CDF("im_hks_ads_20010913000043_20010913235814_cdaweb.cdf")

cdf_vars = ['Epoch', 'BA_FiltSpAttX', 'BA_FiltSpAttY', 'BA_FiltSpAttZ']
data = {}
for var in cdf_vars:
    data[var] = cdf.varget(var)

epochs = np.vectorize(lambda e: e / 1000 - 62167219200 - 3600)(data["Epoch"])
sc_pos = np.zeros((data['BA_FiltSpAttX'].shape[0], 3))
for i in range(sc_pos.shape[0]):
    sc_pos[i] = np.array([data['BA_FiltSpAttX'][i], data['BA_FiltSpAttY'][i], data['BA_FiltSpAttZ'][i]])
    print(sc_pos[i], time.ctime(epochs[i]))
