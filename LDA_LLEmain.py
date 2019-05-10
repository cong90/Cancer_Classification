import numpy as np
import pandas as pd
from numpy import genfromtxt
from NLDR import PCA, LDA, LLE
import matplotlib.pyplot as plt
import pickle



if __name__ == '__main__':


    data_mtx = pd.read_csv('Igsub.csv')
    data_label = data_mtx.values[:, -1]
    data_mtx = data_mtx.values[:,1:-1]
    data_mtx = np.array(data_mtx, dtype = 'float')

    data_label[np.where(data_label == 'PRAD')] = 0
    data_label[np.where(data_label == 'LUAD')] = 1
    data_label[np.where(data_label == 'BRCA')] = 2
    data_label[np.where(data_label == 'KIRC')] = 3
    data_label[np.where(data_label == 'COAD')] = 4


    #  Apply LDA on data
    data_mtx_lda = LDA(data_mtx.T, data_label, num_c = 5, normal_ = 0, reduced_dim = 750)
    colors = (1, 0.5, 1)
    plt.scatter(data_mtx_lda[0,:1000], data_mtx_lda[1,:1000], s = np.pi, c='black', alpha=0.4)
    colors = (1, 1, 0)
    plt.scatter(data_mtx_lda[0,1000:2000], data_mtx_lda[1,1000:2000], s = np.pi, c='red', alpha=0.4)
    colors = (0.3, 0.5, 1)
    plt.scatter(data_mtx_lda[0,2000:], data_mtx_lda[1,2000:], s = np.pi, c='yellow', alpha=0.4)
    plt.show()

    f = open('./data_mtx_lda.pkl', 'wb')
    pickle.dump(data_mtx_lda, f)

    data_lle = LLE(data_mtx.T, K = 750, d = 750)
    f = open('./data_lle_750.pkl', 'wb')
    pickle.dump(data_lle, f)

    f = open('./data_lle_750.pkl', 'rb')
    data_mtx_lle = pickle.load(f)

    print(data_mtx_lle.shape)
    np.savetxt("data_lle_750.csv", data_mtx_lle, delimiter=",")
    np.savetxt("data_lle_750.csv", data_mtx_lda, delimiter=",")








