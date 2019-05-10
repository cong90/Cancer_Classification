import numpy as np
import scipy.io as sio
c = 1e3  # the constant control non-spasity
import math


def substitute_gamma_a(gamma):
    N = len(gamma)
    a = np.ones([N, 1]) * c

    a[gamma == False] = 1

    return a

if __name__ == "__main__":

    BRCA0 = sio.loadmat('BRCA.mat')
    x0 = sio.loadmat('x1.mat')

    X = x0['M'].T
    y = BRCA0['BRCA'].T

    # N: number of variables, T: number of observations
    [T,N] = X.shape

    R = X.T@X+np.eye(N)
    R_inv = np.linalg.inv(R)
    # Hyperparameters
    tao =0.01
    K = 500
    sigma_2 = 1 # the noise variance

    gamma = (np.linalg.pinv(X)@y<0.5)
    z = y.reshape([-1,1])
    a = substitute_gamma_a(gamma)
    D = np.diag(a.reshape([-1])*tao)
    scale = int(1/3)
    beta = np.random.multivariate_normal(np.zeros(N),scale*(D@R@D))

    store_gamma = np.zeros([N,K])
    store_beta = np.zeros([N,K])

    A_inv = np.linalg.pinv((X.T@X)*sigma_2)

    for k in range(K):

        print(k)

    # update of gamma
        for ii in range(N):

            gamma_a = gamma
            gamma_a[ii] = 1
            gamma_a = substitute_gamma_a(gamma_a)
            gamma_a = gamma_a.reshape([-1])
            Sigma_a_inv = np.diag(1/(gamma_a * tao))@R_inv @ np.diag(1/(gamma_a * tao))
            aa = np.exp(-beta @ Sigma_a_inv @ beta.T/2)*np.linalg.det(Sigma_a_inv)

            gamma_b = gamma
            gamma_b[ii] = 0
            gamma_b = substitute_gamma_a(gamma_b)
            gamma_b = gamma_b.reshape([-1])
            Sigma_b_inv = np.diag(1/(gamma_b * tao))@R_inv @ np.diag(gamma_b * tao)
            bb = np.linalg.det(Sigma_b_inv) * np.exp(-beta @ Sigma_b_inv @ beta.T/2)

            if aa is 0:
                p = 0
            else:
                p = aa / (aa + bb)
            if math.isnan(p):
                p = 1

            gamma[ii] = np.random.binomial(1, p)

            # update of beta

            a = substitute_gamma_a(gamma)
            D_inv = np.diag(1/(a.reshape([-1])*tao))
            A = A_inv - A_inv*D_inv@np.linalg.inv(R+D_inv@A_inv@D_inv)@D_inv@A_inv
            mu = A@X.T@z
            beta = np.random.multivariate_normal(mu.reshape([-1]),A)

            store_gamma[:, k] = gamma.reshape([-1])
            store_beta[:, k] = beta.reshape([-1])
            # update of Z
            for ii in range(T):

                if y[ii] == 1:
                    try:
                        while True:
                            numb = np.random.multivariate_normal((X[ii, :] @ beta.T).reshape([-1]), sigma_2)
                            if numb > 0:
                                z[ii] = numb
                                break
                    except:
                        z[ii] = 0


                elif y[ii] is 0:
                    try:
                        while True:
                            numb = np.random.multivariate_normal((X[ii,:]@beta.T).rehape([-1]), sigma_2)
                            if numb<0:
                                z[ii] = numb
                                break
                    except:
                        z[ii] = 0


