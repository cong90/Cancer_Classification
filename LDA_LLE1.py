import numpy as np

def PCA(X, num_comp):

    X_norm, S = data_normalization(X)

    U,D,V  = np.linalg.svd(S)

    DR = U[:,:num_comp].T

    X_dr = DR @ X # X_dr = DR @ X

    # Reconstruction error
    R_X = DR.T @ X_dr

    Re_error = np.linalg.norm(X-R_X)/np.linalg.norm(X)

    return X_dr, Re_error


def LDA(X, x_labels, num_c = 3, normal_= 1, reduced_dim = 3):
    #Linear Discriminant Analysis
    # normal determines whether we need to normalize the data variance when calculate S_W
    indx_c = []
    for nc in range(num_c):

        indx_c.append(np.where(x_labels == nc))

    X_c = []
    for nc in range(num_c):

        X_c.append(np.squeeze(X[:, indx_c[nc]]))

    # scatter matrix
    # mean vector for each class
    m_c = []
    X_c_n = []
    S_W = 0

    N_all = X.shape[1]

    for nc in range(num_c):

        m_c.append(np.mean(X_c[nc], axis=1, keepdims=True))

        if normal_ == 0:

            N = X_c[nc].shape[1]

            X_m = np.mean(X_c[nc], axis=1, keepdims=True)

            X_temp = X_c[nc] - X_m

            S_temp = np.dot(X_temp,X_temp.T) / (N - 1)

            X_temp = X_c[nc]

        else:

            X_temp, S_temp= data_normalization(X_c[nc])

        X_c_n.append(X_temp)

        S_W = S_W + S_temp

    S_W = S_W

    m = np.mean(X, axis=1, keepdims=True)
    S_B = 0
    for nc in range(num_c):

        S_B = S_B + len(indx_c[nc])*np.dot((m_c[nc]-m),(m_c[nc]-m).T)

    #eig_vals, eig_vecs = np.linalg.eig(np.dot(np.linalg.inv(S_W),S_B))

    U, S, Vh = np.linalg.svd(np.dot(np.linalg.inv(S_W),S_B))
    print(S)
    #DR = eig_vecs[:,:reduced_dim].T

    DR = U[:,:reduced_dim].T

    X_norm = np.hstack([X_c_n[0],X_c_n[1],X_c_n[2]])

    X_dr = np.dot(DR, X)

    return X_dr


def LLE(X, K = 10, d = 2 ):
    #local linear embedding
    # X : D x N, where D is the dimension and N is the number of samples
    # K : K smallest
    # neighbor

    #X, S = data_normalization(X)

    tol = 1e-3
    N = X.shape[1]

    X_norm = np.linalg.norm(X, axis = 0, keepdims= True, ord =2)**2

    Dist = X_norm.T + X_norm - 2* X.T @ X

    Neighbor_indx = Dist.argsort(axis = 0)[1:(K+1)]

    # weights
    Neighbors_norm = []

    W = np.zeros([K, N])

    for n in range(N):

        Neighbors_norm.append(X[:, Neighbor_indx[:,n]] - X[:,n].reshape([-1,1]))

        C = Neighbors_norm[n].T@Neighbors_norm[n]

        C = C + np.eye(K)*tol*np.trace(C)

        w = np.linalg.pinv(C) @ np.ones([K,1])

        W[:, n] =  (w/np.sum(w, axis = 0)[0]).reshape([-1])

    # representative
    M = np.zeros([N,N])

    for n in range(N):

        w = W[:, n]

        k = Neighbor_indx[:,n]

        M[n, k] = w

    A = (np.eye(N) - M).T @ (np.eye(N) - M)

    eigvalues, Y = np.linalg.eig(A)

    indx = eigvalues.argsort()[0:d]

    Y = Y[:,indx].T*np.sqrt(N)

    return Y


def data_normalization(X):

    N = X.shape[1]

    X_m = np.mean(X, axis = 1, keepdims= True)

    X_norm = X-X_m

    sigma = np.std(np.array(X_norm), axis=1, ddof=1, keepdims= True)

    X_norm = X_norm/sigma

    S = (np.dot(X_norm,X_norm.T))/(N-1) # Bessel's correction

    return X_norm, S