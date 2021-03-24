import torch

def get_covariance_matrix(X):
    '''
    Returns the covariance of the data X
    X should contain a single data point per row of the tensor
    '''
    X_mean = torch.mean(X, dim=0)
    X_mean_matrix = torch.outer(X_mean, X_mean)
    X_corr_matrix = torch.matmul(torch.transpose(X, 0, 1), X)/X.size(0)
    Cov = X_corr_matrix - X_mean_matrix
    return Cov

def get_e_v(Cov):
    '''
    Returns eigenvalues and eigenvectors in descending order by eigenvalue size
    '''
    e, v = torch.symeig(Cov, eigenvectors=True)
    v = torch.transpose(v, 0, 1)
    e_abs = torch.abs(e)

    inds = torch.argsort(e_abs, descending=True)
    e = e_abs[inds]
    v = v[inds]

    return e,v
