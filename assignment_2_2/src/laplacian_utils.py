import igl
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
import numpy.linalg as la

def compute_mass_matrix(V, F):
    ''' Assemble the Mass matrix by computing quantities per face.

    Parameters:
    - V : np.array (#v, 3)
    - F : np.array (#f, 3)

    Returns:
    - M : scipy sparse diagonal matrix (#v, #v)
        Mass matrix
    '''

    M = sparse.diags(..., shape=..., format='csr')
    return M

def compute_cotangent(a, b, c, A):
    ''' Compute the cotangent of an angle in a triangle by using the triangle edge lengths and area only. 
    The input parameters are defined in the handout figure. 
    The purpose of this function is to check that your formula is correct. 
    You might not be able to directly use this in the `compute_cotangent_weights_matrix` function.

    Parameters:
    - a : float
    - b : float
    - c : float
    - A : float

    Returns:
    - cotangent : float
    '''

    return 0

def compute_cotangent_weights_matrix(V, F):
    ''' Assemble the cotangent weights matrix S by computing quantities per face.

    Parameters:
    - V : np.array (#v, 3)
    - F : np.array (#f, 3)

    Returns:
    - S : scipy sparse matrix (#v, #v)
        Cotangent weights matrix
    '''
    
    S = sparse.coo_matrix(...)
    D = sparse.diags(...)

    S -= D
    return S
