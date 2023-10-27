import numpy as np

def get_matrix(alpha, beta, gamma, num_points):
    """Return the matrix for the internal energy minimization.
    # Arguments
        alpha: The alpha parameter.
        beta: The beta parameter.
        gamma: The gamma parameter.
        num_points: The number of points in the curve.
    # Returns
        The matrix for the internal energy minimization. (i.e. A + gamma * I)
    """
    A = np.identity(num_points)
    A = A * (-2*alpha + 6*beta)
    for i in range(num_points-1): 
        A[i, i+1] = (alpha - 4*beta) 
        A[i+1, i] = (alpha - 4*beta)
    A[num_points-1, 0] = (alpha - 4*beta)
    A[0, num_points-1] = (alpha - 4*beta)

    for i in range(num_points-2):
        A[i, i+2] = beta
        A[i+2, i] = beta
    A[num_points-2, 0] = beta
    A[num_points-1, 1] = beta
    A[0, num_points-2] = beta
    A[1, num_points-1] = beta
    # print("A: ", A)
    return np.linalg.inv(A + gamma * np.identity(num_points))
             

    
