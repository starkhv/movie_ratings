import numpy as np

def read_mat(file_name='ex8_movies.mat'):
    """
    Reads the movie ratings MATLAB file and returns the data
    """
    import scipy.io
    temp = scipy.io.loadmat(file_name)
    return (temp['Y'], temp['R'])

def cost_grad(X, Theta, Y, R, lambd):
    """
    Calculates the cost of the current parameters
    """

    squared_error = np.sum((np.multiply(np.dot(X, np.transpose(Theta))-Y, R))**2, axis=(0,1))/2;
    theta_reg = lambd*np.sum(Theta**2, axis=(0,1))/2
    x_reg = lambd*np.sum(X**2, axis=(0,1))/2
    COST = squared_error + theta_reg + x_reg

    X_grad = np.zeros(X.shape)

    Theta_grad = np.zeros(Theta.shape)

    return COST

def rand_init(Y, num_features):
    """
    Randomly initialize X and Theta to small values and return them
    """

    X = np.random.rand(Y.shape[0], num_features)
    Theta = np.random.rand(Y.shape[1], num_features)
    return (X, Theta)
