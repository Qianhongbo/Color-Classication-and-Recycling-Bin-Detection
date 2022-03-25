import numpy as np
import pickle

def computeParameters(blue, not_blue):
    '''
    total: total number of samples
    theta: prior probability = number of examples that are blue / total number of examples
    mu: mean value
    sig2: covariance
    '''
    total = len(blue) + len(not_blue)
    theta_1, theta_2  = len(blue) / total, len(not_blue) / total
    mu_1, mu_2  = np.mean(blue, axis=0), np.mean(not_blue, axis=0)
    sig2_1, sig2_2 = np.cov(blue.T), np.cov(not_blue.T)
    params = [theta_1, theta_2, mu_1, mu_2, sig2_1, sig2_2]

    return params

if __name__ == '__main__':
    with open('rgb_color_data.pkl', 'rb') as f:
        X = pickle.load(f)
        X_blue, X_not_blue = X[0], X[1]
        print(X_blue.shape)
        print(X_not_blue.shape)

    params = computeParameters(X_blue, X_not_blue)
    with open('parameters.pkl', 'wb') as f:
        pickle.dump(params, f)