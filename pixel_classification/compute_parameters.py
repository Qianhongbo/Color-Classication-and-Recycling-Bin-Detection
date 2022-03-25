import numpy as np
import pickle
from generate_rgb_data import read_pixels

def computeParameters(red, green, blue):
    '''
    total: total number of samples
    theta: prior probability = number of examples that are blue / total number of examples
    mu: mean value
    sig2: covariance
    '''
    total = len(red) + len(green) + len(blue)
    theta_1, theta_2, theta_3  = len(red) / total, len(green) / total, len(blue) / total
    mu_1, mu_2, mu_3  = np.mean(red, axis=0), np.mean(green, axis=0), np.mean(blue, axis=0)
    sig2_1, sig2_2, sig2_3 = np.cov(red.T), np.cov(green.T), np.cov(blue.T)
    params = [theta_1, theta_2, theta_3, mu_1, mu_2, mu_3, sig2_1, sig2_2, sig2_3]
    with open('parameters.pkl', 'wb') as f:
        pickle.dump(params, f)
    return params

if __name__ == '__main__':
    with open('parameters.pkl', 'rb') as f:
      params = pickle.load(f)
    print(params[3])
    print(params[4])
    print(params[5])
    print(params[6])
    print(params[7])
    print(params[8])
#     folder = 'data/training'
#     X1 = read_pixels(folder + '/red', verbose=True)
#     X2 = read_pixels(folder + '/green')
#     X3 = read_pixels(folder + '/blue')
#
#     temp2 = computeParameters(X1, X2, X3)
#     print(temp2)