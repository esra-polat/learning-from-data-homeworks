#150116884
#Esra Polat

import numpy as np
import matplotlib.pyplot as plt

run_step = 1000
N_sample = 1000
N_test = 1000

def generate_random(n):
    return np.random.uniform(-1, 1, size = n)

def E_in():
    E_in_total = 0
    for run in range(run_step):
        
        # Choose the inputs x_n of the data set as random points
        X = np.transpose(np.array([np.ones(N_sample), generate_random(N_sample), generate_random(N_sample)]))
        # f(x1,x2) = sign(x1^2 + x2^2 - 0.6)
        y = np.sign(X[:,1] * X[:,1] + X[:,2] * X[:,2] - 0.6)

        # pick 10% = 100 random indices
        indices = list(range(N_sample))
        np.random.shuffle(indices)
        random_indices = indices[:(N_sample // 10)]

        # flip sign in y vector
        for i in random_indices:
            y[i] = (-1) * y[i]

        # w = ((t(X))^-1)t(X)y
        X_dagger = np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T)
        w_lr = np.dot(X_dagger, y)

        # classification according to w found by linear regression
        y_lr = np.sign(np.dot(X, w_lr))

        # Error E_in
        E_in = sum(y_lr != y) / N_sample
        E_in_total += E_in
        
    print("\nThe average error E_in:\t", '{:1.2f}'.format(E_in_total / run_step))

def g_x1_x2():
    X = np.transpose(np.array([np.ones(N_sample), generate_random(N_sample), generate_random(N_sample)]))
    y = np.sign(X[:,1] * X[:,1] + X[:,2] * X[:,2] - 0.6)

    # new feature matrix
    X_trans = np.transpose(np.array([np.ones(N_sample), X[:,1], X[:,2], X[:,1]*X[:,2], X[:,1]*X[:,1], X[:,2]*X[:,2]]))

    # linear regression on the new "feature matrix"
    X_dagger_trans = np.dot(np.linalg.inv(np.dot(X_trans.T, X_trans)), X_trans.T)
    w_lr_trans = np.dot(X_dagger_trans, y)

    print("\nThe weight vector of hypothesis:", w_lr_trans)

def E_out():
    E_out_total = 0

    X = np.transpose(np.array([np.ones(N_sample), generate_random(N_sample), generate_random(N_sample)]))
    y = np.sign(X[:,1] * X[:,1] + X[:,2] * X[:,2] - 0.6)
    # new feature matrix
    X_trans = np.transpose(np.array([np.ones(N_sample), X[:,1], X[:,2], X[:,1]*X[:,2], X[:,1]*X[:,1], X[:,2]*X[:,2]]))
    # linear regression on the new "feature matrix"
    X_dagger_trans = np.dot(np.linalg.inv(np.dot(X_trans.T, X_trans)), X_trans.T)
    w_lr_trans = np.dot(X_dagger_trans, y)

    for run in range(run_step):
        
        # create 1000 random points
        # matrix consisting of feature vectors
        X_test = np.transpose(np.array([np.ones(N_test), generate_random(N_test), generate_random(N_test)]))
        y_test = np.sign(X[:,1] * X_test[:,1] + X_test[:,2] * X_test[:,2] - 0.6)

        # pick 10% = 100 random indices
        indices = list(range(N_test))
        np.random.shuffle(indices)
        random_indices = indices[:(N_test // 10)]

        # flip sign in y vector
        for i in random_indices:
            y_test[i] = (-1) * y_test[i]

        # Compute classification made by my hypothesis from Problem 9
        # first create transformed feature matrix
        X = X_test
        X_trans_test = np.transpose(np.array([np.ones(N_test), X[:,1], X[:,2], X[:,1]*X[:,2], X[:,1]*X[:,1], X[:,2]*X[:,2]]))
        y_lr_trans_test = np.sign(np.dot(X_trans_test, w_lr_trans))
        
        # Compute disagreement between hypothesis and target function
        E_out = sum(y_lr_trans_test != y_test) / N_test
        E_out_total += E_out
        
    print("\nThe average error E_out:", '{:1.2f}'.format(E_out_total / run_step))

E_in()
g_x1_x2()
E_out()
