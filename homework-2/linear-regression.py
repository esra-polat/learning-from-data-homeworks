from matplotlib.pyplot import show
import numpy as np
import matplotlib.pyplot as plt
import click

def generate_random(n): 
    return np.random.uniform(-1, 1, size = n)

def linear_regression(data_size):

    run_step = 1000
    N_sample = data_size
    N_test = 1000
    E_in_total = 0
    E_out_total = 0
    total_iteration = 0
    d = 2 

    for run in range(run_step):

        # in each run, choose a random line in the plane 
        # do this taking two random points theta1, theta2 in [-1,1] x [-1,1]
        # take d = 2
        theta1 = generate_random(2)
        theta2 = generate_random(2)

        # a line formula is y = a*x + b
        # a is the slope
        a = (theta2[1] - theta1[1]) / (theta2[0] - theta1[0])
        b = theta2[1] - a * theta2[0]  
        w = np.array([b, a, -1])

        # Choose the inputs x_n of the data set as random points
        X = np.transpose(np.array([np.ones(N_sample), generate_random(N_sample), generate_random(N_sample)]))
        # Evaluate the target function on each x_n to get the corresponding output y_n
        y = np.sign(np.dot(X, w))
        
        # w = ((t(X))^-1)t(X)y
        X_dagger = np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T)
        w_lr = np.dot(X_dagger, y)
        
        # classification according to w found by linear regression
        y_lr = np.sign(np.dot(X, w_lr))
            
        # Error E_in
        E_in = sum(y_lr != y) / N_sample
        E_in_total += E_in

        # Problem 6: Take 1000 test points (out of sample points) and count disagreement
        # between y_target_test and y_hypo_test
        X_test = np.transpose(np.array([np.ones(N_test), generate_random(N_test), generate_random(N_test)]))
        y_target_test = np.sign(np.dot(X_test, w))
        y_hypo_test = np.sign(np.dot(X_test, w_lr))
        
        E_out = sum(y_hypo_test != y_target_test) / N_test
        E_out_total += E_out

    print("\nAverage of E_in:\t", '{:1.2f}'.format(E_in_total / run_step))
    print("Average of E_out:\t", '{:1.2f}'.format(E_out_total / run_step))

def pla(data_size):
    
    N_sample = 10
    run_step = 1000
    total_iteration = 0
    d = 2 

    for run in range(run_step):
        
        # in each run, choose a random line in the plane 
        # do this taking two random points theta1, theta2 in [-1,1] x [-1,1]
        # take d = 2
        theta1 = generate_random(d)
        theta2 = generate_random(d)

        # a line formula is y = a*x + b
        # a is the slope
        a = (theta2[1] - theta1[1]) / (theta2[0] - theta1[0]) 
        b = theta2[1] - a * theta2[0]  
        y = np.array([b, a, -1])

        # Choose the inputs x_n of the data set as random points
        X = np.transpose(np.array([np.ones(N_sample), generate_random(N_sample), generate_random(N_sample)]))
        # Evaluate the target function on each x_n to get the corresponding output y_n
        y_target = np.sign(np.dot(X, y))
        
        # y = ((t(X))^-1)t(X)y_target
        X_dagger = np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T)
        w_lr = np.dot(X_dagger, y_target)

        # initialize weight vector being all zeros
        weight = np.copy(w_lr)    
        # count number of iterations in PLA    
        count_iteration = 0   

        # Start PLA
        while True:
            
            # classification by hypothesis
            y_hypo = np.sign(np.dot(X, weight))    

            # The number of iterations that PLA takes to converge to g, and the disagreement between f and g which is P[f(x)!=g(x)]
            # the probability that f and g will disagree on their classification of a random point 

            # compare classification with actual data from target function  
            comp = (y_hypo != y_target)                 
            # indices of points with wrong classification by hypothesis h     
            wrong = np.where(comp)[0]                 

            if wrong.size == 0:
                break
            
            # pick a random misclassified point
            random_choice = np.random.choice(wrong)      

            # update weight vector (new hypothesis):
            weight = weight +  y_target[random_choice] * np.transpose(X[random_choice])
            count_iteration += 1

        total_iteration += count_iteration
        
    print("\nAverage of PLA: \t", '{:1.2f}'.format(total_iteration / run_step), "\n")

#----------------------------------------------------

print("\nCalculating for N:", 100, "\n")
# loading bar
with click.progressbar(range(1000000)) as bar:
    for i in bar:
        pass 
linear_regression(100)

#----------------------------------------------------

print("\nCalculating for N:", 10, "\n")
# loading bar
with click.progressbar(range(1000000)) as bar:
    for i in bar:
        pass 
pla(10)
