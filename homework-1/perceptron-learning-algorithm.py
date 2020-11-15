#150116884
#Esra Polat

import numpy as np
import matplotlib.pyplot as plt
import random
import click

def generate_random(n): 
    return np.random.uniform(-1, 1, size = n)

def pla(data_size):
    N_sample = data_size
    N_test = data_size
    run_step = 1000
    total_iteration = 0
    mismatch_total = 0
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
        
        # initialize weight vector being all zeros
        weight = np.zeros(3)     
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

            # update weight vector
            weight = weight +  y_target[random_choice] * np.transpose(X[random_choice])
            count_iteration += 1

        total_iteration += count_iteration
        
        # Calculate error
        x0_test = np.random.uniform(-1,1,N_test)
        x1_test = np.random.uniform(-1,1,N_test)

        X_test = np.array([np.ones(N_test), x0_test, x1_test]).T

        y_target_test = np.sign(X_test.dot(y))
        y_hypo_test = np.sign(X_test.dot(weight))
        
        mismatch_ratio = ((y_target_test != y_hypo_test).sum()) / N_test
        mismatch_total += mismatch_ratio
        
    # return the mismatch avg and iteration avg 
    return (mismatch_total / run_step, total_iteration / run_step) 


def calculator(number):
    dataset = [int(number/10)*i for i in range(1, 11)]
    E_out, iterations = [], []
        
    for size in dataset:
        mismatch_avg, iteration_avg = pla(size)
        E_out.append(mismatch_avg)
        iterations.append(iteration_avg)

    print("\nPLA iterations:\t", '{:1.2f}'.format(iterations[9]))
    print("P(f(x)!=h(x)):\t", '{:1.2f}'.format(E_out[9]), "\n")

# -----------------------------------------------------------------

def start(n):

    print("\nCalculating for N:", n, "\n")
    # loading bar
    with click.progressbar(range(1000000)) as bar:
        for i in bar:
            pass 

    calculator(n)

# -----------------------------------------------------------------

start(10)
start(100)
