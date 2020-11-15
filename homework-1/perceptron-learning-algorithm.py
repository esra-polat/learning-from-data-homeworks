import numpy as np
import matplotlib.pyplot as plt
import random
import click

def generate_random(n): 
    return np.random.uniform(-1, 1, size = n)

def pla(data_size):
    N = data_size
    N_test = 1000
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
        a = (theta2[1] - theta1[1]) / (theta2[0] - theta1[0]) # a is the slope
        b = theta2[1] - a * theta2[0]  
        w = np.array([b, a, -1])

        # Choose the inputs x_n of the data set as random points
        X = np.transpose(np.array([np.ones(N), generate_random(N), generate_random(N)]))
        # Evaluate the target function on each x_n to get the corresponding output y_n
        y = np.sign(np.dot(X, w))
        
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
            comp = (y_hypo != y)                 
            # indices of points with wrong classification by hypothesis h     
            wrong = np.where(comp)[0]                 

            if wrong.size == 0:
                break
            
            # pick a random misclassified point
            random_choice = np.random.choice(wrong)      

            # update weight vector (new hypothesis):
            weight = weight +  y[random_choice] * np.transpose(X[random_choice])
            count_iteration += 1

        total_iteration += count_iteration
        
        # Calculate error
        # Create data "outside" of training data

        x0_test = np.random.uniform(-1,1,N_test)
        x1_test = np.random.uniform(-1,1,N_test)

        X_test = np.array([np.ones(N_test), x0_test, x1_test]).T

        y_target = np.sign(X_test.dot(w))
        y_hypothesis = np.sign(X_test.dot(weight))
        
        mismatch_ratio = ((y_target != y_hypothesis).sum()) / N_test
        mismatch_total += mismatch_ratio
        
    # Average ratio for the mismatch between f(x) and h(x) outside of the training data
    # mismatch_avg, iteration_avg 
    return (mismatch_total / run_step, total_iteration / run_step) 


def calculator(number):
    N = int(number/10)
    dataset = [N*i for i in range(1, 11)]
    E_out, iterations = [], []
        
    for size in dataset:
        mismatch_avg, iteration_avg = pla(size)
        E_out.append(mismatch_avg)
        iterations.append(iteration_avg)

    plt.figure(1)
    plt.plot(dataset, E_out, 'ro')
    plt.ylabel("E_out")
    plt.xlabel("size training")
    plt.savefig('E_out.png')

    plt.figure(2)
    plt.plot(dataset, iterations, 'bo')
    plt.ylabel("PLA iterations")
    plt.xlabel("size training")
    plt.savefig('PLA.png')

    print("PLA iterations:\t", '{:1.2f}'.format(iterations[9]))
    print("P(f(x)!=h(x)):\t", '{:1.2f}'.format(E_out[9]))

    plt.show()

# -----------------------------------------------------------------

def start(n):

    print("\nCalculating for N:", n)
    # loading bar
    with click.progressbar(range(1000000)) as bar:
        for i in bar:
            pass 

    calculator(n)

# -----------------------------------------------------------------

start(10)
start(100)