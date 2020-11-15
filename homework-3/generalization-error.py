#150116884
#Esra Polat

import math

def lower_bound(M, epsilon = 0.05):
    return math.ceil(-1 / (2 * epsilon**2) * math.log(0.03 / (2 * M)))

def calculator(M):
    print("For M:", M, "\tmin N:", lower_bound(M))

calculator(1)
calculator(10)
calculator(100)
