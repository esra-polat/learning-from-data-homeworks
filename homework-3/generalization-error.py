import math

def lower_bound(M, epsilon = 0.05):
    return math.ceil(-1 / (2 * epsilon**2) * math.log(0.03 / (2 * M)))

M = 1
print("For M =", M, ",min N =", lower_bound(M))

M = 10
print("For M =", M, ",min N =", lower_bound(M))

M = 100
print("For M =", M, ",min N =", lower_bound(M))