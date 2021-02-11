
# 150116884 - Esra Polat

# Learning From Data
# Homework 8 Solution

import numpy as np
import sklearn as sk
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sb
from collections import defaultdict

!wget http://www.amlbook.com/data/zip/features.train
!wget http://www.amlbook.com/data/zip/features.test

# Load and parse data into memory
train_set = np.loadtxt("features.train")
test_set = np.loadtxt("features.test")

"""* # **Polynomial Kernels**

**I have defined the necessary variables and functions for Polynomial Kernels observation as shown below.**
"""

# Define an function to get classification data sets for "n-to-all"
def read_data_n(n, type = "train"):
  if type == "train":
    dataset = train_set
  elif type == "test":
    dataset = test_set
  else:
    1/0

  is_n = dataset[:,0] == n
  is_not_n = dataset[:,0] != 0

  X = dataset.copy()
  X[:,0] = 1.0

  y = 2 * is_n - 1

  return X, y

def E_in(a, b, c, d, e):
  for n in (a, b, c, d, e):
    classifier = svm.SVC( C=0.01, kernel="poly", degree=2, gamma=1.0)
    X, y = read_data(n)
    classifier.fit(X, y)
    hypo_y = classifier.predict(X)
    E_in = np.mean(y != hypo_y)
    print("for ", n, ", result ~ {0:0.4f}".format(E_in))

print("Question 2\n")
E_in(0, 2, 4, 6, 8)

"""**Above we see the output of the E_in function. In the calculation we made with the C and Q values ​​given in the question 2, we can say that the highest E_in value is obtained with the option A. Because 0 versus all has the highest in-sample error. So, the answer of question 2 is :**
```
A) 0 versus all
```




"""

print("\nQuestion 3\n")
E_in(1, 3, 5, 7, 9)

"""**Above we see the output of the E_in function. In the calculation we made with the C and Q values ​​given in the question 3, we can say that the lowest E_in value is obtained with the option A. Because 1 versus all has the lowest in-sample error. So, the answer of question 3 is :**
```
A) 1 versus all
```


"""

def get_diff(n):
    classifier = svm.SVC( C=0.01, kernel="poly", degree=2, gamma=1.0)
    X, y = read_data(n)
    classifier.fit(X,y)
    return len(classifier.support_)

print("\nQuestion 4\n")
print("Difference = ", get_diff(0)-get_diff(1))

"""**Above we see the difference between the number of support vectors of these two classıfıers is 1793. We can say that the option C is true answer. So, the answer of question 4 is :**
```
C) 1800
```


"""

# Define an function to get classification data sets for "n-to-n"
def read_data_n_m(n, m, type = "train"):
  if type == "train":
    dataset = train_set
  elif type == "test":
    dataset = test_set
  else:
    1/0

  is_n = dataset[:,0] == n
  is_m = dataset[:,0] == m

  log_or = dataset[np.logical_or(is_n, is_m)]

  X = log_or.copy()
  X[:,0] = 1.0

  y = log_or[:,0].copy()

  y[np.where(log_or[:,0] == n)] = 1
  y[np.where(log_or[:,0] == m)] = -1

  return X, y

X_train, y_train = read_data_n_m(1, 5, "train")
X_test, y_test = read_data_n_m(1, 5, "test")

def Ein_and_Eout(c1, c2, c3, c4, deg):

  for c in (c1, c2, c3, c4):
    classifier = svm.SVC( C=c, kernel="poly", degree=deg, gamma=1.0)
    classifier.fit(X_train, y_train)
    sup_vec = len(classifier.support_)

    E_in = np.mean(classifier.predict(X_train) != y_train)
    E_out = np.mean(classifier.predict(X_test) != y_test)

    print("\nC: {}  \tsvm: {} \tEin: {}  \tEout: {}".format(c, sup_vec, round(E_in, 5), round(E_out,5)))

print("Question 5")

Ein_and_Eout(0.001, 0.01, 0.1, 1.0, 2)

"""**Above we see the outputs of the E_in and E_out functions. In the calculation we made with the C set and Q value ​​given in the question 5, we consider the 1 versus 5 classifier. According to the results, we can say that option D is the correct statement. Because Maximum C gives lowest E_in and other statements are false. So, the answer of question 5 is :**
```
D) Maximum C achieves the lowest E_in
```



"""

print("Question 6")

print("\n1 versus 5 - Q = 2")
Ein_and_Eout(0.0001, 0.001, 0.010, 0.1, 2)

print("\n\n1 versus 5 - Q = 5")
Ein_and_Eout(0.0001, 0.001, 0.010, 0.1, 5)

"""**Above we see the outputs of the E_in and E_out functions. In the calculation we made with the C set and Q value ​​given in the question 6, we consider the 1 versus 5 classifier. According to the results, we can say that option B is the correct statement. Because C = 0.001 causes the svm is lower at Q = 5 and other statements are false.. So, the answer of question 6 is :**
```
B) When C = 0.001, the number of support vectors is lower at Q = 5.
```

* # **Cross Validation**
"""

X_cv, y_cv = read_data_n_m(1, 5, "train")

def helper(a):
  i = 999
  res = 0
  for c in (0.0001, 0.001, 0.01, 0.1, 1):
    if a[c] < i:
      i = a[c]
      res = c
  return res

selected = defaultdict(int)
fin_E_cv = []

for i in range(100):

  # k-fold cross validation version of SVM
  kfold = sk.model_selection.KFold(n_splits=10, shuffle=True)
  E_cv = defaultdict(float)

  for j, k in kfold.split(X_cv):

    for c in (0.0001, 0.001, 0.01, 0.1, 1):

      X_train, y_train = X_cv[j,:], y_cv[j]
      X_test, y_test = X_cv[k,:], y_cv[k]

      classifier = svm.SVC( C=c, kernel="poly", degree=2, gamma=1.0)
      classifier.fit(X_train, y_train)

      E_val = np.mean(classifier.predict(X_test) != y_test)
      E_cv[c] += E_val
    
    fin_c = helper(E_cv)
    selected[fin_c] += 1
    fin_E_cv.append(E_cv[fin_c])
    print(selected)

print("Question 8\n")

print("The average of E_cv over all 100 runs", np.mean(fin_E_cv))

"""**Above we see the output of selected values. According to the results, we can say that C = 0.001 is selected most often. So, the answer of question 7 is :**
```
B) C = 0.001 is selected most often.
```

**According to the results, we can say that the average value of E_cv over the 100 runs is closest to 0.009. Because we find that the average is 0.235. So, the answer of question 8 is :**
```
E) 0.009
```

* # **RBF Kernel**
"""

X_train, y_train = read_data_n_m(1, 5, "train")
X_test, y_test = read_data_n_m(1, 5, "test")

for c in (0.01, 1.0, 100, 10000, 1000000):
    classifier = svm.SVC(C=c, kernel="rbf", gamma=1.0)
    classifier.fit(X_train, y_train)
    E_out = np.mean(classifier.predict(X_test) != y_test)
    E_in = np.mean(classifier.predict(X_train) != y_train)
    print("\nC: {}  \tEin: {}  \tEout: {}".format(c, round(E_in, 5), round(E_out,5)))

"""**Above we see the outputs of the E_in and E_out functions for RBF. According to the results, we can say that the lowest E_in result is in the C = 1000000 and on the other hand the lowest E_out result is in the C = 100. Because in C = 1000000, E_in is 0.00064 and then in C = 100, E_out is 0.01. So, the answers of question 9 and 10 are :**
```
(9)  E)10^6
(10) C)100
```
"""
