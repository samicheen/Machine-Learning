import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import random
from sklearn import svm

def computeError(X, y, theta, b):
    y_pred = np.sign(X * theta.T + b)
    a = np.sum(y==y_pred)
    error = float(len(X) - a)/float(len(X))
    return error*100

def plotPrediction(X, y, X_test, y_test, theta, title =
'train set'):
    X = np.concatenate((X, y), axis=1)
    X_test = np.concatenate((X_test, y_test), axis=1)
    X = np.array(X)
    X_test = np.array(X_test)
    X1 = X[np.ix_(X[:, 2] == -1, (0,1))]
    X2 = X[np.ix_(X[:, 2] == 1, (0,1))]
    X1_test = X_test[np.ix_(X_test[:, 2] == -1, (0,1))]
    X2_test = X_test[np.ix_(X_test[:, 2] == 1, (0,1))]
    minimum1 = np.floor(min(X1.min(), X1_test.min()))
    maximum1 = np.ceil(max(X1.max(), X1_test.max()))
    minimum2 = np.floor(min(X2.min(), X2_test.min()))
    maximum2 = np.ceil(max(X2.max(), X2_test.max()))
    plt.scatter(X1[:, 0], X1[:, 1], marker='+', label="Train Class 1")
    plt.scatter(X2[:, 0], X2[:, 1], marker='o', color="green",
                label="Train Class 2")
    plt.scatter(X1_test[:, 0], X1_test[:, 1], marker='*', color="red",
                label="Test Class 1")
    plt.scatter(X2_test[:, 0], X2_test[:, 1], marker='D', color="orange",
                label="Test Class 2")
    x = np.arange(-3, 3, 0.1)
    theta=np.array(theta)
    slope = -theta[:,1]/theta[:,2]
    intercept = -theta[:, 0]/theta[:,2]
    y_pred = slope*x + intercept
    plt.ylim(minimum2, maximum2)
    plt.plot(x, y_pred, color='black', label="Boundary")
    plt.title(title)
    plt.legend()
    plt.show()

def SMO(X, y, C, tolerance, max_passes, K):
    alpha = np.zeros(len(X))
    b = 0
    passes = 0
    while passes<max_passes:
        num_of_changed_alpha = 0
        for i in range(0, len(X)-1):
            temp = np.matrix(alpha*np.array(y.T))
            Ei = temp*K[:, i] + b -y[i]
            if (Ei*y[i] < -tolerance and alpha[i] < C) or (Ei*y[i] > tolerance
                                                           and alpha[i] > 0):
                r = range(1, i) + range(i+1, len(X))
                j = random.choice(r)
                Ej = temp*K[:, j] + b - y[j]
                old_alphai = alpha[i]
                old_aplhaj = alpha[j]
                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i]+alpha[j]-C)
                    H = min(C, alpha[i]+alpha[j])
                if L == H:
                    continue
                eta = 2*K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue
                alpha[j] -= (y[j]*(Ei-Ej))/eta
                if alpha[j] > H:
                    alpha[j] = H
                elif alpha[j] < L:
                    alpha[j] = L
                if np.abs(alpha[j] - old_aplhaj) < np.power(10, -5):
                    continue
                alpha[i] += y[i]*y[j]*(old_aplhaj - alpha[j])
                b1 = b - Ei - y[i]*(alpha[i] - old_alphai)*K[i, i] - \
                y[j]*(alpha[j] - old_aplhaj)*K[i, j]
                b2 = b - Ej - y[i] * (alpha[i] - old_alphai) * K[i, j] - \
                     y[j] * (alpha[j] - old_aplhaj) * K[j, j]

                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1+b2)/2

                num_of_changed_alpha += 1
        if num_of_changed_alpha == 0:
            passes += 1
        else:
            passes = 0
    return alpha, b

data = sio.loadmat('data1.mat')
X = data['X_trn']
y = data['Y_trn']
X_test = data['X_tst']
y_test = data['Y_tst']

X = np.matrix(X)
y = np.matrix(y)
y = np.where(y == 0, -1, 1)
X_test = np.matrix(X_test)
y_test = np.matrix(y_test)
y_test = np.where(y_test == 0, -1, 1)


K = X*X.T

C = 100
tolerance = 0.0001
max_passes = 3

alpha, b = SMO(X, y, C, tolerance, max_passes, K)

temp = np.matrix(alpha*np.array(y.T))
theta = temp*X
error_train = computeError(X, y, theta, b)

print "Omega for Dataset 1: ", theta
print "Classification error on training with Dataset 1: {:0.2f}%".format(error_train)

error_test = computeError(X_test, y_test, theta, b)
print "Classification error on test with Dataset 1: {:0.2f}%".format(
    error_test)

theta = np.insert(theta, 0, b, axis=1)

plotPrediction(X,y, X_test, y_test, theta, "SMO Train with Dataset 1")

svc = svm.SVC(kernel='linear', C=C).fit(X, y)
intercept_sklearn = svc.intercept_
theta_sklearn = np.insert(svc.coef_, 0, intercept_sklearn, axis=1)

plotPrediction(X, y, X_test, y_test, theta_sklearn,
               "SMO with sklearn with Dataset 1")

data = sio.loadmat('data2.mat')
X = data['X_trn']
y = data['Y_trn']
X_test = data['X_tst']
y_test = data['Y_tst']

X = np.matrix(X)
y = np.matrix(y)
y = np.where(y == 0, -1, 1)
X_test = np.matrix(X_test)
y_test = np.matrix(y_test)
y_test = np.where(y_test == 0, -1, 1)


K = X*X.T

C = 100
tolerance = 0.0001
max_passes = 3

alpha, b = SMO(X, y, C, tolerance, max_passes, K)

temp = np.matrix(alpha*np.array(y.T))
theta = temp*X
print
print "Omega for Dataset 2: ", theta
error_train = computeError(X, y, theta, b)

print "Classification error on training with Dataset 2: {:0.2f}%".format(
    error_train)

error_test = computeError(X_test, y_test, theta, b)
print "Classification error on test with Dataset 2: {:0.2f}%".format(
    error_test)
theta = np.insert(temp*X, 0, b, axis=1)

plotPrediction(X,y, X_test, y_test, theta, "SMO Train with Dataset 2")


svc = svm.SVC(kernel='linear', C=C).fit(X, y)
intercept_sklearn = svc.intercept_
theta_sklearn = np.insert(svc.coef_, 0, intercept_sklearn, axis=1)

plotPrediction(X, y, X_test, y_test, theta_sklearn, "SMO with sklearn with " \
                                                "Dataset 2")