import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def sigmoid(x):
    sig = 1 / (1 + np.exp(-1 * x))
    return sig

def computeError(X, y, theta):
    y_pred = np.round(sigmoid(X* theta.T))
    a = np.sum(y==y_pred)
    error = float(len(X) - a)/float(len(X))
    return error*100

def computeCost(X, y, theta, l):
    y_hat = sigmoid(X*theta.T)
    m = len(X)
    reg = (l/2*m)* np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    cost = (-1/m) * (y.T*np.log(y_hat) + (1-y).T*(np.log(1-y_hat))) + reg
    return cost

def ridgeSGD(X, y, theta, alpha, m, l, precision):
    batches = len(X) // m
    i = 0
    cost = [np.inf]
    while True:
        for b in range(batches):
            random_ind = np.random.choice(len(X), size = batches, replace = False)
            X1 = X[random_ind]
            y1 = y[random_ind]
            error = sigmoid(X1 * theta.T) - y1
            gradient = (X1.T * error).T + l*theta
            theta = theta - (alpha / len(X1)) * gradient
        cost.append(computeCost(X, y, theta, l))
        if (cost[-2] - cost[-1]) < precision and (cost[-2] - cost[-1]) > 0:
            break

        #print("Loss iter", i, ": ", cost[i])
        i = i + 1
    return theta, cost

def KFold(X, y, fold, alpha, theta, m, precision):
    lam = np.arange(0.01, 0.1, 0.01)
    cost_lam = {}
    for l in lam:
        cost = 0
        items = len(X)/fold
        if len(X)%fold != 0:
            items += 1
        for i in range(0,fold-1):
            data = np.concatenate((X, y), axis=1)
            np.random.shuffle(data)
            data_hold = data[i*items:(i+1)*items]
            data_train = np.delete(data, data_hold, axis=0)
            cols = data_train.shape[1]
            X_train = np.matrix(data_train[:, 0:cols-1])
            y_train = np.matrix(data_train[:, cols-1:])
            theta, _ = ridgeSGD(X_train, y_train, theta, alpha, m, l, precision)
            X_hold = np.matrix(data_hold[:, 0:cols-1])
            y_hold = np.matrix(data_hold[:, cols-1:])
            cost = cost + computeCost(X_hold, y_hold, theta, l)
        avg_cost= cost/fold
        cost_lam[l] = avg_cost
        best_lam = min(cost_lam, key=lambda k: cost_lam[k])
    return best_lam

def plotPrediction(X, y, X_test, y_test, theta, title = 'train set'):
    X = np.concatenate((X, y), axis=1)
    X_test = np.concatenate((X_test, y_test), axis=1)
    X = np.array(X)
    X_test = np.array(X_test)
    X1 = X[np.ix_(X[:, 3] == 0, (1,2))]
    X2 = X[np.ix_(X[:, 3] == 1, (1,2))]
    X1_test = X_test[np.ix_(X_test[:, 3] == 0, (1,2))]
    X2_test = X_test[np.ix_(X_test[:, 3] == 1, (1,2))]
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
    x = np.arange(minimum1, maximum1, 0.1)
    theta=np.array(theta)
    slope = -theta[:,1]/theta[:,2]
    intercept = -theta[:, 0]/theta[:,2]
    y_pred = slope*x + intercept
    plt.ylim(minimum2, maximum2)
    plt.plot(x, y_pred, color='black', label="Boundary")
    plt.title(title)
    plt.legend()
    plt.show()

data = sio.loadmat('data1.mat')
X = data['X_trn']
y = data['Y_trn']
X_test = data['X_tst']
y_test = data['Y_tst']

X = np.matrix(X)
X = np.insert(X, 0, 1, axis=1)
y = np.matrix(y)
X_test = np.matrix(X_test)
X_test = np.insert(X_test, 0, 1, axis=1)
y_test = np.matrix(y_test)

alpha = 0.01
precision = 0.01
theta = np.zeros(X.shape[1])
theta = np.matrix(theta)

K = 10
m = 10
l = KFold(X, y, K, alpha, theta, m, precision)
print l
theta, cost = ridgeSGD(X,y,theta, alpha, m, l, precision)

error_train = computeError(X, y, theta)
print "Omega for Dataset 1: ", theta
print "Classification error on training with Dataset 1: {:0.2f}%".format(error_train)

error_test = computeError(X_test, y_test, theta)
print "Classification error on test with Dataset 1: {:0.2f}%".format(
    error_test)

plotPrediction(X,y, X_test, y_test, theta, "Logistic Regression with "
                                       "Dataset 1")

data = sio.loadmat('data2.mat')
X = data['X_trn']
y = data['Y_trn']
X_test = data['X_tst']
y_test = data['Y_tst']

X = np.matrix(X)
X = np.insert(X, 0, 1, axis=1)
y = np.matrix(y)
X_test = np.matrix(X_test)
X_test = np.insert(X_test, 0, 1, axis=1)
y_test = np.matrix(y_test)

alpha = 0.01
precision = 0.01
theta = np.zeros(X.shape[1])
theta = np.matrix(theta)

K = 10
m = 10
l = KFold(X, y, K, alpha, theta, m, precision)
print l
theta, cost = ridgeSGD(X,y,theta, alpha,m, l, precision)

error_train = computeError(X, y, theta)
print
print "Omega for Dataset 2: ", theta
print "Classification error on training with Dataset 2: {:0.2f}%".format(
    error_train)

error_test = computeError(X_test, y_test, theta)
print "Classification error on test with Dataset 2: {:0.2f}%".format(
    error_test)

plotPrediction(X, y, X_test, y_test, theta, "Logistic Regression with "
                                       "Dataset 2")