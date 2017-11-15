import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def computeCost(X, y, theta):
    product = np.power(((X * theta.T) - y),2)
    cost = np.sum(product)/(2*len(X))
    return cost

def computeTheta(X, y):
    theta = (X.T * X).I * (X.T * y)
    return theta

def MeanSquareError(y, y_pred):
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)
    return np.mean((y - y_pred) ** 2)

def KFold(X, y, fold):
    lam = np.arange(0.01, 0.1, 0.01)
    cost_lam = {}
    for l in lam:
        cost = 0
        for i in range(0,fold-1):
            data = np.concatenate((X, y), axis=1)
            data_split = np.split(data, fold)
            data1 = data_split[:i]
            data2 = data_split[i+1:]
            if not data1:
                data_train = np.concatenate((data2[:]), axis=0)
            else:
                data_train = np.concatenate((data1,data2), axis=0)
                data_train = np.concatenate((data_train[:]), axis=0)
            data_hold = data_split[i]
            cols = data_train.shape[1]
            X_train = np.matrix(data_train[:, 0:cols - 1])
            y_train = np.matrix(data_train[:, cols - 1:cols])
            theta = RidgeRegressionClosed(X_train, y_train, l)
            X_hold = np.matrix(data_hold[:, 0:cols - 1])
            y_hold = np.matrix(data_hold[:, cols - 1:cols])
            theta = theta.T
            cost = cost + computeCost(X_hold, y_hold, theta)
        avg_cost= cost/fold
        cost_lam[l] = avg_cost
        best_lam = min(cost_lam, key=lambda k: cost_lam[k])
    return best_lam

def RidgeRegressionClosed(X, y, l):
    X_product = X.T * X
    lam = l*np.identity(X_product.shape[1])
    theta = (X_product + lam).I * (X.T * y)
    return theta

def ridgeSGD(X, y, theta, alpha, m, l, precision):
    batches = len(X) // m
    i = 0
    cost = [np.inf]
    while True:
        for b in range(batches):
            random_ind = np.random.choice(len(X), size = batches, replace = False)
            X1 = X[random_ind]
            y1 = y[random_ind]
            error = (X1 * theta.T) - y1
            gradient = (X1.T * error).T + l*theta
            theta = theta - (alpha / len(X1)) * gradient
        cost.append(computeCost(X, y, theta))
        if (cost[-2] - cost[-1]) < precision and (cost[-2] - cost[-1]) > 0:
            break

        #print("Loss iter", i, ": ", cost[i])
        i = i + 1
    return theta, cost

def transform(X, degree):
    Z = X
    A = X
    for i in range(degree-1):
        Z = Z * A
        X = np.insert(X, [i+1], Z, axis=1)
    X_tran = np.insert(X, 0, 1, axis=1)
    return X_tran

def plotPrediction(y, y_pred, title = 'train set'):
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)
    plt.scatter(y, y_pred)
    plt.plot([min(y), max(y)], [min(y), max(y)], '--')
    plt.xlabel('true value')
    plt.ylabel('predicted value')
    plt.title(title)

data = sio.loadmat('dataset2.mat')
degrees = [2, 3, 5]
alphas = [0.00001, 0.000001, 0.000000001]
precisions = [0.0001, 0.0001, 0.0001]
for i in range(len(degrees)):
    K = [2, 10, len(data['X_trn'])]
    for j in range(len(K)):
        X = data['X_trn']
        y = data['Y_trn']
        X_test = data['X_tst']
        y_test = data['Y_tst']

        degree = degrees[i]
        X = transform(X, degree)
        X_test = transform(X_test, degree)

        X = np.matrix(X)
        y = np.matrix(y)
        X_test = np.matrix(X_test)
        y_test = np.matrix(y_test)

        l = KFold(X, y, K[j])
        theta = np.zeros(degree + 1)
        theta = np.matrix(theta)
        theta = RidgeRegressionClosed(X,y,l)
        print "Ridge Closed Form with polynomial degree "+str(degree)+" with " \
                                        "K = "+str(K[j])+":"
        print "Theta:"
        theta = theta.T
        print theta
        theta = np.matrix(theta)
        y_pred = X*theta.T
        #plotPrediction(y, y_pred, "Ridge Closed Form with polynomial degree "
        #                          ""+str(degree))
        cost_train = computeCost(X, y, theta)
        mse_train = MeanSquareError(y, y_pred)
        print "Training Error:"
        print mse_train
        y_test_pred = X_test*theta.T
        #plotPrediction(y_test, y_test_pred, "Ridge Closed Form with
        # polynomial "
        #                             "degree "
        #                          "" + str(degree))
        cost_test = computeCost(X_test, y_test, theta)
        mse_test = MeanSquareError(y_test, y_test_pred)
        print "Test Error:"
        print mse_test
        print "\nRidge stochastic with polynomial degree "+str(degree)+" with " \
                                        "K = "+ str(K[j])+":"
        alpha = alphas[i]
        m = 10
        precision = precisions[i]
        s_theta, s_cost_train = ridgeSGD(X, y, theta, alpha, m, l, precision)
        print "Optimal lambda:", l
        print "Theta:"
        print s_theta
        print "Training Error:"
        y_stoch_pred_train = X*s_theta.T
        print MeanSquareError(y, y_stoch_pred_train)
        print "Test Error:"
        y_stoch_pred_test = X_test*s_theta.T
        print MeanSquareError(y_test, y_stoch_pred_test)

        print "-------------------------------------------------------\n"
        j = j + 1
    i = i + 1