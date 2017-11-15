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

def stochasticGradientDescent(X, y, theta, alpha, m, precision):
    batches = len(X) // m
    i = 0
    cost = [np.inf]
    while True:
        for b in range(batches):
            random_ind = np.random.choice(len(X), size = batches, replace = False)
            X1 = X[random_ind]
            y1 = y[random_ind]
            error = (X1 * theta.T) - y1
            theta = theta - (alpha / len(X1)) * (X1.T * error).T
        cost.append(computeCost(X, y, theta))
        if (cost[-2] - cost[-1]) < precision and (cost[-2] - cost[-1]) > 0:
            break

        #print("Loss iter", i, ": ", cost[i])
        i = i + 1
    return theta, cost




def plotPrediction(y, y_pred, title = 'train set'):
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)
    plt.scatter(y, y_pred)
    plt.plot([min(y), max(y)], [min(y), max(y)], '--')
    plt.xlabel('true value')
    plt.ylabel('predicted value')
    plt.title(title)

def transform(X, degree):
    Z = X
    A = X
    for i in range(degree-1):
        Z = Z * A
        X = np.insert(X, [i+1], Z, axis=1)
    X_tran = np.insert(X, 0, 1, axis=1)
    return X_tran

data = sio.loadmat('dataset1.mat')
degrees = [2, 3, 5]
alphas = [0.001, 0.0001, 0.00001]
precisions = [0.001, 0.0001, 0.0001]
for i in range(len(degrees)):
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

    print "Closed Form with polynomial degree "+str(degree)+":"
    print "Theta:"
    theta = computeTheta(X, y)
    theta = theta.T
    print theta
    theta = np.matrix(theta)
    y_pred = X*theta.T
    #plotPrediction(y, y_pred, "Closed form training with polynomial degree "
    #                          ""+str(degree))
    cost_train = computeCost(X, y, theta)
    mse_train = MeanSquareError(y, y_pred)
    print "Training Error:"
    print mse_train
    y_pred_test = X_test*theta.T
    #plotPrediction(y_test, y_pred_test, "Closed form test with polynomial "
    #                               "degree " + str(degree))
    cost_test = computeCost(X_test, y_test, theta)
    mse_test = MeanSquareError(y_test, y_pred_test)
    print "Test Error:"
    print mse_test
    print "\nStochastic with polynomial degree "+str(degree)+":"
    alpha = alphas[i]
    m = 10
    theta = np.zeros(degree+1)
    theta = np.matrix(theta)
    precision = precisions[i]
    s_theta, s_cost_train = stochasticGradientDescent(X, y, theta, alpha, m,
                                                      precision)

    print "Theta:"
    print s_theta

    y_pred_train_sch = X*s_theta.T
    print "Training Error:"
    print MeanSquareError(y, y_pred_train_sch)

    y_pred_test_sch = X_test*s_theta.T
    print "Test Error:"
    print MeanSquareError(y_test, y_pred_test_sch)

    #plt.plot(np.arange(len(s_cost_train)), s_cost_train)
    #plt.title("Stochastic with polynomial degree "+str(degree))
    #print s_theta, s_cost_train
    print "------------------------------------------------------------\n"
    i = i + 1
