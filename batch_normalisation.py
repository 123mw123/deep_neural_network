from utilities import *
import matplotlib.pyplot as plt

def initialize_parameters(layers):
    np.random.seed(3)
    parameters = {}
    L = len(layers)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers[l], layers[l - 1]) * np.sqrt(2 / layers[l - 1])
        parameters['b' + str(l)] = np.zeros((layers[l], 1))
    for l in range(1, L-1):
        parameters['gamma' + str(l)] = np.ones((layers[l], 1))
        parameters['beta' + str(l)] = np.ones((layers[l], 1))

    return parameters, len(layers) - 1

def forward_propagation(X, parameters, n):
    caches = []
    batchnorm_cache = []
    A = X
    L =n
    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        Z = np.dot(W, A_prev) + b
        Z, bn_cache = batchnorm_forward(Z, parameters['gamma' + str(l)], parameters['beta'+str(l)] )
        A = np.maximum(0, Z)
        cache = ((A_prev, W, b), Z)
        batchnorm_cache = batchnorm_cache + [bn_cache]
        caches = caches + [cache]
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    ZL = np.dot(W, A) + b
    ZL = ZL - ZL.max(0)
    AL = np.exp(ZL) / np.sum(np.exp(ZL), axis=0)
    cache = ((A, W, b), ZL)
    caches = caches + [cache]
    caches = (caches, batchnorm_cache)
    return AL, caches


def batchnorm_forward(X, gamma, beta):
    mu = np.mean(X, axis=1,keepdims=True)
    var = np.var(X, axis=1,keepdims=True)
    X_norm = (X - mu) / np.sqrt(var + 1e-8)
    out = gamma * X_norm + beta
    cache = (X, X_norm, mu, var, gamma, beta)

    return out, cache


def softmax_loss(AL, Y):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(AL),Y)
    cost = -(1/m)* logprobs.sum()
    return cost

def batchnorm_backward(dout, cache):
    X, X_norm, mu, var, gamma, beta = cache
    D,N = dout.shape
    dbeta = np.sum(dout, axis=1,keepdims=True)
    dgamma = np.sum(dout * X_norm, axis=1, keepdims= True)
    dX_norm = dout * gamma
    dvar = np.sum(dX_norm * (X-mu), axis=1,keepdims=True)
    dxmu1 = dX_norm * 1/var
    dvar = -1. / (var ** 2) * dvar
    dvar = 0.5 * 1. / np.sqrt(var + 1e-8) * dvar
    dsq = 1. / N * np.ones((D, N)) * dvar
    dxmu2 = 2 * (X-mu) * dsq
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1 + dxmu2, axis=1,keepdims=True)
    dx2 = 1. / N * np.ones((D, N)) * dmu
    dX = dx1 + dx2

    return dX, dgamma, dbeta


def backward_propagation(AL, Y, all_caches):
    grads = {}
    caches, batchnorm_cache = all_caches
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    linear_cache, activation_cache = caches[L - 1]

    dZ = (1/m)*(AL-Y)
    A_prev, W, b = linear_cache
    grads["dW" + str(L)] = np.dot(dZ, A_prev.T)
    grads["db" + str(L)] = np.sum(dZ, axis=1, keepdims=True)
    grads["dA" + str(L)] = np.dot(W.T, dZ)


    for l in reversed(range(L - 1)):
        linear_cache, activation_cache = caches[l]
        bn_cache = batchnorm_cache[l]
        dZ = np.array(grads["dA" + str(l + 2)], copy=True)
        dZ[activation_cache <= 0] = 0
        dZ, grads["dgamma" + str(l + 1)], grads["dbeta" + str(l + 1)] = batchnorm_backward(dZ, bn_cache)
        A_prev, W, b = linear_cache
        grads["dW" + str(l + 1)] = np.dot(dZ, A_prev.T)
        grads["db" + str(l + 1)] = np.sum(dZ, axis=1, keepdims=True)
        grads["dA" + str(l + 1)] = np.dot(W.T, dZ)

    return grads


def update_parameters(parameters, grads, learning_rate,n):
    L = n
    for i in range(1, L + 1):
        parameters['W' + str(i)] = parameters['W' + str(i)] - learning_rate * grads['dW' + str(i)]
        parameters['b' + str(i)] = parameters['b' + str(i)] - learning_rate * grads['db' + str(i)]
    for i in range(1,L):
        parameters['gamma' + str(i)] =  parameters['gamma'+ str(i)] - learning_rate * grads["dgamma" + str(i)]
        parameters['beta' + str(i)] = parameters['beta' + str(i)] - learning_rate * grads["dbeta" + str(i)]
    return parameters

def model(trainX, trainY, testX, testY, layers, learning_rate, batchSize, iterations):
    trgCosts = []
    tstCosts = []
    perTrgAccuracy = []
    perTstAccuracy = []
    parameters, n = initialize_parameters(layers)
    numBatches = int(len(trainX) / batchSize)
    for i in range(0, iterations):
        trgcost = 0.0
        for j in range(numBatches):
            # Select the indices for the current batch
            batchIndices = getCurrentBatchIndices(j, batchSize)
            # Select the training vectors
            xData = trainX[batchIndices].T
            yData = trainY[batchIndices].T
            AL, caches = forward_propagation(xData, parameters, n)
            trgcost = trgcost + softmax_loss(AL, yData)
            gradients = backward_propagation(AL, yData, caches)
            parameters = update_parameters(parameters, gradients, learning_rate,n)

        trgcost = trgcost / numBatches


        AL, caches = forward_propagation(trainX.T, parameters,n)
        perTrgAccuracy.append(percentageCorrectPrediction(AL, trainY.T))

        AL, caches = forward_propagation(testX.T, parameters,n)

        perTstAccuracy.append(percentageCorrectPrediction(AL, testY.T))
        trgCosts.append(trgcost)
        tstCosts.append(softmax_loss(AL, testY.T))
        if (i + 1) % 100 == 0:
            print(
            "Epoch : %d,training error : %3.2f,test error : %3.2f,training accuracy: %3.2f per,test accuracy : %3.2f per" \
            % (i+1,trgCosts[i], tstCosts[i], perTrgAccuracy[i], perTstAccuracy[i]))

    f = plt.figure(1)
    plt.plot(trgCosts, 'b-', label='Training Error')
    plt.plot(tstCosts, 'r--', label='Test Error')
    plt.title('Training and Test Errors with batch normalisation')
    plt.xlabel('No of Epochs')
    plt.ylabel('Cross Entropy Error')
    plt.legend(loc='upper right')

    f = plt.figure(2)
    plt.plot(perTrgAccuracy, 'b-', label='Training Accuracy')
    plt.plot(perTstAccuracy, 'r--', label='Test Accuracy')
    plt.title('Training and Test Accuracies  with batch normalisation')
    plt.xlabel('No of Epochs')
    plt.ylabel('Percentage Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    return parameters

layers = [784, 40,30,20, 9]
trainX, trainY = readTrainData()
testX, testY = readTestData()
model(trainX, trainY, testX, testY, layers, learning_rate=0.03, batchSize = 100, iterations=500)

