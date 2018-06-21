from utilities import *
import matplotlib.pyplot as plt


def initialize_parameters(layers):
    np.random.seed(3)
    parameters = {}
    L = len(layers)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers[l], layers[l - 1]) * np.sqrt(2 / layers[l - 1])
        parameters['b' + str(l)] = np.zeros((layers[l], 1))
    return parameters

def forward_propagation(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        Z = np.dot(W, A_prev) + b
        A = sigmoid(Z)
        cache = ((A_prev, W, b), Z)
        caches = caches + [cache]
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    ZL = np.dot(W, A) + b
    ZL = ZL - ZL.max(0)
    AL = np.exp(ZL) / np.sum(np.exp(ZL), axis=0)
    cache = ((A, W, b), ZL)
    caches = caches + [cache]

    return AL, caches


def softmax_loss(AL, Y):
    m = Y.shape[1]
    cost = np.multiply(np.log(AL),Y)
    cost = -(1/m)* cost.sum()
    return cost

def backward_propagation(AL, Y, caches):
    gradients = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    cache, ZL = caches[L - 1]
    dZ = (1/m)*(AL-Y)
    A_prev, W, b = cache
    gradients["dW" + str(L)] = np.dot(dZ, A_prev.T)
    gradients["db" + str(L)] = np.sum(dZ, axis=1, keepdims=True)
    gradients["dA" + str(L)] = np.dot(W.T, dZ)
    for l in reversed(range(L - 1)):
        cache, Z = caches[l]
        A_prev, W, b = cache
        Z = backward_sigmoid(Z)
        dZ = gradients["dA" + str(l + 2)] * Z
        gradients["dW" + str(l + 1)] = np.dot(dZ, A_prev.T)
        gradients["db" + str(l + 1)] = np.sum(dZ, axis=1, keepdims=True)
        gradients["dA" + str(l + 1)] = np.dot(W.T, dZ)
    return gradients

def update_parameters(parameters, gradients, learning_rate):
    L = len(parameters) // 2
    for i in range(1, L + 1):
        parameters['W' + str(i)] = parameters['W' + str(i)] - learning_rate * gradients['dW' + str(i)]
        parameters['b' + str(i)] = parameters['b' + str(i)] - learning_rate * gradients['db' + str(i)]
    return parameters


def model(trainX, trainY, testX, testY, layers, learning_rate, batchSize, iterations):
    trgCosts = []
    tstCosts = []
    perTrgAccuracy = []
    perTstAccuracy = []
    parameters = initialize_parameters(layers)
    numBatches = int(len(trainX) / batchSize)
    for i in range(0, iterations):
        trgcost = 0.0
        for j in range(numBatches):
            # Select the indices for the current batch
            batchIndices = getCurrentBatchIndices(j, batchSize)
            # Select the training vectors
            xData = trainX[batchIndices].T
            yData = trainY[batchIndices].T
            AL, caches = forward_propagation(xData, parameters)
            trgcost = trgcost + softmax_loss(AL, yData)
            gradients = backward_propagation(AL, yData, caches)
            parameters = update_parameters(parameters, gradients, learning_rate)

        trgcost = trgcost / numBatches
        trgCosts.append(trgcost)
        AL, caches = forward_propagation(trainX.T, parameters)
        perTrgAccuracy.append(percentageCorrectPrediction(AL, trainY.T))

        AL, caches = forward_propagation(testX.T, parameters)
        tstCosts.append(softmax_loss(AL, testY.T))
        perTstAccuracy.append(percentageCorrectPrediction(AL, testY.T))

        if (i + 1) % 100 == 0:
            print(
            "Epoch : %d,training error : %3.2f,test error : %3.2f,training accuracy: %3.2f per,test accuracy : %3.2f per" \
            % (i + 1, trgCosts[i], tstCosts[i], perTrgAccuracy[i], perTstAccuracy[i]))

    f = plt.figure(1)
    plt.plot(trgCosts, 'b-', label='Training Error')
    plt.plot(tstCosts, 'r--', label='Test Error')
    plt.title('Training and Test Errors for sigmoid activation function and softmax loss')
    plt.xlabel('No of Epochs')
    plt.ylabel('Cross Entropy Error')
    plt.legend(loc='upper right')

    f = plt.figure(2)
    plt.plot(perTrgAccuracy, 'b-', label='Training Accuracy')
    plt.plot(perTstAccuracy, 'r--', label='Test Accuracy')
    plt.title('Training and Test Accuracies for sigmoid activation function and softmax loss')
    plt.xlabel('No of Epochs')
    plt.ylabel('Percentage Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    return parameters

layers = [784, 20, 9]
trainX, trainY = readTrainData()
textX, textY = readTestData()
model(trainX, trainY, textX, textY, layers, learning_rate=0.03, batchSize = 100, iterations=1000)