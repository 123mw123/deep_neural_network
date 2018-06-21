import  numpy as np
from scipy.io import loadmat

def tanh(Z):
    return np.tanh(Z)

def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    return  1 / (1 + np.exp(-Z))

def backward_tanh(Z):
    return 1-np.square(np.tanh(Z))

def backward_relu(Z):
    Z = Z>0
    return Z

def backward_sigmoid(Z):
    Z = 1 / (1 + np.exp(-Z))
    return Z * (1 - Z)


def percentageCorrectPrediction(Y, Y_):
    corrPred = np.equal(np.argmax(Y, axis = 0), np.argmax(Y_, axis = 0))
    perCorrPred = sum(corrPred.astype(np.float32))
    perCorrPred = (perCorrPred/Y.shape[1])*100
    return (perCorrPred)

def getCurrentBatchIndices(currentIndex, batchSize):
    start = currentIndex*batchSize
    stop = start + batchSize
    indices = np.linspace(start, stop, batchSize, endpoint = False, dtype = int)
    return(indices)

def one_hot(Y):
    onehot_encoded = []
    for u in Y[0]:
        vec = [0 for _ in range(0,9)]
        if (u == 10):
            vec[0] = 1
            onehot_encoded.append(vec)
        elif(u==13):
            vec[1] = 1
            onehot_encoded.append(vec)
        elif(u==16):
            vec[2] =1
            onehot_encoded.append(vec)
        elif (u == 17):
            vec[3] = 1
            onehot_encoded.append(vec)
        elif(u==19):
            vec[4] = 1
            onehot_encoded.append(vec)
        elif (u == 26):
            vec[5] = 1
            onehot_encoded.append(vec)
        elif (u == 27):
            vec[6] = 1
            onehot_encoded.append(vec)
        elif (u == 28):
            vec[7] = 1
            onehot_encoded.append(vec)
        if (u == 33):
            vec[8] = 1
            onehot_encoded.append(vec)

    Y = np.array(onehot_encoded)
    return Y


def readTrainData():
    number_of_samples= {10:0,13:0,16:0,17:0,19:0,26:0,27:0,28:0,33:0}
    data = loadmat('emnist-byclass.mat')
    data_x = data["dataset"][0][0][0][0][0][0]
    data_x = data_x.astype(np.float32)
    data_y = data["dataset"][0][0][0][0][0][1]
    data_x /= 255.0
    train_x = []
    train_y = []
    for i in range(data_y.shape[0]):
        # Read A, D ,G, H,J,Q,R, S ,X  letters
        if (data_y[i][0] == 10 or data_y[i][0] == 13 or data_y[i][0] == 16 or  data_y[i][0] == 17 or
                    data_y[i][0] == 19 or data_y[i][0] == 26 or data_y[i][0] == 27 or
                    data_y[i][0] == 28 or data_y[i][0] == 33 ):
            if(number_of_samples[data_y[i][0]]<2500):
                train_x = train_x + [data_x[i]]
                train_y = train_y + [data_y[i]]
                number_of_samples[data_y[i][0]]= number_of_samples[data_y[i][0]]+1


    train_x = np.array(train_x)
    train_y = np.array(train_y)
    X = train_x
    Y = one_hot(train_y.T)
    return X, Y

def readTestData():
    number_of_samples = {10: 0, 13: 0, 16: 0, 17: 0, 19: 0, 26: 0, 27: 0, 28: 0, 33: 0}
    data = loadmat('emnist-byclass.mat')
    data_x = data["dataset"][0][0][1][0][0][0]
    data_x = data_x.astype(np.float32)
    data_y = data["dataset"][0][0][1][0][0][1]
    data_x /= 255.0
    test_x = []
    test_y = []
    for i in range(data_y.shape[0]):
        # Read A, D ,G, H,J,Q,R, S ,X  letters
        if (data_y[i][0] == 10 or data_y[i][0] == 13 or data_y[i][0] == 16 or  data_y[i][0] == 17 or
                    data_y[i][0] == 19 or data_y[i][0] == 26 or data_y[i][0] == 27 or
                    data_y[i][0] == 28 or data_y[i][0] == 33 ):
            if (number_of_samples[data_y[i][0]] < 400):
                test_x = test_x + [data_x[i]]
                test_y = test_y + [data_y[i]]
                number_of_samples[data_y[i][0]] = number_of_samples[data_y[i][0]] + 1


    test_x = np.array(test_x)
    test_y = np.array(test_y)
    X = test_x
    Y = one_hot(test_y.T)
    return X, Y


def getTraingAndValidationSamples(X, Y, numTrgPtrns):
    randIndex = np.random.permutation(X.shape[0])
    randX = X[randIndex]
    randY = Y[randIndex]
    trainIndex = np.linspace(0, numTrgPtrns, numTrgPtrns, endpoint = False, dtype = int)
    Val_testIndex = np.linspace(numTrgPtrns, X.shape[0], X.shape[0]- numTrgPtrns, endpoint = False, dtype = int)
    trainX = randX[trainIndex]
    trainY = randY[trainIndex]
    Val_testX = randX[Val_testIndex]
    Val_testY = randY[Val_testIndex]
    return trainX, trainY, Val_testX, Val_testY

def getRotatedData():
    data = loadmat('emnist-byclass.mat')
    x_train = data["dataset"][0][0][0][0][0][0]
    x_train = x_train.astype(np.float32)
    y_train = data["dataset"][0][0][0][0][0][1]
    x_train /= 255.0
    train_data = []
    j =0
    for i in range(y_train.shape[0]):
        if (y_train[i][0] == 10):
            if(j==1):
                train_data = train_data + [x_train[i]]
            j = j + 1

    train_x = np.array(train_data)

    return train_x