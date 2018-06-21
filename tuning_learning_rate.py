from dnn_relu_softmax import *


layers = [784, 20, 9]
X, Y = readTrainData()
numOfTrainingSamples = int(0.8* X.shape[0])
numOfTestSamples = X.shape[0] - numOfTrainingSamples
trainX, trainY, Val_testX, Val_testY = getTraingAndValidationSamples(X, Y, numOfTrainingSamples)
alpha_range = [0.005,0.008,0.009,0.01,0.03,0.04,0.05,0.06,0.08,0.09,0.1,0.2,0.3,0.5]
train_set_cost = []
train_set_acc = []
validation_set_cost = []
validation_set_acc = []
for alpha in alpha_range:
    parameters= model(trainX, trainY,None,None,
                       layers, learning_rate=alpha, batchSize = 100, iterations=100,plot_costs=False)
    AL, _ = forward_propagation(trainX.T,parameters)
    train_cost =  softmax_loss(AL,trainY.T)
    train_acc = percentageCorrectPrediction(AL,trainY.T)
    print("cost of train data set for " + str(alpha) + " :" + str(train_cost) + " acc: " + str(train_acc))
    AL, _ = forward_propagation(Val_testX.T, parameters)
    val_cost = softmax_loss(AL, Val_testY.T)
    val_acc = percentageCorrectPrediction(AL, Val_testY.T)
    print("cost of cross_validation data set for " + str(alpha) + " :" + str(val_cost) + " acc: " + str(val_acc))
    train_set_cost.append(train_cost)
    validation_set_cost.append(val_cost)
    train_set_acc.append(train_acc)
    validation_set_acc.append(val_acc)

plt.scatter(alpha_range, train_set_cost)
plt.plot(alpha_range, train_set_cost, 'r-')
plt.scatter(alpha_range, validation_set_cost)
plt.plot(alpha_range, validation_set_cost, 'b--')
plt.title("Tuning Learning Rate")
plt.xlabel("Learning Rate")
plt.ylabel("cross entropy error")
plt.show()

plt.scatter(alpha_range, train_set_acc)
plt.plot(alpha_range, train_set_acc, 'r--')
plt.scatter(alpha_range, validation_set_acc)
plt.plot(alpha_range, validation_set_acc, 'b--')
plt.title("Tuning Learning Rate")
plt.xlabel("Learning Rate")
plt.ylabel("cross entropy error")
plt.show()
