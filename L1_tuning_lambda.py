from  L1_regularization import *


numOfTrainingSamples = int(0.8 * X.shape[0])
numOfTestSamples = X.shape[0] - numOfTrainingSamples
trainX, trainY, Val_testX, Val_testY = getTraingAndValidationSamples(X, Y, numOfTrainingSamples)
Lambda_range =  [0.000001,0.000005,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10]
train_set_cost = []
train_set_acc = []
validation_set_cost = []
validation_set_acc = []
for Lambda_1 in Lambda_range:
    parameters = model(trainX, trainY, None, None,
                       layers, learning_rate=0.03, batchSize=100, iterations=100, Lambda=Lambda_1, plot_costs=False)
    AL, _ = forward_propagation(trainX.T, parameters)
    train_cost = softmax_loss_with_regularization(AL, trainY.T, parameters, Lambda_1)
    train_acc = percentageCorrectPrediction(AL, trainY.T)
    print("cost of train data set for " + str(Lambda_1) + " :" + str(train_cost) + " acc: " + str(train_acc))
    AL, _ = forward_propagation(Val_testX.T, parameters)
    val_cost = softmax_loss_with_regularization(AL, Val_testY.T, parameters, Lambda_1)
    val_acc = percentageCorrectPrediction(AL, Val_testY.T)
    print("cost of cross_validation data set for " + str(Lambda_1) + " :" + str(val_cost) + " acc: " + str(val_acc))
    train_set_cost.append(train_cost)
    validation_set_cost.append(val_cost)
    train_set_acc.append(train_acc)
    validation_set_acc.append(val_acc)

plt.scatter(Lambda_range, train_set_cost)
plt.plot(Lambda_range, train_set_cost, 'r--')
plt.scatter(Lambda_range, validation_set_cost)
plt.plot(Lambda_range, validation_set_cost, 'b--')
plt.title('Tuning lambda for L1 regularization')
plt.xlabel('Lambda')
plt.ylabel('cost')
plt.show()

plt.scatter(Lambda_range, train_set_acc)
plt.plot(Lambda_range, train_set_acc, 'r--')
plt.scatter(Lambda_range, validation_set_acc)
plt.plot(Lambda_range, validation_set_acc, 'b--')
plt.title('Tuning lambda for L1 regularization')
plt.xlabel('Lambda')
plt.ylabel('accuracies')
plt.show()
