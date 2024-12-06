''' 
    Implementing Neural Network Here.
    bins -> input neurons 
    hiddenLayers -> specified as input + output neurons / 2, 16 + 2 / 2 = 9 for bin 16
    output layer -> specified in config, 2
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# logarithmic normalization to avoid 0 since I have padded 0's to the smaller freq dataset when two bins are combined.
def normalizeInput(x):
    for i in range(x.shape[0]):
        x[i] = np.log1p(x[i]) 
    return x

def softmax(yHat):
    return np.exp(yHat) / np.sum(np.exp(yHat), axis=1, keepdims=True)

def neuralNetworkInit(inputNeurons, hiddenLayerList, outputNeurons):
    np.random.seed(0)
    weight1 = np.random.randn(inputNeurons, hiddenLayerList[0]) / np.sqrt(inputNeurons)
    bias1 = np.zeros((1, hiddenLayerList[0]))
    weight2 = np.random.randn(hiddenLayerList[0], outputNeurons) / np.sqrt(hiddenLayerList[0])
    bias2 = np.zeros((1, outputNeurons))

    return weight1, bias1, weight2, bias2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forwardPropagation(x, weight1, bias1, weight2, bias2):
    z1 = x.dot(weight1) + bias1
    a1 =  np.tanh(z1)
    z2 = a1.dot(weight2) + bias2
    a2 = np.tanh(z2)

    return z1, a1, z2, a2

def crossEntropyLoss(x, y, eta, weight1, bias1, weight2, bias2):
    z1, a1, z2, a2 = forwardPropagation(x, weight1, bias1, weight2, bias2)

    epsilon = 1e-15
    a2 = np.clip(a2, epsilon, 1 - epsilon) # clippin predicted prob to prevent log(0)

    correct_logprobs = -np.log(a2[range(x.shape[0]), y])
    data_loss = np.sum(correct_logprobs)
    data_loss += eta / 2 * ( np.sum( np.square(weight1)) + np.sum(np.square(weight2)))

    loss = 1./ x.shape[0] * data_loss

    return z1, a1, z2, a2, loss

def predict(weight1, weight2, bias1, bias2, x):
    _, _, _, a2 = forwardPropagation(x, weight1, bias1, weight2, bias2)

    return np.argmax(a2, axis = 1) # takes the maximum index from each row
    
def backPropagation(weight1, bias1, weight2, bias2, x_train, y_train, x_test, y_test, eta, learningRate, epochs, beta):

    globalLoss_train = []
    globalLoss_test = []
    maxTrainingAccuracy = 0.0
    maxTestingAccuracy = 0.0
    train_accuracies = []
    test_accuracies = []

    # momentum initialization
    mW1 = np.zeros_like(weight1)
    mB1 = np.zeros_like(bias1)
    mW2 = np.zeros_like(weight2)
    mB2 = np.zeros_like(bias2)
    
    for i in range(epochs):
        # z1, a1, z2, predOutput = forwardPropagation(x, weight1, bias1, weight2, bias2)

        # delta3 = np.copy(predOutput)
        # print("tanh Output: \n", delta3)

        # delta3[range(x.shape[0]), y] -= 1

        z1, a1, z2, a2, loss = crossEntropyLoss(x_train, y_train, eta, weight1, bias1, weight2, bias2)
        
        loss_history_train = 0.0
        loss_history_test = 0.0
        
        # Forward and backward pass for training data
        z1, a1, z2, a2, train_loss = crossEntropyLoss(x_train, y_train, eta, weight1, bias1, weight2, bias2)
        loss_history_train += train_loss

        # sample output converted to one hot encoded form
        y_one_hot = np.zeros((y_train.size, 2))
        y_one_hot[np.arange(y_train.size), y_train] = 1

        # calculate the loss with respect to output layer
        delta3 = (a2 - y_one_hot) * (1 - np.power(a2, 2))

        # gradients for the second layer
        dw2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis = 0, keepdims = True)
        
        # calculate loss with respect to hidden layer
        delta2 = delta3.dot(weight2.T) * (1 - np.power(a1, 2))

        # gradients of first layer
        dw1 = np.dot((x_train.T), delta2)
        db1 = np.sum(delta2, axis = 0)
 
        # regularization
        dw2 += eta * weight2
        dw1 += eta * weight1

        # adding momentum
        mW1 = beta * mW1 + (1 - beta) * dw1
        mB1 = beta * mB1 + (1 - beta) * db1
        mW2 = beta * mW2 + (1 - beta) * dw2
        mB2 = beta * mB2 + (1 - beta) * db2


        # Updating weights and biases
        weight1 -= learningRate * mW1
        bias1 -= learningRate * mB1
        weight2 -= learningRate * mW2
        bias2 -= learningRate * mB2
        
        # Calculating testing loss
        _, _, _, a2_test, test_loss = crossEntropyLoss(x_test, y_test, eta, weight1, bias1, weight2, bias2)
        loss_history_test += test_loss

        if( i % 50 == 0 ):
            predictions = predict(weight1, weight2, bias1, bias2, x_train)
            train_accuracy = getAccuracy(predictions, y_train)
            train_accuracies.append(train_accuracy)
            maxTrainingAccuracy = max(maxTrainingAccuracy, train_accuracy)

            test_predictions = predict(weight1, weight2, bias1, bias2, x_test)
            test_accuracy = getAccuracy(test_predictions, y_test)
            test_accuracies.append(test_accuracy)
            maxTestingAccuracy = max(maxTestingAccuracy, test_accuracy)

            # appending training loss
            globalLoss_train.append(loss_history_train/50)
            globalLoss_test.append(loss_history_test/50)

            # globalLoss.append(loss_history/50)
            print(f"Epoch {i}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Train Accuracy={train_accuracy:.4f}, Test Accuracy={test_accuracy:.4f}")

    return weight1, bias1, weight2, bias2, globalLoss_train, globalLoss_test, maxTrainingAccuracy, maxTestingAccuracy

# Getting the accuracy comparing the predicted accuracy with the ground truth labels
def getConfusionMatrix(predictions, y):
    print("Predicted Output: ", predictions)
    print("Ground Labels:    ", y)
    print()
    accuracy = np.sum(predictions == y) / y.shape[0]

    # confusion matrix implementation
    TP = np.sum((predictions == 1) & (y == 1))
    TN = np.sum((predictions == 0) & (y == 0))
    FP = np.sum((predictions == 1) & (y == 0))
    FN = np.sum((predictions == 0) & (y == 1))

    confusion_matrix = np.array([[TN, FP], [FN, TP]])

    plotConfusionMatrix(confusion_matrix)

    return accuracy

def getAccuracy(predictions, y):
    print("Predicted Output: ", predictions)
    print("Ground Labels:    ", y)
    print()

    return np.sum(predictions == y) / y.shape[0]

def plotConfusionMatrix(confusion_matrix):
    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted 0", "Predicted 1"],
                yticklabels=["Actual 0", "Actual 1"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

# Kfold-3 implementation
def trainXtestSet(x, y):
    samples = x.shape[0]
    
    # selects only good and bad motors 
    goodMotors = np.where(y == 0)[0]
    badMotors = np.where(y == 1)[0]

    # shuffles the good and bad motor datasets
    np.random.seed(42)
    np.random.shuffle(goodMotors)
    np.random.shuffle(badMotors)

    # dividing it by 3 since we are doing 3 k-fold
    goodFoldSize = len(goodMotors) // 3
    badFoldSize = len(badMotors) // 3

    goodMotorFolds = []
    badMotorFolds = []

    # creating three seperate dataset for good motors 
    for i in range(samples):
        start = i * goodFoldSize
        if i < 2:
            end = (i+1) * goodFoldSize
        else:
            end = len(goodMotors)
        goodMotorFolds.append(goodMotors[start:end])
    
    # Creating three seperate dataset for bad motors 
    for i in range(samples):
        start = i * badFoldSize
        if i < 2:
            end = (i+1) * badFoldSize
        else:
            end = len(badMotors)
        badMotorFolds.append(badMotors[start:end])

    motors = []

    # combining first two motors as the training dataset
    for i in range(3):
        combineMotors = np.concatenate((goodMotorFolds[i], badMotorFolds[i]))
        motors.append(combineMotors)

    # Shuffeling the combined dataset
    for motor in motors:
        np.random.shuffle(motor)
    
    datasets = []
    for i in range(3):
        # Use the current fold as the test set, others as training set
        test_set = motors[i]
        train_set = np.concatenate([motors[j] for j in range(3) if j != i])

        # Retrieve actual x and y values
        x_train, y_train = x[train_set], y[train_set]
        x_test, y_test = x[test_set], y[test_set]

        # Append to datasets list
        datasets.append((x_train, y_train, x_test, y_test))

    # returns x and y train and test dataset
    return datasets

def neural_network_3_layer_architecture(config, datasets, inputNeurons, hiddenLayerList, outputNeurons, beta):
    
    print(f"Neural Network Architecture: \nInput Layer: {inputNeurons} X {hiddenLayerList[0]}")
    print(f"Hidden Layer: {hiddenLayerList[0]}  X {outputNeurons}")
    print(f"Output Layer: ", outputNeurons)

    '''
        3 - Layer Neural Network Implementation
    '''
    # Weight Initialization
    weight1, bias1, weight2, bias2 = neuralNetworkInit(inputNeurons, hiddenLayerList, outputNeurons)

    eta = float(config['training']['eta'])
    learningRate = float(config['training']['learningRate'])
    epochs = 1500
    
    for i, (x_train, y_train, x_test, y_test) in enumerate(datasets):
        x_train = normalizeInput(x_train)
        x_test = normalizeInput(x_test)
        weight1, bias1, weight2, bias2, trainLoss, testLoss, maxTrainingAccuracy, maxTestingAccuracy = backPropagation(weight1, bias1, weight2, bias2, x_train, y_train, x_test, y_test, eta, learningRate, epochs, beta)

        # train_predictions = predict(weight1, weight2, bias1, bias2, x_train)
        test_predictions = predict(weight1, weight2, bias1, bias2, x_test)

        # getConfusionMatrix(train_predictions, y_train)
        getConfusionMatrix(test_predictions, y_test)
    
        print(f"Training Maximum Accuracy: {maxTrainingAccuracy:.4f}")
        print(f"Testing Maximum Accuracy: {maxTestingAccuracy:.4f}")
        plotGraph(trainLoss, testLoss, 50, i)

def plotGraph(trainLoss, testLoss, epoch, j):
    epochs_recorded = [i * 50 for i in range(len(trainLoss))]
    title_str = f"Fold {j + 1}"

    # Plotting both training and testing loss on the same graph
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_recorded, trainLoss, label="Training Loss", color='b', marker='o')
    plt.plot(epochs_recorded, testLoss, label="Testing Loss", color='r', marker='x')
    
    # Labeling and titles
    plt.xlabel(f"Epochs (every {epoch} epochs)")
    plt.ylabel("Global Average Loss every 50 epochs")
    plt.title("Global Average Loss over 50 Epochs - " + title_str)
    plt.legend()
    plt.grid(True)
    plt.show()
