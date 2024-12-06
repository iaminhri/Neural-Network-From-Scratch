import numpy as np
import neuralNetwork as nn

def neuralNetworkInit(inputNeurons, hiddenLayerList, outputNeurons):
    np.random.seed(0)
    weight1 = np.random.randn(inputNeurons, hiddenLayerList[0]) / np.sqrt(inputNeurons)
    bias1 = np.zeros((1, hiddenLayerList[0]))
    weight2 = np.random.randn(hiddenLayerList[0], hiddenLayerList[1]) / np.sqrt(hiddenLayerList[0])
    bias2 = np.zeros((1, hiddenLayerList[1]))
    weight3 = np.random.randn(hiddenLayerList[1], hiddenLayerList[2]) / np.sqrt(hiddenLayerList[1])
    bias3 = np.zeros((1, hiddenLayerList[2]))
    weight4 = np.random.randn(hiddenLayerList[2], outputNeurons) / np.sqrt(hiddenLayerList[2])
    bias4 = np.zeros((1, outputNeurons))

    print(f"Neural Network Architecture: \nInput Layer: {inputNeurons} X {hiddenLayerList[0]}")
    print(f"Hidden Layer 1: {hiddenLayerList[0]}  X {hiddenLayerList[1]}")
    print(f"Hidden Layer 2: {hiddenLayerList[1]}  X {hiddenLayerList[2]}")
    print(f"Output Layer: {hiddenLayerList[2]}  X {outputNeurons}")
    print(f"Output Layer: ", outputNeurons)

    return weight1, bias1, weight2, bias2, weight3, bias3, weight4, bias4

# def printArchitecture(w1, b1, w2, b2, w3, b3, w4, b4, w5, b5):
    
def tanh_activation(x):
    return ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))

def forwardPropagation(x, w1, b1, w2, b2, w3, b3, w4, b4):
    # input X weight1
    z1 = x.dot(w1) + b1
    # TanH activation
    a1 = tanh_activation(z1)
    
    # First Hidden Layer X Second Hidden Layer 
    z2 = a1.dot(w2) + b2
    # Activation at the second hidden layer
    a2 = tanh_activation(z2)

    # Second Hidden Layer x Third Hidden Layer
    z3 = a2.dot(w3) + b3
    # Activcation at the Third Hidden Layer
    a3 = tanh_activation(z3)

    # Third Hidden Layer X Output Layer
    z4 = a3.dot(w4) + b4

    # Softmax Output    
    expValues = np.exp(z4)

    softmax = expValues / np.sum(expValues, axis = 1, keepdims = True)

    return z1, a1, z2, a2, z3, a3, z4, softmax

def crossEntropyLoss(x, y, eta, w1, b1, w2, b2, w3, b3, w4, b4):
    z1, a1, z2, a2, z3, a3, z4, softmax = forwardPropagation(x, w1, b1, w2, b2, w3, b3, w4, b4)
    
    epsilon = 1e-15
    softmax = np.clip(softmax, epsilon, 1 - epsilon)

    loss = -np.log(softmax[range(x.shape[0]), y])
    totalLoss = np.sum(loss)
    totalLoss += eta / 2 * ( np.sum( np.square(w1)) + np.sum(np.square(w2)) + np.sum(np.square(w3)) + np.sum(np.square(w4)) )

    return 1./x.shape[0] * totalLoss

def predict(x, w1, b1, w2, b2, w3, b3, w4, b4):
    _, _, _, _, _, _, _, softmax = forwardPropagation(x, w1, b1, w2, b2, w3, b3, w4, b4)

    return np.argmax(softmax, axis = 1)

def backPropagation(w1, b1, w2, b2, w3, b3, w4, b4, x_train, y_train, x_test, y_test, eta, learningRate, epochs, beta):
    
    globalLoss_train = []
    globalLoss_test = []
    maxTrainingAccuracy = 0.0
    maxTestingAccuracy = 0.0
    train_accuracies = []
    test_accuracies = []

    mW1, mB1 = np.zeros_like(w1), np.zeros_like(b1)
    mW2, mB2 = np.zeros_like(w2), np.zeros_like(b2)
    mW3, mB3 = np.zeros_like(w3), np.zeros_like(b3)
    mW4, mB4 = np.zeros_like(w4), np.zeros_like(b4)
    
    for i in range(epochs):
        loss_history_train = 0.0
        loss_history_test = 0.0
        train_loss = 0.0

        z1, a1, z2, a2, z3, a3, z4, softmax = forwardPropagation(x_train, w1, b1, w2, b2, w3, b3, w4, b4)
        train_loss = crossEntropyLoss(x_train, y_train, eta, w1, b1, w2, b2, w3, b3, w4, b4)
        loss_history_train += train_loss

        # loss with respect to output
        delta4 = softmax.copy()
        delta4[range(x_train.shape[0]), y_train] -= 1 
        
        # output layer gradient
        dw4 = a3.T.dot(delta4)
        db4 = np.sum(delta4, axis = 0, keepdims = True)

        # dL / dw4
        delta3 = delta4.dot(w4.T) * (1 - np.power(a3, 2))
        dw3 = a2.T.dot(delta3)
        db3 = np.sum(delta3, axis = 0, keepdims = True)

        # dL / dw3
        delta2 = delta3.dot(w3.T) * (1 - np.power(a2, 2))
        dw2 = a1.T.dot(delta2)
        db2 = np.sum(delta2, axis = 0, keepdims = True)

        # dL / dw2
        delta1 = delta2.dot(w2.T) * (1 - np.power(a1, 2))
        dw1 = x_train.T.dot(delta1)
        db1 = np.sum(delta1, axis = 0, keepdims = True)

        # regularization
        dw4 += eta * w4
        dw3 += eta * w3
        dw2 += eta * w2
        dw1 += eta * w1

        # Adding Momentum
        mW1 = beta * mW1 + (1 - beta) * dw1
        mB1 = beta * mB1 + (1 - beta) * db1
        mW2 = beta * mW2 + (1 - beta) * dw2
        mB2 = beta * mB2 + (1 - beta) * db2
        mW3 = beta * mW3 + (1 - beta) * dw3
        mB3 = beta * mB3 + (1 - beta) * db3
        mW4 = beta * mW4 + (1 - beta) * dw4
        mB4 = beta * mB4 + (1 - beta) * db4

        # Gradient Descent
        w1 -= learningRate * mW1
        b1 -= learningRate * mB1
        w2 -= learningRate * mW2
        b2 -= learningRate * mB2        
        w3 -= learningRate * mW3
        b3 -= learningRate * mB3
        w4 -= learningRate * mW4
        b4 -= learningRate * mB4
        
        # Calculating testing loss
        test_loss = crossEntropyLoss(x_test, y_test, eta, w1, b1, w2, b2, w3, b3, w4, b4)
        loss_history_test += test_loss

        # Accuracy and Predictrions
        if( i % 50 == 0 ):
            predictions = predict(x_train, w1, b1, w2, b2, w3, b3, w4, b4)
            train_accuracy = nn.getAccuracy(predictions, y_train)
            train_accuracies.append(train_accuracy)
            maxTrainingAccuracy = max(maxTrainingAccuracy, train_accuracy)

            test_predictions = predict(x_test, w1, b1, w2, b2, w3, b3, w4, b4)
            test_accuracy = nn.getAccuracy(test_predictions, y_test)
            test_accuracies.append(test_accuracy)
            maxTestingAccuracy = max(maxTestingAccuracy, test_accuracy)

            globalLoss_test.append(loss_history_test/50)
            globalLoss_train.append(loss_history_train/50)
            print(f"Epoch {i}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Train Accuracy={train_accuracy:.4f}, Test Accuracy={test_accuracy:.4f}")
        
    return  w1, b1, w2, b2, w3, b3, w4, b4, globalLoss_train, globalLoss_test, maxTrainingAccuracy, maxTestingAccuracy

def neural_network_extended_5_layer(config, datasets, inputNeurons, hiddenLayerList, outputNeurons, beta):
    
    '''
        5 - Layer Neural Network Implementation
    '''
    
    w1, b1, w2, b2, w3, b3, w4, b4 = neuralNetworkInit(inputNeurons, hiddenLayerList, outputNeurons)

    eta = float(config['training']['eta'])
    learningRate = float(config['training']['learningRate'])
    epochs = 2500

    # Testing different datasets fold
    for i, (x_train, y_train, x_test, y_test) in enumerate(datasets):
        x_train = nn.normalizeInput(x_train)
        x_test = nn.normalizeInput(x_test)
        w1, b1, w2, b2, w3, b3, w4, b4, trainLoss, testLoss, maxTrainingAccuracy, maxTestingAccuracy = backPropagation(w1, b1, w2, b2, w3, b3, w4, b4, x_train, y_train, x_test, y_test, eta, learningRate, epochs, beta)
        nn.plotGraph(trainLoss, testLoss, 50, i)
        print(f"Training Maximum Accuracy: {maxTrainingAccuracy:.4f}")
        print(f"Testing Maximum Accuracy: {maxTestingAccuracy:.4f}")
        test_predictions = predict(x_test, w1, b1, w2, b2, w3, b3, w4, b4)
        nn.getConfusionMatrix(test_predictions, y_test)


    

