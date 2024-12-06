import neuralNetwork as nn
import configFunc as cf
import neuralNetworkExtended as nnExtended

if __name__ == '__main__':
    # reading config file
    config = cf.readConfig()

    # loading data file path from config file
    loadDataFile = config['data']['loadDataFile']

    data = ""

    # loading the data, samples, and bin size into a variable.
    samples, bins, loadData = cf.loadFile(loadDataFile, data)

    # loading the data into numpy array.
    # numpyConversion(loadData)
    x, y = cf.numpyConversion(loadData, int(bins))

    ''' 
        K-fold-3 dataset - seperates three dataset of good and bad motors and 
        merges two into training and one for testing
    '''
    datasets = nn.trainXtestSet(x, y)
    
    # number of input neurons is the number of features
    # Neural Network Inputs
    hiddenLayerList = []
    inputNeurons, numberOfHiddenLayers, outputNeurons, hiddenLayerList = cf.getConfigurations() 

    beta = float(config['training']['beta'])

    '''
        3 - Layer Neural Network Implementation
    '''
    if numberOfHiddenLayers == 1:
        nn.neural_network_3_layer_architecture(config, datasets, inputNeurons, hiddenLayerList, outputNeurons, beta)
    elif numberOfHiddenLayers == 3:
        nnExtended.neural_network_extended_5_layer(config, datasets, inputNeurons, hiddenLayerList, outputNeurons, beta)
    else:
        print("Please Change Neural Network's Hidden Layer to 1 or 3...")

