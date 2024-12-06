import configparser
import numpy as np

def readConfig():
    config = configparser.ConfigParser()
    config.read('config.conf')
    return config

def loadFile(loadDataFile, data):
    with open(loadDataFile, 'r', encoding='utf-8') as inputFile:
        readLines = inputFile.readlines()

        samples, bins = readLines[0].strip().split(' ')

        for line in readLines:
            data += line
    return samples, bins, data

def getConfigurations(config=readConfig()):
    inputNeurons = int(config['neural']['inputNeurons'])
    numberOfHL = int(config['neural']['numberOfHiddenLayer'])

    hiddenLayers = []
    try:
        for i in range(numberOfHL):
            hl = f"hiddenLayer{i+1}"
            hiddenLayers.append(int(config['neural'][hl]))
    except Exception as e:
        print("Error: No", e)

    outputNeurons = int(config['neural']['outputNeurons'])

    return inputNeurons, numberOfHL, outputNeurons, hiddenLayers

def numpyConversion(dataStr, binSize):
    cleanStr = dataStr.rstrip("\x1a")
    lines = cleanStr.strip().split('\n')[1:]

    x = []
    y = []

    for line in lines:
        values = list(map(float, line.split()))
        y.append(values[0])
        x.append(values[1:])

    y = np.array(y)
    x = np.array([row[:binSize] for row in x])
    
    y = y.astype(int)
    print(x.shape)

    return x, y