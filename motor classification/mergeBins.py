import configparser

'''
    *  Merging two or more bins.
'''

def mergeBins(getBinNumbers):

    totalSamples = 0
    binSize = 0
    newBinFile = ""
    features = []

    for i in range(getBinNumbers):
        binFile = config['data'][f'bin{i}']
        binPath = f'{binFile}'

        with open(binPath, "r") as f:
            lines = f.readline().strip().split(' ')

            print(lines)

            samples = int(lines[0])
            bins = int(lines[1])

            totalSamples += int(samples)
            binSize += bins

            for j, line in enumerate(f):
                data = line.split()
                if i == 0:
                    features.append(line.strip())
                else:
                    if j < len(features):
                        str1 = features[j]
                        str2 = " ".join(data[1:])
                        str1 += " " + str2
                        features[j] = str1
                    else:
                        break

    newBinFile = "\n".join(features)
    print(newBinFile)
    newBin = str(samples) + " " + str(binSize) + "\n" + newBinFile
    return newBin, samples, binSize


# Main Function 
if __name__ == '__main__':
    # Read Config File
    config = configparser.ConfigParser()
    config.read('config.conf')

    # Variable Declaration
    getBinNumbers = config.getint('data', 'binNumbers')

    # New Bin file, number of sample size, number of bin size returned
    newBin, samples, binSize = mergeBins(getBinNumbers)

    newBinName = f"L30fft_{binSize}.out"
    with open(newBinName, "w") as outputFile:
        outputFile.write(newBin)