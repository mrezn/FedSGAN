import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


class GetDataSet():
    def __init__(self, dataSetName, ):
        self.dataSetName = dataSetName

        self.trainData = None
        self.trainLabel = None
        self.trainDataSize = None

        self.testData = None
        self.testLabel = None
        self.testDataSize = None

        self.dataDistribution()


    def dataDistribution(self, ):
        if self.dataSetName == 'MNIST':
            trainData = []
            trainLabel = []
            trainTemp = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
            self.trainDataSize = len(trainTemp)
            for i in range(len(trainTemp)):
                trainData.append(trainTemp[i][0].numpy().tolist())
                trainLabel.append(trainTemp[i][1])
            testData = []
            testLabel = []
            testTemp = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())
            self.testDataSize = len(testTemp)
            for i in range(len(testTemp)):
                testData.append(testTemp[i][0].numpy().tolist())
                testLabel.append(testTemp[i][1])
            self.trainData = torch.tensor(trainData)
            self.trainLabel = torch.tensor(trainLabel)
            self.testData = torch.tensor(testData)
            self.testLabel = torch.tensor(testLabel)

        elif self.dataSetName == 'CIFAR10':
            trainData = []
            trainLabel = []
            trainTemp = datasets.CIFAR10(root="data", train=True, download=True, transform=ToTensor())
            self.trainDataSize = len(trainTemp)
            for i in range(len(trainTemp)):
                trainData.append(trainTemp[i][0].numpy().tolist())
                trainLabel.append(trainTemp[i][1])
            testData = []
            testLabel = []
            testTemp = datasets.CIFAR10(root="data", train=False, download=True, transform=ToTensor())
            self.testDataSize = len(testTemp)
            for i in range(len(testTemp)):
                testData.append(testTemp[i][0].numpy().tolist())
                testLabel.append(testTemp[i][1])
            self.trainData = torch.tensor(trainData)
            self.trainLabel = torch.tensor(trainLabel)
            self.testData = torch.tensor(testData)
            self.testLabel = torch.tensor(testLabel)
        
        else:
            print("Not valid dataset name!")

        # self.trainData = DataLoader(trainingData, batch_size=self.trainDataSize)
        # self.testData = DataLoader(testData, batch_size=self.testDataSize)
