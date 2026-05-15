import numpy as np

import torch
import torch.nn as nn

from .backbone.resnet import resnet101
from .GraphNeuralNetwork import GatedGNN
from .SemanticDecoupling import SemanticDecoupling
from .Element_Wise_Layer import Element_Wise_Layer


class KGGR(nn.Module):

    def __init__(self, adjacencyMatrix, wordFeatures,
                 imageFeatureDim=1024, intermediaDim=512, outputDim=1024,
                 classNum=80, wordFeatureDim=300, timeStep=3):

        super(KGGR, self).__init__()

        self.backbone = resnet101(pretrained=True)

        if imageFeatureDim != 2048:
            self.changeChannel = nn.Sequential(nn.Conv2d(2048, imageFeatureDim, kernel_size=1, stride=1, bias=False),
                                               nn.BatchNorm2d(imageFeatureDim),)

        self.classNum = classNum
        self.timeStep = timeStep

        self.outputDim = outputDim
        self.intermediaDim = intermediaDim
        self.wordFeatureDim = wordFeatureDim
        self.imageFeatureDim = imageFeatureDim
        
        self.wordFeatures = self.load_features(wordFeatures)
        self.inMatrix, self.outMatrix = self.load_matrix(adjacencyMatrix)

        self.SemanticDecoupling = SemanticDecoupling(self.classNum, self.imageFeatureDim, self.wordFeatureDim, intermediaDim=self.intermediaDim)
        self.FeatureGraphNeuralNetwork = GatedGNN(self.imageFeatureDim, self.timeStep, self.inMatrix, self.outMatrix)
        self.SemanticGraphNeuralNetwork = GatedGNN(self.imageFeatureDim, self.timeStep, self.inMatrix, self.outMatrix)

        self.fc1 = nn.Linear(2 * self.imageFeatureDim, self.outputDim)
        self.fc2 = nn.Linear(2 * self.outputDim, self.outputDim)
        self.classifiers = Element_Wise_Layer(self.classNum, self.outputDim)

    def forward(self, input):

        batchSize = input.size(0)

        featureMap = self.backbone(input)                                            # (BatchSize, Channel, imgSize, imgSize)
        if featureMap.size(1) != self.imageFeatureDim:
            featureMap = self.changeChannel(featureMap)                              # (BatchSize, imgFeatureDim, imgSize, imgSize)

        semanticFeature = self.SemanticDecoupling(featureMap, self.wordFeatures)[0]  # (BatchSize, classNum, imgFeatureDim)
        feature = self.FeatureGraphNeuralNetwork(semanticFeature)                           # (BatchSize, classNum, imgFeatureDim)
        
        # Predict Category
        output = torch.tanh(self.fc1(torch.cat((feature.view(batchSize * self.classNum, -1),
                                                semanticFeature.view(-1, self.imageFeatureDim)),1)))
        weight0 = self.classifiers.weight
        weight = self.SemanticGraphNeuralNetwork(weight0.unsqueeze(0))
        weight = self.fc2(torch.cat((weight.view(self.classNum, -1), weight0), 1))
        weight = weight.contiguous().view(1, self.classNum, self.outputDim)
        output = output.contiguous().view(batchSize, self.classNum, self.outputDim)
        output = output * weight
        result = self.classifiers(output)                                            # (BatchSize, classNum)

        return result

    def load_features(self, wordFeatures):
        return nn.Parameter(torch.from_numpy(wordFeatures.astype(np.float32)), requires_grad=False)

    def load_matrix(self, mat):
        _in_matrix, _out_matrix = mat.astype(np.float32), mat.T.astype(np.float32)
        _in_matrix, _out_matrix = nn.Parameter(torch.from_numpy(_in_matrix), requires_grad=False), nn.Parameter(torch.from_numpy(_out_matrix), requires_grad=False)
        return _in_matrix, _out_matrix
