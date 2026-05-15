import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.resnet import resnet101
from .GraphNeuralNetwork import GatedGNN
from .SemanticDecoupling import SemanticDecoupling
from .Element_Wise_Layer import Element_Wise_Layer


class Normalize(nn.Module):
    def __init__(self, power=2, dim=-1):
        super(Normalize, self).__init__()
        self.power = power
        self.dim = dim

    def forward(self, x):
        norm = x.pow(self.power).sum(self.dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class SSGRL_Proto(nn.Module):

    def __init__(self, adjacencyMatrix, wordFeatures,
                 inputDim=2048, imageFeatureDim=2048, intermediaDim=1024, outputDim=2048,
                 classNum=80, wordFeatureDim=300, timeStep=3,
                 proto_momentum=0.999, low_dim=128, queue_len=8192,
                 temperature=0.1, theta1=1., theta2=0., alpha1=1., alpha2=0., dtl=True, dtl_mode='high'):

        super(SSGRL_Proto, self).__init__()

        self.backbone = resnet101(pretrained=True)

        if imageFeatureDim != inputDim:
            self.changeChannel = nn.Sequential(nn.Conv2d(inputDim, imageFeatureDim, kernel_size=1, stride=1, bias=False),
                                               nn.BatchNorm2d(imageFeatureDim),)

        self.classNum = classNum
        self.timeStep = timeStep

        self.outputDim = outputDim
        self.intermediaDim = intermediaDim
        self.wordFeatureDim = wordFeatureDim
        self.imageFeatureDim = imageFeatureDim
        self.low_dim = low_dim
        self.proto_momentum = proto_momentum
        self.queue_len = queue_len
        self.temperature = temperature
        
        self.load_features(wordFeatures)
        self.load_matrix(adjacencyMatrix)

        self.SemanticDecoupling = SemanticDecoupling(self.classNum, self.imageFeatureDim, self.wordFeatureDim, intermediaDim=self.intermediaDim)
        self.GraphNeuralNetwork = GatedGNN(self.imageFeatureDim, self.timeStep, self.inMatrix, self.outMatrix)
        self.fc = nn.Linear(2 * self.imageFeatureDim, self.outputDim)
        self.classifiers = Element_Wise_Layer(self.classNum, self.outputDim)

        self.register_buffer("queue", torch.randn(self.low_dim, self.classNum, self.queue_len))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(self.classNum, dtype=torch.int64))
        
        self.register_buffer('prototypes', torch.zeros(self.classNum, self.low_dim))

        self.mlp = nn.Sequential(
            nn.Linear(self.imageFeatureDim, self.imageFeatureDim),
            nn.ReLU(),
            nn.Linear(self.imageFeatureDim, self.low_dim),
            Normalize())
        
        if dtl:
            self.register_buffer('theta1', torch.full([classNum], theta1))
            self.register_buffer('theta2', torch.full([classNum], theta2))
            self.register_buffer('alpha1', torch.full([classNum], alpha1))
            self.register_buffer('alpha2', torch.full([classNum], alpha2))
        else:
            self.theta1 = theta1
            self.theta2 = theta2
            self.alpha1 = alpha1
            self.alpha2 = alpha2
        
        self.dtl_mode = dtl_mode

    def forward(self, input):

        batch_size = input.size(0)

        img_feature = self.backbone(input)                                            # (BatchSize, Channel, imgSize, imgSize)
        if img_feature.size(1) != self.imageFeatureDim:
            img_feature = self.changeChannel(img_feature)                              # (BatchSize, imgFeatureDim, imgSize, imgSize)

        semantic_feature = self.SemanticDecoupling(img_feature, self.wordFeatures)[0]  # (BatchSize, classNum, imgFeatureDim)
        graph_feature = self.GraphNeuralNetwork(semantic_feature)                           # (BatchSize, classNum, imgFeatureDim)
        
        # Predict Category
        fused_feature = self.fc(torch.cat((
            graph_feature.view(batch_size * self.classNum, -1),
            semantic_feature.view(-1, self.imageFeatureDim)),1))

        fused_feature = fused_feature.contiguous().view(batch_size, self.classNum, self.outputDim)
        cls_logits = self.classifiers(fused_feature)                                            # (BatchSize, classNum)

        semantic_feature = self.mlp(semantic_feature.view(-1, self.imageFeatureDim))
        semantic_feature = semantic_feature.contiguous().view(batch_size, self.classNum, self.low_dim)

        return cls_logits, semantic_feature

    def load_features(self, wordFeatures):
        self.wordFeatures = nn.Parameter(torch.from_numpy(wordFeatures.astype(np.float32)), requires_grad=False)

    def load_matrix(self, mat):
        _in_matrix, _out_matrix = mat.astype(np.float32), mat.T.astype(np.float32)
        _in_matrix, _out_matrix = nn.Parameter(torch.from_numpy(_in_matrix), requires_grad=False), nn.Parameter(torch.from_numpy(_out_matrix), requires_grad=False)
        self.inMatrix, self.outMatrix = _in_matrix, _out_matrix
    
    @torch.no_grad()
    def dequeue_and_enqueue(self, keys, gt_labels):
        # gather keys before updating queue
        keys = keys.detach().clone()

        batch_size = keys.shape[0]
        for b in range(batch_size):
            for i in range(self.classNum):
                if gt_labels[b, i] == 1:
                    continue
                
                ptr = int(self.queue_ptr[i])

                # replace the keys at ptr (dequeue and enqueue)
                self.queue[:, i, ptr] = keys[b, i]
                ptr = (ptr + 1) % self.queue_len  # move pointer

                self.queue_ptr[i] = ptr

    @torch.no_grad()
    def update_prototype(self, features, gt_labels):
        for feat, label in zip(features, gt_labels):
            for i in range(self.classNum):
                if label[i] == 1:
                    self.prototypes[i, :] = self.prototypes[i, :]*self.proto_momentum + (1-self.proto_momentum)*feat[i]

        # self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
    
    @torch.no_grad()
    def update_thresh(self, cls_score, similarity, gt_label):
        for i in range(self.classNum):
            pos_mask = gt_label[:, i] > 0
            neg_mask = ~pos_mask
            if pos_mask.any():
                if self.dtl_mode == 'high':
                    pos_score = torch.sort(cls_score[pos_mask, i], descending=True).values
                    theta1_ = pos_score[:max(1, round(len(pos_score)*0.5))].mean()
                    pos_sim = torch.sort(similarity[pos_mask, i], descending=True).values
                    alpha1_ = pos_sim[:max(1, round(len(pos_sim)*0.5))].mean()
                else:
                    theta1_ = cls_score[pos_mask, i].mean()
                    alpha1_ = similarity[pos_mask, i].mean()
                self.theta1[i] = self.theta1[i]*self.proto_momentum + (1-self.proto_momentum)*theta1_
                self.alpha1[i] = self.alpha1[i]*self.proto_momentum + (1-self.proto_momentum)*alpha1_
            if neg_mask.any():
                if self.dtl_mode == 'high':
                    neg_score = torch.sort(cls_score[neg_mask, i], descending=False).values
                    theta2_ = neg_score[:max(1, round(len(neg_score)*0.5))].mean()
                    neg_sim = torch.sort(similarity[neg_mask, i], descending=False).values
                    alpha2_ = neg_sim[:max(1, round(len(neg_sim)*0.5))].mean()
                else:
                    theta2_ = cls_score[neg_mask, i].mean()
                    alpha2_ = similarity[neg_mask, i].mean()
                self.theta2[i] = self.theta2[i]*self.proto_momentum + (1-self.proto_momentum)*theta2_
                self.alpha2[i] = self.alpha2[i]*self.proto_momentum + (1-self.proto_momentum)*alpha2_
    
    def test(self, img, return_proto=True):
        cls_logits, semantic_feature = self(img)
        cls_score = cls_logits.sigmoid()

        if return_proto:
            prototypes = self.prototypes.unsqueeze(0).repeat(cls_logits.shape[0], 1, 1)
            similarity = F.cosine_similarity(semantic_feature, prototypes, dim=2)
            return cls_score, similarity
        
        return cls_score
