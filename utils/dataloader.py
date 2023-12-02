import os
import PIL
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.coco import COCO

from config import prefixPathCOCO2014, prefixPathWebML, prefixPathVOC2007

def get_graph_file(labels):

    graph = np.zeros((labels.shape[1], labels.shape[1]), dtype=np.float)

    for index in range(labels.shape[0]):
        indexs = np.where(labels[index] == 1)[0]
        for i in indexs:
            for j in indexs:
                graph[i, j] += 1

    for i in range(labels.shape[1]):
        graph[i] /= graph[i, i]

    np.nan_to_num(graph)

    return graph


def get_graph_and_word_file(args, labels):
    if args.dataset == 'COCO2014':
        WordFilePath = './data/coco/vectors.npy'

    elif args.dataset == 'DualCOCO2014':
        WordFilePath = './data/coco/vectors.npy'
    
    elif args.dataset == 'WebCOCO':
        WordFilePath = './data/coco/vectors.npy'
    
    elif args.dataset == 'DualWebCOCO':
        WordFilePath = './data/coco/vectors.npy'
    
    elif args.dataset == 'WebPascal':
        WordFilePath = './data/voc_devkit/VOC2007/voc07_vector.npy'
    
    elif args.dataset == 'DualWebPascal':
        WordFilePath = './data/voc_devkit/VOC2007/voc07_vector.npy'
        
    GraphFile = get_graph_file(labels)
    WordFile = np.load(WordFilePath)

    return GraphFile, WordFile


def get_data_path(dataset, ann_file):

    if dataset == 'COCO2014':
        prefixPath = prefixPathCOCO2014
        train_dir, train_anno = os.path.join(prefixPath, 'train2014'), os.path.join(prefixPath, ann_file)
        test_dir, test_anno = os.path.join(prefixPath, 'val2014'), os.path.join(prefixPath, 'annotations/instances_val2014.json')
    
    elif dataset == 'DualCOCO2014':
        prefixPath = prefixPathCOCO2014
        ann_file = ann_file.split(',')
        train_dir, train_anno = os.path.join(prefixPath, 'train2014'), (os.path.join(prefixPath, ann_file[0]), os.path.join(prefixPath, ann_file[1]))
        test_dir, test_anno = os.path.join(prefixPath, 'val2014'), os.path.join(prefixPath, 'annotations/instances_val2014.json')
    
    elif dataset == 'WebCOCO':
        train_dir, train_anno = prefixPathWebML, os.path.join(prefixPathWebML, ann_file)
        test_dir, test_anno = os.path.join(prefixPathCOCO2014, 'val2014'), os.path.join(prefixPathCOCO2014, 'annotations/instances_val2014.json')
    
    elif dataset == 'DualWebCOCO':
        ann_file = ann_file.split(',')
        train_dir, train_anno = prefixPathWebML, (os.path.join(prefixPathWebML, ann_file[0]), os.path.join(prefixPathWebML, ann_file[1]))
        test_dir, test_anno = os.path.join(prefixPathCOCO2014, 'val2014'), os.path.join(prefixPathCOCO2014, 'annotations/instances_val2014.json')
    
    elif dataset == 'WebPascal':
        train_dir, train_anno = prefixPathWebML, os.path.join(prefixPathWebML, ann_file)
        test_dir, test_anno = prefixPathVOC2007, os.path.join(prefixPathVOC2007, 'test2007.json')
    
    elif dataset == 'DualWebPascal':
        ann_file = ann_file.split(',')
        train_dir, train_anno = prefixPathWebML, (os.path.join(prefixPathWebML, ann_file[0]), os.path.join(prefixPathWebML, ann_file[1]))
        test_dir, test_anno = prefixPathVOC2007, os.path.join(prefixPathVOC2007, 'test2007.json')

    return train_dir, train_anno, test_dir, test_anno


def get_data_loader(args):
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    randomCropList = [transforms.RandomCrop(Size) for Size in [640, 576, 512, 448, 384, 320]] if args.scaleSize == 640 else \
                     [transforms.RandomCrop(Size) for Size in [512, 448, 384, 320, 256]]
    train_data_transform = transforms.Compose([transforms.Resize((args.scaleSize, args.scaleSize), interpolation=PIL.Image.BICUBIC),
                                               transforms.RandomChoice(randomCropList),
                                               transforms.Resize((args.cropSize, args.cropSize), interpolation=PIL.Image.BICUBIC),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               normalize])
    
    test_data_transform = transforms.Compose([transforms.Resize((args.cropSize, args.cropSize), interpolation=PIL.Image.BICUBIC),
                                              transforms.ToTensor(),
                                              normalize])
 
    train_dir, train_anno, test_dir, test_anno = get_data_path(args.dataset, args.ann_file)

    if args.dataset in ('COCO2014', 'WebCOCO', 'WebPascal'):  
        print(f"==> Loading {args.dataset}...")
        train_set = COCO('train',
                             train_dir, train_anno,
                             input_transform=train_data_transform)
        test_set = COCO('val',
                            test_dir, test_anno,
                            input_transform=test_data_transform)
        train_loader = DataLoader(dataset=train_set,
                              num_workers=args.workers,
                              batch_size=args.batchSize,
                              pin_memory=True,
                              drop_last=True,
                              shuffle=True)
        test_loader = DataLoader(dataset=test_set,
                             num_workers=args.workers,
                             batch_size=args.batchSize,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=False)

        return train_loader, test_loader
    elif args.dataset in ('DualCOCO2014', 'DualWebCOCO', 'DualWebPascal'):
        print(f"==> Loading {args.dataset}...")

        train_set1 = COCO('train',
                                 train_dir, train_anno[0],
                                 input_transform=train_data_transform)
        train_set2 = COCO('train',
                                 train_dir, train_anno[1],
                                 input_transform=train_data_transform)
        test_set = COCO('val',
                            test_dir, test_anno,
                            input_transform=test_data_transform)

        train_loader1 = DataLoader(dataset=train_set1,
                              num_workers=args.workers,
                              batch_size=args.batchSize,
                              pin_memory=True,
                              drop_last=True,
                              shuffle=True)
        train_loader2 = DataLoader(dataset=train_set2,
                              num_workers=args.workers,
                              batch_size=args.batchSize,
                              pin_memory=True,
                              drop_last=True,
                              shuffle=True)
        test_loader = DataLoader(dataset=test_set,
                             num_workers=args.workers,
                             batch_size=args.batchSize,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=False)

        return train_loader1, train_loader2, test_loader

    
