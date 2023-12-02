import os
import gc
import shutil

import torch


def load_pretrained_model(model, args):
    pretrainedModel = torch.load(args.pretrainedModel)['state_dict']
    model.load_state_dict(pretrainedModel)

    return model


def save_code_file(args):
    return

    prefixPath = os.path.join('exp/code/', args.post)
    if os.path.exists(prefixPath):
        shutil.rmtree(prefixPath)
    os.makedirs(prefixPath)

    fileNames = ['scripts/{}.sh', '{}.py', 'model/{}.py', 'loss/{}.py', 'config.py']

    for fileName in fileNames:
        fileName = fileName.format(args.mode)
        if not os.path.exists(fileName):
            continue
        dstFile = os.path.join(prefixPath, fileName)
        os.makedirs(os.path.dirname(dstFile), exist_ok=True)
        shutil.copyfile(fileName, dstFile)


def save_checkpoint(args, state, name='last'):

    outputPath = os.path.join('exp/checkpoint/', args.post)
    os.makedirs(outputPath, exist_ok=True)

    torch.save(state, os.path.join(outputPath, name+'.pth'))
