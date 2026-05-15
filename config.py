"""
Configuration file!
"""

import logging
import warnings
import argparse

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Dataset Path
# =============================================================================
prefixPathCOCO2014 = '../../datasets/COCO2014/'
prefixPathWebML = '../../datasets/Web_ML/'
prefixPathVOC2007 = '../../datasets/VOCdevkit/VOC2007/'
# =============================================================================

# ClassNum of Dataset
# =============================================================================
_ClassNum = {'COCO2014': 80,
             'DualCOCO2014': 80,
             'WebCOCO': 80,
             'DualWebCOCO': 80,
             'WebPascal': 20,
             'DualWebPascal': 20,
             'VOC2007': 20,
            }
# =============================================================================


# Argument Parse
# =============================================================================
def str2bool(input):
    if isinstance(input, bool):
        return input

    if input.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def show_args(args):

    logger.info("==========================================")
    logger.info("==========       CONFIG      =============")
    logger.info("==========================================")

    for arg, content in args.__dict__.items():
        logger.info("{}: {}".format(arg, content))

    logger.info("==========================================")
    logger.info("===========        END        ============")
    logger.info("==========================================")

    logger.info("\n")


def arg_parse(mode):
    assert mode in ('Base', 'DBMLCL', 'CCD')

    parser = argparse.ArgumentParser(description='HCP Multi-label Image Recognition with Partial Labels')

    # Basic Augments
    parser.add_argument('--post', type=str, default='', help='postname of save model')
    parser.add_argument('--printFreq', type=int, default='1000', help='number of print frequency (default: 1000)')

    parser.add_argument('--mode', type=str, default=mode, choices=['Base', 'DBMLCL', 'CCD'], help='mode of experiment (default: Base)')
    parser.add_argument('--dataset', type=str, default='COCO2014', choices=list(_ClassNum.keys()), help='dataset for training and testing')
    parser.add_argument('--ann_file', type=str, help='training annotation file path')

    parser.add_argument('--pretrainedModel', type=str, default='None', help='path to pretrained model (default: None)')
    parser.add_argument('--resumeModel', type=str, default='None', help='path to resume model (default: None)')
    parser.add_argument('--evaluate', type=str2bool, default='False', help='whether to evaluate model (default: False)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'npu', 'cpu'], help='device type for training and testing')
    parser.add_argument('--deviceIds', type=str, default='0', help='visible device ids, e.g. 0 or 0,1')

    parser.add_argument('--epochs', type=int, default=20, help='number of total epochs to run (default: 20)')
    parser.add_argument('--startEpoch', type=int, default=0, help='manual epoch number (default: 0)')
    parser.add_argument('--stepEpoch', type=int, default=10, help='decend the lr in epoch number (default: 10)')

    parser.add_argument('--batchSize', type=int, default=8, help='mini-batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weightDecay', type=float, default=1e-4, help='weight decay (default: 0.0001)')
    parser.add_argument('--warmupEpoch', type=int, default=1, help='warmup epoch')

    parser.add_argument('--cropSize', type=int, default=448, help='size of crop image')
    parser.add_argument('--scaleSize', type=int, default=512, help='size of rescale image')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 4)')

    if mode == 'DBMLCL':
        parser.add_argument('--prototypeNum', type=int, default=10, help='number of prototype clusters per class')
        parser.add_argument('--cleanEpoch', type=int, default=5, help='epoch to start label cleaning')
        parser.add_argument('--theta1', type=float, default=0.8, help='positive score thresh for label cleaning')
        parser.add_argument('--theta2', type=float, default=0.01, help='negative score thresh for label cleaning')
        parser.add_argument('--alpha', type=float, default=0.5, help='balance factor for cls score and proto score')
        parser.add_argument('--instLossWeight', type=float, default=0.01, help='intra loss weight')
        parser.add_argument('--protoLossWeight', type=float, default=1, help='intra loss weight')
    
    if mode == 'CCD':
        parser.add_argument('--clipModelPath', type=str, default='./pretrained/RN50x64.pt', help='path to CLIP RN50x64 checkpoint')
        parser.add_argument('--lrMult', type=float, default=10, help='learning rate multiplier for CCD 1x1 conv head')
        parser.add_argument('--offsetSize', type=int, default=40, help='offset size for CAM crop boxes')
        parser.add_argument('--coeff', type=float, default=0.8, help='CCD alpha coefficient')
        parser.add_argument('--lossCoeff', type=float, default=0.1, help='loss coefficient for uncertain samples')
        parser.add_argument('--ratio', type=float, default=0.8, help='EMA ratio for pseudo label updates')
        parser.add_argument('--updateLabel', type=str2bool, default=True, help='whether to update pseudo labels with CAM crops')
        parser.add_argument('--infNum', type=int, default=1, help='number of epochs to run CCD inference updates after epoch 2')
        parser.add_argument('--globalTemp', type=str2bool, default=True, help='whether to use global CLIP temperature correction')
        parser.add_argument('--localTemp', type=str2bool, default=True, help='whether to use local CLIP temperature correction')
        parser.add_argument('--useConsist', type=float, default=1, help='weight of consistency loss')
        parser.add_argument('--LS', type=str2bool, default=False, help='whether to apply label smoothing floor')
        parser.add_argument('--LSCoeff', type=int, default=80, help='inverse label smoothing floor coefficient')
        parser.add_argument('--bound', type=int, default=4, help='target lower bound for multiplier calibration')
        parser.add_argument('--freezeBackbone', type=str2bool, default=False, help='whether to freeze CCD backbone')
   
    args = parser.parse_args()
    args.classNum = _ClassNum[args.dataset]    

    return args
# =============================================================================
