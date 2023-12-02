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
prefixPathCOCO2014 = '../datasets/coco2014/'
prefixPathWebML = '../datasets/web_ml/'
prefixPathVOC2007 = '../datasets/VOCdevkit/VOC2007/'
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
    assert mode in ('Base', 'DBMLCL')

    parser = argparse.ArgumentParser(description='HCP Multi-label Image Recognition with Partial Labels')

    # Basic Augments
    parser.add_argument('--post', type=str, default='', help='postname of save model')
    parser.add_argument('--printFreq', type=int, default='1000', help='number of print frequency (default: 1000)')

    parser.add_argument('--mode', type=str, default='Base', choices=['Base', 'DBMLCL'], help='mode of experiment (default: Base)')
    parser.add_argument('--dataset', type=str, default='COCO2014', choices=list(_ClassNum.keys()), help='dataset for training and testing')
    parser.add_argument('--ann_file', type=str, help='training annotation file path')

    parser.add_argument('--pretrainedModel', type=str, default='None', help='path to pretrained model (default: None)')
    parser.add_argument('--resumeModel', type=str, default='None', help='path to resume model (default: None)')
    parser.add_argument('--evaluate', type=str2bool, default='False', help='whether to evaluate model (default: False)')

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
   
    args = parser.parse_args()
    args.classNum = _ClassNum[args.dataset]    

    return args
# =============================================================================
