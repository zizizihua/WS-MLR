# WS-MLR
Webly Supervised Multi-Label Recognition: Evaluation Benchmark and Dual-Branch Multi-Label Contrastive Learning.

## Preliminary
1. Download WebML data [[baidu cloud](https://pan.baidu.com/s/1Ipue3jpsFfqcUOf8JZJBTw?pwd=hjt3): hjt3].
2. Donwload corresponding datasets (eg. COCO2014) for validation.
3. Modify the lines 16-18 in config.py.

## Usage
```
# modify experiment settings in scripts, than run the script

./scripts/DBMLCL.sh
```

## Pretrained Models
All models are trained on WebCOCO dataset and test on coco2014 val.

| model | mAP | CF1 | OF1 | weights |
|  ---  | --- | --- | --- |   ---   |
| SSGRL |  64.7  |  57.3  |  60.8  |  [link](https://github.com/zizizihua/WS-MLR/releases/download/v1.0.0/SSGRL-webcoco.pth)  |
| ML-GCN |  69.4  |  64.7  |  53.5  |  [link](https://github.com/zizizihua/WS-MLR/releases/download/v1.0.0/ML-GCN-webcoco.pth)  |
| ASL |  68.9  |  60.7  |  60.9  |  [link](https://github.com/zizizihua/WS-MLR/releases/download/v1.0.0/ASL-webcoco.pth)  |
| P-GCN |  69.4  |  64.5  |  66.0  |  [link](https://github.com/zizizihua/WS-MLR/releases/download/v1.0.0/P-GCN-webcoco.pth)  |
| CSRA |  70.2  |  65.7  |  66.2  |  [link](https://github.com/zizizihua/WS-MLR/releases/download/v1.0.0/CSRA-webcoco.pth)  |
| KGGR |  67.8  |  61.5  |  63.6  |  [link](https://github.com/zizizihua/WS-MLR/releases/download/v1.0.0/KGGR-webcoco.pth)  |
| DBMLCL |  71.4  |  66.5  |  66.7  |  [link](https://github.com/zizizihua/WS-MLR/releases/download/v1.0.0/DBMLCL-webcoco.pth)  |