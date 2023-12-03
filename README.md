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
| SSGRL |  -  |  -  |  -  |  [link]()  |
| ML-GCN |  -  |  -  |  -  |  [link]()  |
| ASL |  -  |  -  |  -  |  [link]()  |
| P-GCN |  -  |  -  |  -  |  [link]()  |
| CSRA |  -  |  -  |  -  |  [link]()  |
| KGGR |  -  |  -  |  -  |  [link]()  |
| DBMLCL |  -  |  -  |  -  |  [link]()  |