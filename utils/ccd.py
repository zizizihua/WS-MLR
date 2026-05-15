import torch
import torch.nn.functional as F


PASCAL_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'dining table', 'dog', 'horse', 'motorbike', 'person',
    'potted plant', 'sheep', 'sofa', 'train', 'tv monitor'
]

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

TEMPLATES = ['a photo of the {}']


def get_class_names(dataset):
    if dataset in ('WebPascal', 'DualWebPascal', 'VOC2007'):
        return PASCAL_CLASSES
    if dataset in ('COCO2014', 'DualCOCO2014', 'WebCOCO', 'DualWebCOCO'):
        return COCO_CLASSES
    raise ValueError('Unsupported CCD dataset: {}'.format(dataset))


def build_text_weights(clip_module, clip_model, class_names, device):
    text_weights = []
    with torch.no_grad():
        for className in class_names:
            texts = [template.format(className) for template in TEMPLATES]
            texts = clip_module.tokenize(texts).to(device)
            text_features = clip_model.encode_text(texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.mean(dim=0)
            text_features = text_features / text_features.norm()
            text_weights.append(text_features)

    return torch.stack(text_weights, dim=0).to(device)


def clip_softscore(clip_model, image, text_weights, height, width):
    image_features = clip_model.encode_image(image, height, width)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    similarity = 100.0 * image_features @ text_weights.T
    return torch.softmax(similarity, dim=1).float()


def cropping_box(image, pos1, pos2, offset):
    y1, x1 = pos1
    y2, x2 = pos2
    height, width = image.shape[-2], image.shape[-1]

    ymin, ymax = sorted((int(y1), int(y2)))
    xmin, xmax = sorted((int(x1), int(x2)))

    ymin = max(ymin - offset, 0)
    ymax = min(ymax + offset, height)
    xmin = max(xmin - offset, 0)
    xmax = min(xmax + offset, width)

    cropy = max(ymax - ymin, 1)
    cropx = max(xmax - xmin, 1)
    scale = 640 / max(cropy, cropx)

    cropped_img = F.interpolate(
        image[:, :, ymin:ymax, xmin:xmax],
        (max(int(cropy * scale), 1), max(int(cropx * scale), 1)),
        mode='bilinear',
        align_corners=False)

    return cropped_img, (xmin, ymin, xmax, ymax)
