import os
import json
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.utils.data as data
from pycocotools.coco import COCO as COCO_


class COCO(data.Dataset):

    def __init__(self, mode,
                 image_dir, anno_path,
                 input_transform=None,
                 input_transform2=None):

        assert mode in ('train', 'val')

        self.mode = mode
        self.input_transform = input_transform
        self.input_transform2 = input_transform2

        self.root = image_dir
        self.coco = COCO_(anno_path)
        self.ids = list(self.coco.imgs.keys())
     
        self.category_map = {id: i for i, id in enumerate(self.coco.cats.keys())}

        # labels : numpy.ndarray, shape->(len(coco), 80)
        # value range->(0 means label don't exist, 1 means label exist)
        self.labels = []
        for i in range(len(self.ids)):
            img_id = self.ids[i]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)
            self.labels.append(getLabelVector(getCategoryList(target), self.category_map))
        self.labels = np.array(self.labels)

    def __getitem__(self, index):
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        input_ = Image.open(os.path.join(self.root, path)).convert('RGB')
        input = self.input_transform(input_.copy())
        if self.input_transform2 is not None:
            input2 = self.input_transform2(input_.copy())
            return index, input, input2, self.labels[index]

        return index, input, self.labels[index]

    def __len__(self):
        return len(self.ids)

# =============================================================================
# Help Functions
# =============================================================================
def getCategoryList(item):
    categories = set()
    for t in item:
        categories.add(t['category_id'])
    return list(categories)


def getLabelVector(categories, category_map):
    label = np.zeros(len(category_map))
    for c in categories:
        label[category_map[c]] = 1.0
    return label

