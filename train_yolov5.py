import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from copy import deepcopy
import time
import random
import glob
import json
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision

# device 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# seed 설정
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

if device == 'cuda':
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

train_dir = './train'
test_dir = './test'
annotation_file = './instances_default.json'


class COCODataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        img_path = os.path.join(self.root, path)

        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            return None, None

        img = Image.open(img_path).convert('RGB')

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return tuple(zip(*batch))


# 데이터 변환 설정
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# 데이터셋 로드
train_dataset = COCODataset(root=train_dir, annotation_file=annotation_file, transforms=transform)

# 데이터 로더 설정
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_fn)

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 모델을 학습 모드로 설정
model.train()

num_epochs = 100

# 손실 값을 저장할 리스트
losses_list = []

# 그래프 설정
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-')
ax.set_xlim(0, num_epochs)
ax.set_ylim(0, 10)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.title('Training Loss over Epochs')

for epoch in range(num_epochs):
    running_loss = 0.0
    i = 0
    for imgs, annotations in train_loader:
        # 객체가 없는 이미지는 건너뜁니다
        valid_indices = [idx for idx, target in enumerate(annotations) if len(target['boxes']) > 0]
        if not valid_indices:
            continue

        imgs = [imgs[idx] for idx in valid_indices]
        annotations = [annotations[idx] for idx in valid_indices]

        i += 1
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

        # YOLOv5 모델의 forward 메서드 호출
        loss_dict = model.forward(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())

        if torch.isnan(losses):
            print("Loss is NaN, stopping training")
            break

        model.optimizer.zero_grad()
        losses.backward()
        model.optimizer.step()

        running_loss += losses.item()
        print(f'Epoch: {epoch + 1}, Iteration: {i}/{len(train_loader)}, Loss: {losses.item()}')

    # epoch당 평균 손실 값 저장
    epoch_loss = running_loss / i
    losses_list.append(epoch_loss)

    # 그래프 업데이트
    line.set_xdata(range(len(losses_list)))
    line.set_ydata(losses_list)
    ax.set_ylim(0, max(losses_list) * 1.1)  # y축 범위 동적 설정
    fig.canvas.draw()
    fig.canvas.flush_events()

print("Training completed.")

# 손실 값 그래프 그리기
plt.ioff()
plt.show()
