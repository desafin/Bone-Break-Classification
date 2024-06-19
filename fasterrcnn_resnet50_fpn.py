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

seed = 42  # seed 값 설정
random.seed(seed)  # 파이썬 난수 생성기
os.environ['PYTHONHASHSEED'] = str(seed)  # 해시 시크릿값 고정
np.random.seed(seed)  # 넘파이 난수 생성기

torch.manual_seed(seed)  # 파이토치 CPU 난수 생성기
torch.backends.cudnn.deterministic = True  # 확정적 연산 사용 설정
torch.backends.cudnn.benchmark = False  # 벤치마크 기능 사용 해제
torch.backends.cudnn.enabled = False  # cudnn 기능 사용 해제

if device == 'cuda':
    torch.cuda.manual_seed(seed)  # 파이토치 GPU 난수 생성기
    torch.cuda.manual_seed_all(seed)  # 파이토치 멀티 GPU 난수 생성기

train_dir = './new_train'
test_dir = './test'
annotation_file = './annotations/instances_default.json'


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
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

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
    return tuple(zip(*batch))


# 데이터 변환 설정
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),  # 수평 뒤집기
    transforms.RandomVerticalFlip(),  # 수직 뒤집기 추가
    transforms.RandomRotation(30),  # 임의의 각도로 회전
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 색상 변형
    transforms.RandomGrayscale(p=0.1),  # 일부 이미지를 회색조로 변환
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # 원근 왜곡
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 임의의 평행 이동
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화 추가
])


# 데이터셋 로드
train_dataset = COCODataset(root=train_dir, annotation_file=annotation_file, transforms=transform)

# 데이터 로더 설정
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_fn)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main():
    # 2 classes; Only target class or background
    num_classes = 2
    num_epochs = 15000
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # parameters
    params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = torch.optim.Adadelta(params, lr=0.1, weight_decay=0.0001)
    #optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.0001)# 실패
    #optimizer = torch.optim.AdamW(params, lr=0.01, weight_decay=0.0001)#실패
    #optimizer = torch.optim.RMSprop(params, lr=0.0001, weight_decay=0.0001, momentum=0.9)#실패
    optimizer = torch.optim.Adagrad(params, lr=0.01, weight_decay=0.0001)#값 너무튐
    # optimizer = torch.optim.Adamax(params, lr=0.002, weight_decay=0.0001)
    # optimizer = torch.optim.NAdam(params, lr=0.0001, weight_decay=0.0001)실패

    # #스케쥴러 시도
    # optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # optimizer = torch.optim.Adadelta(params, lr=1.0, weight_decay=0.0001)
    len_dataloader = len(train_loader)

    # 손실 값을 저장할 리스트
    losses_list = []
    best_loss = float('inf')  # 최적 손실 값을 무한대로 초기화
    save_path = 'best_model.pth'  # 모델 저장 경로

    # 그래프 설정
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'r-')
    ax.set_xlim(0, num_epochs)
    ax.set_ylim(0, 10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.title('Training Loss over Epochs')

    for epoch in range(num_epochs):  # 수정된 부분: for 루프 추가
        model.train()
        i = 0
        running_loss = 0.0
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
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            if torch.isnan(losses):
                print("Loss is NaN, stopping training")
                break

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            running_loss += losses.item()
            print(f'Epoch: {epoch + 1}, Iteration: {i}/{len_dataloader}, Loss: {losses.item()}')

        # epoch당 평균 손실 값 저장
        epoch_loss = running_loss / i
        losses_list.append(epoch_loss)

        # 최적 모델 저장
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with loss: {best_loss}")

        # 그래프 업데이트
        line.set_xdata(range(len(losses_list)))
        line.set_ydata(losses_list)
        ax.set_ylim(0, max(losses_list) * 1.1)  # y축 범위 동적 설정
        fig.canvas.draw()
        fig.canvas.flush_events()

        # 손실 값이 0.01 이하로 내려가면 학습 중단
        if best_loss < 0.01:
            print("Loss has decreased below 0.01, stopping training.")
            break

    print("Training completed.")

    # 손실 값 그래프 그리기
    plt.ioff()
    plt.show()

import cv2
import numpy as np
from PIL import Image

def load_model(model_path, num_classes):
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model, img, threshold=0.5):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(img)

    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()
    pred_labels = prediction[0]['labels'].cpu().numpy()

    pred_boxes = pred_boxes[pred_scores >= threshold].astype(int)
    pred_labels = pred_labels[pred_scores >= threshold]
    return pred_boxes, pred_labels

def draw_boxes(img_path, boxes, labels, class_names):
    img = cv2.imread(img_path)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, class_names[label], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        # 박스 좌표를 터미널에 출력
        print(f"Detected {class_names[label]} at Box: ({x1}, {y1}), ({x2}, {y2})")
    return img

def main_test():
    test_images = glob.glob(os.path.join(test_dir, '*.jpg'))
    num_classes = 2
    model_path = 'best_model.pth'
    model = load_model(model_path, num_classes)
    class_names = {1: 'bone_break'}  # 클래스 이름 설정

    for img_path in test_images:
        img = Image.open(img_path).convert('RGB')
        boxes, labels = predict(model, img, threshold=0.1)
        result_img = draw_boxes(img_path, boxes, labels, class_names)
        cv2.imshow("Object Detection", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main_test()
