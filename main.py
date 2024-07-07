import torch
import cv2
import os
import torchvision
import argparse
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim import AdamW
import torchvision.transforms as transforms

from dataset import TrainingDataset

def train(args):
    num_epochs = args.epoch
    device = args.device

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((512, 512)),
        # transforms.RandomCrop((256, 256)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        # transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = TrainingDataset(args.train_df, args.train_img_root, transform=train_transform)
    val_dataset = TrainingDataset(args.val_df, args.val_img_root, transform=val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    model = torchvision.models.convnext_tiny(num_classes=2)
    model.train().to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.5, 0.99), weight_decay=0.01)

    ce_criteria = torch.nn.CrossEntropyLoss()
    w_ce = 1.0

    for epoch in range(num_epochs):

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        running_ce_loss = 0.0
        acc = 0
        total = 0

        model.train()

        for idx, (img, label) in enumerate(pbar):
            img = img.to(device)
            label = label.to(device)

            output = model(img)

            loss = 0
            ce_loss = ce_criteria(output, label)
            running_ce_loss += ce_loss.item()
            loss += w_ce * ce_loss

            
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pred = torch.argmax(output, dim=1)
            acc += pred.eq(label).sum().item()
            total += pred.size(0)

            pbar.set_postfix(loss=running_loss/total, acc=acc/total)


        pbar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1} Validation")
        running_loss = 0.0
        running_ce_loss = 0.0
        acc = 0
        total = 0

        model.eval()

        for idx, (img, label) in enumerate(pbar):
            img = img.to(device)
            label = label.to(device)

            output = model(img)
            ce_loss = ce_criteria(output, label)
            ce_loss = w_ce * ce_loss.item()
            running_ce_loss += ce_loss
            running_loss += ce_loss

            pred = torch.argmax(output, dim=1)
            acc += pred.eq(label).sum()
            total += pred.size(0)

            pbar.set_postfix(loss=running_loss/total, acc=acc/total)

        print(f"Epoch {epoch + 1} Validation Acc: {acc/total}")

        torch.save(model, f"model_epoch_{epoch}.pth")

    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_df', default='/mnt/e/Downloads/waitan2024_deepfake_challenge__赛道1对外发布数据集_phase1/phase1/trainset_label.txt')
    parser.add_argument('--val_df', default='/mnt/e/Downloads/waitan2024_deepfake_challenge__赛道1对外发布数据集_phase1/phase1/valset_label.txt')
    parser.add_argument('--train_img_root', default='/mnt/e/Downloads/waitan2024_deepfake_challenge__赛道1对外发布数据集_phase1/phase1/trainset')
    parser.add_argument('--val_img_root', default='/mnt/e/Downloads/waitan2024_deepfake_challenge__赛道1对外发布数据集_phase1/phase1/valset')
    parser.add_argument('--epoch', default=50)
    parser.add_argument('--device', default='cuda')


    args = parser.parse_args()


    train(args)