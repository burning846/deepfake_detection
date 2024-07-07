import torch
import cv2
import os
import torchvision
import argparse
from tqdm import tqdm
import pandas as pd

from torch.utils.data import DataLoader
from torch.optim import AdamW
import torchvision.transforms as transforms

from dataset import InferenceDataset

def inference(args):
    device = args.device

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
    ])

    df = pd.read_csv(args.input_df)
    val_dataset = InferenceDataset(args.input_df, args.img_root, transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = torch.load(args.checkpoint)
    model.eval().to(device)


    pbar = tqdm(val_dataloader, desc=f"Prediction")
    preds = []
    with torch.no_grad():
        for idx, img in enumerate(pbar):
            img = img.to(device)

            output = model(img)
            output = torch.softmax(output, dim=1)
            # pred = torch.argmax(output, dim=1)
            preds.extend(output.cpu().numpy()[:, 1])

    df['y_pred'] = preds
    df.to_csv(args.save_df, index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_df', default='/mnt/e//Downloads/multi-ffdi/prediction.txt.csv')
    parser.add_argument('--img_root', default='/mnt/e/Downloads/waitan2024_deepfake_challenge__赛道1对外发布数据集_phase1/phase1/valset')
    parser.add_argument('--save_df', default='./predict_output.csv')
    parser.add_argument('--checkpoint', default='./model_epoch_8.pth')
    parser.add_argument('--device', default='cuda')


    args = parser.parse_args()


    inference(args)