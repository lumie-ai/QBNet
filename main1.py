import torch
import argparse
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torch import nn, optim
from torchvision.transforms import transforms
from sklearn.model_selection import KFold
from QBNet import Unet
from dataset import LiverDataset
from common_tools import transform_invert
from torch.profiler import profile, record_function, ProfilerActivity
import random
from sklearn.metrics import roc_curve, auc

model_name = "cnn1"
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.ToTensor()
y_transforms = transforms.ToTensor()

train_losses = []
valid_losses = []
best_val_loss = float('inf')
val_interval = 1

def makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100, fold=0):
    global best_val_loss, train_losses, valid_losses
    train_losses = []
    valid_losses = []
    best_val_loss = float('inf')

    model_dir = f"./model/{model_name}"
    makedir(model_dir)
    model_path = os.path.join(model_dir, f"weights_fold{fold}_epoch20.pth")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        start_epoch = 20
        print('Loaded successfully!')
    else:
        start_epoch = 0
        print('No saved model, training from scratch!')
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    print(f"Train size: {len(train_loader)}")
    print(f"Validation size: {len(val_loader)}")    
    for epoch in range(start_epoch + 1, num_epochs):
        print(f'Epoch {epoch}/{num_epochs} (Fold {fold + 1}/5)')
        print('-' * 10)
        dt_size = len(train_loader.dataset)
        epoch_loss = 0
        step = 0
        model.train()
        for x, y in train_loader:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            optimizer.zero_grad()
            with record_function("forward_pass"):
                outputs = model(inputs)
            with record_function("loss_calculation"):
                loss = criterion(outputs, labels)
            with record_function("backward_pass"):
                loss.backward()
            with record_function("optimizer_step"):
                optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_loader)}, train_loss: {loss.item():.3f}")
        avg_epoch_loss = epoch_loss / step
        train_losses.append(avg_epoch_loss)
        print(f"epoch {epoch} loss: {avg_epoch_loss:.3f}")

        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f"weights_fold{fold}_epoch{epoch + 1}.pth"))

        if (epoch + 1) % val_interval == 0:
            val_loss = 0.
            model.eval()
            with torch.no_grad():
                step_val = 0
                for x, y in val_loader:
                    step_val += 1
                    inputs = x.to(device)
                    labels = y.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                avg_val_loss = val_loss / step_val
                valid_losses.append(avg_val_loss)
                print(f"epoch {epoch} valid_loss: {avg_val_loss:.3f}")

                save_path = os.path.join(model_dir, f"best_model_fold{fold}.pth")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), save_path)
                    print(f"Best model saved with validation loss: {best_val_loss}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 0.5)
    plt.title(f'Training and Validation Loss - Model: {model_name} (Fold {fold})')
    plt.legend()
    plt.savefig(os.path.join(model_dir, f"loss_curve_fold{fold}.png"))
    plt.close()

def train(args):
    liver_dataset = LiverDataset("./data/train", mode="train", transform=x_transforms, target_transform=y_transforms)
    print(f"Total dataset size: {len(liver_dataset)}")

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(liver_dataset)):
        print(f"Fold {fold + 1}: Train indices length={len(train_idx)}, Val indices length={len(val_idx)}")

        model = Unet(1, 1).to(device)
        batch_size = args.batch_size
        print(f"Batch size: {batch_size}")
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print(f"Fold {fold + 1}/5")
        train_subset = Subset(liver_dataset, train_idx)
        print("len(train_subset)",len(train_subset))
        val_subset = Subset(liver_dataset, val_idx)
        print("len(val_subset)",len(val_subset))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=16)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=16)

        print(f"Train dataset size: {len(train_loader.dataset)}")
        print(f"Number of batches in train_loader: {len(train_loader)}")

        train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=200, fold=fold)

def test(args):
    model = Unet(1, 1)
    model_dir = f"./model/{model_name}"
    liver_dataset = LiverDataset("data/test", mode="test", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)

    save_root = f'./data/{model_name}_predict'
    makedir(save_root)

    all_sensitivity_means = []
    all_specificity_means = []
    all_accuracy_means = []
    all_auc_means = []
    all_f1_means = []
    all_iou_means = []

    for fold in range(5):
        model_path = os.path.join(model_dir, f"best_model_fold{fold}.pth")
        if not os.path.exists(model_path):
            print(f"Model for fold {fold} not found!")
            continue

        model.load_state_dict(torch.load(model_path, map_location='cuda:4'))
        model.eval()
        plt.ion()

        sensitivity = []
        specificity = []
        accuracy = []
        auc_scores = []
        f1 = []
        iou = []

        all_ground_truths = []
        all_probabilities = []

        index = 0
        with torch.no_grad():
            for x, ground in dataloaders:
                x = x.type(torch.FloatTensor)
                y = model(x)

                y_np = y.cpu().numpy().squeeze() * 255
                y_np = y_np.astype(np.uint8)

                _, binary_output = cv2.threshold(y_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                threshold = _ / 255

                binary_output = (y > threshold).float()

                x = torch.squeeze(x)
                x = x.unsqueeze(0)
                ground = torch.squeeze(ground)
                ground = ground.unsqueeze(0)
                img_ground = transform_invert(ground, y_transforms)
                img_x = transform_invert(x, x_transforms)
                img_y = torch.squeeze(binary_output).numpy()

                save_path = os.path.join(save_root, f"predict_fold{fold}_{index}_o.png")
                ground_path = os.path.join(save_root, f"predict_fold{fold}_{index}_g.png")
                img_ground.save(ground_path)
                cv2.imwrite(save_path, img_y * 255)
                index += 1

                img_ground_np = np.array(img_ground)
                img_predict_np = img_y

                img_ground_np = (img_ground_np > 127).astype(np.uint8)
                img_predict_np = (img_predict_np > 0.5).astype(np.uint8)

                tp = np.logical_and(img_ground_np, img_predict_np).sum()
                fp = np.logical_and(1 - img_ground_np, img_predict_np).sum()
                fn = np.logical_and(img_ground_np, 1 - img_predict_np).sum()
                tn = np.logical_and(1 - img_ground_np, 1 - img_predict_np).sum()

                if tp + fn == 0:
                    current_sensitivity = 1
                else:
                    current_sensitivity = tp / (tp + fn)
                sensitivity.append(current_sensitivity)

                if tn + fp == 0:
                    current_specificity = 1
                else:
                    current_specificity = tn / (tn + fp)
                specificity.append(current_specificity)

                total_pixels = img_ground_np.size
                correct_pixels = (img_ground_np == img_predict_np).sum()
                current_accuracy = correct_pixels / total_pixels
                accuracy.append(current_accuracy)

                if tp + fp + fn == 0:
                    current_f1 = 1
                else:
                    current_f1 = (2 * tp) / (2 * tp + fp + fn)
                f1.append(current_f1)

                intersection = tp
                union = tp + fp + fn
                if union == 0:
                    current_iou = 1
                else:
                    current_iou = intersection / union
                iou.append(current_iou)

                all_ground_truths.append(img_ground_np.flatten())
                all_probabilities.append(y.cpu().numpy().squeeze().flatten())

        all_ground_truths = np.concatenate(all_ground_truths)
        all_probabilities = np.concatenate(all_probabilities)
        fpr, tpr, thresholds = roc_curve(all_ground_truths, all_probabilities)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve - Fold {fold}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_root, f"roc_curve_fold{fold}.png"))
        plt.close()

        auc_scores.append(roc_auc)

        sensitivity_mean = np.mean(sensitivity)
        specificity_mean = np.mean(specificity)
        accuracy_mean = np.mean(accuracy)
        f1_mean = np.mean(f1)
        iou_mean = np.mean(iou)
        auc_mean = np.mean(auc_scores)

        all_sensitivity_means.append(sensitivity_mean)
        all_specificity_means.append(specificity_mean)
        all_accuracy_means.append(accuracy_mean)
        all_auc_means.append(auc_mean)
        all_f1_means.append(f1_mean)
        all_iou_means.append(iou_mean)

        print(f"Fold {fold}:")
        print("Best Threshold:", threshold)
        print("Sensitivity Mean:", sensitivity_mean)
        print("Specificity Mean:", specificity_mean)
        print("Accuracy Mean:", accuracy_mean)
        print("F1 Score Mean:", f1_mean)
        print("IOU Mean:", iou_mean)
        print("AUC Mean:", auc_mean)

    avg_sensitivity_means = np.mean(all_sensitivity_means)
    avg_specificity_means = np.mean(all_specificity_means)
    avg_accuracy_means = np.mean(all_accuracy_means)
    avg_auc_means = np.mean(all_auc_means)
    avg_f1_means = np.mean(all_f1_means)
    avg_iou_means = np.mean(all_iou_means)

    print("\nAverage Metrics Across All Folds:")
    print("Average Sensitivity Mean:", avg_sensitivity_means)
    print("Average Specificity Mean:", avg_specificity_means)
    print("Average Accuracy Mean:", avg_accuracy_means)
    print("Average AUC Mean:", avg_auc_means)
    print("Average F1 Score Mean:", avg_f1_means)
    print("Average IOU Mean:", avg_iou_means)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, help="train, test or dice", default="train")
    parse.add_argument("--batch_size", type=int, default=4)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file", default=f"./model/{model_name}/weights_20.pth")
    args = parse.parse_args()

    if args.action == "train":
        train(args)
    elif args.action == "test":
        test(args)