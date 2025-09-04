import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

from net4class.NASNet import SequenceFusion3BranchNetworkNASNet
from net4class.RegNet import SequenceFusion3BranchNetworkRegNet
from net4class.SeFTNet import SequenceFusion3BranchNetworkSeFTNet
from net4class.Xception import SequenceFusion3BranchNetworkXceptionNet
from net4class.SqueezeNet import SequenceFusion3BranchNetworkSqueezeNet
from net4class.InceptionResNet import SequenceFusion3BranchNetworkInceptionResNet
from net4class.DenseNet import SequenceFusion3BranchNetworkDenseNet
from net4class.EfficientNet import SequenceFusion3BranchNetworkEfficientNet
from net4class.SEBlock import SequenceFusion3BranchNetworkSE
from net4class.MobileNetBranch import SequenceFusion3BranchNetworkMobileNet
from net4class.SequenceFusion2n2 import SequenceFusion3BranchNetworkSelfAtt, SequenceFusion3BranchNetwork, \
    SequenceFusion3BranchNetworkAlexNet, SequenceFusion3BranchNetworkGoogLeNet, SequenceFusion3BranchNetworkLeNet, \
    SequenceFusion3BranchNetworkResNet, SequenceFusion3BranchNetworkVGGNet, SequenceFusion3BranchNetworkWithGRU, \
    SequenceFusion3BranchNetworkResidualFusion, SequenceFusion3BranchNetworkStackedFusion, SeqTransFusionNet
from dataprocess.dataload_4class import SequenceDataset3Branchbothlabel
from sklearn.model_selection import train_test_split
from net4class.newnet import UTransNet
from net4class.newnet1 import UTransNet1

class CustomNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


def split_dataset(dataset, val_ratio=0.2):
    labels1 = [dataset[i][3] for i in range(len(dataset))]
    labels2 = [dataset[i][4] for i in range(len(dataset))]
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=val_ratio, stratify=labels1, random_state=42
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset


def calculate_metrics(labels, predictions, probabilities, num_classes):
    sensitivity = []
    specificity = []
    class_acc = []

    cm = confusion_matrix(labels, predictions)
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)

        sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        class_acc.append((tp + tn) / cm.sum())

    return sensitivity, specificity, class_acc


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    num_classes_task1 = 2
    num_classes_task2 = 2
    model = SequenceFusion3BranchNetworkWithGRU(num_classes_task1=num_classes_task1, num_classes_task2=num_classes_task1).to(device)
    class_weights_task1 = torch.tensor([1, 3], dtype=torch.float).to(device)
    class_weights_task2 = torch.tensor([1, 3], dtype=torch.float).to(device)
    criterion_task1 = nn.CrossEntropyLoss(weight=class_weights_task1)
    criterion_task2 = nn.CrossEntropyLoss(weight=class_weights_task2)
    base_lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    scaler = GradScaler()
    transform = CustomNormalize(mean=0.5, std=0.5)
    dataset = SequenceDataset3Branchbothlabel(
        sheet1_path='curve1.csv',
        sheet2_path='curve2.csv',
        sheet3_path='curve3.csv',
        label_path1='Task1.xlsx',
        label_path2='Task2.xlsx',
        transform=transform
    )
    train_dataset, val_dataset = split_dataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8)

    num_epochs = 100
    patience = 20
    trigger_times = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss_task1, train_loss_task2 = 0.0, 0.0
        all_labels_task1, all_predictions_task1, all_probabilities_task1 = [], [], []
        all_labels_task2, all_predictions_task2, all_probabilities_task2 = [], [], []

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{num_epochs}, Current Learning Rate: {current_lr:.8f}")

        for seq1, seq2, seq3, labels_task1, labels_task2 in train_loader:
            seq1, seq2, seq3 = seq1.to(device), seq2.to(device), seq3.to(device)
            labels_task1, labels_task2 = labels_task1.to(device).long(), labels_task2.to(device).long()
            optimizer.zero_grad()

            with autocast():
                outputs_task1, outputs_task2 = model(seq1, seq2, seq3)
                outputs_task1 = outputs_task1.float()
                outputs_task2 = outputs_task2.float()

                loss_task1 = criterion_task1(outputs_task1, labels_task1)
                loss_task2 = criterion_task2(outputs_task2, labels_task2)
                loss_task1 += model.regularization_loss()
                loss_task2 += model.regularization_loss()

            total_loss = loss_task1 + loss_task2
            alpha_task1 = loss_task2 / total_loss
            alpha_task2 = loss_task1 / total_loss

            loss = alpha_task1 * loss_task1 + alpha_task2 * loss_task2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_task1 += loss_task1.item() * seq1.size(0)
            train_loss_task2 += loss_task2.item() * seq1.size(0)

            _, predicted_task1 = torch.max(outputs_task1, 1)
            _, predicted_task2 = torch.max(outputs_task2, 1)

            probabilities_task1 = torch.softmax(outputs_task1, dim=1)[:, 1]
            probabilities_task2 = torch.softmax(outputs_task2, dim=1)[:, 1]

            all_labels_task1.extend(labels_task1.cpu().numpy())
            all_predictions_task1.extend(predicted_task1.cpu().numpy())
            all_probabilities_task1.extend(probabilities_task1.detach().cpu().numpy())

            all_labels_task2.extend(labels_task2.cpu().numpy())
            all_predictions_task2.extend(predicted_task2.cpu().numpy())
            all_probabilities_task2.extend(probabilities_task2.detach().cpu().numpy())

        sensitivity_train_task1, specificity_train_task1, class_acc_train_task1 = calculate_metrics(
            all_labels_task1, all_predictions_task1, all_probabilities_task1, num_classes_task1)
        sensitivity_train_task2, specificity_train_task2, class_acc_train_task2 = calculate_metrics(
            all_labels_task2, all_predictions_task2, all_probabilities_task2, num_classes_task2)

        acc_train_task1 = accuracy_score(all_labels_task1, all_predictions_task1)
        auc_train_task1 = roc_auc_score(all_labels_task1, all_probabilities_task1)
        acc_train_task2 = accuracy_score(all_labels_task2, all_predictions_task2)
        auc_train_task2 = roc_auc_score(all_labels_task2, all_probabilities_task2)

        model.eval()
        val_loss_task1, val_loss_task2 = 0.0, 0.0
        all_labels_task1, all_predictions_task1, all_probabilities_task1 = [], [], []
        all_labels_task2, all_predictions_task2, all_probabilities_task2 = [], [], []

        with torch.no_grad():
            for seq1, seq2, seq3, labels_task1, labels_task2 in val_loader:
                seq1, seq2, seq3 = seq1.to(device), seq2.to(device), seq3.to(device)
                labels_task1, labels_task2 = labels_task1.to(device).long(), labels_task2.to(device).long()

                outputs_task1, outputs_task2 = model(seq1, seq2, seq3)
                outputs_task1 = outputs_task1.float()
                outputs_task2 = outputs_task2.float()

                loss_task1 = criterion_task1(outputs_task1, labels_task1)
                loss_task2 = criterion_task2(outputs_task2, labels_task2)
                loss_task1 += model.regularization_loss()
                loss_task2 += model.regularization_loss()

                val_loss_task1 += loss_task1.item() * seq1.size(0)
                val_loss_task2 += loss_task2.item() * seq1.size(0)

                _, predicted_task1 = torch.max(outputs_task1, 1)
                _, predicted_task2 = torch.max(outputs_task2, 1)

                probabilities_task1 = F.softmax(outputs_task1, dim=1)[:, 1]
                probabilities_task2 = F.softmax(outputs_task2, dim=1)[:, 1]

                all_labels_task1.extend(labels_task1.cpu().numpy())
                all_predictions_task1.extend(predicted_task1.cpu().numpy())
                all_probabilities_task1.extend(probabilities_task1.detach().cpu().numpy())

                all_labels_task2.extend(labels_task2.cpu().numpy())
                all_predictions_task2.extend(predicted_task2.cpu().numpy())
                all_probabilities_task2.extend(probabilities_task2.detach().cpu().numpy())

        sensitivity_val_task1, specificity_val_task1, class_acc_val_task1 = calculate_metrics(
            all_labels_task1, all_predictions_task1, all_probabilities_task1, num_classes_task1)
        sensitivity_val_task2, specificity_val_task2, class_acc_val_task2 = calculate_metrics(
            all_labels_task2, all_predictions_task2, all_probabilities_task2, num_classes_task2)

        acc_val_task1 = accuracy_score(all_labels_task1, all_predictions_task1)
        auc_val_task1 = roc_auc_score(all_labels_task1, all_probabilities_task1)
        acc_val_task2 = accuracy_score(all_labels_task2, all_predictions_task2)
        auc_val_task2 = roc_auc_score(all_labels_task2, all_probabilities_task2)

        avg_train_loss_task1 = train_loss_task1 / len(train_loader.dataset)
        avg_train_loss_task2 = train_loss_task2 / len(train_loader.dataset)
        avg_val_loss_task1 = val_loss_task1 / len(val_loader.dataset)
        avg_val_loss_task2 = val_loss_task2 / len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Task 1 - Train Loss: {avg_train_loss_task1:.4f}, Train Acc: {acc_train_task1:.4f}, Train AUC: {auc_train_task1:.4f}')
        print(f'Task 1 - Val Loss: {avg_val_loss_task1:.4f}, Val Acc: {acc_val_task1:.4f}, Val AUC: {auc_val_task1:.4f}')
        print(f'Task 2 - Train Loss: {avg_train_loss_task2:.4f}, Train Acc: {acc_train_task2:.4f}, Train AUC: {auc_train_task2:.4f}')
        print(f'Task 2 - Val Loss: {avg_val_loss_task2:.4f}, Val Acc: {acc_val_task2:.4f}, Val AUC: {auc_val_task2:.4f}')

        print(f'Task 1 - Train Sensitivity: {sensitivity_train_task1}, Train Specificity: {specificity_train_task1}')
        print(f'Task 1 - Val Sensitivity: {sensitivity_val_task1}, Val Specificity: {specificity_val_task1}')
        print(f'Task 2 - Train Sensitivity: {sensitivity_train_task2}, Train Specificity: {specificity_train_task2}')
        print(f'Task 2 - Val Sensitivity: {sensitivity_val_task2}, Val Specificity: {specificity_val_task2}')

        scheduler.step(avg_val_loss_task1 + avg_val_loss_task2)

        if avg_val_loss_task1 + avg_val_loss_task2 < best_val_loss:
            best_val_loss = avg_val_loss_task1 + avg_val_loss_task2
            trigger_times = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print('Best model saved!')
        else:
            trigger_times += 1
            print(f'Early stopping trigger times: {trigger_times}/{patience}')

        if trigger_times >= patience:
            print('Early stopping!')
            break

    print("Training finished.")


if __name__ == '__main__':
    main()
