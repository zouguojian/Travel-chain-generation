import torch
import torch.nn as nn
import numpy as np


def compute_macro_metrics(y_true, y_pred):
    """
    Compute macro-averaged Precision, Recall, and F1 score for multi-class classification.

    Parameters:
    y_true (list or np.array): True labels.
    y_pred (list or np.array): Predicted labels.

    Returns:
    tuple: (macro_precision, macro_recall, macro_f1)
    """
    # Get unique labels
    labels = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(labels)

    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    # Initialize confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)

    # Populate confusion matrix
    for true, pred in zip(y_true, y_pred):
        cm[label_to_idx[true], label_to_idx[pred]] += 1

    # Compute per-class metrics
    precisions = []
    recalls = []
    f1s = []

    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        # Precision for class i
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(precision)

        # Recall for class i
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)

        # F1 for class i
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    # Macro averages
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    return macro_precision, macro_recall, macro_f1


def _compute_cla_loss(preds, labels):
    '''
    example:
        # 模型输出（未经过Softmax的原始分数）
        outputs = model(inputs)  # shape: [batch_size, num_classes]
        # 真实标签（类别索引，不是one-hot编码）
        labels = torch.tensor([1, 3, 0, 2])  # shape: [batch_size]
        loss = criterion(outputs, labels)
    '''
    criterion = nn.CrossEntropyLoss()
    return criterion(preds, labels)

'''
y_pred = np.load('/Users/zouguojian/Travel-chain-generation/data/results/DMTLN-YINCHUAN.npz', allow_pickle=True)['prediction']
y_true = np.load('/Users/zouguojian/Travel-chain-generation/data/results/DMTLN-YINCHUAN.npz', allow_pickle=True)['truth']
y_true, y_pred = np.reshape(y_true,[-1]), np.reshape(y_pred,[-1])
indices = np.where(y_true == 0)
print(y_true[indices])

macro_precision, macro_recall, macro_f1 = compute_macro_metrics(y_true, y_pred)
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
'''

'''
def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)

        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label.astype(np.float32))
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape

for i in range(10):
    print('the %d-th route'%i)
    preds = np.load('/Users/zouguojian/Travel-chain-generation/data/results/DMTLN-'+str(i)+'-YINCHUAN.npz', allow_pickle=True)['prediction'][:,0]
    labels = np.load('/Users/zouguojian/Travel-chain-generation/data/results/DMTLN-'+str(i)+'-YINCHUAN.npz', allow_pickle=True)['truth'][:,0]
    preds = np.reshape(preds[:, 0:1], (-1))
    labels = np.reshape(labels[:, 0:1], (-1))
    mae, rmse, mape = metric(preds, labels)
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}")
'''

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

# statistic model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# metric
def metric(pred, label):
    mask = torch.ne(label, 0)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    mae = torch.abs(torch.sub(pred, label)).type(torch.float32)
    rmse = mae ** 2
    mape = mae / label
    mae = torch.mean(mae)
    rmse = rmse * mask
    rmse = torch.sqrt(torch.mean(rmse))
    mape = mape * mask
    mape = torch.mean(mape)
    return mae, rmse, mape

def _compute_loss(y_true, y_predicted):
    return masked_mae(y_predicted, y_true, 0.0)


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)