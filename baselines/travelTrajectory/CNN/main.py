import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from Dataload import load_dataset

from model import MultiLayerCNN

import argparse
from utils.utils_ import log_string,count_parameters, _compute_cla_loss, compute_macro_metrics
import random

# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--num_his', type=int, default=12, help='history steps')
parser.add_argument('--L', type=int, default=1, help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=8, help='number of attention heads')
parser.add_argument('--d', type=int, default=8, help='dims of each head attention outputs')
parser.add_argument('--patience', type=int, default=10, help='patience for early stop')
parser.add_argument('--decay_epoch', type=int, default=10, help='decay epoch')

parser.add_argument('--max_seg', type=int, default=108, help='number of segments')
parser.add_argument('--seed', type=int, default=10, help='random seed')
parser.add_argument('--num_features', type=int, default=5, help='total number of features')
parser.add_argument('--emb_size', type=int, default=32, help='size of embedding')
parser.add_argument('--class_num', type=int, default=10, help='number of classes')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--max_R', type=int, default=8, help='max length of routes')
parser.add_argument('--max_S', type=int, default=10, help='max length of sequence for position embedding')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--traffic_file', default='./data/', help='traffic file')
parser.add_argument('--model_file', default='./baselines/travelTrajectory/CNN/best_model.pth', help='save the model to disk')
parser.add_argument('--segment_station_with_distance', default='./data/states/segment_station_with_distance.csv', help='segment station with distance')
parser.add_argument('--log_file', default='log', help='log file')
parser.add_argument("--test", action="store_true", help="test program")
args = parser.parse_args()

log = open(args.log_file, 'w')
log_string(log, str(args)[10: -1])

if args.seed is not None:
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

train_loader, val_loader, test_loader, max_city, max_plate, max_v_type, max_dow, max_mod = load_dataset(args.traffic_file, args.batch_size, test_batch_size=128)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# 训练函数
def train_model(model, train_loader, val_loader, epochs, lr, device, log):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        iteration = 1
        for batch_x, batch_cla, batch_l in train_loader:
            batch_x, batch_cla = (batch_x.to(device), batch_cla.to(device))
            optimizer.zero_grad()
            pred = model(batch_x)

            # 将标签中的-1替换为0以满足loss的要求
            loss = _compute_cla_loss(pred, batch_cla.view(-1))
            # print(f'Epoch {epoch + 1}/{epochs}, Epoch {iteration}/{train_loader.__len__()}, Train Loss: {loss:.4f}')
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            iteration += 1

        # 验证
        val_loss = validate_model(model, val_loader, device)

        log_string(log,f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, '
              f'Val Loss: {val_loss / len(val_loader):.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_file)
            print(f'Saved best model at epoch {epoch + 1}')


# 验证函数
def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_cla, batch_l in val_loader:
            batch_x, batch_cla = (batch_x.to(device), batch_cla.to(device))
            pred = model(batch_x) # pred1是路段时间，pred2是路线时间

            # 计算每个任务的损失
            loss = _compute_cla_loss(pred, batch_cla.view(-1))
            val_loss += loss.item()
    return val_loss

# 测试函数
def test_model(model, test_loader, device):
    model.load_state_dict(torch.load(args.model_file, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    cla_preds = []
    cla_labels = []
    with torch.no_grad():
        for batch_x, batch_cla, batch_l in test_loader:
            batch_x, batch_cla = (batch_x.to(device), batch_cla.to(device))
            pred = model(batch_x)
            for j in range(batch_cla.shape[0]):
                cla_preds.append(torch.argmax(pred[j:j + 1], dim=1).view(-1).cpu().numpy())
                cla_labels.append(batch_cla.view(-1).cpu().numpy()[j:j + 1])

        cla_preds = np.array(cla_preds, dtype=np.int32)
        cla_labels = np.array(cla_labels, dtype=np.int32)
        np.savez_compressed('data/results/DMTLN-YINCHUAN', **{'prediction': cla_preds, 'truth': cla_labels})
        macro_precision, macro_recall, macro_f1 = compute_macro_metrics(np.reshape(cla_labels, [-1]),
                                                                        np.reshape(cla_preds, [-1]))

        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")

# 主函数
def main():
    # 超参数
    field_dims = [max_city, max_plate, max_v_type, max_dow, max_mod]  # 前六个field的取值范围

    # 初始化模型
    model = MultiLayerCNN(field_dims=field_dims,
                          num_features = args.num_features,
                          embed_dim=args.emb_size,
                          num_classes=args.class_num)
    parameters = count_parameters(model)
    log_string(log, 'trainable parameters: {:,}'.format(parameters))

    # 训练和验证
    if not args.test:  # training
        train_model(model, train_loader, val_loader, args.num_epochs, args.lr, device, log)
    else:
    # 测试
        test_model(model, test_loader, device)


if __name__ == "__main__":
    main()