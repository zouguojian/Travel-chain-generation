import torch
import torch.nn as nn
import numpy as np
from sympy import print_rcode

from data.Dataload import load_dataset
from model.DMTLN import DmtlnModel
import argparse
from utils.utils_ import log_string,count_parameters, _compute_loss

parser = argparse.ArgumentParser()
parser.add_argument('--num_his', type=int, default=12, help='history steps')
parser.add_argument('--num_pred', type=int, default=12, help='prediction steps')
parser.add_argument('--L', type=int, default=1, help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=8, help='number of attention heads')
parser.add_argument('--d', type=int, default=8, help='dims of each head attention outputs')
parser.add_argument('--patience', type=int, default=10, help='patience for early stop')
parser.add_argument('--decay_epoch', type=int, default=10, help='decay epoch')

parser.add_argument('--num_features', type=int, default=7, help='total number of features')
parser.add_argument('--emb_size', type=int, default=32, help='size of embedding')
parser.add_argument('--class_num', type=int, default=11, help='number of classes')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--max_len', type=int, default=10, help='weight decay')
parser.add_argument('--traffic_file', default='./data/', help='traffic file')
parser.add_argument('--model_file', default='./data/best_model.pth', help='save the model to disk')
parser.add_argument('--log_file', default='./data/log', help='log file')
args = parser.parse_args()

log = open(args.log_file, 'w')
log_string(log, str(args)[10: -1])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集加载
log_string(log, 'loading data...')
train_loader, val_loader, test_loader, max_city, max_plate, max_v_type, max_dow, max_mod, max_rote, ytra_mean, ytra_std, ytol_mean, ytol_std = load_dataset(args.traffic_file, args.batch_size, test_batch_size=1)
log_string(log, f'max_city:   {max_city}\t\tmax_plate:   {max_plate}\t\tmax_v_type:   {max_v_type}\t\tmax_rote:   {max_rote}')
log_string(log, f'ytra_mean:   {ytra_mean:.4f}\t\tytra_std:   {ytra_std:.4f}')
log_string(log, f'ytol_mean:   {ytol_mean:.4f}\t\tytol_std:   {ytol_std:.4f}')
log_string(log, 'data loaded!')

# 训练函数
def train_model(model, train_loader, val_loader, epochs, lr, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y, batch_l in train_loader:
            batch_x, batch_y = (batch_x.to(device), batch_y.to(device))
            optimizer.zero_grad()
            pred1, pred2, pred3 = model(batch_x, batch_l)

            # 计算每个任务的损失
            loss1 = _compute_loss(pred1, batch_y[:,1:])
            loss2 = _compute_loss(pred2, batch_y[:,:1])
            # loss3 = criterion_task3(out3.squeeze(), batch_y3.squeeze())
            loss = 0.5 * loss1 + 0.5 * loss2  # 简单加和
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        val_loss = validate_model(model, val_loader, device)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, '
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
        for batch_x, batch_y, batch_l in val_loader:
            batch_x, batch_y, batch_l = (batch_x.to(device), batch_y.to(device), batch_l.to(device))
            pred1, pred2, pred3 = model(batch_x, batch_l)

            # 计算每个任务的损失
            loss1 = _compute_loss(pred1, batch_y[:,1:])
            loss2 = _compute_loss(pred2, batch_y[:,:1])
            # loss3 = criterion_task3(out3.squeeze(), batch_y3.squeeze())
            loss = 0.5 * loss1 + 0.5 * loss2  # 简单加和
            val_loss += loss.item()
    return val_loss

# 测试函数
def test_model(model, test_loader, device, load_path='best_model.pth'):
    model.load_state_dict(torch.load(load_path))
    model = model.to(device)
    model.eval()

    criterion_task1 = nn.CrossEntropyLoss()
    criterion_task2 = nn.MSELoss()
    criterion_task3 = nn.MSELoss()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y1, batch_y2, batch_y3 in test_loader:
            batch_x, batch_y1, batch_y2, batch_y3 = (
                batch_x.to(device), batch_y1.to(device),
                batch_y2.to(device), batch_y3.to(device)
            )
            out1, out2, out3 = model(batch_x)

            # 分类任务准确率
            _, predicted = torch.max(out1, 1)
            total += batch_y1.size(0)
            correct += (predicted == batch_y1).sum().item()

            # 计算损失
            loss1 = criterion_task1(out1, batch_y1)
            loss2 = criterion_task2(out2.squeeze(), batch_y2.squeeze())
            loss3 = criterion_task3(out3.squeeze(), batch_y3.squeeze())
            loss = loss1 + loss2 + loss3
            test_loss += loss.item()

    accuracy = correct / total
    print(f'Test Loss: {test_loss / len(test_loader):.4f}, Task 1 Accuracy: {accuracy:.4f}')


# 主函数
def main():
    # 超参数
    field_dims = [max_city, max_plate, max_v_type, max_dow, max_mod, max_rote]  # 前六个field的取值范围

    # 初始化模型
    model = DmtlnModel(
        field_dims=field_dims,
        num_features= args.num_features,
        embed_size=args.emb_size,
        class_num=args.class_num,
        max_len = args.max_len
    )
    parameters = count_parameters(model)
    log_string(log, 'trainable parameters: {:,}'.format(parameters))

    # 训练和验证
    train_model(model, train_loader, val_loader, args.num_epochs, args.lr, device)

    # 测试
    # test_model(model, test_loader, device)


if __name__ == "__main__":
    main()