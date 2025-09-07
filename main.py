import torch
import numpy as np
from data.Dataload import load_dataset, load_adjs
from model.DMTLN import DmtlnModel
import argparse
from utils.utils_ import log_string,count_parameters, _compute_loss, _compute_cla_loss
import pandas as pd
import random

parser = argparse.ArgumentParser()
parser.add_argument('--num_his', type=int, default=12, help='history steps')
parser.add_argument('--L', type=int, default=1, help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=8, help='number of attention heads')
parser.add_argument('--d', type=int, default=8, help='dims of each head attention outputs')
parser.add_argument('--patience', type=int, default=10, help='patience for early stop')
parser.add_argument('--decay_epoch', type=int, default=10, help='decay epoch')

parser.add_argument('--seed', type=int, default=10, help='random seed')
parser.add_argument('--num_features', type=int, default=7, help='total number of features')
parser.add_argument('--emb_size', type=int, default=32, help='size of embedding')
parser.add_argument('--class_num', type=int, default=10, help='number of classes')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--max_len', type=int, default=10, help='weight decay')
parser.add_argument('--traffic_file', default='./data/', help='traffic file')
parser.add_argument('--model_file', default='best_model.pth', help='save the model to disk')
parser.add_argument('--segment_station_with_distance', default='./data/states/segment_station_with_distance.csv', help='segment station with distance')
parser.add_argument('--log_file', default='./data/log', help='log file')
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

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
log_string(log, 'device using: {:}'.format(device))

# 数据集加载
log_string(log, 'loading data...')
train_loader, val_loader, test_loader, max_city, max_plate, max_v_type, max_dow, max_mod, max_route, ytra_mean, ytra_std, ytol_mean, ytol_std = load_dataset(args.traffic_file, args.batch_size, test_batch_size=1)
log_string(log, f'max_city:   {max_city}\t\tmax_plate:   {max_plate}\t\tmax_v_type:   {max_v_type}\t\tmax_rote:   {max_route}')
log_string(log, f'ytra_mean:   {ytra_mean:.4f}\t\tytra_std:   {ytra_std:.4f}')
log_string(log, f'ytol_mean:   {ytol_mean:.4f}\t\tytol_std:   {ytol_std:.4f}')
geo_flow_adj, sem_flow_adj = load_adjs(geo_adj_fil = 'data/states/geo_flow_adj.npz', sem_adj_fil = 'data/states/sem_flow_adj.npz', device=device)
geo_speed_adj, sem_speed_adj = load_adjs(geo_adj_fil = 'data/states/geo_speed_adj.npz', sem_adj_fil = 'data/states/sem_speed_adj.npz', device=device)
df = pd.read_csv(args.segment_station_with_distance, encoding='utf-8')
log_string(log, 'data loaded!')

# 训练函数
def train_model(model, train_loader, val_loader, epochs, lr, device, log):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        iteration = 1
        for batch_x, batch_flow, batch_speed, batch_dow, batch_mod, batch_y, batch_cla, batch_l in train_loader:
            batch_x, batch_flow, batch_speed, batch_dow, batch_mod, batch_y, batch_cla = (batch_x.to(device), batch_flow.to(device), batch_speed.to(device), batch_dow.to(device), batch_mod.to(device), batch_y.to(device), batch_cla.to(device))
            optimizer.zero_grad()
            pred1, pred2, pred3 = model(batch_x, batch_flow, batch_speed, batch_dow, batch_mod, batch_cla, batch_l, df, adjs = [geo_flow_adj, sem_flow_adj, geo_speed_adj, sem_speed_adj])

            # 将标签中的-1替换为0以满足loss的要求
            valid_batch_y = torch.where(batch_y[:,1:] == -1, torch.zeros_like(batch_y[:,1:]), batch_y[:,1:])  # [B, current_max_L, 1]
            loss1 = _compute_loss(pred1, valid_batch_y)
            loss2 = _compute_loss(pred2, batch_y[:,:1])
            # loss3 = criterion_task3(out3.squeeze(), batch_y3.squeeze())
            loss = 0.3 * loss1 + 0.7 * loss2  # 简单加和
            print(f'Epoch {epoch + 1}/{epochs}, Epoch {iteration}/{train_loader.__len__()}, Train Loss: {loss:.4f}')
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
        for batch_x, batch_flow, batch_speed, batch_dow, batch_mod, batch_y, batch_cla, batch_l in val_loader:
            batch_x, batch_flow, batch_speed, batch_dow, batch_mod, batch_y, batch_cla = (batch_x.to(device), batch_flow.to(device), batch_speed.to(device), batch_dow.to(device), batch_mod.to(device), batch_y.to(device), batch_cla.to(device))
            pred1, pred2, pred3 = model(batch_x, batch_flow, batch_speed, batch_dow, batch_mod, batch_cla, batch_l, df, adjs = [geo_flow_adj, sem_flow_adj, geo_speed_adj, sem_speed_adj]) # pred1是路段时间，pred2是路线时间

            # 计算每个任务的损失
            # 将标签中的-1替换为0以满足loss的要求
            valid_batch_y = torch.where(batch_y[:,1:] == -1, torch.zeros_like(batch_y[:,1:]), batch_y[:,1:])  # [B, current_max_L, 1]
            loss1 = _compute_loss(pred1, valid_batch_y)
            loss2 = _compute_loss(pred2, batch_y[:,:1])
            # loss3 = criterion_task3(out3.squeeze(), batch_y3.squeeze())
            loss = 0.3 * loss1 + 0.7 * loss2  # 简单加和
            val_loss += loss.item()
    return val_loss

# 测试函数
def test_model(model, test_loader, device):
    model.load_state_dict(torch.load(args.model_file, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    test1_loss = 0
    test2_loss = 0
    labels_dict = {i:[] for i in range(max_route)}
    preds_dict = {i:[] for i in range(max_route)}
    with torch.no_grad():
        i = 0
        for batch_x, batch_flow, batch_speed, batch_dow, batch_mod, batch_y, batch_cla, batch_l in test_loader:
            batch_x, batch_flow, batch_speed, batch_dow, batch_mod, batch_y, batch_cla = (batch_x.to(device), batch_flow.to(device), batch_speed.to(device), batch_dow.to(device), batch_mod.to(device), batch_y.to(device), batch_cla.to(device))
            pred1, pred2, pred3 = model(batch_x, batch_flow, batch_speed, batch_dow, batch_mod, batch_cla, batch_l, df, adjs = [geo_flow_adj, sem_flow_adj, geo_speed_adj, sem_speed_adj])
            # print(batch_x.cpu().numpy()[0, batch_cla.cpu().numpy()[0, 0], 0, 5]) # [1, routes, max_length, num_fields]
            preds_dict[int(batch_x.cpu().numpy()[0, batch_cla.cpu().numpy()[0, 0], 0, 5])].append(np.concatenate((pred2.cpu().numpy(), pred1.cpu().numpy()), axis=1))
            labels_dict[int(batch_x.cpu().numpy()[0, batch_cla.cpu().numpy()[0, 0], 0, 5])].append(batch_y.cpu().numpy())

            # 计算每个任务的损失, 将标签中的-1替换为0以满足loss的要求
            valid_batch_y = torch.where(batch_y[:,1:] == -1, torch.zeros_like(batch_y[:,1:]), batch_y[:,1:])  # [B, current_max_L, 1]
            loss1 = _compute_loss(pred1, valid_batch_y) # 路段
            loss2 = _compute_loss(pred2, batch_y[:,:1]) # 路线
            # loss3 = criterion_task3(out3.squeeze(), batch_y3.squeeze())
            test1_loss += loss1.item()
            test2_loss += loss2.item()
            i+=1
            if i == 5000: break

    for i in range(len(labels_dict)):
        preds_dict[i] = np.array(preds_dict[i], dtype=np.float32)
        labels_dict[i] = np.array(labels_dict[i], dtype=np.float32)
        np.savez_compressed('data/results/DMTLN-' + str(i) + '-YINCHUAN', **{'prediction': preds_dict[i], 'truth': labels_dict[i]})
    print(f'Test1 Loss: {test1_loss / len(test_loader):.4f}, Task 2 Loss: {test2_loss / len(test_loader):.4f}')

# 主函数
def main():
    # 超参数
    field_dims = [max_city, max_plate, max_v_type, max_dow, max_mod, max_route]  # 前六个field的取值范围

    # 初始化模型
    model = DmtlnModel(
        field_dims=field_dims,
        num_features= args.num_features,
        embed_size=args.emb_size,
        class_num=args.class_num,
        max_len = args.max_len,
        ytra_mean = ytra_mean, ytra_std = ytra_std, ytol_mean = ytol_mean, ytol_std = ytol_std
    )
    parameters = count_parameters(model)
    log_string(log, 'trainable parameters: {:,}'.format(parameters))

    # 训练和验证
    # train_model(model, train_loader, val_loader, args.num_epochs, args.lr, device, log)

    # 测试
    test_model(model, test_loader, device)


if __name__ == "__main__":
    main()