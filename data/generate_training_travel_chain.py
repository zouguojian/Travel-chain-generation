import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import argparse
import os

# 自定义函数：将时间四舍五入到最近的五分钟
def round_to_five_minutes(time_str):
    # 将字符串转换为 datetime 对象
    dt = pd.to_datetime(time_str)
    # 计算分钟数
    minutes = dt.minute
    # 计算最近的五分钟刻度
    rounded_minutes = round(minutes / 5) * 5
    # 如果四舍五入到60分钟，需处理小时进位
    if rounded_minutes == 60:
        dt = dt + timedelta(hours=1)
        rounded_minutes = 0
    # 设置分钟和秒钟
    dt = dt.replace(minute=rounded_minutes, second=0, microsecond=0)
    # 返回格式化的时间字符串
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def process_travel_data(args):
    # 读取轨迹 CSV 文件
    df = pd.read_csv(args.trajectory_df_filename)
    # 提取出发时间（轨迹中的时间1）并转换为datetime用于排序
    df['departure'] = df['trajectory'].apply(lambda x: datetime.strptime(eval(x)[1], '%Y-%m-%d %H:%M:%S'))

    # 读取流量 CSV 文件
    df_flow = pd.read_csv(args.flow_df_filename)
    df_flow['datetime'] = pd.to_datetime(df_flow['time'])
    # 计算星期几（0=周一, 6=周日）
    df_flow['day_of_week'] = df_flow['datetime'].dt.dayofweek
    # 计算一天中的分钟数
    df_flow['minute_of_day'] = (df_flow['datetime'].dt.hour * 60 + df_flow['datetime'].dt.minute) // 5 # 五为时间颗粒度
    # 重新排列列，使 day_of_week 和 minute_of_day 位于第一列和第二列
    columns = ['day_of_week', 'minute_of_day'] + [col for col in df_flow.columns if col not in ['day_of_week', 'minute_of_day']]
    df_flow = df_flow[columns]
    # 将datatime列作为字典的索引, 行程时间rounded_time列可以根据字典的键获取值, 例如: 2021-06-01 00:15:00 : 0
    datetime_dic = {}
    for row in df_flow.values:
        datetime_dic[str(row[-1])] = len(datetime_dic)
    df_flow.drop('datetime', axis=1, inplace=True)
    data_flow = df_flow.values
    # 读取速度 CSV 文件
    data_speed = pd.read_csv(args.speed_df_filename).values
    # 归一化
    flow_mean = np.mean(data_flow[:, 3:])
    flow_std = np.std(data_flow[:, 3:])
    speed_mean = np.mean(data_speed[:, 1:])
    speed_std = np.std(data_speed[:, 1:])
    data_flow[:, 3:] = (data_flow[:, 3:] - flow_mean) / flow_std      # 流量归一化
    data_speed[:, 1:] = (data_speed[:, 1:] - speed_mean) / speed_std  # 速度归一化

    # 应用时间四舍五入函数, 例如: 2021-06-01 00:27:46取下，得到2021-06-01 00:25:00
    df['rounded_time'] = df['departure'].apply(round_to_five_minutes)
    # 按出发时间排序
    df = df.sort_values(by='departure').reset_index(drop=True)

    # 初始化x和y列表
    x_data = []
    y_data = []
    l_data = []
    speed_x_data = []
    flow_x_data = []
    dow_data = [] # day of week
    mod_data = [] # minute of day
    city_dict = dict() # 用于记录城市和索引的字典
    plate_dict = dict() # 用于记录车牌和索引的字典

    dis = []
    tra_times = []
    for _, row in df.iterrows():
        # 提取车牌、车型
        plate = row['vehicle_plate']  # 第一个整型
        city = plate[0]  # 城市信息（车牌第一个字符），第二个整型
        vehicle_type = row['vehicle_type'] # 第三个整型
        length = row['length']

        # 车牌和车型字典生成
        if city in city_dict: city = city_dict[city]
        else:
            city_dict[city]=len(city_dict)
            city = city_dict[city]
        if plate in plate_dict: plate = plate_dict[plate]
        else:
            plate_dict[plate]=len(plate_dict)
            plate = plate_dict[plate]

        # 提取轨迹信息（假设轨迹信息存储为字符串，需要解析）
        trajectory = eval(row['trajectory'])  # 假设轨迹列名为'轨迹'，存储为字符串格式

        # 获取出发时间（第一个时间）
        start_time_str = trajectory[1]  # 时间1
        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')

        if str(row['rounded_time']) not in datetime_dic:
            print(str(row['rounded_time']))
            print('匹配不上')
            continue
        elif datetime_dic[str(row['rounded_time'])] - 12 < 0:
            continue

        index = datetime_dic[str(row['rounded_time'])]
        flow_x_data.append(data_flow[index - 12: index, 3:])    # 添加流量数据
        speed_x_data.append(data_speed[index - 12: index, 1:])  # 添加速度数据
        dow_data.append(data_flow[index - 12: index, 1:2]) # 添加天信息
        mod_data.append(data_flow[index - 12: index, 0:1]) # 添加分钟信息

        # 计算day of week（0=周一, 6=周日）
        day_of_week = start_time.weekday() # 第四个整型

        # 计算minute of day
        minute_of_day = start_time.hour * 60 + start_time.minute # 第五个整型

        # 车子经过的路线
        route_type = row['route']  # 第六个整型

        # 提取所有距离
        distances = [float(trajectory[i]) for i in range(2, len(trajectory), 3)]  # 距离1, 距离2, ..., 距离n-1
        dis += distances

        # 计算行程时间（时间k+1 - 时间k，单位：分钟）
        travel_times = []
        for i in range(1, len(trajectory) - 3, 3):
            time_k = datetime.strptime(trajectory[i], '%Y-%m-%d %H:%M:%S')
            time_k1 = datetime.strptime(trajectory[i + 3], '%Y-%m-%d %H:%M:%S')
            travel_time = (time_k1 - time_k).total_seconds() / 60.0  # 转换为分钟
            travel_times.append(round(travel_time, 3)) # 保留小数点后面三位

        # 构建x：城市信息、车牌、车型、day of week、minute of day、距离列表
        x = [city, plate, vehicle_type, day_of_week, minute_of_day, route_type] + [distances]
        y = [travel_times, round(sum(travel_times), 3)]
        tra_times += travel_times

        x_data.append(x)
        y_data.append(y)
        l_data.append(length)

    del df

    # 转换为numpy数组
    x_data = np.array(x_data, dtype=object)
    flow_x_data = np.array(flow_x_data, dtype=np.float32)
    speed_x_data = np.array(speed_x_data, dtype=np.float32)
    dow_data = np.array(dow_data, dtype=np.int32)
    mod_data = np.array(mod_data, dtype=np.int32)
    y_data = np.array(y_data, dtype=object)
    l_data = np.array(l_data, dtype=np.int32)

    xdis_mean, xdis_std = np.mean(np.array(dis, dtype=np.float32)), np.std(np.array(dis, dtype=np.float32))
    ytra_mean, ytra_std = np.mean(np.array(tra_times, dtype=np.float32)), np.std(np.array(tra_times, dtype=np.float32))
    ytol_mean, ytol_std = np.mean(np.array(y_data[:, 1], dtype=np.float32)), np.std(np.array(y_data[:, 1], dtype=np.float32))
    del dis, tra_times

    # 划分轨迹数据集：70%训练，10%验证，20%测试
    num_samples = y_data.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    # train
    x_train, flow_x_train, speed_x_train, dow_train, mod_train, y_train, l_train = (
        x_data[:num_train],
        flow_x_data[:num_train],
        speed_x_data[:num_train],
        dow_data[:num_train],
        mod_data[:num_train],
        y_data[:num_train],
        l_data[:num_train]
    )
    # val
    x_val, flow_x_val, speed_x_val, dow_val, mod_val, y_val, l_val = (
        x_data[num_train: num_train + num_val],
        flow_x_data[num_train: num_train + num_val],
        speed_x_data[num_train: num_train + num_val],
        dow_data[num_train: num_train + num_val],
        mod_data[num_train: num_train + num_val],
        y_data[num_train: num_train + num_val],
        l_data[num_train: num_train + num_val]
    )
    # test
    x_test, flow_x_test, speed_x_test, dow_test, mod_test, y_test, l_test = (
        x_data[-num_test:],
        flow_x_data[-num_test:],
        speed_x_data[-num_test:],
        dow_data[-num_test:],
        mod_data[-num_test:],
        y_data[-num_test:],
        l_data[-num_test:]
    )

    for cat in ["train", "val", "test"]:
        _x, _flow_x, _speed_x, _dow, _mod, _y, _l = (
            locals()["x_" + cat],
            locals()["flow_x_" + cat],
            locals()["speed_x_" + cat],
            locals()["dow_" + cat],
            locals()["mod_" + cat],
            locals()["y_" + cat],
            locals()["l_" + cat]
        )
        print(cat, "x: ", _x.shape, "y:", _y.shape, "flow:", _flow_x.shape, "speed:", _speed_x.shape, "dow:", _dow.shape, "mod:", _mod.shape, "l:", _l.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            l=_l,
            flow_x=_flow_x,
            speed_x=_speed_x,
            dow=_dow,
            mod=_mod,
            xdis_mean = xdis_mean,
            xdis_std = xdis_std,
            ytra_mean = ytra_mean,
            ytra_std = ytra_std,
            ytol_mean = ytol_mean,
            ytol_std = ytol_std
        )

    return x_train, flow_x_train, speed_x_train, dow_train, mod_train, y_train, l_train, x_val, flow_x_val, speed_x_val, dow_val, mod_val, l_val, x_test, flow_x_test, speed_x_test, dow_test, mod_test, l_test, xdis_mean, xdis_std, ytra_mean, ytra_std, ytol_mean, ytol_std


# 示例用法
if __name__ == "__main__":
    print("Generating training data")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="", help="Output directory."
    )
    parser.add_argument(
        "--trajectory_df_filename",
        type=str,
        default="/Users/zouguojian/Travel-link/MT/data/travel.csv",
        help="Raw traffic readings.",
    )
    parser.add_argument(
        "--flow_df_filename",
        type=str,
        default="states/train_flow.csv",
        help="Raw traffic flow readings.",
    )

    parser.add_argument(
        "--speed_df_filename",
        type=str,
        default="states/train_speed.csv",
        help="Raw traffic speed readings.",
    )
    args = parser.parse_args()

    x_train, flow_x_train, speed_x_train, dow_train, mod_train, y_train, l_train, x_val, flow_x_val, speed_x_val, dow_val, mod_val, l_val, x_test, flow_x_test, speed_x_test, dow_test, mod_test, l_test, xdis_mean, xdis_std, ytra_mean, ytra_std, ytol_mean, ytol_std = process_travel_data(args)
    print("训练集大小:", len(x_train))
    print("流量训练集大小:", len(flow_x_train))
    print("速度训练集大小:", len(speed_x_train))
    print("验证集大小:", len(x_val))
    print("测试集大小:", len(x_test))
    print("距离的mean和std", xdis_mean, xdis_std)
    print("路段时间的mean和std", ytra_mean, ytra_std)
    print("路程时间的mean和std", ytol_mean, ytol_std)
    print("示例x_train[0]:", x_train[0])
    print("示例y_train[0]:", y_train[0])