'''本功能用于行程时间预测数据集划分和生成'''
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import os

def process_travel_data(args):
    # 读取CSV文件
    df = pd.read_csv(args.traffic_df_filename)
    # 提取出发时间（轨迹中的时间1）并转换为datetime用于排序
    df['departure'] = df['trajectory'].apply(lambda x: datetime.strptime(eval(x)[1], '%Y-%m-%d %H:%M:%S'))
    print(df)
    # 按出发时间排序
    df = df.sort_values(by='departure').reset_index(drop=True)

    # 初始化x和y列表
    x_data = []
    y_data = []
    l_data = []
    city_dict = dict() # 用于记录城市和索引的字典
    plate_dict = dict() # 用于记录车牌和索引的字典
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

        # 计算day of week（0=周一, 6=周日）
        day_of_week = start_time.weekday() # 第四个整型

        # 计算minute of day
        minute_of_day = start_time.hour * 60 + start_time.minute # 第五个整型

        # 车子经过的路线
        route_type = row['route']  # 第六个整型

        # 提取所有距离
        distances = [float(trajectory[i]) for i in range(2, len(trajectory), 3)]  # 距离1, 距离2, ..., 距离n-1

        # 计算行程时间（时间k+1 - 时间k，单位：分钟）
        travel_times = []
        for i in range(1, len(trajectory) - 3, 3):
            time_k = datetime.strptime(trajectory[i], '%Y-%m-%d %H:%M:%S')
            time_k1 = datetime.strptime(trajectory[i + 3], '%Y-%m-%d %H:%M:%S')
            travel_time = (time_k1 - time_k).total_seconds() / 60.0  # 转换为分钟
            travel_times.append(round(travel_time, 3)) # 保留小数点后面三位

        # 构建x：城市信息、车牌、车型、day of week、minute of day、距离列表
        x = [city, plate, vehicle_type, day_of_week, minute_of_day, route_type] + [distances]
        y = [travel_times, sum(travel_times)]
        len_ = [length]

        x_data.append(x)
        y_data.append(y)
        l_data.append(len_)

    del df

    # 转换为numpy数组
    x_data = np.array(x_data, dtype=object)
    y_data = np.array(y_data, dtype=object)

    # 划分轨迹数据集：70%训练，10%验证，20%测试
    num_samples = y_data.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    # train
    x_train, y_train, l_train = x_data[:num_train], y_data[:num_train], l_data[:num_train]
    # val
    x_val, y_val, l_val = (
        x_data[num_train: num_train + num_val],
        y_data[num_train: num_train + num_val],
        l_data[num_train: num_train + num_val]
    )
    # test
    x_test, y_test, l_test = x_data[-num_test:], y_data[-num_test:], l_data[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y
        )

    return x_train, y_train, l_train, x_val, y_val, l_val, x_test, y_test, l_test


# 示例用法
if __name__ == "__main__":
    print("Generating training data")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="/Users/zouguojian/Travel-link/MT/data/travel.csv",
        help="Raw traffic readings.",
    )
    args = parser.parse_args()

    x_train, y_train, l_train, x_val, y_val, l_val, x_test, y_test, l_test = process_travel_data(args)
    print("训练集大小:", len(x_train))
    print("验证集大小:", len(x_val))
    print("测试集大小:", len(x_test))
    print("示例x_train[0]:", x_train[0])
    print("示例y_train[0]:", y_train[0])