import rosbag
import pandas as pd
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from scipy.interpolate import interp1d
from mavros_msgs.msg import RCOut  # 导入 mavros_msgs 中的 RCOut 消息类型

# 设置输入和输出文件路径
bag_file1 = '/home/garronliu/2_Tracking_control/mysindy/identification/Complex_odom.bag'
bag_file2 = '/home/garronliu/2_Tracking_control/mysindy/identification/Complex_rc.bag'
odom_csv_dir = '/home/garronliu/2_Tracking_control/mysindy/odom.csv'  # 输出CSV文件路径
rc_csv_dir = '/home/garronliu/2_Tracking_control/mysindy/rc.csv'

# 初始化数据存储列表
odom_data = []
rc_data = []

# 定义一个函数来处理rosbag文件
def process_bag(bag_file):
    with rosbag.Bag(bag_file, 'r') as bag:
        # 遍历rosbag中的每个消息
        for topic, msg, t in bag.read_messages():
            # 解析 /odom 数据（船舶模型的 u, v, r）
            if topic == '/mavros/local_position/odom':  # 假设船舶的odom话题是/odom
                u = msg.twist.twist.linear.x  # 船舶前进速度（线速度）
                v = msg.twist.twist.linear.y  # 船舶横向速度
                r = msg.twist.twist.angular.z  # 船舶角速度（转角速度）
                timestamp = t.to_sec()  # 获取时间戳（秒）
                odom_data.append({'time': timestamp, 'u': u, 'v': v, 'r': r})

            # 解析 /rc/out 数据（控制器输入）
            elif topic == '/mavros/rc/out':
                throttle_left = msg.channels[0]  # 获取通道数据
                throttle_right = msg.channels[2]  # 获取通道数据
                timestamp = t.to_sec()  # 获取时间戳（秒）
                assert 1000 <= throttle_left <= 2000, f"throttle_left out of range: {throttle_left}"
                assert 1000 <= throttle_right <= 2000, f"throttle_right out of range: {throttle_right}"
                throttle_left = 2 * (throttle_left - 1000) / (2000 - 1000) - 1
                throttle_right = 2 * (throttle_right - 1000) / (2000 - 1000) - 1
                rc_data.append({'time': timestamp, 'throttle_left': throttle_left, 'throttle_right': throttle_right})

# 处理两个rosbag文件
process_bag(bag_file1)
process_bag(bag_file2)

# 检查是否有数据
if not odom_data:
    print("No odom data collected from bag files.")
else:
    # 根据timestamp升序对odom_data进行重新排序
    odom_data = sorted(odom_data, key=lambda x: x['time'])
    # 转换为 pandas DataFrame
    df = pd.DataFrame(odom_data)
    # 保存为CSV文件
    df.to_csv(odom_csv_dir, index=False)
    print(f'Data saved to {odom_csv_dir}')

if not rc_data:
    print("No rc data collected from bag files.")
else:
    rc_data = sorted(rc_data, key=lambda x: x['time'])
    # 转换为 pandas DataFrame
    df = pd.DataFrame(rc_data)
    # 保存为CSV文件
    df.to_csv(rc_csv_dir, index=False)
    print(f'Data saved to {rc_csv_dir}')


# 检查combined_data的time数据是否严格递增
times = [entry['time'] for entry in rc_data]
assert all(x < y for x, y in zip(times, times[1:])), "时间戳数据不是严格递增的！"

times = [entry['time'] for entry in odom_data]
assert all(x < y for x, y in zip(times, times[1:])), "时间戳数据不是严格递增的！"


# 获取rc_data和odom_data的开始时间戳
odom_start_time = odom_data[0]['time']
rc_start_time = rc_data[0]['time']
# 定义采样频率
sampling_frequency = 10  # 5 Hz
sampling_period = 1.0 / sampling_frequency

# 获取插值采样的时间戳范围
start_time = max(odom_start_time, rc_start_time)
end_time = min(odom_data[-1]['time'], rc_data[-1]['time'])
sample_times = np.arange(start_time, end_time, sampling_period)

# 提取原始数据的时间戳和对应的值
odom_times = [data['time'] for data in odom_data]
odom_u = [data['u'] for data in odom_data]
odom_v = [data['v'] for data in odom_data]
odom_r = [data['r'] for data in odom_data]

rc_times = [data['time'] for data in rc_data]
throttle_left = [data['throttle_left'] for data in rc_data]
throttle_right = [data['throttle_right'] for data in rc_data]

# 创建插值函数
interp_odom_u = interp1d(odom_times, odom_u, kind='linear', fill_value="extrapolate")
interp_odom_v = interp1d(odom_times, odom_v, kind='linear', fill_value="extrapolate")
interp_odom_r = interp1d(odom_times, odom_r, kind='linear', fill_value="extrapolate")

interp_throttle_left = interp1d(rc_times, throttle_left, kind='linear', fill_value="extrapolate")
interp_throttle_right = interp1d(rc_times, throttle_right, kind='linear', fill_value="extrapolate")

# 对采样时间戳进行插值采样
sampled_odom_u = interp_odom_u(sample_times)
sampled_odom_v = interp_odom_v(sample_times)
sampled_odom_r = interp_odom_r(sample_times)

sampled_throttle_left = interp_throttle_left(sample_times)
sampled_throttle_right = interp_throttle_right(sample_times)

# 组合插值后的数据
interpolated_data = []
for i, t in enumerate(sample_times):
    interpolated_data.append({
        'time': t,
        'u': sampled_odom_u[i],
        'v': sampled_odom_v[i],
        'r': sampled_odom_r[i],
        'throttle_left': sampled_throttle_left[i],
        'throttle_right': sampled_throttle_right[i]
    })

# 转换为 pandas DataFrame
df_interpolated = pd.DataFrame(interpolated_data)

# 保存为CSV文件
interpolated_csv_dir = '/home/garronliu/2_Tracking_control/GP-MPC/libgp-master/script/combined.csv'
df_interpolated.to_csv(interpolated_csv_dir, index=False)
print(f'Interpolated data saved to {interpolated_csv_dir}')

