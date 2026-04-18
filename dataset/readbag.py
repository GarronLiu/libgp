import rosbag
import pandas as pd
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from scipy.interpolate import interp1d
from mavros_msgs.msg import RCOut
import math
from tf.transformations import euler_from_quaternion

# 设置输入和输出文件路径
bag_prefix = "8mUSV_ZigZag20deg-20deg"
bag_file = "dataset/rosbag/" + bag_prefix + ".bag"
csv_dir = "dataset/rosbag/" + bag_prefix + ".csv"  # 输出CSV文件路径

# 初始化数据存储列表
odom_data = []
rc_data = []

# 定义一个函数来处理rosbag文件
def process_bag(bag_file):
    with rosbag.Bag(bag_file, "r") as bag:
        # 遍历rosbag中的每个消息
        for topic, msg, t in bag.read_messages():
            # 解析 /odom 数据（船舶模型的 u, v, r, x, y, psi）
            if topic == "/mavros/local_position/odom":
                u = msg.twist.twist.linear.x  # 船舶前进速度（线速度）
                v = msg.twist.twist.linear.y  # 船舶横向速度
                r = msg.twist.twist.angular.z  # 船舶角速度（转角速度）
                x = msg.pose.pose.position.x  # 船舶x坐标
                y = msg.pose.pose.position.y  # 船舶y坐标
                
                # 计算 psi (yaw)
                orientation_q = msg.pose.pose.orientation
                orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
                (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
                psi = yaw

                timestamp = t.to_sec()  # 获取时间戳（秒）
                odom_data.append(
                    {"time": timestamp, "u": u, "v": v, "r": r, "x": x, "y": y, "psi": psi}
                )

            # 解析 /rc/out 数据（控制器输入）
            elif topic == "/mavros/rc/out": # 注意：用户提示中提到 rc/out，但代码原逻辑是 rc/override，此处保留原逻辑或根据实际bag修改
                throttle_left = msg.channels[0]  # 获取通道数据
                throttle_right = msg.channels[2]  # 获取通道数据
                timestamp = t.to_sec()  # 获取时间戳（秒）
                
                # 归一化处理
                throttle_left = (throttle_left - 1000) / (2000 - 1000) - 0.5
                throttle_right = (throttle_right - 1000) / (2000 - 1000) - 0.5
                
                rc_data.append(
                    {
                        "time": timestamp,
                        "throttle_left": throttle_left,
                        "throttle_right": throttle_right,
                    }
                )

# 处理rosbag文件
process_bag(bag_file)

# ---------------- 数据清洗与排序 ----------------

# 舍弃 odom_data 中 timestamp 异常的点
if odom_data:
    odom_data = [
        data
        for i, data in enumerate(odom_data)
        if i == 0 or data["time"] > odom_data[i - 1]["time"]
    ]
    odom_data = sorted(odom_data, key=lambda x: x["time"])
else:
    print("No odom data collected.")

# 舍弃 rc_data 中 timestamp 异常的点
if rc_data:
    rc_data = [
        data
        for i, data in enumerate(rc_data)
        if i == 0 or data["time"] > rc_data[i - 1]["time"]
    ]
    rc_data = sorted(rc_data, key=lambda x: x["time"])
else:
    print("No rc data collected.")

# ---------------- 时间对齐与截取 ----------------

if not odom_data or not rc_data:
    print("Insufficient data to proceed.")
    exit()

# 获取公共时间范围
start_time_abs = max(odom_data[0]["time"], rc_data[0]["time"])
end_time_abs = min(odom_data[-1]["time"], rc_data[-1]["time"])

# 如果需要截取特定片段（如原代码中的后半段），可以在这里调整 start_time_abs
# duration = end_time_abs - start_time_abs
# start_time_abs = start_time_abs + 0.5 * duration 

# 过滤数据以匹配时间范围
odom_data = [d for d in odom_data if start_time_abs <= d["time"] <= end_time_abs]
rc_data = [d for d in rc_data if start_time_abs <= d["time"] <= end_time_abs]

# ---------------- 插值重采样 ----------------

# 定义采样频率
sampling_frequency = 20  # Hz
sampling_period = 1.0 / sampling_frequency

# 生成采样时间点
sample_times = np.arange(start_time_abs, end_time_abs, sampling_period)

# 提取原始数据用于插值
odom_times = [d["time"] for d in odom_data]
odom_u = [d["u"] for d in odom_data]
odom_v = [d["v"] for d in odom_data]
odom_r = [d["r"] for d in odom_data]
odom_x = [d["x"] for d in odom_data]
odom_y = [d["y"] for d in odom_data]
odom_psi = [d["psi"] for d in odom_data]

rc_times = [d["time"] for d in rc_data]
rc_left = [d["throttle_left"] for d in rc_data]
rc_right = [d["throttle_right"] for d in rc_data]

# 创建插值函数
f_u = interp1d(odom_times, odom_u, kind="linear", fill_value="extrapolate")
f_v = interp1d(odom_times, odom_v, kind="linear", fill_value="extrapolate")
f_r = interp1d(odom_times, odom_r, kind="linear", fill_value="extrapolate")
f_x = interp1d(odom_times, odom_x, kind="linear", fill_value="extrapolate")
f_y = interp1d(odom_times, odom_y, kind="linear", fill_value="extrapolate")
f_psi = interp1d(odom_times, odom_psi, kind="linear", fill_value="extrapolate")

f_th_left = interp1d(rc_times, rc_left, kind="linear", fill_value="extrapolate")
f_th_right = interp1d(rc_times, rc_right, kind="linear", fill_value="extrapolate")

# 执行插值
res_u = f_u(sample_times)
res_v = f_v(sample_times)
res_r = f_r(sample_times)
res_x = f_x(sample_times)
res_y = f_y(sample_times)
res_psi = f_psi(sample_times)
res_th_left = f_th_left(sample_times)
res_th_right = f_th_right(sample_times)

#对 u，v，r 进行平滑处理（可选）
window_size = 20  # 平滑窗口大小
res_u = np.convolve(res_u, np.ones(window_size) / window_size, mode="same")
res_v = np.convolve(res_v, np.ones(window_size) / window_size, mode="same")
res_r = np.convolve(res_r, np.ones(window_size) / window_size, mode="same")


# ---------------- 组合数据并保存 ----------------

# 时间归零（以第一帧时间戳为原点）
relative_times = sample_times - sample_times[0]

interpolated_data = []
for i in range(len(sample_times)):
    interpolated_data.append({
        "time": relative_times[i],
        "x": res_x[i],
        "y": res_y[i],
        "psi": res_psi[i],
        "u": res_u[i],
        "v": res_v[i],
        "r": res_r[i],
        "throttle_left": res_th_left[i],
        "throttle_right": res_th_right[i]
    })

# 转换为 pandas DataFrame
df_interpolated = pd.DataFrame(interpolated_data)

# 按照要求的列顺序排列
columns_order = ["time", "x", "y", "psi", "u", "v", "r", "throttle_left", "throttle_right"]
df_interpolated = df_interpolated[columns_order]

# 保存为CSV文件
df_interpolated.to_csv(csv_dir, index=False)
print(f"Interpolated data saved to {csv_dir}")
