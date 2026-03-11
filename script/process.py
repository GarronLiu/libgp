import pandas as pd
from datetime import datetime

sample_num = 200
data_start_col_id = 1

# 读取combined.csv文件
# df = pd.read_excel('DATA.xlsx')

df = pd.read_csv('/home/garronliu/2_Tracking_control/GP-MPC/libgp-master/script/combined.csv')                                       

df = df.iloc[:,data_start_col_id:]

df_test = df.copy()
df_test.insert(0, 'new_column', 0)

import matplotlib.pyplot as plt

# 绘制 u, v, r, throttle_left, throttle_right 的时历图
plt.figure(figsize=(12, 8))

plt.subplot(5, 1, 1)
plt.plot(df.index, df.iloc[:, 0], label='u')
plt.legend()
plt.title('u over time')

plt.subplot(5, 1, 2)
plt.plot(df.index, df.iloc[:, 1], label='v')
plt.legend()
plt.title('u over time')

plt.subplot(5, 1, 3)
plt.plot(df.index, df.iloc[:, 2], label='r')
plt.legend()
plt.title('u over time')

plt.subplot(5, 1, 4)
plt.plot(df.index, df.iloc[:, 3], label='throttle_left')
plt.legend()
plt.title('Throttle Left over time')

plt.subplot(5, 1, 5)
plt.plot(df.index, df.iloc[:, 4], label='throttle_right')
plt.legend()
plt.title('Throttle Right over time')

plt.tight_layout()
plt.show()
# u
df1 = df.copy()
df1.insert(0, 'new_column', range(1, len(df) + 1))

# 向上移动一行，并删除最后一行
df1.iloc[:, 0] = df.iloc[:, 0].shift(-1)
df1 = df1.dropna()
df1 = df1.sample(n=sample_num, random_state=1)

# v
df2 = df.copy()
df2.insert(0, 'new_column', range(1, len(df) + 1))

# 向上移动一行，并删除最后一行
df2.iloc[:, 0] = df.iloc[:, 1].shift(-1)
df2 = df2.dropna()
df2 = df2.sample(n=sample_num, random_state=1)

# r
df3 = df.copy()
df3.insert(0, 'new_column', range(1, len(df) + 1))

# 向上移动一行，并删除最后一行
df3.iloc[:, 0] = df.iloc[:, 2].shift(-1)
df3 = df3.dropna()
df3 = df3.sample(n=sample_num, random_state=1)
# 获取当前时间
current_time = datetime.now().strftime("# %a %b %d %H:%M:%S %Y")

# 定义文件头部信息
header = f"""{current_time}

# input dimensionality
5

# covariance function
CovSum(CovSEard, CovNoise)

# log-hyperparameter
0 0 0 0 0 0 -2.249110744 

# data (target value in first column) u v r tl tr"""

absolut_pat = "/home/garronliu/2_Tracking_control/GP-MPC/libgp-master/config/"

# 将数据转换为指定格式
data = df1.to_csv(sep=' ', index=False, header=False)

# 将结果写入新文件
with open(absolut_pat+'output_u_sim.txt', 'w') as f:
    f.write(header + '\n' + data)

data = df2.to_csv(sep=' ', index=False, header=False)

# 将结果写入新文件
with open(absolut_pat+'output_v_sim.txt', 'w') as f:
    f.write(header + '\n' + data)

data = df3.to_csv(sep=' ', index=False, header=False)

# 将结果写入新文件
with open(absolut_pat+'output_r_sim.txt', 'w') as f:
    f.write(header + '\n' + data)

data = df_test.to_csv(sep=' ', index=False, header=False)

# 将结果写入新文件
with open(absolut_pat+'output_test.txt', 'w') as f:
    f.write(header + '\n' + data)