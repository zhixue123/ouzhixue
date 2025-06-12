import pandas as pd
import matplotlib.pyplot as plt

file_path = '1.xlsx'  # 替换为你的Excel文件路径
sheet_name = 'cycle'   # 替换为你的工作表名
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 正确选择列（假设第1列是X轴，第6列是Y轴）
x_column = data.columns[0]  # 第1列的列名
y_column = data.columns[5]  # 第6列的列名

target_value = "1000"
# 查找第一次出现"400"的行索引（Python从0开始）
first_target_row = data[data[x_column].astype(str) == target_value].index[0]

start_row = 1  # 从第2行开始（Python索引1）
end_row = first_target_row  # 到第一次出现"400"的行

# 提取指定行范围的列数据（使用列名）
x_data = data.loc[start_row:end_row, x_column].values  # 转为numpy数组
y_data = data.loc[start_row:end_row, y_column].values  # 转为numpy数组

# 绘制曲线图
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, marker='o', linestyle='-', color='b', label='Data')

# 添加标题和标签
plt.title(f'Data Plot (Rows {start_row+1} to {end_row+1})', fontsize=16)
plt.xlabel(x_column, fontsize=12)
plt.ylabel(y_column, fontsize=12)
plt.legend()
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()