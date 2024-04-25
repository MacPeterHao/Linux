# # -*- coding: gbk -*-
import re
import matplotlib.pyplot as plt
import numpy as np
import random

# 中文字体支持
plt.rc("font", family='SimSun')


def extract_act_values(file_path):
    act_values = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'Act:\s+(\d+)', line)
            if match:
                act_values.append(int(match.group(1)))
    return act_values


def log_normal_pdf(x, mu, sigma):
    return np.exp(-((np.log(x) - mu) ** 2 / (2 * sigma ** 2))) / (x * sigma * np.sqrt(2 * np.pi))


def plot_combined_histogram(datasets):
    plt.figure(figsize=(10, 5))

    for dataset_name, color in datasets:
        file_path = f'data/100%/{dataset_name}.txt'
        act_values = extract_act_values(file_path)
        act_values = [random.uniform(0.001, 8) if val == 0 else val * 0.001 for val in act_values]

        # 转换为对数形式
        log_act_values = np.log(act_values)

        # 计算对数变换后的均值和标准差
        mean_log = np.mean(log_act_values)
        std_log = np.std(log_act_values)

        act_mean = np.mean(act_values)
        act_std = np.std(act_values)

        # 原始尺度上的值范围
        x_values = np.linspace(min(act_values), max(act_values), 1000)

        # 计算概率密度
        pdf_values = log_normal_pdf(x_values, mean_log, std_log)

        if dataset_name == 'Linux':
            bins = 800
        else:
            bins = 30

        # 绘制直方图和概率密度函数
        plt.hist(act_values, bins=bins, alpha=0.5, color=color, edgecolor='black', density=True,
                 label=f'{dataset_name} - Mean: {act_mean:.2f}, SD: {act_std:.2f}')
        plt.plot(x_values, pdf_values, color=color, linewidth=2)

    plt.xlim((0, 120))
    plt.xlabel('响应延时 (us)', fontsize=16)
    plt.ylabel('概率密度', fontsize=16)
    plt.title("不同系统100% CPU负载下延时直方图与对数正态分布模型拟合", fontsize=16)
    plt.legend()
    plt.savefig("./img/combined_histogram.svg", dpi=450)
    plt.show()


def plot_fit_curves(datasets):
    plt.figure(figsize=(10, 5))

    for dataset_name, color in datasets:
        file_path = f'data/100%/{dataset_name}.txt'
        act_values = extract_act_values(file_path)
        act_values = [random.uniform(0.001, 8) if val == 0 else val * 0.001 for val in act_values]

        # 转换为对数形式
        log_act_values = np.log(act_values)

        # 计算对数变换后的均值和标准差
        mean_log = np.mean(log_act_values)
        std_log = np.std(log_act_values)

        act_mean = np.mean(act_values)
        act_std = np.std(act_values)

        # 原始尺度上的值范围
        x_values = np.linspace(min(act_values), max(act_values), 1000)

        # 计算概率密度
        pdf_values = log_normal_pdf(x_values, mean_log, std_log)

        # 绘制对数正态分布曲线
        plt.plot(x_values, pdf_values, color=color, label=f'{dataset_name} - 平均值: {act_mean:.2f}, 标准差: {act_std:.2f}',
                 linewidth=2)

    plt.xlim((0, 120))
    plt.xlabel('响应延时 (us)', fontsize=16)
    plt.ylabel('概率密度', fontsize=16)
    plt.title("不同系统100% CPU负载下任务响应延迟的对数正态分布模型拟合", fontsize=16)
    plt.legend()
    plt.savefig("./img/fit_curves.svg", dpi=450)
    plt.show()


def plot_hist(datasets):
    for dataset_name, color in datasets:
        file_path = f'data/100%/{dataset_name}.txt'
        act_values = extract_act_values(file_path)
        act_values = [random.uniform(0.001, 8) if val == 0 else val * 0.001 for val in act_values]

        # 转换为对数形式
        log_act_values = np.log(act_values)

        # 假设对数变换后的均值和标准差
        mean_log = np.mean(log_act_values)
        std_log = np.std(log_act_values)

        act_mean = np.mean(act_values)
        act_std = np.std(act_values)

        # 原始尺度上的值范围
        x_values = np.linspace(min(act_values), max(act_values), 1000)

        # 对每个x值计算概率密度
        pdf_values = log_normal_pdf(x_values, mean_log, std_log)

        print(act_values)

        # 绘制直方图
        if dataset_name == 'Linux':
            bins = 800
        else:
            bins = 30
        plt.figure(figsize=(10, 5))
        plt.hist(act_values, bins=bins, alpha=0.5, color='blue', edgecolor='black', density=True)
        plt.xlim((0, 120))
        plt.xlabel('响应延时 (us)', fontsize=16)
        plt.ylabel('概率密度', fontsize=16)

        # 绘制对数正态分布曲线
        plt.plot(x_values, pdf_values, 'k', linewidth=2)
        plt.title(f"{dataset_name} 系统100% CPU负载下延时直方图与对数正态分布模型拟合\n "
                  f"平均延时= {act_mean:.2f}, 标准差 = {act_std:.2f}", fontsize=16)

        plt.savefig(f"./img/{dataset_name}_100.svg", dpi=450)
        plt.show()


def plot_cdf(datasets):
    plt.figure(figsize=(10, 5))

    for dataset_name, color in datasets:
        file_path = f'data/100%/{dataset_name}.txt'
        act_values = extract_act_values(file_path)
        act_values = [random.uniform(0.001, 8) if val == 0 else val * 0.001 for val in act_values]

        # 排序数据
        act_values_sorted = np.sort(act_values)
        # 计算CDF
        yvals = np.arange(len(act_values_sorted)) / float(len(act_values_sorted) - 1)

        # 绘制CDF
        plt.plot(act_values_sorted, yvals, label=f'{dataset_name}', color=color, linewidth=2)

        # 找到95%累计概率对应的延迟
        value_95th = np.interp(0.95, yvals, act_values_sorted)
        plt.axvline(x=value_95th, color=color, linestyle='--', linewidth=1)
        plt.annotate(f'{value_95th:.3f}', xy=(value_95th, 0.95), xytext=(value_95th, 0.8),
                     textcoords='data', arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black'),
                     color='black')

        # 添加95%的水平虚线
        plt.axhline(y=0.95, color='gray', linestyle='--', linewidth=1)
        plt.text(min(plt.xlim()) - 2, 0.92, '0.95', fontsize=10, va='bottom', ha='left')

    plt.xlim((0, 120))
    plt.xlabel('响应延时 (us)', fontsize=16)
    plt.ylabel('累计概率', fontsize=16)
    plt.title("不同系统100% CPU负载下任务响应延迟的累计分布", fontsize=16)
    plt.legend(fontsize=12)
    # plt.grid(True)
    plt.savefig("./img/cdf_comparison.svg", dpi=450)
    plt.show()


def main():
    datasets = [("Xenomai", 'blue'), ("Linux", 'green'), ("Preempt_RT", 'red')]
    # plot_combined_histogram(datasets)
    plot_fit_curves(datasets)
    # plot_hist(datasets)
    plot_cdf(datasets)


if __name__ == '__main__':
    main()

