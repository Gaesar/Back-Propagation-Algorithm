import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter
from tkinter import filedialog, ttk


from bp import bp

# BP神经网络训练函数，返回损失和拟合数据
def train_neural_network(*args):
    losses, losses2, re_predicted_values, re_actual_values, predicted_values, actual_values, epoched = bp(*args)
    # 这里只是一个示例，你需要根据实际情况实现神经网络的训练逻辑
    # 并返回损失值列表和拟合数据
    # losses = [0.1, 0.05, 0.02, 0.01, 0.005]  # 示例损失值
    # predictions = [1, 2, 3, 4, 5]  # 示例拟合数据
    return losses, losses2, re_predicted_values, re_actual_values, predicted_values, actual_values, epoched


# 绘制曲线的函数
def plot_curves(losses, losses2, re_predicted_values, re_actual_values, predicted_values, actual_values):
    global canvas, image_window  # 用于更新图像的全局变量
    # 创建一个新的Figure对象和1个子图
    # 假设我们有三个不同的数据集用于绘图
    # data_sets = [[losses], [predictions, actual_values]]

    # # 遍历所有画布，并更新它们
    # for i, (losses, predictions) in enumerate(data_sets):
    # 创建一个格式化器实例
    y_formatter = FormatStrFormatter('%1.2f')

    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.cla()
    ax.plot(losses, label='mse-train')
    ax.plot(losses2, label='mse-test')
    ax.set_title("Training")
    ax.yaxis.set_major_formatter(y_formatter)
    ax.legend(loc='upper right')
    last_x = len(losses) - 1  # 最后一个点的x坐标（索引）
    last_y = losses[-1]  # 最后一个点的y坐标（损失值）
    last_x2 = len(losses2) - 1  # 最后一个点的x坐标（索引）
    last_y2 = losses2[-1]  # 最后一个点的y坐标（损失值）

    # 在最后一个点的上方添加注释，显示其值
    ax.annotate(losses[-1],  # 注释文本，格式化为两位小数
                xy=(last_x, last_y),  # 注释位置
                xytext=(last_x-last_x*0.125, last_y + 0.01),  # 文本显示位置，稍微偏移
                textcoords='data',  # 使用数据坐标
                arrowprops=dict(facecolor='black', arrowstyle='->'), color='blue')  # 箭头样
    ax.annotate(losses2[-1],  # 注释文本，格式化为两位小数
                xy=(last_x2, last_y2),  # 注释位置
                xytext=(last_x2-last_x2*0.25, last_y2 + 0.05),  # 文本显示位置，稍微偏移
                textcoords='data',  # 使用数据坐标
                arrowprops=dict(facecolor='black', arrowstyle='->'), color='red')  # 箭头样
    # 清除旧的图像
    canvases[0].get_tk_widget().destroy()

    # 创建新的图像画布
    canvases[0] = FigureCanvasTkAgg(fig, master=canvas_frames[0])
    canvases[0].draw()
    canvases[0].get_tk_widget().pack(side="left", fill="both", expand=True)

    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.cla()
    ax.plot(re_actual_values, label='actual', )
    ax.plot(re_predicted_values, label='predict')
    ax.set_title("Recalling")
    ax.yaxis.set_major_formatter(y_formatter)
    ax.legend(loc='upper right')
    # 清除旧的图像
    canvases[1].get_tk_widget().destroy()

    # 创建新的图像画布
    canvases[1] = FigureCanvasTkAgg(fig, master=canvas_frames[1])
    canvases[1].draw()
    canvases[1].get_tk_widget().pack(side="left", fill="both", expand=True)

    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.cla()
    ax.plot(actual_values, label='actual')
    ax.plot(predicted_values, label="predict")
    ax.legend(loc='upper right')
    ax.yaxis.set_major_formatter(y_formatter)
    ax.set_title("Fitting")
    # 清除旧的图像
    canvases[2].get_tk_widget().destroy()

    # 创建新的图像画布
    canvases[2] = FigureCanvasTkAgg(fig, master=canvas_frames[2])
    canvases[2].draw()
    canvases[2].get_tk_widget().pack(side="left", fill="both", expand=True)




# Tkinter界面的主函数
def main():
    global window, canvas, image_window,canvases,canvas_frames # 全局变量声明
    window = tk.Tk()
    window.title("BP神经网络可视化系统")

    image_window = tk.Frame(window)
    image_window.pack(side="top", fill="both", expand=True)

    # 初始化三个空白的图像画布，每个都是1x1的布局
    blank_figs = [Figure(figsize=(3, 3), dpi=100) for _ in range(3)]
    # 为每个画布创建一个Frame作为容器，并将它们水平排列
    canvas_frames = []
    canvases = []
    for i in range(len(blank_figs)):
        canvas_frame = tk.Frame(image_window)
        canvas_frames.append(canvas_frame)
        canvas_frame.pack(side="left", fill="both", expand=True)
        canvas = FigureCanvasTkAgg(blank_figs[i], master=canvas_frame)
        canvases.append(canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

    # 将canvas_frames水平排列在image_window中
    tk.Frame(image_window, width=10).pack(side="left")  # 一个10像素宽的分隔条
    for canvas_frame in canvas_frames:
         canvas_frame.pack(side="left", fill="both", expand=True, padx=10)

    # 创建一个Frame用于放置参数输入框

    parameters_frame = tk.Frame(window)
    parameters_frame.pack(pady=20, anchor='center', fill="x")

    # 创建参数输入框和对应的标签
    labels = ["学习率：", "迭代次数：", "均方误差：", "隐藏层神经元个数："]
    entry_widgets = []
    for i, label in enumerate(labels):
        tk.Label(parameters_frame, text=label).grid(row=i, column=0, padx=10, pady=5, sticky='e')
        entry_widgets.append(tk.Entry(parameters_frame))
        entry_widgets[i].grid(row=i, column=1, padx=10)

    # 添加数据集路径输入框和选择按钮
    dataset_label = tk.Label(parameters_frame, text="数据集路径：")
    dataset_label.grid(row=0, column=2, padx=10, pady=5, sticky='e')
    dataset_entry = tk.Entry(parameters_frame, width=30)
    dataset_entry.grid(row=0, column=3, padx=10)
    entry_widgets.append(dataset_entry)

    def open_file_dialog_dataset():
        file_path = filedialog.askopenfilename()
        dataset_entry.delete(0, tk.END)
        dataset_entry.insert(0, file_path)

    file_button_dataset = tk.Button(parameters_frame, text="选择文件", command=open_file_dialog_dataset)
    file_button_dataset.grid(row=0, column=4, padx=10)

    # # 添加模型路径输入框和选择按钮
    # model_label = tk.Label(parameters_frame, text="模型路径：")
    # model_label.grid(row=1, column=2, padx=10, pady=5, sticky='e')
    # model_entry = tk.Entry(parameters_frame, width=30)
    # model_entry.grid(row=1, column=3, padx=10)
    # entry_widgets.append(model_entry)
    # def open_file_dialog_model():
    #     file_path = filedialog.asksaveasfilename()
    #     model_entry.delete(0, tk.END)
    #     model_entry.insert(0, file_path)
    #
    # file_button_model = tk.Button(parameters_frame, text="选择文件", command=open_file_dialog_model)
    # file_button_model.grid(row=1, column=4, padx=10)

    # 添加激活函数下拉框
    activation_label = tk.Label(parameters_frame, text="激活函数：")
    activation_label.grid(row=1, column=2, padx=10, pady=5, sticky='e')
    activation_var = tk.StringVar()
    entry_widgets.append(activation_var)
    activation_options = ["sigmoid", "tanh"]
    activation_dropdown = ttk.Combobox(parameters_frame, textvariable=activation_var, values=activation_options)
    activation_dropdown.grid(row=1, column=3, padx=10, pady=5, sticky='ew')
    activation_dropdown.current(0)  # 默认选择第一个选项


    # 创建一个按钮，点击时获取参数并绘制曲线
    def on_show_click():
        try:
            # 获取输入参数，对于最后一个参数，我们假设它是一个以逗号分隔的字符串
            params = [float(w.get()) for i, w in enumerate(entry_widgets) if i < 3]
            custom_param_str = entry_widgets[3].get()  # 获取自定义参数的字符串
            custom_params = [int(x.strip()) for x in
                             custom_param_str.split(',')] if custom_param_str else []  # 将字符串分割并转换为整数列表
            params.append(custom_params)  # 将自定义参数列表添加到参数列表中

            for i in custom_params:
                if i <= 0:
                    raise ValueError
            if int(params[1]) <= 1 or params[2] >= 0.1:
                raise ValueError
            for i in range(4, 6):
                params.append(entry_widgets[i].get())
            if params[4] == "":
                raise ValueError
            losses, losses2, re_predicted_values, re_actual_values, predicted_values, actual_values, epoched = train_neural_network(*params)
            plot_curves(losses, losses2, re_predicted_values, re_actual_values, predicted_values, actual_values)
            if epoched != params[1]:
                tk.messagebox.showinfo("Success!", "提前达到预计均方误差！")
            else:
                tk.messagebox.showinfo("Success!", "未达到预计均方误差，到达预定迭代次数！")
        except ValueError:
            tk.messagebox.showerror("Input Error", "输入参数错误！")


    show_button = tk.Button(parameters_frame, text="Train", command=on_show_click)
    show_button.grid(row=4, column=2, columnspan=2, padx=10, pady=5)

    window.mainloop()


def run():
    main()

