# 反向传播算法可视化实现

## 目录结构

```
—BP-Algorithm
|-bp.py # 实现反向传播算法
|-main.py # 程序的入口
|-pannel.py # 实现可视化界面
|-dataset # 训练数据集
  |-f=sinx.npy # 训练数据y=sinx
  |-f=x1+x2+x3.npy # 训练数据y=x1+x2+x3
  |-f=x2.npy # 训练数据y=x^2
```

## 运行代码

运行main.py程序的入口函数即可。

## 界面展示

![Snipaste_2024-06-30_21-17-45](https://cdn.jsdelivr.net/gh/Gaesar/Gaesar.github.io@main/pic/202406302147050.jpg)

### 参数说明：

1.学习率：输入浮点数。

2.迭代次数：输入整数。

3.均方误差：输入浮点数。

4.隐藏层神经元个数：用','分隔，表示每层有多少个神经元，例如: 3,3,3，表示有3层，每层3个神经元。

5.数据集路径：以npy文件保存，每个数据项由[x1, x2, ..., xn, y]组成。

6.激活函数：支持两种激活函数sigmoid、tanh。

## 示例

![image-20240630215516514](https://cdn.jsdelivr.net/gh/Gaesar/Gaesar.github.io@main/pic/202406302155590.png)

1.Training展示的是训练曲线，记录了每次迭代的均方误差

2.Recalling展示的回想曲线，记录了在验证集上的拟合曲线

3.Fitting展示的是测试曲线，记录了在测试集上的拟合曲线

### 注意

点击Train按钮后会卡顿一小段时间，请耐心等待。
