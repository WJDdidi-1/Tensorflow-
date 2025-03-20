# TensorFlow 核心知识点整理
> 说明：本汇总涵盖掌握 TensorFlow 需要直接记忆的关键知识点，包括核心概念、常用 API、模型训练优化技巧和部署方案等。内容以 TensorFlow 2.x 为主，并提及与 TensorFlow 1.x 的差异，配有示例代码便于理解。 

---

## 目录
- [核心概念](#1.核心概念)
- [常用API记忆点](#2.常用API记忆点)
- [模型优化与调优](#3.模型优化与调优)
- [模型部署](#4.模型部署)
- [总结](#总结)
- [具体实例](#具体实例)




---
# 1.核心概念

---

## 张量（Tensor）
TensorFlow 的基本数据结构，用于表示多维数组，即“张量”。张量在计算图中作为数据在节点之间传递的载体，本质上类似于 NumPy 的多维数组 。每个张量具有静态的形状(shape)和数据类型(dtype)。TensorFlow 中可以创建常量张量（如 tf.constant）和变量张量（如 tf.Variable）：常量一旦定义其值不可变，适合表示固定超参数等；变量则用于存储模型参数等需要在训练过程中更新的值（变量定义后值可变但维度不变）。 

---
## 计算图（Computational Graph）与会话（Session）
在 TensorFlow 1.x 中，采用静态计算图模式——需要先定义计算图，然后启动会话执行图中的运算 。计算图中的节点表示操作（op），边表示张量数据依赖；使用 tf.Session().run() 提交图的部分或整个计算。Session 负责在设备（CPU/GPU）上执行图并返回结果。 

---
## Eager Execution（即时执行）
TensorFlow 2.x 默认启用了动态图机制，即命令式执行模式。运算会立即执行并返回结果，不再需要显式构建计算图和开启会话 。这种动态计算图让 TensorFlow 代码的执行流程与普通 Python 代码类似，易于调试和打印中间结果，如同使用 NumPy 一样 。动态图的缺点是每次运算需要在 Python 与底层C++运行时之间频繁通信，可能导致性能略低于静态图 。 

---
## tf.function（图函数）
在 TF2.x 中，可通过 @tf.function 装饰器将 Python 函数转换为 TensorFlow 的计算图以提升性能 。被 tf.function 装饰的函数会被编译成静态图（这一机制称为 AutoGraph），随后调用该函数相当于在TF1.x中使用会话执行计算图。简而言之，tf.function 可以让动态图代码获得接近静态图的执行效率。

```python
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x**2 + 2*x + 1            # 前向计算: y = x^2 + 2x + 1
dy_dx = tape.gradient(y, x)       # 自动计算 dy/dx
print(dy_dx)  # 输出: tf.Tensor(8.0, shape=(), dtype=float32)
```

上例中，梯度磁带记录了 y 关于 x 的计算，求得导数 dy/dx = 2*x + 2 在 x=3 时的值为 8.0。在 TF1.x 中，自动微分通过在静态图中使用 tf.gradients 或优化器的 minimize() 隐式完成；而 TF2.x 的梯度磁带机制更加直观 。大部分内置的 TF API（如 Keras 优化器）也会利用自动微分来更新模型参数。 

---

# 2.常用 API 记忆点 

---
## Keras 高级 API（tf.keras）
Keras 是 TensorFlow 官方的高级 API，提供易于使用且高效的接口来构建和训练模型 。Keras 覆盖了机器学习工作流的各个步骤，包括数据准备、模型定义、训练、评估和部署。构建模型时常用 Sequential 顺序模型或函数式 API/子类化 API：定义层（tf.keras.layers）并将其堆叠成模型。使用 model.compile(optimizer, loss, metrics) 配置训练过程，然后使用 model.fit(...) 进行模型训练，model.evaluate(...) 评估模型，model.predict(...) 进行推断。Keras 简化了模型训练的常见操作，默认启用即时执行，并与 TensorFlow 无缝集成（例如支持在 TPU/GPU 上训练，导出 SavedModel 部署等） 。 
示例：使用 Keras Sequential API 构建和训练一个简单的全连接网络： 
例如，以下代码定义了一个两层的全连接模型并编译： 
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 构建顺序模型
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 构建顺序模型
model = Sequential([
    Dense(10, activation='relu', input_shape=(2,)),  # 隐藏层，输入维度2
    Dense(1, activation='sigmoid')                   # 输出层
])
# 编译模型（指定优化器、损失函数和评估指标）
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```
上述模型的结构通过 model.summary() 输出如下：
```
Layer (type)         Output Shape     Param #  
=============================================  
dense (Dense)        (None, 10)       30      
dense_1 (Dense)      (None, 1)        11      
=============================================  
Total params: 41  (全为可训练参数)
```
使用 .fit 方法可以训练模型，例如：
```python
# 构造伪数据并训练模型
import numpy as np
x_train = np.random.rand(1000, 2)
y_train = np.random.randint(0, 2, size=(1000, 1))
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
```
Keras 还提供方便的模型保存接口，如 model.save('model.h5') 将整个模型保存为 H5 文件，或保存为 TensorFlow SavedModel 格式用于部署；加载时使用 tf.keras.models.load_model 等即可恢复模型。

---
## 数据输入管道（tf.data）
TensorFlow 提供 tf.data API 构建高效的数据输入管道，用于加载和预处理数据。核心抽象是 Dataset：可通过工厂方法从内存中的数组、生成器或者文件（如 TFRecord）创建数据集（如 tf.data.Dataset.from_tensor_slices）。常用Dataset转换函数包括：map（逐条数据预处理）、batch（批量组合）、shuffle（随机打乱）、repeat（重复数据集）、prefetch（预取批次加速）。这些转换可以链式调用。例如：
```python
# 从 NumPy 数组创建数据集
ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds = ds.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)
```
上述代码创建了一个数据集，将数据随机打乱后按批次32提供，并预取批次以提高吞吐。构建好的 Dataset 可用于模型训练，例如直接传入 Keras 模型：model.fit(ds, epochs=10)。使用 tf.data 可以处理大型数据集并利用流水线并行加速数据供给，使训练过程更高效。

---
## 自定义训练循环
除了使用 model.fit, TensorFlow 2.x 允许通过 自定义训练循环 实现更灵活的训练流程。这通常需要用到 tf.GradientTape 计算梯度和优化器手动更新参数，以及可选的 tf.function 加速。基本模式是：
```python
optimizer = tf.keras.optimizers.SGD()
loss_fn = tf.keras.losses.MeanSquaredError()

# 单步训练函数
@tf.function  # 将训练步骤编译成图，加速执行
def train_step(x, y):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = loss_fn(y, pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# 自定义训练循环
for epoch in range(5):
    for x_batch, y_batch in ds:  # ds 为上述批次数据集
        loss = train_step(x_batch, y_batch)
    print(f"Epoch {epoch+1}, last batch loss = {loss.numpy():.4f}")
```
上面示例中，我们通过梯度磁带获取模型参数的梯度并使用优化器的 apply_gradients 更新参数。使用 @tf.function 将训练步骤函数编译，可以显著提高自定义循环的执行效率（类似 TF1.x 静态图的性能）。自定义训练循环适合需要对每步更新有细粒度控制的场景，比如实现自定义损失、梯度裁剪、动态学习率调整等。

---

# 3.模型优化与调优

---
## 优化器（Optimizer）与梯度下降
训练神经网络通常使用梯度下降算法及其变种，通过反向传播计算损失对参数的梯度，并沿梯度下降方向更新参数，使损失逐步减小。TensorFlow 的优化器位于 tf.keras.optimizers 下，例如 SGD（随机梯度下降）、Adam、RMSProp、Adagrad 等。随机梯度下降 (SGD) 每次用一个小批量样本的梯度来更新参数，简单高效；带动量的 SGD 在更新中加入动量项以加速收敛和减少震荡；Adam 则综合了动量和自适应学习率等优点，能够在大多数情况下高效收敛。优化器负责根据损失函数的梯度调整模型权重，从而逐步最小化损失 ￼。使用 Keras 时可通过 model.compile(optimizer='adam', ...) 简便地指定优化器，也可在自定义训练循环中手动调用优化器（如上例）。

---
## 学习率调度（Learning Rate Schedule）
学习率是梯度下降中每次更新的步长，其选择对训练效果至关重要。学习率调度指在训练过程中动态调整学习率策略，例如在训练进程中逐步衰减学习率以获得更稳健的收敛。常见策略有：指数衰减（每隔一定步数乘以固定衰减率，如 tf.keras.optimizers.schedules.ExponentialDecay）、分段常数（分阶段降低学习率）、余弦退火、在验证指标停滞时降低（如 ReduceLROnPlateau 回调按监测指标降低学习率）等。使用 Keras 可通过回调 LearningRateScheduler 或预定义 schedule 类指定调度策略。合理的学习率调度有助于在保持初期快速学习的同时，在后期细致收敛到更优解。

---
## 正则化（Regularization）
防止模型过拟合的技术统称为正则化。常用方法包括：
### 权重衰减（Weight Decay）
通过在损失函数中加入模型权重的范数惩罚项来约束权重大小，相当于 L2 正则（L1 正则则使用权重绝对值和）。在 Keras 中可通过层参数 kernel_regularizer=tf.keras.regularizers.l2(lambda) 等直接在模型中加入正则项，训练时这些惩罚会加到总损失上。
### Dropout
训练时随机将一部分神经元的输出设为0（按设定的比例丢弃），迫使网络的剩余连接更健壮，降低对某些特定神经元的依赖 ￼。Dropout 在 Keras 中由层 tf.keras.layers.Dropout(rate) 实现，只在训练时生效（预测时自动关闭）。适当的 Dropout 可以有效减少过拟合。
### 早停（Early Stopping）
在验证集性能不再提升时提前停止训练，以避免模型在训练集上过度拟合。Keras 提供 EarlyStopping 回调实现该策略。
通过以上正则化技巧，能够在一定程度上提升模型的泛化能力，缓解过拟合。需注意正则化项或过强的 Dropout 可能导致欠拟合，需要平衡。

---
## 批量归一化（Batch Normalization，BN）
批量归一化是一种加速深层网络训练并稳定模型收敛的技巧。它在每次训练迭代时，对小批量的激活值按照该批次的均值和标准差进行标准化，然后再通过可学习的缩放和平移参数恢复尺度和偏移 ￼。这种对中间层输入分布的调整使得各层输入更稳定，缓解了“内部协变量偏移”问题，从而允许使用较高的学习率并加速训练。Batch Norm 还具有轻微的正则化效果（因为每批数据加入了随机噪声的归一化）。在实践中，BN 层通常插入在全连接或卷积层之后、激活函数之前。TensorFlow 提供 tf.keras.layers.BatchNormalization 实现 BN，使用时需注意在训练和推理阶段的不同行为（训练时依据批统计量，推理时使用移动平均的全局统计量）。

---
# 4.模型部署

---
TensorFlow 模型训练完成后通常导出为 SavedModel（包含计算图和权重的序列化格式），可通过多种方式部署以供预测服务使用。下面概述了 TensorFlow 2.x 中从训练到部署的整体流程（训练阶段利用 tf.data 构建输入管道、使用 tf.keras 或 Estimator 训练模型并导出 SavedModel；部署阶段可选择不同方案）：
![](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEih3dh8TfcreBO1iYiVYsVl0KyhxL_Lv77mGLB6xmLDXd4eFd-l9Az9Jx-U2mZRXopNOtJoRGYX8fFk_XdTSGjfVsKKat9erGMcIjOQ-G4ERedqcv7-z4urSJS-0XZrp1SUszOmi7U-GHc/s1600/model.png)

---
## TensorFlow Serving
用于服务端部署机器学习模型的高性能系统。TensorFlow Serving 是 Google 开源的灵活、高性能的模型服务框架，专为生产环境设计。借助 TF Serving，可以方便地将训练后的模型作为在线服务提供推理接口（支持 gRPC 或 REST API）。它支持热加载新模型版本、A/B 测试等，能够管理模型生命周期并高效利用硬件（CPU/GPU）。典型流程是将模型保存为 SavedModel，然后由 TensorFlow Serving 加载提供服务。TF Serving 与 TensorFlow 无缝集成，也可扩展服务其他类型的模型。

## TensorFlow Lite (TFLite)
面向移动端和物联网嵌入式设备的超轻量级推理框架。TFLite 是 TensorFlow 的一部分，专为在 Android、iOS、嵌入式 Linux、微控制器（MCU）等设备上高效运行机器学习模型而设计。通过减少文件体积和优化推理速度，TFLite 实现了端侧设备上的低延迟、高性能推理，并且由于在本地运行还能提高隐私（无需将数据发送到服务器）。使用 TFLite 部署时，需要先将训练好的模型转换为 .tflite 格式（使用 tf.lite.TFLiteConverter 转换 SavedModel 等），然后在移动/嵌入式应用中利用 TFLite 解释器加载模型进行推理。TFLite 还支持模型量化和剪枝等优化，以进一步减小模型并加速推理。

## TensorFlow.js
用于在浏览器或 Node.js 中部署和训练模型的库。TensorFlow.js 可以将 TensorFlow 模型（通常需转换为 TF.js 特定格式）加载到浏览器中直接利用客户端的 CPU/GPU 进行推理，或在 Node.js 后端运行。这样可实现前端网页中的机器学习推理，与用户直接交互且不需后端服务器支持。同样，它适用于跨平台部署，如将模型嵌入到网页或微信小程序中。

## 其他部署方式
TensorFlow 提供多语言支持，可以在 C++、Java、Go、C#、Rust 等语言中加载 SavedModel 并进行推理。这允许在现有后端服务中集成训练好的模型。此外，TensorFlow Extended (TFX) 提供了完整的机器学习工作流管理，用于将训练、验证、部署串联起来实现流水线。在实际应用中，选择部署方式取决于使用场景：服务器端高并发服务通常选用 TensorFlow Serving，移动端独立应用选用 TFLite，Web 应用则使用 TensorFlow.js。

---

# TensorFlow 1.x 与 2.x 差异提示
TensorFlow 2.x 在易用性上相对于 1.x 有较大改变，需要注意以下不同：

---
## 计算图执行方式
TF1.x 默认使用静态计算图，需要先构建 tf.Graph 再通过 tf.Session 执行；TF2.x 默认即时执行（Eager），无需显式构建图或会话。例如，在 TF1.x 中相加两个数：
```python
a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)
c = a + b
with tf.Session() as sess:
    result = sess.run(c, feed_dict={a:5, b:3})
    print(result)  # 8
```
在 TF2.x 中则可以直接：
```python
a = tf.constant(5)
b = tf.constant(3)
c = a + b
print(c.numpy())  # 8
```
不再需要 placeholder、feed_dict 和会话来获取结果。

## 高阶 API
TF1.x 除 Keras 外常用 tf.layers 构建模型，训练上大量采用 Estimator API 封装训练流程；TF2.x 强调以 tf.keras 为统一的高阶 API 来构建和训练模型，并进一步简化了 API 接口。Estimator 在 TF2.x 中依然可用（通过 tf.estimator 模块），但官方推荐优先使用 Keras。
## 自动微分和定制化
TF1.x 没有 tf.GradientTape，用户若需自行计算梯度需借助 tf.gradients 或构造优化器图，定制训练相对繁琐。TF2.x 引入了更加直观的梯度磁带接口，支持动态计算梯度，方便实现自定义训练循环。
## API 清理与兼容
TF2.x 移除了 TF1.x 中大量冗余或不一致的 API（例如 tf.app, tf.flags, tf.logging 等模块在2.x中不再提供）。如果需要使用旧代码，可通过 tf.compat.v1 模块在 TF2 中访问 TF1.x 的接口。总体而言，TF2.x 对 TF1.x 不完全向后兼容，但提供了辅助迁移的工具和模块。
## 性能与分布式
由于即时执行的开销，某些纯 Python 控制逻辑频繁的任务 TF2 相比 TF1 可能略慢，不过通过 tf.function 可以恢复性能。另一方面，TF2.x 提供了更高级的 分布式策略 (tf.distribute.Strategy) 来方便地在多 GPU/TPU 上进行分布式训练，这是 TF1.x 不具备或需要手动实现的改进。此外，TensorFlow 社区和官方对 TF2.x 提供了更多支持和优化更新，使其在大多数情况下表现优于 TF1.x。

---
# 总结
TensorFlow 2.x 强调易用性和与实践接轨的简洁接口，如利用 Keras 迅速构建模型、即时执行便于调试、自动微分与自定义训练更灵活，同时保留了通过 tf.function 获得高性能的选项。掌握上述核心概念和常用接口，有助于高效地使用 TensorFlow 进行深度学习模型的开发与部署。各知识点在实践中可结合官方文档和示例进一步加深理解.

---

# 具体实例（expression detection）
通过实际的例子可以快速理解tensorflow的入门和使用，这是一个用tensorflow训练开放数据集，并调用电脑摄像头的例子，我使用的是MacOS，如果有需要可以自行更改。

---
## Import 需要用到的函数等，由于会要调用摄像头，所以我们这里调用了opencv：
```python
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import requests
import zipfile

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
```

下面这部分代码用于导入下载的数据文件，并对其进行解压。这部分代码按需使用。我是因为使用colab来做的示范，所以多出这一步用于解压文件以进行后续训练：
```python
print("TensorFlow version:", tf.__version__)
zip_file = "/archive.zip"  # 修改为你的文件路径
extract_folder = "/content/extracted_folder"

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

print("解压完成！")
```





