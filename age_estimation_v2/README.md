# 年龄估计项目 V2

这是对旧版年龄预测项目的重构版本。新版项目不再使用“分类模型 + 回归模型 + 手动融合”的复杂结构，而是使用一个统一的年龄估计模型，并将数据处理、训练、评估、推理和 GUI 拆分为独立模块。

## 项目目标

- 使用 UTKFace 数据集训练一个可复现的年龄估计模型。
- 使用固定 CSV 文件划分 `train / val / test`，避免每次运行随机划分导致结果不可复现。
- 将训练、评估、推理和 GUI 代码分离，降低项目复杂度。
- 推理随机人脸照片时，优先进行人脸检测与裁剪，减少背景和构图差异对预测的影响。

## 项目结构

```text
age_estimation_v2/
  configs/
    train.yaml          # 训练配置
    infer.yaml          # 推理配置

  data/
    split_dataset.py    # 生成固定数据划分 CSV
    datasets.py         # PyTorch Dataset
    transforms.py       # 训练/验证图像预处理
    face_crop.py        # 推理阶段人脸检测与裁剪
    splits/
      utkface_split.csv # 已生成的 UTKFace 数据划分文件

  models/
    age_model.py        # 年龄估计模型
    losses.py           # 损失函数

  training/
    train.py            # 训练入口
    evaluate.py         # 评估入口
    metrics.py          # MAE、RMSE、年龄段误差等指标
    utils.py            # checkpoint 保存与加载

  inference/
    predictor.py        # 推理封装，供命令行和 GUI 复用

  gui/
    app.py              # 图形界面

  tools/
    test_single_image.py # 单张图片命令行测试

  checkpoints/          # 训练后自动保存模型
  runs/                 # 训练日志和验证指标
```

## 模型说明

当前默认模型使用 `EfficientNet-B0` 作为主干网络，模型输出 `0-100` 岁的年龄分布。旧版 baseline 使用 `ResNet18`，如需回退对比，可以在 `configs/train.yaml` 中把 `model.backbone` 改回 `resnet18`。

当前支持的主干网络包括：

- `resnet18`
- `resnet34`
- `efficientnet_b0`
- `convnext_tiny`

预测年龄不是直接回归得到，而是由年龄分布计算期望：

```text
预测年龄 = sum(年龄 * 对应概率)
```

当前默认训练损失为：

```text
KLDiv(pred_age_distribution, gaussian_age_distribution)
+ reg_weight * Huber(expected_age, true_age)
```

其中 Gaussian soft label 会让相邻年龄也获得部分监督信号，比完全 hard class 更符合年龄的连续有序特点。项目仍保留旧的 `CrossEntropy + L1` 损失，可通过 `configs/train.yaml` 中的 `loss_type: ce_l1` 回退。

## 环境准备

推荐使用独立 Anaconda 环境，例如：

```powershell
conda create -n cnn python=3.11 -y
conda activate cnn
```

进入项目父目录。后续 `python -m age_estimation_v2...` 命令都应在这个目录运行：

```powershell
cd "D:\Projects\age-estimation-cnn"
```

安装项目运行依赖：

```powershell
pip install -r age_estimation_v2\requirements.txt
```

如果需要指定 CUDA 版本，建议先按 PyTorch 官网命令安装匹配当前显卡驱动的 `torch` 和 `torchvision`，再执行上面的 `pip install -r` 命令。

如需使用 notebook、数据分析或可视化工具，再安装可选依赖：

```powershell
pip install -r age_estimation_v2\requirements-dev.txt
```

验证 PyTorch 是否可以使用 GPU：

```powershell
python -c "import torch; import torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 数据集

当前默认使用 UTKFace：

```text
UTKface/UTKface_Aligned&cropped/UTKFace
```

本仓库不提交 UTKFace 原始数据集。当前本地项目通过目录联接引用外部数据集：

```text
D:\Projects\age-estimation-cnn\UTKface -> D:\UTKface
```

如果更换电脑或移动数据集，需要重新创建该目录联接，或直接修改 `configs/train.yaml` 中的 `data_dir`。

固定划分文件：

```text
age_estimation_v2/data/splits/utkface_split.csv
```

`configs/train.yaml` 中已开启按年龄段分层划分。训练或评估时，如果 `split_csv` 指向的文件不存在，会自动按配置生成固定划分文件。已有划分文件不会被自动覆盖。

当前划分：

```text
train: 16580
val:   3553
test:  3554
```

如需重新生成划分：

```powershell
python -m age_estimation_v2.data.split_dataset --data-dir "UTKface\UTKface_Aligned&cropped\UTKFace" --output "age_estimation_v2\data\splits\utkface_split_stratified.csv" --stratify-age
```

如果要替换当前训练使用的划分文件，先确认新 CSV 的样本数量和年龄段分布，再修改 `configs/train.yaml` 中的 `split_csv`。

## 开始训练

在项目根目录运行：

```powershell
cd "D:\Projects\age-estimation-cnn"
python -m age_estimation_v2.training.train --config "age_estimation_v2\configs\train.yaml"
```

训练过程会自动保存：

```text
age_estimation_v2/checkpoints/best_model.pth
age_estimation_v2/checkpoints/last_model.pth
age_estimation_v2/runs/train_log.csv
age_estimation_v2/runs/best_val_metrics.json
```

其中 `best_model.pth` 是验证集 MAE 最低的模型。`train_log.csv` 会记录总 loss、KL/CE loss、Huber/年龄 MAE loss、训练 MAE、验证 MAE/RMSE、误差阈值比例、学习率和各年龄段验证 MAE。

当前训练配置启用了：

- `EfficientNet-B0` 预训练主干
- Gaussian soft label + Huber 年龄回归损失
- CUDA 可用时自动启用 AMP 混合精度
- `ReduceLROnPlateau` 学习率调度
- 梯度裁剪

首次使用 `EfficientNet-B0` 预训练权重时，`torchvision` 会自动下载权重到本机缓存目录。如果下载失败，通常是网络或 SSL 临时问题，可以重试训练命令；也可以临时把 `configs/train.yaml` 中的 `model.pretrained` 改为 `false`，但这样一般会降低效果。

## ConvNeXt-Tiny 对比实验

当前 EfficientNet-B0 测试集结果：

```text
MAE:      4.6732
RMSE:     6.7439
within_5: 65.79%
```

如需只更换 backbone、保持 loss 和训练策略不变，可以训练 `ConvNeXt-Tiny` 对比实验：

```powershell
python -m age_estimation_v2.training.train --config "age_estimation_v2\configs\train_convnext_tiny.yaml"
```

训练结果会保存到：

```text
age_estimation_v2/checkpoints/convnext_tiny/
age_estimation_v2/runs/convnext_tiny/
```

测试集评估：

```powershell
python -m age_estimation_v2.training.evaluate --config "age_estimation_v2\configs\train_convnext_tiny.yaml" --checkpoint "age_estimation_v2\checkpoints\convnext_tiny\best_model.pth" --split test
```

如果 ConvNeXt-Tiny 的 `MAE < 4.6732` 且 `within_5 >= 65.79%`，说明它比当前 EfficientNet-B0 更适合作为主模型。

## 评估模型

训练完成后，在测试集上评估：

```powershell
python -m age_estimation_v2.training.evaluate --config "age_estimation_v2\configs\train.yaml" --checkpoint "age_estimation_v2\checkpoints\best_model.pth" --split test
```

评估结果包括：

- MAE
- RMSE
- 误差小于 3 岁、5 岁、10 岁的比例
- 不同年龄段的 MAE

## 单张图片预测

```powershell
python -m age_estimation_v2.tools.test_single_image --image "path\to\face.jpg"
```

输出内容包括：

- 预测年龄
- 置信提示
- 测试时增强波动
- 人脸裁剪状态

单张图片预测默认读取 `configs/infer.yaml` 中的 `checkpoint`、`image_size`、`face_margin` 和 `device`。如需临时指定模型：

```powershell
python -m age_estimation_v2.tools.test_single_image --image "path\to\face.jpg" --checkpoint "age_estimation_v2\checkpoints\best_model.pth"
```

## 启动 GUI

训练完成并生成 `best_model.pth` 后运行：

```powershell
python -m age_estimation_v2.gui.app
```

GUI 默认读取 `configs/infer.yaml`。它会自动尝试检测并裁剪最大人脸，然后显示预测年龄和稳定性信息。

## 打包 GUI 为 exe

GUI 应用可以单独打包，不需要把训练数据集、训练入口和 runs 日志一起打进 exe。打包内容主要包括：

- GUI 入口 `gui/app.py`
- 推理代码 `inference/`
- 模型定义 `models/`
- 推理配置 `configs/`
- 当前最佳模型 `checkpoints/best_model.pth`

先安装打包依赖：

```powershell
pip install -r age_estimation_v2\requirements-build.txt
```

从项目父目录运行打包命令：

```powershell
cd "D:\Projects\age-estimation-cnn"
python -m age_estimation_v2.tools.build_gui_exe
```

默认生成文件夹版应用：

```text
age_estimation_v2/dist/AgeEstimatorV2/AgeEstimatorV2.exe
```

文件夹版启动速度更快，也更适合包含 PyTorch 这类大依赖的应用。如果需要单文件 exe，可以运行：

```powershell
python -m age_estimation_v2.tools.build_gui_exe --onefile
```

单文件版会生成：

```text
age_estimation_v2/dist/AgeEstimatorV2.exe
```

注意：PyTorch 和 torchvision 体积较大，打包后的 exe 或应用文件夹会比较大，这是正常现象。首次启动也可能比直接运行 Python 慢。

## 预处理说明

训练增强使用随机裁剪、水平翻转、轻微旋转和颜色扰动。验证、评估和推理使用等比例缩放后中心裁剪，避免直接拉伸成正方形导致人脸比例变形。


