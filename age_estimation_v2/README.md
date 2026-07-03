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

当前 baseline 使用 `ResNet18` 作为主干网络，模型输出 `0-100` 岁的年龄分布。

预测年龄不是直接回归得到，而是由年龄分布计算期望：

```text
预测年龄 = sum(年龄 * 对应概率)
```

训练损失为：

```text
CrossEntropy(age_class) + mae_weight * L1(expected_age, true_age)
```

这种方式比纯回归更稳定，也比旧版双模型融合更清晰。

## 环境准备

推荐使用独立 Anaconda 环境，例如：

```powershell
conda create -n cnn python=3.11 -y
conda activate cnn
```

安装 PyTorch 后，再安装项目依赖：

```powershell
cd "D:\Graduation project"
pip install numpy pillow pyyaml opencv-python==4.10.0.84 tqdm matplotlib pandas scikit-learn seaborn ipykernel
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

当前划分：

```text
train: 16580
val:   3553
test:  3554
```

如需重新生成划分：

```powershell
python -m age_estimation_v2.data.split_dataset --data-dir "UTKface\UTKface_Aligned&cropped\UTKFace" --output "age_estimation_v2\data\splits\utkface_split.csv"
```

## 开始训练

在项目根目录运行：

```powershell
cd "D:\Graduation project"
python -m age_estimation_v2.training.train --config "age_estimation_v2\configs\train.yaml"
```

训练过程会自动保存：

```text
age_estimation_v2/checkpoints/best_model.pth
age_estimation_v2/checkpoints/last_model.pth
age_estimation_v2/runs/train_log.csv
age_estimation_v2/runs/best_val_metrics.json
```

其中 `best_model.pth` 是验证集 MAE 最低的模型。

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
python -m age_estimation_v2.tools.test_single_image --image "path\to\face.jpg" --checkpoint "age_estimation_v2\checkpoints\best_model.pth"
```

输出内容包括：

- 预测年龄
- 置信提示
- 测试时增强波动
- 人脸裁剪状态

## 启动 GUI

训练完成并生成 `best_model.pth` 后运行：

```powershell
python -m age_estimation_v2.gui.app --checkpoint "age_estimation_v2\checkpoints\best_model.pth"
```

GUI 会自动尝试检测并裁剪最大人脸，然后显示预测年龄和稳定性信息。


