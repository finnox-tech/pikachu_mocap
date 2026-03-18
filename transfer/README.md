# 姿态转换模块 (Pose Transfer Module)

## 概述

本模块提供了不同参考系之间的姿态转换功能，用于将MediaPipe检测的人体姿态转换为Pikachu机器人的骨骼姿态或URDF机器人姿态。

## 目录结构

```
transfer/
├── __init__.py              # 模块入口，导出所有公共接口
├── data_structures.py       # 参考系数据结构定义
├── utils.py                 # 工具函数
├── humanoid2skeleton.py      # Humanoid -> Skeleton 转换
├── humanoid2urdf.py         # Humanoid -> URDF 转换
├── urdf2skeleton.py         # URDF -> Skeleton 转换
├── main.py                  # 主转换接口和测试代码
└── README.md                # 本文档
```

## 参考系说明

### 1. Humanoid参考系
- **来源**: MediaPipe人体姿态检测
- **坐标系**: 右手坐标系，Y轴向上，X轴向右，Z轴向前
- **数据格式**: XYZ欧拉角（度）
- **关键点**: LEFT_SHOULDER, RIGHT_HIP, HEAD_LINK等

### 2. URDF参考系
- **来源**: URDF机器人模型
- **坐标系**: 右手坐标系，通常Z轴向上（取决于URDF定义）
- **数据格式**: 单自由度关节角度（弧度）
- **关节**: left_hip_yaw, right_knee_pitch等

### 3. Skeleton参考系
- **来源**: Pikachu骨骼动画
- **坐标系**: 游戏引擎坐标系约定
- **数据格式**: XYZ欧拉角（度）
- **骨骼**: upper_arm_fk.L, head_link等

## 快速开始

### 安装依赖

```bash
pip install numpy
```

### 基本使用

#### 方式1：使用兼容接口（向后兼容）

```python
from transfer import map_humanoid_to_pikachu

# Humanoid姿态数据
humanoid_angles = {
    "LEFT_SHOULDER": (0, 10, 5),
    "RIGHT_HIP": (0, -10, 0)
}

# 目标骨骼
target_bones = ["upper_arm_fk.L", "hip.R"]

# 执行转换
result = map_humanoid_to_pikachu(humanoid_angles, target_bones)
print(result)
```

#### 方式2：使用转换类

```python
from transfer import Humanoid2Skeleton, HumanoidPoseData

# 创建Humanoid姿态数据
humanoid_data = HumanoidPoseData(angles=humanoid_angles)

# 创建转换器
converter = Humanoid2Skeleton()

# 执行转换
skeleton_data = converter.convert(humanoid_data, target_bones)

# 访问结果
for bone_name in skeleton_data.bone_names:
    angles = skeleton_data.get_bone_angle(bone_name)
    print(f"{bone_name}: {angles}")
```

## API文档

### 数据结构类

#### HumanoidPoseData
```python
from transfer import HumanoidPoseData

data = HumanoidPoseData()
data.set_angle("LEFT_SHOULDER", (0, 10, 5))
angle = data.get_angle("LEFT_SHOULDER")
has_key = data.has_key("LEFT_SHOULDER")
```

#### UrdfPoseData
```python
from transfer import UrdfPoseData

data = UrdfPoseData(joint_limits={"left_hip_yaw": (-1.57, 1.57)})
data.set_joint_angle("left_hip_yaw", 0.5)
angle = data.get_joint_angle("left_hip_yaw")
clamped = data.clamp_angle("left_hip_yaw", 2.0)  # 限制在范围内
```

#### SkeletonPoseData
```python
from transfer import SkeletonPoseData

data = SkeletonPoseData()
data.set_bone_angle("upper_arm_fk.L", (0, 10, 5))
angle = data.get_bone_angle("upper_arm_fk.L")
has_bone = data.has_bone("upper_arm_fk.L")
```

### 转换类

#### Humanoid2Skeleton
```python
from transfer import Humanoid2Skeleton

converter = Humanoid2Skeleton()
skeleton_data = converter.convert(humanoid_data, target_bones)

# 特定骨骼转换
head_angles = converter.convert_head_link(humanoid_data)
shoulder_angles = converter.convert_shoulder("LEFT", humanoid_data)
```

#### Humanoid2Urdf
```python
from transfer import Humanoid2Urdf

converter = Humanoid2Urdf()
urdf_data = converter.convert(humanoid_data, target_joints, joint_limits)

# 特定关节转换
head_angle = converter.convert_head_joint(humanoid_data)
hip_angle = converter.convert_hip_joint("LEFT", humanoid_data)
```

#### Urdf2Skeleton
```python
from transfer import Urdf2Skeleton

converter = Urdf2Skeleton()
skeleton_data = converter.convert(urdf_data, target_bones)

# 特定骨骼转换
head_angles = converter.convert_head_link(urdf_data)
hip_angles = converter.convert_hip_joint_to_bone("LEFT", urdf_data)
```

### 主转换接口

#### map_humanoid_to_pikachu
```python
from transfer import map_humanoid_to_pikachu

result = map_humanoid_to_pikachu(humanoid_angles, target_bones)
```

#### convert_humanoid_to_urdf
```python
from transfer import convert_humanoid_to_urdf

joint_limits = {"left_hip_yaw": (-1.57, 1.57)}
result = convert_humanoid_to_urdf(humanoid_angles, target_joints, joint_limits)
```

#### convert_urdf_to_skeleton
```python
from transfer import convert_urdf_to_skeleton

result = convert_urdf_to_skeleton(urdf_angles, target_bones)
```

### 工具函数

```python
from transfer import (
    side_from_name,          # 提取侧边信息
    euler_xyz_to_matrix,     # 欧拉角转旋转矩阵
    matrix_to_euler_xyz,     # 旋转矩阵转欧拉角
    swap_xy_basis,           # 交换X和Y基向量
    wrap_angle,              # 角度归一化
    avg_angles               # 计算平均角度
)
```

## 转换流程

### Humanoid -> Skeleton
1. 从Humanoid姿态中提取相关关键点的角度
2. 进行参考系转换（坐标系对齐、基向量变换）
3. 应用骨骼特定的轴重映射和角度调整
4. 输出Skeleton姿态数据

### Humanoid -> URDF
1. 从Humanoid姿态中提取相关关键点的角度
2. 进行参考系转换（坐标系对齐、基向量变换）
3. 应用关节特定的轴重映射和角度调整
4. 将欧拉角转换为单自由度关节角度（弧度）
5. 输出URDF姿态数据

### URDF -> Skeleton
1. 从URDF姿态中提取关节角度（弧度）
2. 将单自由度关节角度转换为XYZ欧拉角
3. 进行参考系转换（坐标系对齐）
4. 应用骨骼特定的轴重映射和角度调整
5. 输出Skeleton姿态数据

## 映射关系

### MediaPipe关键点 -> 骨骼

| 骨骼类型 | 左侧 | 右侧 | 双侧 |
|---------|------|------|------|
| 耳朵 | LEFT_EAR | RIGHT_EAR | LEFT_EAR, RIGHT_EAR |
| 头部 | HEAD_LINK | HEAD_LINK | HEAD_LINK |
| 颈部 | LEFT_SHOULDER, RIGHT_SHOULDER | - | LEFT_SHOULDER, RIGHT_SHOULDER |
| 肩部 | LEFT_SHOULDER | RIGHT_SHOULDER | LEFT_SHOULDER, RIGHT_SHOULDER |
| 前臂 | LEFT_ELBOW | RIGHT_ELBOW | LEFT_ELBOW, RIGHT_ELBOW |
| 手部 | LEFT_WRIST | RIGHT_WRIST | LEFT_WRIST, RIGHT_WRIST |
| 脚部 | LEFT_ANKLE | RIGHT_ANKLE | LEFT_ANKLE, RIGHT_ANKLE |

### 骨骼特定规则

- **躯干骨骼**（hip, pelvis, torso, chest, spine）：只保留Z轴旋转
- **头部骨骼**：特殊的XYZ映射 `(y, z-90, x)`

## 示例代码

### 完整示例1：Humanoid -> Skeleton

```python
from transfer import Humanoid2Skeleton, HumanoidPoseData

# 创建Humanoid姿态数据
humanoid_data = HumanoidPoseData()
humanoid_data.set_angle("LEFT_SHOULDER", (0, 10, 5))
humanoid_data.set_angle("RIGHT_SHOULDER", (0, -10, -5))
humanoid_data.set_angle("HEAD_LINK", (5, 0, 0))

# 创建转换器
converter = Humanoid2Skeleton()

# 定义目标骨骼
target_bones = ["upper_arm_fk.L", "upper_arm_fk.R", "head_link"]

# 执行转换
skeleton_data = converter.convert(humanoid_data, target_bones)

# 输出结果
for bone_name in skeleton_data.bone_names:
    angles = skeleton_data.get_bone_angle(bone_name)
    print(f"{bone_name}: X={angles[0]:.2f}, Y={angles[1]:.2f}, Z={angles[2]:.2f}")
```

### 完整示例2：Humanoid -> URDF

```python
from transfer import Humanoid2Urdf, HumanoidPoseData

# 创建Humanoid姿态数据
humanoid_data = HumanoidPoseData(angles={
    "LEFT_HIP": (0, 10, 5),
    "RIGHT_KNEE": (0, -45, 0)
})

# 定义关节限制
joint_limits = {
    "left_hip_yaw": (-1.57, 1.57),
    "right_knee_pitch": (-2.36, 0)
}

# 创建转换器
converter = Humanoid2Urdf()

# 定义目标关节
target_joints = ["left_hip_yaw", "right_knee_pitch"]

# 执行转换
urdf_data = converter.convert(humanoid_data, target_joints, joint_limits)

# 输出结果
for joint_name in urdf_data.joint_names:
    angle = urdf_data.get_joint_angle(joint_name)
    print(f"{joint_name}: {angle:.3f} rad")
```

### 完整示例3：URDF -> Skeleton

```python
from transfer import Urdf2Skeleton, UrdfPoseData

# 创建URDF姿态数据
urdf_data = UrdfPoseData(joint_angles={
    "left_hip_yaw": 0.5,
    "right_knee_pitch": -0.8
})

# 创建转换器
converter = Urdf2Skeleton()

# 定义目标骨骼
target_bones = ["hip.L", "shin.R"]

# 执行转换
skeleton_data = converter.convert(urdf_data, target_bones)

# 输出结果
for bone_name in skeleton_data.bone_names:
    angles = skeleton_data.get_bone_angle(bone_name)
    print(f"{bone_name}: X={angles[0]:.2f}, Y={angles[1]:.2f}, Z={angles[2]:.2f}")
```

## 注意事项

1. **坐标系差异**: 不同参考系使用不同的坐标系约定，转换时需要特别注意
2. **角度单位**: Humanoid和Skeleton使用度，URDF使用弧度
3. **关节限制**: URDF关节通常有角度限制，使用`clamp_angle()`方法确保角度在范围内
4. **侧边信息**: 骨骼和关节名称包含侧边信息（.L, .R, _L, _R），使用`side_from_name()`提取

## 测试

运行测试代码：

```bash
python -m transfer.main
```

或者：

```bash
python transfer/main.py
```

## 版本历史

- **v1.0.0** (2026-03-12)
  - 初始版本
  - 实现Humanoid2Skeleton转换
  - 实现Humanoid2Urdf转换
  - 实现Urdf2Skeleton转换
  - 提供向后兼容接口

## 作者

Auto-generated

## 许可证

根据项目许可证使用。