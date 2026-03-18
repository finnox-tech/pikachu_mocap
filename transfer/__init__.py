"""
姿态转换模块

提供不同参考系之间的姿态转换功能，用于将MediaPipe检测的人体姿态
转换为Pikachu机器人的骨骼姿态或URDF机器人姿态。

主要包含三个参考系和三个转换类：
- Humanoid参考系：MediaPipe人体姿态检测的坐标系
- URDF参考系：URDF机器人模型的坐标系
- Skeleton参考系：Pikachu骨骼动画的坐标系

转换类：
- Humanoid2Skeleton: Humanoid -> Skeleton
- Humanoid2Urdf: Humanoid -> URDF
- Urdf2Skeleton: URDF -> Skeleton

使用示例:
    >>> # 方式1：使用兼容接口（向后兼容）
    >>> from transfer import map_humanoid_to_pikachu
    >>> result = map_humanoid_to_pikachu(humanoid_angles, target_bones)
    
    >>> # 方式2：使用转换类
    >>> from transfer import Humanoid2Skeleton, HumanoidPoseData
    >>> converter = Humanoid2Skeleton()
    >>> humanoid_data = HumanoidPoseData(angles=humanoid_angles)
    >>> skeleton_data = converter.convert(humanoid_data, target_bones)

作者: Auto-generated
日期: 2026-03-12
"""

# 导入数据结构
from .data_structures import (
    HumanoidPoseData,
    UrdfPoseData,
    SkeletonPoseData
)

# 导入转换类
from .humanoid2skeleton import Humanoid2Skeleton
from .humanoid2urdf import Humanoid2Urdf
from .urdf2skeleton import Urdf2Skeleton

# 导入工具函数
from .utils import (
    side_from_name,
    euler_xyz_to_matrix,
    matrix_to_euler_xyz,
    swap_xy_basis,
    wrap_angle,
    avg_angles
)

# 导入主转换接口
from .main import (
    map_humanoid_to_pikachu,
    convert_humanoid_to_urdf,
    convert_urdf_to_skeleton
)

__all__ = [
    # 数据结构
    'HumanoidPoseData',
    'UrdfPoseData',
    'SkeletonPoseData',
    
    # 转换类
    'Humanoid2Skeleton',
    'Humanoid2Urdf',
    'Urdf2Skeleton',
    
    # 工具函数
    'side_from_name',
    'euler_xyz_to_matrix',
    'matrix_to_euler_xyz',
    'swap_xy_basis',
    'wrap_angle',
    'avg_angles',
    
    # 主转换接口
    'map_humanoid_to_pikachu',
    'convert_humanoid_to_urdf',
    'convert_urdf_to_skeleton',
]

__version__ = '1.0.0'