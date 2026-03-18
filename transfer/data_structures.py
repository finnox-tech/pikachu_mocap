"""
姿态转换数据结构模块

定义了三个参考系的数据结构：
- HumanoidPoseData: Humanoid参考系姿态数据（MediaPipe人体姿态）
- UrdfPoseData: URDF参考系姿态数据（URDF机器人模型）
- SkeletonPoseData: Skeleton参考系姿态数据（Pikachu骨骼动画）

作者: Auto-generated
日期: 2026-03-12
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class HumanoidPoseData:
    """
    Humanoid参考系姿态数据结构
    
    表示从MediaPipe检测的人体姿态数据，包含各个关键点的欧拉角。
    MediaPipe使用右手坐标系，Y轴向上，X轴向右，Z轴向前。
    
    Attributes:
        angles (Dict[str, Tuple[float, float, float]]): 
            关键点名称到欧拉角(XYZ, 单位:度)的映射
            示例: {"LEFT_SHOULDER": (0.0, 10.0, 5.0)}
        
        landmarks (Optional[Dict[str, Tuple[float, float, float]]]): 
            可选：3D关键点位置，用于高级计算
    
    Example:
        >>> data = HumanoidPoseData()
        >>> data.set_angle("LEFT_SHOULDER", (0, 10, 5))
        >>> print(data.get_angle("LEFT_SHOULDER"))
        (0, 10, 5)
    """
    angles: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    landmarks: Optional[Dict[str, Tuple[float, float, float]]] = None
    
    def get_angle(self, key: str) -> Optional[Tuple[float, float, float]]:
        """
        获取指定关键点的欧拉角
        
        Args:
            key: 关键点名称
            
        Returns:
            欧拉角(x, y, z)，单位:度，如果不存在则返回None
        """
        return self.angles.get(key.upper())
    
    def set_angle(self, key: str, angles: Tuple[float, float, float]):
        """
        设置指定关键点的欧拉角
        
        Args:
            key: 关键点名称
            angles: 欧拉角(x, y, z)，单位:度
        """
        self.angles[key.upper()] = angles
    
    def has_key(self, key: str) -> bool:
        """
        检查是否存在指定关键点
        
        Args:
            key: 关键点名称
            
        Returns:
            如果存在返回True，否则返回False
        """
        return key.upper() in self.angles


@dataclass
class UrdfPoseData:
    """
    URDF参考系姿态数据结构
    
    表示URDF机器人模型的姿态数据，包含各个关节的角度值。
    URDF使用右手坐标系，通常Z轴向上，但具体取决于URDF文件定义。
    
    Attributes:
        joint_angles (Dict[str, float]): 
            关节名称到关节角度(单位:弧度)的映射
            URDF关节通常是单自由度的，只存储一个角度值
            示例: {"left_hip_yaw": 0.5, "right_knee": -0.3}
        
        joint_names (List[str]): 
            关节名称列表，用于遍历所有关节
        
        joint_limits (Optional[Dict[str, Tuple[float, float]]]): 
            可选：关节角度限制(最小值, 最大值)，单位:弧度
    
    Example:
        >>> data = UrdfPoseData()
        >>> data.set_joint_angle("left_hip_yaw", 0.5)
        >>> print(data.get_joint_angle("left_hip_yaw"))
        0.5
    """
    joint_angles: Dict[str, float] = field(default_factory=dict)
    joint_names: List[str] = field(default_factory=list)
    joint_limits: Optional[Dict[str, Tuple[float, float]]] = None
    
    def set_joint_angle(self, joint_name: str, angle: float):
        """
        设置指定关节的角度（弧度）
        
        Args:
            joint_name: 关节名称
            angle: 关节角度，单位:弧度
        """
        self.joint_angles[joint_name] = angle
        if joint_name not in self.joint_names:
            self.joint_names.append(joint_name)
    
    def get_joint_angle(self, joint_name: str) -> Optional[float]:
        """
        获取指定关节的角度（弧度）
        
        Args:
            joint_name: 关节名称
            
        Returns:
            关节角度，单位:弧度，如果不存在则返回None
        """
        return self.joint_angles.get(joint_name)
    
    def has_joint(self, joint_name: str) -> bool:
        """
        检查是否存在指定关节
        
        Args:
            joint_name: 关节名称
            
        Returns:
            如果存在返回True，否则返回False
        """
        return joint_name in self.joint_angles
    
    def clamp_angle(self, joint_name: str, angle: float) -> float:
        """
        将角度限制在关节范围内
        
        Args:
            joint_name: 关节名称
            angle: 关节角度，单位:弧度
            
        Returns:
            限制后的角度值
        """
        if self.joint_limits and joint_name in self.joint_limits:
            lower, upper = self.joint_limits[joint_name]
            return max(lower, min(upper, angle))
        return angle


@dataclass
class SkeletonPoseData:
    """
    Skeleton参考系姿态数据结构
    
    表示Pikachu骨骼动画的姿态数据，包含各个骨骼的欧拉角。
    Skeleton参考系通常使用游戏引擎的坐标系约定。
    
    Attributes:
        bone_angles (Dict[str, Tuple[float, float, float]]): 
            骨骼名称到欧拉角(XYZ, 单位:度)的映射
            示例: {"upper_arm_fk.L": (0.0, 10.0, 5.0)}
        
        bone_names (List[str]): 
            骨骼名称列表，用于遍历所有骨骼
        
        bone_limits (Optional[Dict[str, Dict[str, Tuple[float, float]]]]): 
            可选：骨骼角度限制，格式: {bone_name: {"x": (min, max), "y": (min, max), "z": (min, max)}}
    
    Example:
        >>> data = SkeletonPoseData()
        >>> data.set_bone_angle("upper_arm_fk.L", (0, 10, 5))
        >>> print(data.get_bone_angle("upper_arm_fk.L"))
        (0, 10, 5)
    """
    bone_angles: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    bone_names: List[str] = field(default_factory=list)
    bone_limits: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
    
    def set_bone_angle(self, bone_name: str, angles: Tuple[float, float, float]):
        """
        设置指定骨骼的欧拉角（度）
        
        Args:
            bone_name: 骨骼名称
            angles: 欧拉角(x, y, z)，单位:度
        """
        self.bone_angles[bone_name] = angles
        if bone_name not in self.bone_names:
            self.bone_names.append(bone_name)
    
    def get_bone_angle(self, bone_name: str) -> Optional[Tuple[float, float, float]]:
        """
        获取指定骨骼的欧拉角（度）
        
        Args:
            bone_name: 骨骼名称
            
        Returns:
            欧拉角(x, y, z)，单位:度，如果不存在则返回None
        """
        return self.bone_angles.get(bone_name)
    
    def has_bone(self, bone_name: str) -> bool:
        """
        检查是否存在指定骨骼
        
        Args:
            bone_name: 骨骼名称
            
        Returns:
            如果存在返回True，否则返回False
        """
        return bone_name in self.bone_angles