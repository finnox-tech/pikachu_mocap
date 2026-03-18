"""
URDF到Skeleton转换模块

将URDF机器人的关节角度转换为Pikachu骨骼动画姿态。

转换流程：
1. 从URDF姿态中提取关节角度（弧度）
2. 将单自由度关节角度转换为XYZ欧拉角
3. 进行参考系转换（坐标系对齐）
4. 应用骨骼特定的轴重映射和角度调整
5. 输出Skeleton姿态数据

作者: Auto-generated
日期: 2026-03-12
"""

import math
from typing import List, Optional, Tuple

from .data_structures import UrdfPoseData, SkeletonPoseData
from .utils import (
    side_from_name,
    wrap_angle
)


class Urdf2Skeleton:
    """
    URDF参考系到Skeleton参考系的姿态转换类
    
    将URDF机器人的关节角度转换为Pikachu骨骼动画姿态。
    
    使用示例:
        >>> converter = Urdf2Skeleton()
        >>> urdf_data = UrdfPoseData(joint_angles={"left_hip_yaw": 0.5})
        >>> skeleton_data = converter.convert(urdf_data, ["upper_arm_fk.L"])
    """
    
    # URDF关节到骨骼的映射关系
    JOINT_TO_BONE_MAP = {
        "head": ["head_link"],
        "neck": ["neck"],
        "chest": ["chest", "spine"],
        "torso": ["torso", "spine"],
        "hip": ["hip"],
        "pelvis": ["pelvis"],
        "shoulder": ["upper_arm"],
        "upper_arm": ["upper_arm", "shoulder"],
        "elbow": ["forearm", "lower_arm"],
        "forearm": ["forearm", "lower_arm"],
        "wrist": ["hand", "wrist"],
        "hand": ["hand", "wrist"],
        "knee": ["shin", "knee"],
        "ankle": ["foot", "ankle"],
        "foot": ["foot", "ankle"]
    }
    
    def __init__(self):
        """初始化转换器"""
        pass
    
    def convert(
        self,
        urdf_data: UrdfPoseData,
        target_bones: List[str]
    ) -> SkeletonPoseData:
        """
        执行完整的姿态转换
        
        Args:
            urdf_data: URDF参考系姿态数据
            target_bones: 目标骨骼名称列表
            
        Returns:
            Skeleton参考系姿态数据
        """
        if not urdf_data.joint_angles:
            return SkeletonPoseData()
        
        result = SkeletonPoseData()
        
        for bone in target_bones:
            # 查找对应的URDF关节
            joints = self._get_joints_for_bone(bone)
            if not joints:
                continue
            
            # 从关节获取角度
            angle = None
            for joint in joints:
                if urdf_data.has_joint(joint):
                    angle = urdf_data.get_joint_angle(joint)
                    break
            
            if angle is None:
                continue
            
            # 转换角度
            converted_angles = self._convert_bone_angle(bone, angle)
            result.set_bone_angle(bone, converted_angles)
        
        return result
    
    def _get_joints_for_bone(self, bone_name: str) -> List[str]:
        """
        获取骨骼对应的URDF关节
        
        Args:
            bone_name: 骨骼名称
            
        Returns:
            URDF关节名称列表
        """
        n = bone_name.lower()
        side = side_from_name(bone_name)
        
        for bone_type, joints in self.JOINT_TO_BONE_MAP.items():
            if bone_type in n:
                result = []
                for joint in joints:
                    if side == "LEFT":
                        result.append(f"left_{joint}")
                    elif side == "RIGHT":
                        result.append(f"right_{joint}")
                    else:
                        result.append(joint)
                return result
        
        return []
    
    def _convert_bone_angle(
        self,
        bone_name: str,
        angle: float
    ) -> Tuple[float, float, float]:
        """
        转换单个骨骼的角度
        
        Args:
            bone_name: 骨骼名称
            angle: 原始关节角度，单位:弧度
            
        Returns:
            转换后的欧拉角(x, y, z)，单位:度
        """
        if angle is None:
            return (0.0, 0.0, 0.0)
        
        # 1. 将弧度转换为角度
        angle_deg = math.degrees(angle)
        
        # 2. 将单自由度角度扩展为XYZ欧拉角
        xyz_angles = self._expand_to_xyz(bone_name, angle_deg)
        
        # 3. 应用骨骼特定的规则
        xyz_angles = self._apply_bone_specific_rules(bone_name, xyz_angles)
        
        # 4. 角度归一化
        return tuple(wrap_angle(v) for v in xyz_angles)
    
    def _expand_to_xyz(
        self,
        bone_name: str,
        angle: float
    ) -> Tuple[float, float, float]:
        """
        将单自由度角度扩展为XYZ欧拉角
        
        Args:
            bone_name: 骨骼名称
            angle: 角度值，单位:度
            
        Returns:
            (x, y, z)欧拉角
        """
        n = bone_name.lower()
        
        # 根据骨骼名称确定主轴
        if "head" in n:
            # 头部：三个轴都有角度
            return (0.0, angle, 0.0)
        elif "shoulder" in n:
            # 肩部：主轴是roll
            return (angle, 0.0, 0.0)
        elif "elbow" in n or "forearm" in n:
            # 肘部：主轴是pitch
            return (0.0, angle, 0.0)
        elif "hip" in n:
            # 髋部：主轴是yaw
            return (0.0, 0.0, angle)
        elif "knee" in n:
            # 膝盖：主轴是pitch
            return (0.0, angle, 0.0)
        elif "ankle" in n or "foot" in n:
            # 踝部：主轴是roll
            return (angle, 0.0, 0.0)
        else:
            # 默认：只保留Z轴
            return (0.0, 0.0, angle)
    
    def _apply_bone_specific_rules(
        self,
        bone_name: str,
        angles: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """
        应用骨骼特定的角度调整规则
        
        Args:
            bone_name: 骨骼名称
            angles: 调整前的角度
            
        Returns:
            调整后的角度
        """
        n = bone_name.lower()
        x, y, z = angles
        
        # 躯干骨骼：只保留Z轴旋转
        if any(key in n for key in ("hip", "pelvis", "torso", "chest", "spine", "root")):
            return (0.0, 0.0, z)
        
        # 头部骨骼：特殊的XYZ映射
        if "head" in n:
            return (y, z - 90, x)
        
        return angles
    
    # ==================== 特定骨骼的转换方法 ====================
    
    def convert_head_link(
        self,
        urdf_data: UrdfPoseData
    ) -> Tuple[float, float, float]:
        """
        转换head_link骨骼
        
        Args:
            urdf_data: URDF参考系姿态数据
            
        Returns:
            (x, y, z)欧拉角，单位:度
        """
        # 查找头部关节
        for joint in urdf_data.joint_names:
            if "head" in joint.lower():
                angle = urdf_data.get_joint_angle(joint)
                if angle is not None:
                    return self._convert_bone_angle("head_link", angle)
        
        return (0.0, 0.0, 0.0)
    
    def convert_hip_joint_to_bone(
        self,
        side: str,
        urdf_data: UrdfPoseData
    ) -> Tuple[float, float, float]:
        """
        转换髋部关节到骨骼
        
        Args:
            side: "LEFT" 或 "RIGHT"
            urdf_data: URDF参考系姿态数据
            
        Returns:
            (x, y, z)欧拉角，单位:度
        """
        joint_name = f"{side.lower()}_hip_yaw"
        bone_name = f"hip.{side.lower()}"
        
        angle = urdf_data.get_joint_angle(joint_name)
        if angle is None:
            return (0.0, 0.0, 0.0)
        
        return self._convert_bone_angle(bone_name, angle)