"""
姿态转换主接口模块

提供主转换接口和测试代码。

作者: Auto-generated
日期: 2026-03-12
"""

from typing import Dict, List, Optional, Tuple

from .data_structures import HumanoidPoseData, UrdfPoseData, SkeletonPoseData
from .humanoid2skeleton import Humanoid2Skeleton
from .humanoid2urdf import Humanoid2Urdf
from .urdf2skeleton import Urdf2Skeleton


def map_humanoid_to_pikachu(
    humanoid_angles: Dict[str, Tuple[float, float, float]],
    target_bones: List[str]
) -> Dict[str, Tuple[float, float, float]]:
    """
    将Humanoid姿态映射到Pikachu骨骼姿态（兼容接口）
    
    这是一个向后兼容的接口，保持与原函数相同的签名和行为。
    
    Args:
        humanoid_angles: Humanoid姿态角度字典，格式: {key: (x, y, z)}
        target_bones: 目标骨骼名称列表
        
    Returns:
        Pikachu骨骼姿态角度字典，格式: {bone_name: (x, y, z)}
        
    Example:
        >>> angles = {"LEFT_SHOULDER": (0, 10, 5), "RIGHT_HIP": (5, -10, 0)}
        >>> result = map_humanoid_to_pikachu(angles, ["upper_arm_fk.L", "hip.L"])
        >>> print(result)
        {'upper_arm_fk.L': (10.0, 5.0, 0.0), 'hip.L': (0.0, 0.0, 0.0)}
    """
    # 创建Humanoid姿态数据
    humanoid_data = HumanoidPoseData(angles=humanoid_angles)
    
    # 创建转换器
    converter = Humanoid2Skeleton()
    
    # 执行转换
    skeleton_data = converter.convert(humanoid_data, target_bones)
    
    # 返回字典格式（向后兼容）
    return skeleton_data.bone_angles


def convert_humanoid_to_urdf(
    humanoid_angles: Dict[str, Tuple[float, float, float]],
    target_joints: List[str],
    joint_limits: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, float]:
    """
    将Humanoid姿态转换为URDF关节角度
    
    Args:
        humanoid_angles: Humanoid姿态角度字典，格式: {key: (x, y, z)}
        target_joints: 目标URDF关节名称列表
        joint_limits: 可选的关节角度限制，格式: {joint_name: (min, max)}
        
    Returns:
        URDF关节角度字典，格式: {joint_name: angle_radians}
        
    Example:
        >>> angles = {"LEFT_HIP": (0, 10, 5), "RIGHT_KNEE": (0, -45, 0)}
        >>> limits = {"left_hip_yaw": (-1.57, 1.57), "right_knee_pitch": (-2.36, 0)}
        >>> result = convert_humanoid_to_urdf(angles, ["left_hip_yaw", "right_knee_pitch"], limits)
        >>> print(result)
        {'left_hip_yaw': 0.087, 'right_knee_pitch': -0.785}
    """
    # 创建Humanoid姿态数据
    humanoid_data = HumanoidPoseData(angles=humanoid_angles)
    
    # 创建转换器
    converter = Humanoid2Urdf()
    
    # 执行转换
    urdf_data = converter.convert(humanoid_data, target_joints, joint_limits)
    
    # 返回字典格式
    return urdf_data.joint_angles


def convert_urdf_to_skeleton(
    urdf_angles: Dict[str, float],
    target_bones: List[str]
) -> Dict[str, Tuple[float, float, float]]:
    """
    将URDF关节角度转换为Skeleton骨骼姿态
    
    Args:
        urdf_angles: URDF关节角度字典，格式: {joint_name: angle_radians}
        target_bones: 目标骨骼名称列表
        
    Returns:
        Skeleton骨骼姿态角度字典，格式: {bone_name: (x, y, z)}
        
    Example:
        >>> angles = {"left_hip_yaw": 0.5, "right_knee_pitch": -0.5}
        >>> result = convert_urdf_to_skeleton(angles, ["upper_arm_fk.L", "hip.L"])
        >>> print(result)
        {'upper_arm_fk.L': (0.0, 0.0, 0.0), 'hip.L': (0.0, 0.0, 28.6)}
    """
    # 创建URDF姿态数据
    urdf_data = UrdfPoseData(joint_angles=urdf_angles)
    
    # 创建转换器
    converter = Urdf2Skeleton()
    
    # 执行转换
    skeleton_data = converter.convert(urdf_data, target_bones)
    
    # 返回字典格式
    return skeleton_data.bone_angles


# ============================================================================
# 测试和示例
# ============================================================================


if __name__ == "__main__":
    # 示例：使用Humanoid2Skeleton转换器
    print("=" * 60)
    print("姿态转换模块测试")
    print("=" * 60)
    
    # 1. Humanoid -> Skeleton 转换示例
    print("\n1. Humanoid -> Skeleton 转换示例")
    humanoid_angles = {
        "LEFT_SHOULDER": (0, 10, 5),
        "RIGHT_SHOULDER": (0, -10, -5),
        "HEAD_LINK": (5, 0, 0),
        "LEFT_HIP": (0, 5, 0),
        "RIGHT_HIP": (0, -5, 0)
    }
    target_bones = ["upper_arm_fk.L", "upper_arm_fk.R", "head_link", "hip.L", "hip.R"]
    
    skeleton_result = map_humanoid_to_pikachu(humanoid_angles, target_bones)
    print(f"输入: {humanoid_angles}")
    print(f"输出: {skeleton_result}")
    
    # 2. Humanoid -> URDF 转换示例
    print("\n2. Humanoid -> URDF 转换示例")
    target_joints = ["left_hip_yaw", "right_hip_yaw", "left_knee_pitch", "right_knee_pitch"]
    joint_limits = {
        "left_hip_yaw": (-1.57, 1.57),
        "right_hip_yaw": (-1.57, 1.57),
        "left_knee_pitch": (-2.36, 0),
        "right_knee_pitch": (-2.36, 0)
    }
    
    urdf_result = convert_humanoid_to_urdf(humanoid_angles, target_joints, joint_limits)
    print(f"输入: {humanoid_angles}")
    print(f"输出: {urdf_result}")
    
    # 3. URDF -> Skeleton 转换示例
    print("\n3. URDF -> Skeleton 转换示例")
    urdf_angles = {
        "left_hip_yaw": 0.5,
        "right_hip_yaw": -0.3,
        "left_knee_pitch": -0.8,
        "right_knee_pitch": -0.6
    }
    
    skeleton_from_urdf = convert_urdf_to_skeleton(urdf_angles, target_bones)
    print(f"输入: {urdf_angles}")
    print(f"输出: {skeleton_from_urdf}")
    
    # 4. 使用类的完整示例
    print("\n4. 使用Humanoid2Skeleton类的完整示例")
    humanoid_data = HumanoidPoseData(angles=humanoid_angles)
    converter = Humanoid2Skeleton()
    skeleton_data = converter.convert(humanoid_data, target_bones)
    print(f"转换结果:")
    for bone_name in skeleton_data.bone_names:
        angles = skeleton_data.get_bone_angle(bone_name)
        print(f"  {bone_name}: {angles}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)