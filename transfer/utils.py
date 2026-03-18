"""
姿态转换工具函数模块

提供姿态转换过程中使用的各种工具函数，包括：
- 侧边信息提取
- 欧拉角和旋转矩阵转换
- 基向量变换
- 角度归一化
- 角度平均计算

作者: Auto-generated
日期: 2026-03-12
"""

import math
from typing import Dict, List, Optional, Tuple


def side_from_name(name: str) -> Optional[str]:
    """
    从名称中提取侧边信息
    
    Args:
        name: 骨骼或关节名称
        
    Returns:
        "LEFT", "RIGHT" 或 None
        
    Example:
        >>> side_from_name("left_arm")
        'LEFT'
        >>> side_from_name("right_arm")
        'RIGHT'
        >>> side_from_name("center")
        None
    """
    n = name.lower()
    # URDF-style prefix: left_hip_yaw, right_knee_pitch
    if n.startswith("left_") or n.startswith("left."):
        return "LEFT"
    if n.startswith("right_") or n.startswith("right."):
        return "RIGHT"
    # Blender-style suffix: upper_arm_fk.L, hip.R, toe_l
    if n.endswith(".l") or n.endswith("_l") or n.endswith("-l") or ".l_" in n:
        return "LEFT"
    if n.endswith(".r") or n.endswith("_r") or n.endswith("-r") or ".r_" in n:
        return "RIGHT"
    if ".l" in n:
        return "LEFT"
    if ".r" in n:
        return "RIGHT"
    return None


def euler_xyz_to_matrix(angles: Tuple[float, float, float]) -> List[List[float]]:
    """
    将XYZ欧拉角转换为旋转矩阵
    
    使用ZYX旋转顺序（即绕Z轴、Y轴、X轴旋转）。
    
    Args:
        angles: (x, y, z)欧拉角，单位:度
        
    Returns:
        3x3旋转矩阵
        
    Example:
        >>> mat = euler_xyz_to_matrix((90, 0, 0))
        >>> # 绕X轴旋转90度的旋转矩阵
    """
    ax, ay, az = [math.radians(a) for a in angles]
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    return [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]


def matrix_to_euler_xyz(mat: List[List[float]]) -> Tuple[float, float, float]:
    """
    将旋转矩阵转换为XYZ欧拉角
    
    Args:
        mat: 3x3旋转矩阵
        
    Returns:
        (x, y, z)欧拉角，单位:度
        
    Example:
        >>> angles = matrix_to_euler_xyz([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        >>> print(angles)
        (90.0, 0.0, 0.0)
    """
    r00, r10, r20 = mat[0][0], mat[1][0], mat[2][0]
    r01, r11, r21 = mat[0][1], mat[1][1], mat[2][1]
    r02, r12, r22 = mat[0][2], mat[1][2], mat[2][2]

    sy = -r20
    cy = math.sqrt(r00 * r00 + r10 * r10)

    if cy > 1e-6:
        x = math.atan2(r21, r22)
        y = math.atan2(sy, cy)
        z = math.atan2(r10, r00)
    else:
        x = math.atan2(-r12, r11)
        y = math.atan2(sy, cy)
        z = 0.0

    return (math.degrees(x), math.degrees(y), math.degrees(z))


def swap_xy_basis(mat: List[List[float]]) -> List[List[float]]:
    """
    交换旋转矩阵的X和Y轴
    
    用于不同参考系之间的基向量转换。
    
    Args:
        mat: 3x3旋转矩阵
        
    Returns:
        交换X和Y轴后的旋转矩阵
        
    Example:
        >>> mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> swapped = swap_xy_basis(mat)
        >>> # swapped的X轴是原矩阵的Y轴，Y轴是原矩阵的X轴
    """
    return [
        [mat[0][1], mat[0][0], mat[0][2]],
        [mat[1][1], mat[1][0], mat[1][2]],
        [mat[2][1], mat[2][0], mat[2][2]],
    ]


def wrap_angle(angle: float) -> float:
    """
    将角度归一化到[-180, 180]范围
    
    Args:
        angle: 角度值，单位:度
        
    Returns:
        归一化后的角度值
        
    Example:
        >>> wrap_angle(270)
        -90.0
        >>> wrap_angle(-200)
        160.0
    """
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def avg_angles(
    sources: List[str],
    angles_map: Dict[str, Tuple[float, float, float]]
) -> Optional[Tuple[float, float, float]]:
    """
    计算多个源角度的平均值
    
    Args:
        sources: 源关键点名称列表
        angles_map: 关键点名称到角度的映射
        
    Returns:
        平均角度(x, y, z)，如果没有有效数据则返回None
        
    Example:
        >>> angles_map = {
        ...     "LEFT_SHOULDER": (10, 20, 30),
        ...     "RIGHT_SHOULDER": (20, 10, 40)
        ... }
        >>> avg = avg_angles(["LEFT_SHOULDER", "RIGHT_SHOULDER"], angles_map)
        >>> print(avg)
        (15.0, 15.0, 35.0)
    """
    if not sources:
        return None
    acc = [0.0, 0.0, 0.0]
    count = 0
    for src in sources:
        ang = angles_map.get(src)
        if ang is None:
            continue
        acc[0] += ang[0]
        acc[1] += ang[1]
        acc[2] += ang[2]
        count += 1
    if count == 0:
        return None
    return (acc[0] / count, acc[1] / count, acc[2] / count)