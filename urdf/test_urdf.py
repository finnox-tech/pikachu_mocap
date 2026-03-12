#!/usr/bin/env python3
"""
简单的URDF测试脚本，不依赖GUI
用于验证URDF文件的基本结构和关节信息
"""

import os
import sys

def test_urdf_basic():
    """测试URDF文件的基本结构"""
    print("=" * 60)
    print("URDF文件基本结构测试")
    print("=" * 60)
    
    urdf_path = "/home/finnox/Pikachu/Mocap/urdf/robot/Pikachu_V025/urdf/Pikachu_V025_flat_21dof.urdf"
    
    if not os.path.exists(urdf_path):
        print(f"错误: URDF文件不存在: {urdf_path}")
        return False
    
    # 读取URDF文件
    with open(urdf_path, 'r') as f:
        content = f.read()
    
    # 统计基本信息
    num_links = content.count('<link')
    num_joints = content.count('<joint')
    num_revolute = content.count('type="revolute"')
    num_meshes = content.count('<mesh')
    
    print(f"URDF文件: {urdf_path}")
    print(f"Link数量: {num_links}")
    print(f"Joint数量: {num_joints}")
    print(f"旋转关节数量: {num_revolute}")
    print(f"Mesh数量: {num_meshes}")
    print()
    
    # 检查mesh文件是否存在
    import re
    mesh_pattern = r'<mesh\s+filename="([^"]+)"'
    mesh_files = re.findall(mesh_pattern, content)
    
    urdf_dir = os.path.dirname(urdf_path)
    mesh_dir = os.path.join(urdf_dir, "..", "meshes")
    
    print("检查mesh文件:")
    missing_meshes = []
    for mesh_file in set(mesh_files):  # 去重
        mesh_path = os.path.join(mesh_dir, mesh_file)
        if os.path.exists(mesh_path):
            print(f"  ✓ {mesh_file}")
        else:
            print(f"  ✗ {mesh_file} (缺失)")
            missing_meshes.append(mesh_file)
    
    print()
    
    # 提取关节信息
    print("关节信息:")
    joint_pattern = r'<joint\s+name="([^"]+)"\s+type="([^"]+)".*?<axis\s+xyz="([^"]+)".*?<limit.*?lower="([^"]+)".*?upper="([^"]+)"'
    joints = re.findall(joint_pattern, content, re.DOTALL)
    
    for name, jtype, axis, lower, upper in joints:
        print(f"  {name}: type={jtype}, axis={axis}, limits=[{lower}, {upper}]")
    
    print()
    print("=" * 60)
    
    if missing_meshes:
        print(f"警告: {len(missing_meshes)} 个mesh文件缺失")
        return False
    else:
        print("所有mesh文件都存在")
        return True

def test_visual_origins():
    """测试visual origin是否存在"""
    print("=" * 60)
    print("Visual Origin测试")
    print("=" * 60)
    
    urdf_path = "/home/finnox/Pikachu/Mocap/urdf/robot/Pikachu_V025/urdf/Pikachu_V025_flat_21dof.urdf"
    
    with open(urdf_path, 'r') as f:
        content = f.read()
    
    # 检查visual origin
    import re
    visual_pattern = r'<visual>.*?<origin\s+xyz="([^"]+)"\s+rpy="([^"]+)".*?</visual>'
    visuals = re.findall(visual_pattern, content, re.DOTALL)
    
    print(f"找到 {len(visuals)} 个visual定义")
    for i, (xyz, rpy) in enumerate(visuals[:5], 1):  # 只显示前5个
        print(f"  Visual {i}: xyz={xyz}, rpy={rpy}")
    
    if len(visuals) > 5:
        print(f"  ... 还有 {len(visuals) - 5} 个visual")
    
    print()
    
    # 检查joint origin
    joint_pattern = r'<joint[^>]*>.*?<origin\s+xyz="([^"]+)"\s+rpy="([^"]+)".*?</joint>'
    joint_origins = re.findall(joint_pattern, content, re.DOTALL)
    
    print(f"找到 {len(joint_origins)} 个joint origin定义")
    for i, (xyz, rpy) in enumerate(joint_origins[:5], 1):  # 只显示前5个
        print(f"  Joint {i}: xyz={xyz}, rpy={rpy}")
    
    if len(joint_origins) > 5:
        print(f"  ... 还有 {len(joint_origins) - 5} 个joint")
    
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        success = test_urdf_basic()
        print()
        test_visual_origins()
        
        if success:
            print("\n✓ URDF文件结构正常")
            sys.exit(0)
        else:
            print("\n✗ URDF文件存在问题")
            sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)