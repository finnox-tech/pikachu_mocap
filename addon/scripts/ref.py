import bpy
from mathutils import Euler
from math import radians, degrees

# =====================================================
# 基础接口
# =====================================================

def get_armature(armature_name="rig"):
    """
    获取Armature对象
    """
    arm = bpy.data.objects.get(armature_name)

    if arm is None:
        raise ValueError(f"Armature {armature_name} not found")

    if arm.type != 'ARMATURE':
        raise TypeError(f"{armature_name} is not an ARMATURE")

    return arm


def ensure_pose_mode(arm):
    """
    确保进入Pose模式
    """
    bpy.context.view_layer.objects.active = arm

    if bpy.context.object.mode != 'POSE':
        bpy.ops.object.mode_set(mode='POSE')


def get_pose_bone(armature_name, bone_name):
    """
    获取骨骼
    """
    arm = get_armature(armature_name)

    ensure_pose_mode(arm)

    bone = arm.pose.bones.get(bone_name)

    if bone is None:
        raise ValueError(f"Bone {bone_name} not found")

    return arm, bone


# =====================================================
# 单关节控制
# =====================================================

def set_joint_angle(
        armature_name,
        bone_name,
        angle,
        axis='x',
        degrees_input=True):
    """
    设置单个关节角

    参数
    ----------
    angle : 角度
    axis  : 'x','y','z'
    degrees_input : True=角度 False=弧度
    """

    arm, bone = get_pose_bone(armature_name, bone_name)

    bone.rotation_mode = 'XYZ'

    if degrees_input:
        angle = radians(angle)

    e = bone.rotation_euler

    if axis == 'x':
        e.x = angle
    elif axis == 'y':
        e.y = angle
    elif axis == 'z':
        e.z = angle
    else:
        raise ValueError("axis must be x y z")

    bone.rotation_euler = e

    bpy.context.view_layer.update()


def get_joint_angle(
        armature_name,
        bone_name,
        axis='x',
        degrees_output=True):
    """
    获取关节角
    """

    arm, bone = get_pose_bone(armature_name, bone_name)

    bone.rotation_mode = 'XYZ'

    e = bone.rotation_euler

    if axis == 'x':
        angle = e.x
    elif axis == 'y':
        angle = e.y
    elif axis == 'z':
        angle = e.z
    else:
        raise ValueError("axis must be x y z")

    if degrees_output:
        angle = degrees(angle)

    return angle


# =====================================================
# 多关节控制（机器人模式）
# =====================================================

def set_joint_angles(armature_name, joint_dict, degrees_input=True):
    """
    一次控制多个关节

    示例：
    {
        "upper_arm_fk.L": (30,0,0),
        "forearm_fk.L": (45,0,0),
        "hand_fk.L": (10,0,0)
    }
    """

    arm = get_armature(armature_name)

    ensure_pose_mode(arm)

    for bone_name, angles in joint_dict.items():

        bone = arm.pose.bones.get(bone_name)

        if bone is None:
            print(f"Warning: bone {bone_name} not found")
            continue

        bone.rotation_mode = 'XYZ'

        x, y, z = angles

        if degrees_input:
            x = radians(x)
            y = radians(y)
            z = radians(z)

        bone.rotation_euler = Euler((x, y, z), 'XYZ')

    bpy.context.view_layer.update()


# =====================================================
# 增量旋转（类似机器人速度控制）
# =====================================================

def add_joint_rotation(
        armature_name,
        bone_name,
        delta,
        axis='x',
        degrees_input=True):

    arm, bone = get_pose_bone(armature_name, bone_name)

    bone.rotation_mode = 'XYZ'

    if degrees_input:
        delta = radians(delta)

    e = bone.rotation_euler

    if axis == 'x':
        e.x += delta
    elif axis == 'y':
        e.y += delta
    elif axis == 'z':
        e.z += delta

    bone.rotation_euler = e

    bpy.context.view_layer.update()


# =====================================================
# 调试接口
# =====================================================

def print_armatures():
    print("\n===== Armatures =====")
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            print(obj.name)


def print_bones(armature_name="rig"):

    arm = get_armature(armature_name)

    print(f"\n===== Bones in {armature_name} =====")

    for b in arm.pose.bones:
        print(b.name)


# =====================================================
# 示例（机器人控制）
# =====================================================

if __name__ == "__main__":

    ARMATURE = "rig"

    # 查看骨骼
    print_armatures()
    print_bones(ARMATURE)

#    # 单关节控制
#    set_joint_angle(
#        ARMATURE,
#        "forearm_fk.L",
#        40,
#        axis='x'
#    )

    # # 增量控制
    # add_joint_rotation(
    #     ARMATURE,
    #     "shoulder.L",
    #     10,
    #     axis='x'
    # )

   # 多关节控制（机器人模式）
    set_joint_angles(
       ARMATURE,
       {
           "upper_arm_fk.L": (10, 0, 0),
           "forearm_fk.L": (10, 0, 0),
           "hand_fk.L": (10, 0, 0),
           "upper_arm_fk.R": (0, 0, 0),
           "forearm_fk.R": (0, 0, 0),
           "hand_fk.R": (0, 0, 0)
       }
   )






#shoulder.L
#upper_arm.L
#forearm.L
#hand.L
#f_index.01.L
#f_index.02.L
#f_index.03.L
#thumb.01.L
#thumb.02.L
#thumb.03.L
#f_middle.01.L
#f_middle.02.L
#f_middle.03.L
#f_ring.01.L
#f_ring.02.L
#f_ring.03.L
#shoulder.R
#upper_arm.R
#forearm.R
#hand.R
#f_index.01.R
#f_index.02.R
#f_index.03.R
#thumb.01.R
#thumb.02.R
#thumb.03.R
#f_middle.01.R
#f_middle.02.R
#f_middle.03.R
#f_ring.01.R
#f_ring.02.R
#f_ring.03.R
#pelvis.L
#pelvis.R
#thigh.L
#shin.L
#foot.L
#toe.L
#heel.02.L
#thigh.R
#shin.R
#foot.R
#toe.R
#heel.02.R
