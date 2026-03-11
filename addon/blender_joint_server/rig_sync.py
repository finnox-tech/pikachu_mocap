import bpy
import json
from math import radians, degrees
from mathutils import Euler, Matrix

from . import server


ARMATURE_NAME = "rig"

def _tag_redraw():

    wm = bpy.context.window_manager

    if wm is None:
        return

    for window in wm.windows:

        screen = window.screen

        if screen is None:
            continue

        for area in screen.areas:

            if area.type == 'VIEW_3D':
                area.tag_redraw()


def get_armature(armature_name=ARMATURE_NAME):

    if armature_name:
        arm = bpy.data.objects.get(armature_name)
        if arm and arm.type == 'ARMATURE':
            return arm

    obj = bpy.context.object
    if obj and obj.type == 'ARMATURE':
        return obj

    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            return obj

    return None


def _send_debug(message):
    try:
        if hasattr(server, "add_log"):
            server.add_log(message)
        server.send_message({
            "type": "debug",
            "message": str(message)
        })
    except Exception:
        pass


def ensure_pose_mode(arm):

    bpy.context.view_layer.objects.active = arm

    if bpy.context.object.mode != 'POSE':
        try:
            bpy.ops.object.mode_set(mode='POSE')
        except Exception as e:
            for window in bpy.context.window_manager.windows:
                screen = window.screen
                for area in screen.areas:
                    if area.type != 'VIEW_3D':
                        continue
                    for region in area.regions:
                        if region.type != 'WINDOW':
                            continue
                        override = {
                            "window": window,
                            "screen": screen,
                            "area": area,
                            "region": region,
                            "scene": bpy.context.scene,
                            "view_layer": bpy.context.view_layer,
                            "active_object": arm,
                            "object": arm
                        }
                        try:
                            bpy.ops.object.mode_set(override, mode='POSE')
                            break
                        except Exception:
                            pass
            if bpy.context.object.mode != 'POSE':
                print("Pose mode error:", e)
                return False

    return True


def get_pose_bone(armature_name, bone_name):

    arm = get_armature(armature_name)

    if arm is None:
        raise ValueError(f"Armature {armature_name} not found")

    ensure_pose_mode(arm)

    bone = arm.pose.bones.get(bone_name)

    if bone is None:
        raise ValueError(f"Bone {bone_name} not found")

    return arm, bone


# ==========================
# 设置骨骼
# ==========================

def set_joint(bone_name, axis, angle):

    try:
        arm, bone = get_pose_bone(ARMATURE_NAME, bone_name)
    except Exception as e:
        print("Set joint error:", e)
        _send_debug(f"Set joint error: {e}")
        return

    bone.rotation_mode = 'XYZ'

    e = bone.rotation_euler

    angle = radians(angle)

    if axis == "x":
        e.x = angle
    elif axis == "y":
        e.y = angle
    elif axis == "z":
        e.z = angle

    bone.rotation_euler = e

    bpy.context.view_layer.update()


# ==========================
# UI helpers
# ==========================

def get_active_pose_bone(context=None):

    ctx = context or bpy.context
    pb = getattr(ctx, "active_pose_bone", None)

    if pb:
        return pb

    obj = ctx.object
    if obj and obj.type == 'ARMATURE' and obj.pose:
        for b in obj.pose.bones:
            if b.bone.select:
                return b

    return None


def get_bone_angles(pbone):

    e = pbone.rotation_euler
    return (
        degrees(e.x),
        degrees(e.y),
        degrees(e.z)
    )


# ==========================
# 获取骨骼姿态
# ==========================

def get_pose():

    arm = get_armature()

    if arm is None:
        return {}

    pose = {}

    for bone in arm.pose.bones:

        e = bone.rotation_euler

        pose[bone.name] = [
            degrees(e.x),
            degrees(e.y),
            degrees(e.z)
        ]

    return pose


# ==========================
# 获取骨骼树
# ==========================

def get_bone_tree():

    arm = get_armature()

    if arm is None:
        return []

    bones = []

    for b in arm.data.bones:

        bones.append({
            "name": b.name,
            "parent": b.parent.name if b.parent else None
        })

    return bones


# ==========================
# 获取骨骼变换
# ==========================

def _matrix_to_list(mat):
    return [list(row) for row in mat]


def get_bone_transforms(bone_names=None, debug=False):

    arm = get_armature()

    if arm is None:
        if debug:
            _send_debug("Armature not found for transforms.")
        return []

    if not ensure_pose_mode(arm):
        if debug:
            _send_debug("Failed to enter pose mode.")
        return []

    bpy.context.view_layer.update()

    transforms = []
    name_filter = None
    if bone_names:
        try:
            name_filter = set()
            for name in bone_names:
                bone = arm.pose.bones.get(name)
                while bone:
                    name_filter.add(bone.name)
                    bone = bone.parent
        except Exception:
            name_filter = None

    for bone in arm.pose.bones:
        if name_filter is not None and bone.name not in name_filter:
            continue

        parent = bone.parent
        if parent:
            local = parent.matrix.inverted_safe() @ bone.matrix
            parent_name = parent.name
        else:
            local = bone.matrix.copy()
            parent_name = None

        world_matrix = arm.matrix_world @ bone.matrix
        head = arm.matrix_world @ bone.head
        tail = arm.matrix_world @ bone.tail

        transforms.append({
            "name": bone.name,
            "parent": parent_name,
            "head": [float(v) for v in head],
            "tail": [float(v) for v in tail],
            "matrix": _matrix_to_list(world_matrix),
            "local_matrix": _matrix_to_list(local)
        })

    return transforms


# ==========================
# 处理消息
# ==========================

def handle_message(msg):

    try:

        data = json.loads(msg)

        if data["type"] == "set_joint":

            set_joint(
                data["bone"],
                data["axis"],
                data["angle"]
            )

        elif data["type"] == "request_pose":

            pose = get_pose()

            server.send_message({
                "type": "pose",
                "data": pose
            })

        elif data["type"] == "request_bones":

            bones = get_bone_tree()

            server.send_message({
                "type": "bones",
                "data": bones
            })

        elif data["type"] == "request_transforms":

            requested = data.get("bones") or []
            transforms = get_bone_transforms(requested, debug=True)
            _send_debug(f"Send transforms: {len(transforms)} (requested {len(requested)} + ancestors)")

            server.send_message({
                "type": "transforms",
                "data": transforms
            })

    except Exception as e:

        print("Message error:", e)
        _send_debug(f"Message error: {e}")


# ==========================
# 主循环
# ==========================

def blender_loop():

    while not server.msg_queue.empty():

        msg = server.msg_queue.get()

        handle_message(msg)

    if server.state_dirty:
        server.state_dirty = False
        _tag_redraw()

    return 0.02
