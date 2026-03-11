import math


def _side_from_name(name: str):
    n = name.lower()
    if n.endswith(".l") or n.endswith("_l") or n.endswith("-l") or ".l_" in n:
        return "LEFT"
    if n.endswith(".r") or n.endswith("_r") or n.endswith("-r") or ".r_" in n:
        return "RIGHT"
    if ".l" in n:
        return "LEFT"
    if ".r" in n:
        return "RIGHT"
    return None


def _euler_xyz_to_matrix(angles):
    ax, ay, az = [math.radians(a) for a in angles]
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    return [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]


def _matrix_to_euler_xyz(mat):
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


def _swap_xy_basis(mat):
    return [
        [mat[0][1], mat[0][0], mat[0][2]],
        [mat[1][1], mat[1][0], mat[1][2]],
        [mat[2][1], mat[2][0], mat[2][2]],
    ]


def _wrap_angle(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def _avg_angles(sources, angles_map):
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


def _sources_for_bone(bone_name: str):
    n = bone_name.lower()
    side = _side_from_name(bone_name)

    if "ear" in n:
        if side == "LEFT":
            return ["LEFT_EAR"]
        if side == "RIGHT":
            return ["RIGHT_EAR"]
        return ["LEFT_EAR", "RIGHT_EAR"]

    if "head" in n:
        return ["HEAD_LINK"]

    if "neck" in n:
        return ["LEFT_SHOULDER", "RIGHT_SHOULDER"]

    if "chest" in n or "torso" in n:
        return ["LEFT_SHOULDER", "RIGHT_SHOULDER"]

    if "hip" in n or "pelvis" in n:
        return ["LEFT_HIP", "RIGHT_HIP"]

    if "shoulder" in n:
        if side == "LEFT":
            return ["LEFT_SHOULDER"]
        if side == "RIGHT":
            return ["RIGHT_SHOULDER"]
        return ["LEFT_SHOULDER", "RIGHT_SHOULDER"]

    if "upper_arm" in n or "upperarm" in n:
        if side == "LEFT":
            return ["LEFT_SHOULDER"]
        if side == "RIGHT":
            return ["RIGHT_SHOULDER"]
        return ["LEFT_SHOULDER", "RIGHT_SHOULDER"]

    if "forearm" in n or "lower_arm" in n or "lowerarm" in n:
        if side == "LEFT":
            return ["LEFT_ELBOW"]
        if side == "RIGHT":
            return ["RIGHT_ELBOW"]
        return ["LEFT_ELBOW", "RIGHT_ELBOW"]

    if "hand" in n or "wrist" in n:
        if side == "LEFT":
            return ["LEFT_WRIST"]
        if side == "RIGHT":
            return ["RIGHT_WRIST"]
        return ["LEFT_WRIST", "RIGHT_WRIST"]

    if "foot" in n or "ankle" in n:
        if side == "LEFT":
            return ["LEFT_ANKLE"]
        if side == "RIGHT":
            return ["RIGHT_ANKLE"]
        return ["LEFT_ANKLE", "RIGHT_ANKLE"]

    if "toe" in n:
        if side == "LEFT":
            return ["LEFT_FOOT_INDEX"]
        if side == "RIGHT":
            return ["RIGHT_FOOT_INDEX"]
        return ["LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"]

    if "tail" in n:
        return ["LEFT_HIP", "RIGHT_HIP"]

    return []


def _axis_remap(bone_name: str, angles):
    if angles is None:
        return None

    mat = _euler_xyz_to_matrix(angles)
    mat = _swap_xy_basis(mat)
    remapped = _matrix_to_euler_xyz(mat)

    n = bone_name.lower()
    if any(key in n for key in ("hip", "pelvis", "torso", "chest", "spine", "root")):
        remapped = (0.0, 0.0, remapped[2])

    # head完全映射xyz三个角度
    # if "head" in n:
    #     remapped = (0.0, 0.0, remapped[2])

    return tuple(_wrap_angle(v) for v in remapped)


def map_humanoid_to_pikachu(humanoid_angles, target_bones):
    if not humanoid_angles:
        return {}

    angles_map = {k.upper(): v for k, v in humanoid_angles.items()}
    result = {}

    for bone in target_bones:
        sources = _sources_for_bone(bone)
        angles = _avg_angles(sources, angles_map)
        if angles is None:
            continue
        result[bone] = _axis_remap(bone, angles)

    return result
