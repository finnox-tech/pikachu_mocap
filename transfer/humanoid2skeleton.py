"""
Humanoid到Skeleton转换模块

将MediaPipe检测的人体姿态转换为Pikachu骨骼动画姿态（度）。

转换公式：output = wrap_angle((converted + bias) * scale)，各轴独立。

微调接口：
  converter.set_bias("upper_arm_fk.L", x=0, y=0, z=10)
  converter.set_scale("upper_arm_fk.L", x=1.0, y=0.8, z=1.0)
"""

from typing import Dict, List, Tuple

from .data_structures import HumanoidPoseData, SkeletonPoseData
from .utils import (
    euler_xyz_to_matrix,
    matrix_to_euler_xyz,
    swap_xy_basis,
    wrap_angle,
    avg_angles,
)


class Humanoid2Skeleton:
    """
    Humanoid 参考系 → Skeleton 参考系的姿态转换类。

    使用示例：
        converter = Humanoid2Skeleton()
        converter.set_bias("upper_arm_fk.L", z=10)    # Z轴+10°偏置
        converter.set_scale("upper_arm_fk.L", y=0.5)  # Y轴幅度减半
        skeleton_data = converter.convert(humanoid_data, ["upper_arm_fk.L"])
    """

    def __init__(self):
        # 各骨骼角度偏置（度）：{bone_name: (dx, dy, dz)}
        self.bone_bias:  Dict[str, Tuple[float, float, float]] = {}
        # 各骨骼角度缩放（0~1）：{bone_name: (sx, sy, sz)}
        self.bone_scale: Dict[str, Tuple[float, float, float]] = {}
        # 骨骼名 → 具体转换方法（供 convert() 分发，setdefault 在方法内触发）
        self._dispatch = {
            "head":           self.convert_head,
            "neck":           self.convert_neck,
            "chest":          self.convert_chest,
            "hips":           self.convert_hips,
            "tail":           self.convert_tail,
            "shoulder.L":     self.convert_shoulder_l,
            "shoulder.R":     self.convert_shoulder_r,
            "upper_arm_fk.L": self.convert_upper_arm_l,
            "upper_arm_fk.R": self.convert_upper_arm_r,
            "forearm_fk.L":   self.convert_forearm_l,
            "forearm_fk.R":   self.convert_forearm_r,
            "hand_fk.L":      self.convert_hand_l,
            "hand_fk.R":      self.convert_hand_r,
            "foot_ik.L":      self.convert_foot_l,
            "foot_ik.R":      self.convert_foot_r,
            "toe.L":          self.convert_toe_l,
            "toe.R":          self.convert_toe_r,
            "ear.L":          self.convert_ear_l,
            "ear.R":          self.convert_ear_r,
        }

    # ── bias / scale 管理 ────────────────────────────────────────────────────

    def set_bias(self, bone_name: str, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """设置骨骼角度偏置（度）。先加偏置再缩放：output = (converted + bias) * scale"""
        self.bone_bias[bone_name] = (x, y, z)

    def clear_bias(self, bone_name: str = None):
        """清除偏置。bone_name=None 时清除所有。"""
        if bone_name is None:
            self.bone_bias.clear()
        else:
            self.bone_bias.pop(bone_name, None)

    def set_scale(self, bone_name: str, x: float = 1.0, y: float = 1.0, z: float = 1.0):
        """设置骨骼各轴缩放（推荐范围 0~1）。先加偏置再缩放：output = (converted + bias) * scale"""
        self.bone_scale[bone_name] = (x, y, z)

    def clear_scale(self, bone_name: str = None):
        """清除缩放。bone_name=None 时清除所有。"""
        if bone_name is None:
            self.bone_scale.clear()
        else:
            self.bone_scale.pop(bone_name, None)

    # ── 批量转换入口 ──────────────────────────────────────────────────────────

    def convert(
        self,
        humanoid_data: HumanoidPoseData,
        target_bones: List[str],
    ) -> SkeletonPoseData:
        """批量转换，返回 SkeletonPoseData。"""
        if not humanoid_data.angles:
            return SkeletonPoseData()

        result = SkeletonPoseData()
        for bone in target_bones:
            method = self._dispatch.get(bone)
            if method is None:
                continue
            result.set_bone_angle(bone, method(humanoid_data))

        return result

    # ── 核心转换辅助 ──────────────────────────────────────────────────────────

    def _to_skeleton_axes(
        self,
        angles: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """
        绝对坐标系转换：MediaPipe 欧拉角 → Skeleton 参考系欧拉角（度）。
        流程：euler_xyz → 旋转矩阵 → swap_xy_basis → euler_xyz
        """
        mat = euler_xyz_to_matrix(angles)
        mat = swap_xy_basis(mat)
        return matrix_to_euler_xyz(mat)

    def _apply_bias_scale(
        self,
        bone_name: str,
        x: float,
        y: float,
        z: float,
    ) -> Tuple[float, float, float]:
        """
        加偏置、乘缩放、归一化到 [-180, 180]。
        output = wrap_angle((converted + bias) * scale)，各轴独立。
        """
        bx, by, bz = self.bone_bias.get(bone_name,  (0.0, 0.0, 0.0))
        sx, sy, sz = self.bone_scale.get(bone_name, (1.0, 1.0, 1.0))
        return (
            wrap_angle((x + bx) * sx),
            wrap_angle((y + by) * sy),
            wrap_angle((z + bz) * sz),
        )

    def _avg(
        self,
        keys: List[str],
        humanoid_data: HumanoidPoseData,
    ) -> Tuple[float, float, float]:
        """从 humanoid_data 中取多个关键点角度并平均，返回平均欧拉角（度）。"""
        angles_map = {k.upper(): v for k, v in humanoid_data.angles.items()}
        return avg_angles(keys, angles_map)

    # ── 各骨骼显式转换方法（从头到脚）──────────────────────────────────────────
    # 每个方法均可独立调用，也可通过 convert() 批量调用。
    # 偏置/缩放默认值写在每个方法内（setdefault：外部已设置时不覆盖）。

    # ── 头部 ────────────────────────────────────────────────────────────────

    def convert_head(self, humanoid_data: HumanoidPoseData) -> Tuple[float, float, float]:
        """
        Skeleton bone: head
        MediaPipe 源: HEAD_LINK
        轴重映射: (x,y,z) → out_x=y, out_y=z-90, out_z=x
        """
        self.bone_bias.setdefault( "head", (-5.0,  0.0, -5.0))
        self.bone_scale.setdefault("head", ( 0.8,  0.8,  0.5))
        raw = humanoid_data.get_angle("HEAD_LINK")
        if raw is None:
            return (0.0, 0.0, 0.0)
        x, y, z = self._to_skeleton_axes(raw)
        return self._apply_bias_scale("head", y, z - 90.0, x)

    def convert_neck(self, humanoid_data: HumanoidPoseData) -> Tuple[float, float, float]:
        """
        Skeleton bone: neck
        MediaPipe 源: LEFT_SHOULDER + RIGHT_SHOULDER（平均）
        躯干类：只保留 Z 轴（yaw），X/Y 清零
        """
        self.bone_bias.setdefault( "neck", (40.0, 0.0, -180.0))
        self.bone_scale.setdefault("neck", ( 0.8, 0.8,    0.1))
        raw = self._avg(["LEFT_SHOULDER", "RIGHT_SHOULDER"], humanoid_data)
        if raw is None:
            return (0.0, 0.0, 0.0)
        x, y, z = self._to_skeleton_axes(raw)
        return self._apply_bias_scale("neck", 0.0, 0.0, 0)

    def convert_chest(self, humanoid_data: HumanoidPoseData) -> Tuple[float, float, float]:
        """
        Skeleton bone: chest
        MediaPipe 源: LEFT_SHOULDER + RIGHT_SHOULDER（平均）
        躯干类：只保留 Z 轴（yaw），X/Y 清零
        """
        self.bone_bias.setdefault( "chest", (0.0, 0.0, 0.0))
        self.bone_scale.setdefault("chest", (1.0, 1.0, 1.0))
        raw = self._avg(["LEFT_SHOULDER", "RIGHT_SHOULDER"], humanoid_data)
        if raw is None:
            return (0.0, 0.0, 0.0)
        x, y, z = self._to_skeleton_axes(raw)
        return self._apply_bias_scale("chest", 0.0, 0.0, 0)

    def convert_hips(self, humanoid_data: HumanoidPoseData) -> Tuple[float, float, float]:
        """
        Skeleton bone: hips
        MediaPipe 源: LEFT_HIP + RIGHT_HIP（平均）
        躯干类：只保留 Z 轴（yaw），X/Y 清零
        """
        self.bone_bias.setdefault( "hips", (0.0, 0.0, 0.0))
        self.bone_scale.setdefault("hips", (1.0, 1.0, 1.0))
        raw = self._avg(["LEFT_HIP", "RIGHT_HIP"], humanoid_data)
        if raw is None:
            return (0.0, 0.0, 0.0)
        x, y, z = self._to_skeleton_axes(raw)
        return self._apply_bias_scale("hips", 0.0, 0.0, z)

    def convert_tail(self, humanoid_data: HumanoidPoseData) -> Tuple[float, float, float]:
        """
        Skeleton bone: tail
        MediaPipe 源: LEFT_HIP + RIGHT_HIP（平均）
        躯干类：只保留 Z 轴（yaw），X/Y 清零
        """
        self.bone_bias.setdefault( "tail", (0.0, 0.0, 0.0))
        self.bone_scale.setdefault("tail", (1.0, 1.0, 1.0))
        raw = self._avg(["LEFT_HIP", "RIGHT_HIP"], humanoid_data)
        if raw is None:
            return (0.0, 0.0, 0.0)
        x, y, z = self._to_skeleton_axes(raw)
        return self._apply_bias_scale("tail", 0.0, 0.0, 0)

    # ── 左臂 ────────────────────────────────────────────────────────────────

    def convert_shoulder_l(self, humanoid_data: HumanoidPoseData) -> Tuple[float, float, float]:
        """
        Skeleton bone: shoulder.L
        MediaPipe 源: RIGHT_SHOULDER
        """
        self.bone_bias.setdefault( "shoulder.L", (0.0, 0.0, 0.0))
        self.bone_scale.setdefault("shoulder.L", (1.0, 1.0, 1.0))
        raw = humanoid_data.get_angle("RIGHT_SHOULDER")
        if raw is None:
            return (0.0, 0.0, 0.0)
        x, y, z = self._to_skeleton_axes(raw)
        return self._apply_bias_scale("shoulder.L", x, y, z)

    def convert_upper_arm_l(self, humanoid_data: HumanoidPoseData) -> Tuple[float, float, float]:
        """
        Skeleton bone: upper_arm_fk.L
        MediaPipe 源: RIGHT_SHOULDER
        """
        self.bone_bias.setdefault( "upper_arm_fk.L", (85.0, 0.0, 120.0))
        self.bone_scale.setdefault("upper_arm_fk.L", (1.0, 1.0, 1.0))
        raw = humanoid_data.get_angle("RIGHT_SHOULDER")
        if raw is None:
            return (0.0, 0.0, 0.0)
        x, y, z = self._to_skeleton_axes(raw)
        return self._apply_bias_scale("upper_arm_fk.L", x, y, z)

    def convert_forearm_l(self, humanoid_data: HumanoidPoseData) -> Tuple[float, float, float]:
        """
        Skeleton bone: forearm_fk.L
        MediaPipe 源: RIGHT_ELBOW
        """
        self.bone_bias.setdefault( "forearm_fk.L", (60.0, 0.0, -10.0))
        self.bone_scale.setdefault("forearm_fk.L", (1.0, 1.0, 1.0))
        raw = humanoid_data.get_angle("RIGHT_ELBOW")
        if raw is None:
            return (0.0, 0.0, 0.0)
        x, y, z = self._to_skeleton_axes(raw)
        return self._apply_bias_scale("forearm_fk.L", x, z, y)

    def convert_hand_l(self, humanoid_data: HumanoidPoseData) -> Tuple[float, float, float]:
        """
        Skeleton bone: hand_fk.L
        MediaPipe 源: RIGHT_WRIST
        """
        self.bone_bias.setdefault( "hand_fk.L", (0.0, 0.0, 0.0))
        self.bone_scale.setdefault("hand_fk.L", (1.0, 1.0, 1.0))
        raw = humanoid_data.get_angle("RIGHT_WRIST")
        if raw is None:
            return (0.0, 0.0, 0.0)
        x, y, z = self._to_skeleton_axes(raw)
        return self._apply_bias_scale("hand_fk.L", z, y, x)

    # ── 右臂 ────────────────────────────────────────────────────────────────

    def convert_shoulder_r(self, humanoid_data: HumanoidPoseData) -> Tuple[float, float, float]:
        """
        Skeleton bone: shoulder.R
        MediaPipe 源: LEFT_SHOULDER
        """
        self.bone_bias.setdefault( "shoulder.R", (0.0, 0.0, 0.0))
        self.bone_scale.setdefault("shoulder.R", (1.0, 1.0, 1.0))
        raw = humanoid_data.get_angle("LEFT_SHOULDER")
        if raw is None:
            return (0.0, 0.0, 0.0)
        x, y, z = self._to_skeleton_axes(raw)
        return self._apply_bias_scale("shoulder.R", x, y, z)

    def convert_upper_arm_r(self, humanoid_data: HumanoidPoseData) -> Tuple[float, float, float]:
        """
        Skeleton bone: upper_arm_fk.R
        MediaPipe 源: LEFT_SHOULDER
        """
        self.bone_bias.setdefault( "upper_arm_fk.R", (85, 0.0, 120.0))
        # self.bone_bias.setdefault( "upper_arm_fk.R", (0, 40.0, -5.0))

        self.bone_scale.setdefault("upper_arm_fk.R", (1.0, 1.0, 1.0))
        raw = humanoid_data.get_angle("LEFT_SHOULDER")
        if raw is None:
            return (0.0, 0.0, 0.0)
        x, y, z = self._to_skeleton_axes(raw)
        return self._apply_bias_scale("upper_arm_fk.R", x, y, z)

    def convert_forearm_r(self, humanoid_data: HumanoidPoseData) -> Tuple[float, float, float]:
        """
        Skeleton bone: forearm_fk.R
        MediaPipe 源: LEFT_ELBOW
        """
        self.bone_bias.setdefault( "forearm_fk.R", (60.0, 0.0, -10.0))
        self.bone_scale.setdefault("forearm_fk.R", (1.0, 1.0, 1.0))
        raw = humanoid_data.get_angle("LEFT_ELBOW")
        if raw is None:
            return (0.0, 0.0, 0.0)
        x, y, z = self._to_skeleton_axes(raw)
        return self._apply_bias_scale("forearm_fk.R", x, z, y)

    def convert_hand_r(self, humanoid_data: HumanoidPoseData) -> Tuple[float, float, float]:
        """
        Skeleton bone: hand_fk.R
        MediaPipe 源: LEFT_WRIST
        """
        self.bone_bias.setdefault( "hand_fk.R", (0.0, 0.0, 0.0))
        self.bone_scale.setdefault("hand_fk.R", (1.0, 1.0, -1.0))
        raw = humanoid_data.get_angle("LEFT_WRIST")
        if raw is None:
            return (0.0, 0.0, 0.0)
        x, y, z = self._to_skeleton_axes(raw)
        return self._apply_bias_scale("hand_fk.R", z, y, x)

    # ── 左腿 ────────────────────────────────────────────────────────────────

    def convert_foot_l(self, humanoid_data: HumanoidPoseData) -> Tuple[float, float, float]:
        """
        Skeleton bone: foot_ik.L
        MediaPipe 源: RIGHT_ANKLE
        """
        self.bone_bias.setdefault( "foot_ik.L", (0.0, 0.0, 0.0))
        self.bone_scale.setdefault("foot_ik.L", (1.0, 1.0, 1.0))
        raw = humanoid_data.get_angle("RIGHT_ANKLE")
        if raw is None:
            return (0.0, 0.0, 0.0)
        x, y, z = self._to_skeleton_axes(raw)
        return self._apply_bias_scale("foot_ik.L", x, y, z)

    def convert_toe_l(self, humanoid_data: HumanoidPoseData) -> Tuple[float, float, float]:
        """
        Skeleton bone: toe.L
        MediaPipe 源: RIGHT_FOOT_INDEX
        """
        self.bone_bias.setdefault( "toe.L", (0.0, 0.0, 0.0))
        self.bone_scale.setdefault("toe.L", (1.0, 1.0, 1.0))
        raw = humanoid_data.get_angle("RIGHT_FOOT_INDEX")
        if raw is None:
            return (0.0, 0.0, 0.0)
        x, y, z = self._to_skeleton_axes(raw)
        return self._apply_bias_scale("toe.L", x, y, z)

    # ── 右腿 ────────────────────────────────────────────────────────────────

    def convert_foot_r(self, humanoid_data: HumanoidPoseData) -> Tuple[float, float, float]:
        """
        Skeleton bone: foot_ik.R
        MediaPipe 源: LEFT_ANKLE
        """
        self.bone_bias.setdefault( "foot_ik.R", (0.0, 0.0, 0.0))
        self.bone_scale.setdefault("foot_ik.R", (1.0, 1.0, 1.0))
        raw = humanoid_data.get_angle("LEFT_ANKLE")
        if raw is None:
            return (0.0, 0.0, 0.0)
        x, y, z = self._to_skeleton_axes(raw)
        return self._apply_bias_scale("foot_ik.R", x, y, z)

    def convert_toe_r(self, humanoid_data: HumanoidPoseData) -> Tuple[float, float, float]:
        """
        Skeleton bone: toe.R
        MediaPipe 源: LEFT_FOOT_INDEX
        """
        self.bone_bias.setdefault( "toe.R", (0.0, 0.0, 0.0))
        self.bone_scale.setdefault("toe.R", (1.0, 1.0, 1.0))
        raw = humanoid_data.get_angle("LEFT_FOOT_INDEX")
        if raw is None:
            return (0.0, 0.0, 0.0)
        x, y, z = self._to_skeleton_axes(raw)
        return self._apply_bias_scale("toe.R", x, y, z)

    # ── 耳朵 ────────────────────────────────────────────────────────────────

    def convert_ear_l(self, humanoid_data: HumanoidPoseData) -> Tuple[float, float, float]:
        """
        Skeleton bone: ear.L
        MediaPipe 源: LEFT_EAR
        """
        self.bone_bias.setdefault( "ear.L", (0.0, 0.0, 0.0))
        self.bone_scale.setdefault("ear.L", (1.0, 1.0, 1.0))
        raw = humanoid_data.get_angle("LEFT_EAR")
        if raw is None:
            return (0.0, 0.0, 0.0)
        x, y, z = self._to_skeleton_axes(raw)
        return self._apply_bias_scale("ear.L", x, y, z)

    def convert_ear_r(self, humanoid_data: HumanoidPoseData) -> Tuple[float, float, float]:
        """
        Skeleton bone: ear.R
        MediaPipe 源: RIGHT_EAR
        """
        self.bone_bias.setdefault( "ear.R", (0.0, 0.0, 0.0))
        self.bone_scale.setdefault("ear.R", (1.0, 1.0, 1.0))
        raw = humanoid_data.get_angle("RIGHT_EAR")
        if raw is None:
            return (0.0, 0.0, 0.0)
        x, y, z = self._to_skeleton_axes(raw)
        return self._apply_bias_scale("ear.R", x, y, z)
