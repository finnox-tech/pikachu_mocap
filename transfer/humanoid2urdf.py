"""
Humanoid → URDF 转换模块

将 MediaPipe 检测的人体姿态转换为 URDF 机器人的关节角度。

转换流程（每个关节）：
  1. 从 HumanoidPoseData 读取对应 MediaPipe 关键点角度（度）
  2. 欧拉角 → 旋转矩阵
  3. swap_xy_basis（参考系对齐）
  4. 旋转矩阵 → 欧拉角
  5. 根据关节名取主轴（pitch→Y / roll→X / yaw→Z）
  6. 转弧度，先加 bias，再乘 scale：output = (converted + bias) * scale

URDF 关节列表（Pikachu_V025_flat_21dof）：
  头部：  head_yaw_joint, head_pitch_joint, head_roll_joint
  左臂：  left_arm_pitch_joint, left_arm_roll_joint, left_arm_yaw_joint, left_elbow_ankle_joint
  右臂：  right_arm_pitch_joint, right_arm_roll_joint, right_arm_yaw_joint, right_elbow_ankle_joint
  左腿：  left_hip_pitch_joint, left_hip_roll_joint, left_hip_yaw_joint, left_knee_joint, left_ankle_joint
  右腿：  right_hip_pitch_joint, right_hip_roll_joint, right_hip_yaw_joint, right_knee_joint, right_ankle_joint
"""

import math
from typing import Dict, List, Optional, Tuple

from .data_structures import HumanoidPoseData, UrdfPoseData
from .utils import euler_xyz_to_matrix, matrix_to_euler_xyz, swap_xy_basis


class Humanoid2Urdf:
    """
    Humanoid 参考系 → URDF 参考系姿态转换。

    使用示例：
        converter = Humanoid2Urdf()
        converter.set_bias("head_pitch_joint", math.radians(10))
        converter.set_scale("head_pitch_joint", 0.8)
        urdf_data = converter.convert(humanoid_data, ["head_pitch_joint", "left_hip_pitch_joint"])
    """

    # ── 各 URDF 关节 → MediaPipe 源关键点（从头到脚）──────────────────────
    _JOINT_SOURCE: Dict[str, str] = {
        # 头部
        "head_yaw_joint":            "HEAD_LINK",
        "head_pitch_joint":          "HEAD_LINK",
        "head_roll_joint":           "HEAD_LINK",
        # 左臂
        "left_arm_pitch_joint":      "LEFT_SHOULDER",
        "left_arm_roll_joint":       "LEFT_SHOULDER",
        "left_arm_yaw_joint":        "LEFT_SHOULDER",
        "left_elbow_ankle_joint":    "LEFT_ELBOW",
        # 右臂
        "right_arm_pitch_joint":     "RIGHT_SHOULDER",
        "right_arm_roll_joint":      "RIGHT_SHOULDER",
        "right_arm_yaw_joint":       "RIGHT_SHOULDER",
        "right_elbow_ankle_joint":   "RIGHT_ELBOW",
        # 左腿
        "left_hip_pitch_joint":      "LEFT_HIP",
        "left_hip_roll_joint":       "LEFT_HIP",
        "left_hip_yaw_joint":        "LEFT_HIP",
        "left_knee_joint":           "LEFT_KNEE",
        "left_ankle_joint":          "LEFT_ANKLE",
        # 右腿
        "right_hip_pitch_joint":     "RIGHT_HIP",
        "right_hip_roll_joint":      "RIGHT_HIP",
        "right_hip_yaw_joint":       "RIGHT_HIP",
        "right_knee_joint":          "RIGHT_KNEE",
        "right_ankle_joint":         "RIGHT_ANKLE",
    }

    def __init__(self):
        # 每个关节的角度偏置（弧度）
        self.joint_bias:  Dict[str, float] = {}
        # 每个关节的幅度缩放（推荐 0~1）
        self.joint_scale: Dict[str, float] = {}
        # 关节名 → 具体转换方法（供 convert() 分发，setdefault 在方法内触发）
        self._dispatch = {
            "head_yaw_joint":              self.convert_head_yaw_joint,
            "head_pitch_joint":            self.convert_head_pitch_joint,
            "head_roll_joint":             self.convert_head_roll_joint,
            "left_arm_pitch_joint":        self.convert_left_arm_pitch_joint,
            "left_arm_roll_joint":         self.convert_left_arm_roll_joint,
            "left_arm_yaw_joint":          self.convert_left_arm_yaw_joint,
            "left_elbow_ankle_joint":      self.convert_left_elbow_ankle_joint,
            "right_arm_pitch_joint":       self.convert_right_arm_pitch_joint,
            "right_arm_roll_joint":        self.convert_right_arm_roll_joint,
            "right_arm_yaw_joint":         self.convert_right_arm_yaw_joint,
            "right_elbow_ankle_joint":     self.convert_right_elbow_ankle_joint,
            "left_hip_pitch_joint":        self.convert_left_hip_pitch_joint,
            "left_hip_roll_joint":         self.convert_left_hip_roll_joint,
            "left_hip_yaw_joint":          self.convert_left_hip_yaw_joint,
            "left_knee_joint":             self.convert_left_knee_joint,
            "left_ankle_joint":            self.convert_left_ankle_joint,
            "right_hip_pitch_joint":       self.convert_right_hip_pitch_joint,
            "right_hip_roll_joint":        self.convert_right_hip_roll_joint,
            "right_hip_yaw_joint":         self.convert_right_hip_yaw_joint,
            "right_knee_joint":            self.convert_right_knee_joint,
            "right_ankle_joint":           self.convert_right_ankle_joint,
        }

    # ── bias / scale 管理 ────────────────────────────────────────────────────

    def set_bias(self, joint_name: str, offset_rad: float = 0.0):
        """设置关节角度偏置（弧度）。先加偏置再缩放：output = (converted + bias) * scale"""
        self.joint_bias[joint_name] = offset_rad

    def clear_bias(self, joint_name: str = None):
        """清除偏置。joint_name=None 时清除所有。"""
        if joint_name is None:
            self.joint_bias.clear()
        else:
            self.joint_bias.pop(joint_name, None)

    def set_scale(self, joint_name: str, scale: float = 1.0):
        """设置关节角度缩放（推荐 0~1）。先加偏置再缩放：output = (converted + bias) * scale"""
        self.joint_scale[joint_name] = scale

    def clear_scale(self, joint_name: str = None):
        """清除缩放。joint_name=None 时清除所有。"""
        if joint_name is None:
            self.joint_scale.clear()
        else:
            self.joint_scale.pop(joint_name, None)

    # ── 主入口 ───────────────────────────────────────────────────────────────

    def convert(
        self,
        humanoid_data: HumanoidPoseData,
        target_joints: List[str],
        joint_limits: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> UrdfPoseData:
        """
        将 humanoid_data 中的角度转换为指定关节的 URDF 角度。

        Args:
            humanoid_data: MediaPipe 姿态数据
            target_joints: 目标关节名列表（须在 _JOINT_SOURCE 中）
            joint_limits: 可选的关节角度限制 {joint_name: (lower_rad, upper_rad)}
        Returns:
            UrdfPoseData
        """
        if not humanoid_data.angles:
            return UrdfPoseData()

        result = UrdfPoseData(joint_limits=joint_limits)

        for joint in target_joints:
            method = self._dispatch.get(joint)
            if method is None:
                continue
            angle = method(humanoid_data)
            if joint_limits and joint in joint_limits:
                angle = result.clamp_angle(joint, angle)
            result.set_joint_angle(joint, angle)

        return result

    # ── 核心转换 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _mat3_T(M):
        """3x3 矩阵转置"""
        return [[M[j][i] for j in range(3)] for i in range(3)]

    @staticmethod
    def _mat3_mul(A, B):
        """3x3 矩阵乘法"""
        return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    def _to_urdf_axes(
        self,
        angles: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """
        绝对坐标系转换：MediaPipe 欧拉角 → URDF 参考系欧拉角（度）
        返回 (x_deg, y_deg, z_deg)，适用于无明确父骨骼的关节（肩、髋）。
        """
        mat = euler_xyz_to_matrix(angles)
        mat = swap_xy_basis(mat)
        return matrix_to_euler_xyz(mat)

    def _relative_to(
        self,
        parent_angles: Tuple[float, float, float],
        child_angles:  Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """
        相对坐标系：child 在 parent 局部坐标系下的旋转（度）。
        计算公式：R_rel = R_parent^T @ R_child
        适用于有明确父子骨骼的关节（肘、膝），消除绝对朝向的影响。
        返回 (x_deg, y_deg, z_deg)
        """
        Rp = euler_xyz_to_matrix(parent_angles)
        Rc = euler_xyz_to_matrix(child_angles)
        R_rel = self._mat3_mul(self._mat3_T(Rp), Rc)
        return matrix_to_euler_xyz(R_rel)

    def _apply_bias_scale(self, joint_name: str, deg: float) -> float:
        """
        将已选好的角度（度）加偏置、乘缩放，返回弧度。
        output = (radians(deg) + bias) * scale
        """
        bias  = self.joint_bias.get(joint_name,  0.0)
        scale = self.joint_scale.get(joint_name, 1.0)
        return (math.radians(deg) + bias) * scale

    # ── 各关节显式转换方法（从头到脚）────────────────────────────────────────
    # 每个方法均可独立调用，也可通过 convert() 批量调用。
    # 偏置/缩放默认值写在每个方法内（setdefault：外部已设置时不覆盖）。

    # ── 头部 ────────────────────────────────────────────────────────────────

    def convert_head_yaw_joint(self, humanoid_data: HumanoidPoseData) -> float:
        """
        URDF joint: head_yaw_joint   （头部左右转，Z 轴）
        MediaPipe 源: HEAD_LINK → Z 轴 (yaw)
        """
        self.joint_bias.setdefault( "head_yaw_joint", -1.55)
        self.joint_scale.setdefault("head_yaw_joint", 0.8)
        raw = humanoid_data.get_angle("HEAD_LINK")
        if raw is None:
            return 0.0
        x, y, z = self._to_urdf_axes(raw)
        return self._apply_bias_scale("head_yaw_joint", z)

    def convert_head_pitch_joint(self, humanoid_data: HumanoidPoseData) -> float:
        """
        URDF joint: head_pitch_joint  （头部前后仰，Y 轴）
        MediaPipe 源: HEAD_LINK → Y 轴 (pitch)
        """
        self.joint_bias.setdefault( "head_pitch_joint", 0.0)
        self.joint_scale.setdefault("head_pitch_joint", -1.0)
        raw = humanoid_data.get_angle("HEAD_LINK")
        if raw is None:
            return 0.0
        x, y, z = self._to_urdf_axes(raw)
        return self._apply_bias_scale("head_pitch_joint", y)

    def convert_head_roll_joint(self, humanoid_data: HumanoidPoseData) -> float:
        """
        URDF joint: head_roll_joint   （头部侧倾，X 轴）
        MediaPipe 源: HEAD_LINK → X 轴 (roll)
        """
        self.joint_bias.setdefault( "head_roll_joint", 0.0)
        self.joint_scale.setdefault("head_roll_joint", 1.0)
        raw = humanoid_data.get_angle("HEAD_LINK")
        if raw is None:
            return 0.0
        x, y, z = self._to_urdf_axes(raw)
        return self._apply_bias_scale("head_roll_joint", x)

    # ── 左臂 ────────────────────────────────────────────────────────────────

    def convert_left_arm_pitch_joint(self, humanoid_data: HumanoidPoseData) -> float:
        """
        URDF joint: left_arm_pitch_joint  （左肩前后摆，Y 轴）
        MediaPipe 源: LEFT_ELBOW 相对 BASE_LINK → Z 轴 (pitch)
        """
        self.joint_bias.setdefault( "left_arm_pitch_joint", 0.5)
        self.joint_scale.setdefault("left_arm_pitch_joint", 1.0)
        raw_base = humanoid_data.get_angle("BASE_LINK")
        raw = humanoid_data.get_angle("LEFT_ELBOW")
        if raw_base is None or raw is None:
            return 0.0
        x, y, z = self._relative_to(raw_base, raw)
        return self._apply_bias_scale("left_arm_pitch_joint", z)

    def convert_left_arm_roll_joint(self, humanoid_data: HumanoidPoseData) -> float:
        """
        URDF joint: left_arm_roll_joint   （左肩侧展，X 轴）
        MediaPipe 源: LEFT_ELBOW 相对 BASE_LINK → X 轴 (roll)
        """
        self.joint_bias.setdefault( "left_arm_roll_joint", 0.0)
        self.joint_scale.setdefault("left_arm_roll_joint", 1.0)
        raw_base = humanoid_data.get_angle("BASE_LINK")
        raw = humanoid_data.get_angle("LEFT_ELBOW")
        if raw_base is None or raw is None:
            return 0.0
        x, y, z = self._relative_to(raw_base, raw)
        return self._apply_bias_scale("left_arm_roll_joint", x)

    def convert_left_arm_yaw_joint(self, humanoid_data: HumanoidPoseData) -> float:
        """
        URDF joint: left_arm_yaw_joint    （左臂内/外旋，Z 轴）
        MediaPipe 源: LEFT_ELBOW 相对 BASE_LINK → Y 轴 (yaw)
        """
        self.joint_bias.setdefault( "left_arm_yaw_joint", 0.0)
        self.joint_scale.setdefault("left_arm_yaw_joint", 1.0)
        raw_base = humanoid_data.get_angle("BASE_LINK")
        raw = humanoid_data.get_angle("LEFT_ELBOW")
        if raw_base is None or raw is None:
            return 0.0
        x, y, z = self._relative_to(raw_base, raw)
        return self._apply_bias_scale("left_arm_yaw_joint", y)

    def convert_left_elbow_ankle_joint(self, humanoid_data: HumanoidPoseData) -> float:
        """
        URDF joint: left_elbow_ankle_joint  （左肘弯曲）
        父骨骼: LEFT_SHOULDER（上臂方向）
        子骨骼: LEFT_ELBOW（前臂方向）
        方法: R_upper_arm^T @ R_forearm → Y 轴（肘弯角）
        """
        self.joint_bias.setdefault( "left_elbow_ankle_joint", 0.0)
        self.joint_scale.setdefault("left_elbow_ankle_joint", 1.0)
        raw_parent = humanoid_data.get_angle("LEFT_SHOULDER")
        raw_child  = humanoid_data.get_angle("LEFT_ELBOW")
        if raw_parent is None or raw_child is None:
            return 0.0
        x, y, z = self._relative_to(raw_parent, raw_child)
        return self._apply_bias_scale("left_elbow_ankle_joint", x)

    # ── 右臂 ────────────────────────────────────────────────────────────────

    def convert_right_arm_pitch_joint(self, humanoid_data: HumanoidPoseData) -> float:
        """
        URDF joint: right_arm_pitch_joint  （右肩前后摆，Y 轴）
        MediaPipe 源: RIGHT_ELBOW 相对 BASE_LINK → Z 轴 (pitch)
        """
        self.joint_bias.setdefault( "right_arm_pitch_joint", 0.5)
        self.joint_scale.setdefault("right_arm_pitch_joint", 1.0)
        raw_base = humanoid_data.get_angle("BASE_LINK")
        raw = humanoid_data.get_angle("RIGHT_ELBOW")
        if raw_base is None or raw is None:
            return 0.0
        x, y, z = self._relative_to(raw_base, raw)
        return self._apply_bias_scale("right_arm_pitch_joint", z)

    def convert_right_arm_roll_joint(self, humanoid_data: HumanoidPoseData) -> float:
        """
        URDF joint: right_arm_roll_joint   （右肩侧展，X 轴）
        MediaPipe 源: RIGHT_ELBOW 相对 BASE_LINK → X 轴 (roll)
        """
        self.joint_bias.setdefault( "right_arm_roll_joint", 0.0)
        self.joint_scale.setdefault("right_arm_roll_joint", 1.0)
        raw_base = humanoid_data.get_angle("BASE_LINK")
        raw = humanoid_data.get_angle("RIGHT_ELBOW")
        if raw_base is None or raw is None:
            return 0.0
        x, y, z = self._relative_to(raw_base, raw)
        return self._apply_bias_scale("right_arm_roll_joint", x)

    def convert_right_arm_yaw_joint(self, humanoid_data: HumanoidPoseData) -> float:
        """
        URDF joint: right_arm_yaw_joint    （右臂内/外旋，Z 轴）
        MediaPipe 源: RIGHT_ELBOW 相对 BASE_LINK → Y 轴 (yaw)
        """
        self.joint_bias.setdefault( "right_arm_yaw_joint", 0.0)
        self.joint_scale.setdefault("right_arm_yaw_joint", 1.0)
        raw_base = humanoid_data.get_angle("BASE_LINK")
        raw = humanoid_data.get_angle("RIGHT_ELBOW")
        if raw_base is None or raw is None:
            return 0.0
        x, y, z = self._relative_to(raw_base, raw)
        return self._apply_bias_scale("right_arm_yaw_joint", y)

    def convert_right_elbow_ankle_joint(self, humanoid_data: HumanoidPoseData) -> float:
        """
        URDF joint: right_elbow_ankle_joint  （右肘弯曲）
        父骨骼: RIGHT_SHOULDER（上臂方向）
        子骨骼: RIGHT_ELBOW（前臂方向）
        方法: R_upper_arm^T @ R_forearm → Y 轴（肘弯角）
        """
        self.joint_bias.setdefault( "right_elbow_ankle_joint", 0.0)
        self.joint_scale.setdefault("right_elbow_ankle_joint", 1.0)
        raw_parent = humanoid_data.get_angle("RIGHT_SHOULDER")
        raw_child  = humanoid_data.get_angle("RIGHT_ELBOW")
        if raw_parent is None or raw_child is None:
            return 0.0
        x, y, z = self._relative_to(raw_parent, raw_child)
        return self._apply_bias_scale("right_elbow_ankle_joint", x)

    # ── 左腿 ────────────────────────────────────────────────────────────────

    def convert_left_hip_pitch_joint(self, humanoid_data: HumanoidPoseData) -> float:
        """
        URDF joint: left_hip_pitch_joint  （左髋前后摆，Y 轴）
        MediaPipe 源: LEFT_HIP 相对 BASE_LINK → Y 轴 (pitch)
        """
        self.joint_bias.setdefault( "left_hip_pitch_joint", 0.0)
        self.joint_scale.setdefault("left_hip_pitch_joint", 1.0)
        raw_base = humanoid_data.get_angle("BASE_LINK")
        raw = humanoid_data.get_angle("LEFT_HIP")
        if raw_base is None or raw is None:
            return 0.0
        x, y, z = self._relative_to(raw_base, raw)
        return self._apply_bias_scale("left_hip_pitch_joint", y)

    def convert_left_hip_roll_joint(self, humanoid_data: HumanoidPoseData) -> float:
        """
        URDF joint: left_hip_roll_joint   （左髋侧展，X 轴）
        MediaPipe 源: LEFT_HIP 相对 BASE_LINK → X 轴 (roll)
        """
        self.joint_bias.setdefault( "left_hip_roll_joint", 0.0)
        self.joint_scale.setdefault("left_hip_roll_joint", 1.0)
        raw_base = humanoid_data.get_angle("BASE_LINK")
        raw = humanoid_data.get_angle("LEFT_HIP")
        if raw_base is None or raw is None:
            return 0.0
        x, y, z = self._relative_to(raw_base, raw)
        return self._apply_bias_scale("left_hip_roll_joint", x)

    def convert_left_hip_yaw_joint(self, humanoid_data: HumanoidPoseData) -> float:
        """
        URDF joint: left_hip_yaw_joint    （左髋内/外旋，Z 轴）
        MediaPipe 源: LEFT_HIP 相对 BASE_LINK → Z 轴 (yaw)
        """
        self.joint_bias.setdefault( "left_hip_yaw_joint", 0.0)
        self.joint_scale.setdefault("left_hip_yaw_joint", 1.0)
        raw_base = humanoid_data.get_angle("BASE_LINK")
        raw = humanoid_data.get_angle("LEFT_HIP")
        if raw_base is None or raw is None:
            return 0.0
        x, y, z = self._relative_to(raw_base, raw)
        return self._apply_bias_scale("left_hip_yaw_joint", z)

    def convert_left_knee_joint(self, humanoid_data: HumanoidPoseData) -> float:
        """
        URDF joint: left_knee_joint   （左膝弯曲）
        父骨骼: LEFT_HIP（大腿方向）
        子骨骼: LEFT_KNEE（小腿方向）
        方法: R_thigh^T @ R_shin → Y 轴（膝弯角）
        """
        self.joint_bias.setdefault( "left_knee_joint", 0.0)
        self.joint_scale.setdefault("left_knee_joint", 1.0)
        raw_parent = humanoid_data.get_angle("LEFT_HIP")
        raw_child  = humanoid_data.get_angle("LEFT_KNEE")
        if raw_parent is None or raw_child is None:
            return 0.0
        x, y, z = self._relative_to(raw_parent, raw_child)
        return self._apply_bias_scale("left_knee_joint", y)

    def convert_left_ankle_joint(self, humanoid_data: HumanoidPoseData) -> float:
        """
        URDF joint: left_ankle_joint  （左踝，X 轴）
        MediaPipe 源: LEFT_ANKLE → X 轴 (roll)
        """
        self.joint_bias.setdefault( "left_ankle_joint", 0.0)
        self.joint_scale.setdefault("left_ankle_joint", 1.0)
        raw = humanoid_data.get_angle("LEFT_ANKLE")
        if raw is None:
            return 0.0
        x, y, z = self._to_urdf_axes(raw)
        return self._apply_bias_scale("left_ankle_joint", x)

    # ── 右腿 ────────────────────────────────────────────────────────────────

    def convert_right_hip_pitch_joint(self, humanoid_data: HumanoidPoseData) -> float:
        """
        URDF joint: right_hip_pitch_joint  （右髋前后摆，Y 轴）
        MediaPipe 源: RIGHT_HIP 相对 BASE_LINK → Y 轴 (pitch)
        """
        self.joint_bias.setdefault( "right_hip_pitch_joint", 0.0)
        self.joint_scale.setdefault("right_hip_pitch_joint", 1.0)
        raw_base = humanoid_data.get_angle("BASE_LINK")
        raw = humanoid_data.get_angle("RIGHT_HIP")
        if raw_base is None or raw is None:
            return 0.0
        x, y, z = self._relative_to(raw_base, raw)
        return self._apply_bias_scale("right_hip_pitch_joint", y)

    def convert_right_hip_roll_joint(self, humanoid_data: HumanoidPoseData) -> float:
        """
        URDF joint: right_hip_roll_joint   （右髋侧展，X 轴）
        MediaPipe 源: RIGHT_HIP 相对 BASE_LINK → X 轴 (roll)
        """
        self.joint_bias.setdefault( "right_hip_roll_joint", 0.0)
        self.joint_scale.setdefault("right_hip_roll_joint", 1.0)
        raw_base = humanoid_data.get_angle("BASE_LINK")
        raw = humanoid_data.get_angle("RIGHT_HIP")
        if raw_base is None or raw is None:
            return 0.0
        x, y, z = self._relative_to(raw_base, raw)
        return self._apply_bias_scale("right_hip_roll_joint", x)

    def convert_right_hip_yaw_joint(self, humanoid_data: HumanoidPoseData) -> float:
        """
        URDF joint: right_hip_yaw_joint    （右髋内/外旋，Z 轴）
        MediaPipe 源: RIGHT_HIP 相对 BASE_LINK → Z 轴 (yaw)
        """
        self.joint_bias.setdefault( "right_hip_yaw_joint", 0.0)
        self.joint_scale.setdefault("right_hip_yaw_joint", 1.0)
        raw_base = humanoid_data.get_angle("BASE_LINK")
        raw = humanoid_data.get_angle("RIGHT_HIP")
        if raw_base is None or raw is None:
            return 0.0
        x, y, z = self._relative_to(raw_base, raw)
        return self._apply_bias_scale("right_hip_yaw_joint", z)

    def convert_right_knee_joint(self, humanoid_data: HumanoidPoseData) -> float:
        """
        URDF joint: right_knee_joint   （右膝弯曲）
        父骨骼: RIGHT_HIP（大腿方向）
        子骨骼: RIGHT_KNEE（小腿方向）
        方法: R_thigh^T @ R_shin → Y 轴（膝弯角）
        """
        self.joint_bias.setdefault( "right_knee_joint", 0.0)
        self.joint_scale.setdefault("right_knee_joint", 1.0)
        raw_parent = humanoid_data.get_angle("RIGHT_HIP")
        raw_child  = humanoid_data.get_angle("RIGHT_KNEE")
        if raw_parent is None or raw_child is None:
            return 0.0
        x, y, z = self._relative_to(raw_parent, raw_child)
        return self._apply_bias_scale("right_knee_joint", y)

    def convert_right_ankle_joint(self, humanoid_data: HumanoidPoseData) -> float:
        """
        URDF joint: right_ankle_joint  （右踝，X 轴）
        MediaPipe 源: RIGHT_ANKLE → X 轴 (roll)
        """
        self.joint_bias.setdefault( "right_ankle_joint", 0.0)
        self.joint_scale.setdefault("right_ankle_joint", 1.0)
        raw = humanoid_data.get_angle("RIGHT_ANKLE")
        if raw is None:
            return 0.0
        x, y, z = self._to_urdf_axes(raw)
        return self._apply_bias_scale("right_ankle_joint", x)

