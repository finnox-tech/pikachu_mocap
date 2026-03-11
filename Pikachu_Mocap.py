import sys
import socket
import json
import ast
import os
import math
import cv2
import mediapipe as mp
from typing import Any

from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


HOST = "127.0.0.1"
PORT = 9999
BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "addon", "scripts", "joint_config.yaml")
SKELETON_PATH = os.path.join(BASE_DIR, "addon", "scripts", "pikachu_skeleton.yaml")
MEDIA_DIR = os.path.join(BASE_DIR, "pose", "MediaPipe")

if MEDIA_DIR not in sys.path:
    sys.path.append(MEDIA_DIR)

from MediaPipe_detect import MediaPipeDetector
from Humanoid_frame import HumanoidPlotter
from Pikachu_frame import PikachuPlotter

try:
    import yaml
    _HAVE_YAML = True
except Exception:
    yaml = None
    _HAVE_YAML = False


def _parse_yaml_value(value: str) -> Any:

    if value is None:
        return ""

    v = value.strip()
    if not v:
        return ""

    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]

    if (v.startswith("[") and v.endswith("]")) or (v.startswith("{") and v.endswith("}")):
        try:
            return ast.literal_eval(v)
        except Exception:
            pass

    try:
        return int(v)
    except ValueError:
        pass

    try:
        return float(v)
    except ValueError:
        pass

    return v


def _parse_axis_spec(value):

    if value is None:
        return 0, -180, 180

    if isinstance(value, (int, float)):
        return int(value), -180, 180

    text = str(value).strip()
    if not text:
        return 0, -180, 180

    parts = text.split(",", 1)
    angle = int(float(parts[0].strip())) if parts[0].strip() else 0

    lower = -180
    upper = 180

    if len(parts) > 1:
        lim = parts[1].strip()
        lim = lim.lstrip("(").rstrip(")")
        if "," in lim:
            a, b = lim.split(",", 1)
            try:
                lower = int(float(a.strip()))
            except ValueError:
                lower = -180
            try:
                upper = int(float(b.strip()))
            except ValueError:
                upper = 180

    if lower > upper:
        lower, upper = upper, lower

    angle = max(lower, min(upper, angle))

    return angle, lower, upper


def _load_yaml_fallback(text: str) -> dict:

    cfg = {"bones": []}
    current = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("bones:"):
            continue
        if line.startswith("-"):
            if current:
                cfg["bones"].append(current)
            current = {}
            rest = line[1:].strip()
            if rest and ":" in rest:
                key, value = rest.split(":", 1)
                current[key.strip()] = _parse_yaml_value(value)
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            if current is not None:
                current[key.strip()] = _parse_yaml_value(value)
            else:
                cfg[key.strip()] = _parse_yaml_value(value)

    if current:
        cfg["bones"].append(current)

    return cfg


def _dump_yaml_fallback(cfg: dict) -> str:

    lines = ["bones:"]
    for bone in cfg.get("bones", []):
        name = bone.get("name", "")
        x = bone.get("x", "0,(-180,180)")
        y = bone.get("y", "0,(-180,180)")
        z = bone.get("z", "0,(-180,180)")
        lines.append(f"  - name: {name}")
        lines.append(f"    x: {x}")
        lines.append(f"    y: {y}")
        lines.append(f"    z: {z}")
    return "\n".join(lines) + "\n"


def _dump_skeleton_fallback(data: dict) -> str:

    lines = ["bones:"]
    for bone in data.get("bones", []):
        name = bone.get("name", "")
        parent = bone.get("parent") or ""
        head = bone.get("head", [0.0, 0.0, 0.0])
        tail = bone.get("tail", [0.0, 0.0, 0.0])
        matrix = bone.get("matrix")
        local_matrix = bone.get("local_matrix")
        lines.append(f"  - name: {name}")
        lines.append(f"    parent: {parent}")
        lines.append(f"    head: [{head[0]}, {head[1]}, {head[2]}]")
        lines.append(f"    tail: [{tail[0]}, {tail[1]}, {tail[2]}]")
        if matrix is not None:
            lines.append(f"    matrix: {json.dumps(matrix)}")
        if local_matrix is not None:
            lines.append(f"    local_matrix: {json.dumps(local_matrix)}")
    return "\n".join(lines) + "\n"


def _parse_vec3(value):

    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return [float(value[0]), float(value[1]), float(value[2])]

    if value is None:
        return [0.0, 0.0, 0.0]

    text = str(value).strip().lstrip("[").rstrip("]")
    if not text:
        return [0.0, 0.0, 0.0]

    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) < 3:
        return [0.0, 0.0, 0.0]

    try:
        return [float(parts[0]), float(parts[1]), float(parts[2])]
    except ValueError:
        return [0.0, 0.0, 0.0]


def _parse_matrix_value(value):
    if isinstance(value, list):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
        try:
            return ast.literal_eval(text)
        except Exception:
            try:
                return json.loads(text)
            except Exception:
                return None
    return None


def load_skeleton():

    if not os.path.exists(SKELETON_PATH):
        return None

    try:
        with open(SKELETON_PATH, "r", encoding="utf-8") as f:
            data = f.read()
            if _HAVE_YAML:
                return yaml.safe_load(data) or {}
            return _load_yaml_fallback(data)
    except Exception as e:
        print("Skeleton load error:", e)
        return None


def save_skeleton(data: dict):

    try:
        normalized = {"bones": []}
        for bone in data.get("bones", []):
            if not isinstance(bone, dict):
                continue
            out = {
                "name": bone.get("name"),
                "parent": bone.get("parent"),
                "head": bone.get("head"),
                "tail": bone.get("tail")
            }
            matrix = bone.get("matrix")
            if isinstance(matrix, list):
                out["matrix"] = json.dumps(matrix)
            elif matrix is not None:
                out["matrix"] = str(matrix)
            local_matrix = bone.get("local_matrix")
            if isinstance(local_matrix, list):
                out["local_matrix"] = json.dumps(local_matrix)
            elif local_matrix is not None:
                out["local_matrix"] = str(local_matrix)
            normalized["bones"].append(out)
        with open(SKELETON_PATH, "w", encoding="utf-8") as f:
            if _HAVE_YAML:
                yaml.safe_dump(normalized, f, allow_unicode=True, sort_keys=False)
            else:
                f.write(_dump_skeleton_fallback(normalized))
    except Exception as e:
        print("Skeleton save error:", e)


def load_config():

    if not os.path.exists(CONFIG_PATH):
        default = {
            "bones": [
                {"name": "upper_arm_fk.L", "x": "0,(-180,180)", "y": "0,(-180,180)", "z": "0,(-180,180)"},
                {"name": "forearm_fk.L", "x": "0,(-180,180)", "y": "0,(-180,180)", "z": "0,(-180,180)"},
                {"name": "hand_fk.L", "x": "0,(-180,180)", "y": "0,(-180,180)", "z": "0,(-180,180)"},
                {"name": "upper_arm_fk.R", "x": "0,(-180,180)", "y": "0,(-180,180)", "z": "0,(-180,180)"},
                {"name": "forearm_fk.R", "x": "0,(-180,180)", "y": "0,(-180,180)", "z": "0,(-180,180)"},
                {"name": "hand_fk.R", "x": "0,(-180,180)", "y": "0,(-180,180)", "z": "0,(-180,180)"}
            ]
        }
        save_config(default)
        return default

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = f.read()
            if _HAVE_YAML:
                return yaml.safe_load(data) or {"bones": []}
            return _load_yaml_fallback(data)
    except Exception as e:
        print("Config load error:", e)
        return {"bones": []}


def save_config(cfg):

    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            if _HAVE_YAML:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
            else:
                f.write(_dump_yaml_fallback(cfg))
    except Exception as e:
        print("Config save error:", e)


class BlenderClient:

    def __init__(self, on_message=None):

        self.sock = None
        self.connected = False
        self.on_message = on_message
        self._buffer = ""

        self._retry_timer = QTimer()
        self._retry_timer.setInterval(1000)
        self._retry_timer.timeout.connect(self._try_connect)

        self._recv_timer = QTimer()
        self._recv_timer.setInterval(30)
        self._recv_timer.timeout.connect(self._poll_socket)

        self._try_connect()

    def send(self, data):

        if not self.connected or self.sock is None:
            return

        try:
            self.sock.sendall((json.dumps(data)+"\n").encode())
        except Exception:
            self._handle_disconnect()

    def set_joint(self, bone, axis, angle):

        self.send({
            "type": "set_joint",
            "bone": bone,
            "axis": axis,
            "angle": angle
        })

    def request_pose(self):
        self.send({"type": "request_pose"})

    def request_bones(self):
        self.send({"type": "request_bones"})

    def request_transforms(self, bones=None):
        payload = {"type": "request_transforms"}
        if bones:
            payload["bones"] = list(bones)
        self.send(payload)

    def _try_connect(self):

        if self.connected:
            return

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(1.0)
            print("Connecting to Blender server...")
            self.sock.connect((HOST,PORT))
            self.sock.setblocking(False)
            self.connected = True
            self._retry_timer.stop()
            self._recv_timer.start()
            print("Connected!")
        except Exception as e:
            if not self._retry_timer.isActive():
                self._retry_timer.start()
            print("Connect failed:", e)
            self._handle_disconnect(retry=False)

    def _poll_socket(self):

        if not self.connected or self.sock is None:
            return

        try:
            data = self.sock.recv(4096)
        except BlockingIOError:
            return
        except Exception:
            self._handle_disconnect()
            return

        if not data:
            self._handle_disconnect()
            return

        self._buffer += data.decode(errors="replace")

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                print("Socket decode error:", line[:200])
                continue
            if self.on_message:
                self.on_message(msg)

    def _handle_disconnect(self, retry=True):

        self.connected = False
        self._recv_timer.stop()

        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

        if retry and not self._retry_timer.isActive():
            self._retry_timer.start()


class AxisSlider(QWidget):

    def __init__(self, label, on_change):

        super().__init__()

        layout = QHBoxLayout()

        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label)
        self.label.setFixedWidth(40)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(-180)
        self.slider.setMaximum(180)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(on_change)

        self.value = QLabel("0")
        self.value.setFixedWidth(40)
        self.value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.range_label = QLabel("(-180,180)")
        self.range_label.setFixedWidth(90)
        self.range_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.value)
        layout.addWidget(self.range_label)

        self.setLayout(layout)

    def set_value(self, val):
        self.slider.setValue(val)
        self.value.setText(str(val))

    def update_value_label(self, val):
        self.value.setText(str(val))

    def set_range(self, lower, upper):
        self.slider.setMinimum(lower)
        self.slider.setMaximum(upper)
        self.range_label.setText(f"({lower},{upper})")


class JointPanel(QWidget):

    def __init__(self, client, on_axis_change, get_angles, get_limits):

        super().__init__()

        self.client = client
        self.on_axis_change = on_axis_change
        self.get_angles = get_angles
        self.get_limits = get_limits
        self.bone = None
        self.axis_sliders = {}

        layout = QVBoxLayout()

        self.title = QLabel("Select a bone")
        self.title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.title.setStyleSheet("font-size: 16px; font-weight: 600;")

        layout.addWidget(self.title)

        for axis in ["x", "y", "z"]:
            slider = AxisSlider(axis.upper(), self._make_on_change(axis))
            self.axis_sliders[axis] = slider
            layout.addWidget(slider)

        layout.addStretch(1)

        self.setLayout(layout)

        self.setEnabled(False)

    def _make_on_change(self, axis):

        def _on_change(val):
            if not self.bone:
                return
            self.axis_sliders[axis].update_value_label(val)
            self.client.set_joint(self.bone, axis, val)
            self.on_axis_change(self.bone, axis, val)

        return _on_change

    def set_bone(self, bone):

        self.bone = bone
        self.title.setText(f"Bone: {bone}")
        angles = self.get_angles(bone)
        limits = self.get_limits(bone)
        self._set_limits(limits)
        self._set_angles(self._clamp_angles(angles, limits))
        self.setEnabled(True)

    def _set_angles(self, angles):

        for axis, val in zip(["x", "y", "z"], angles):
            slider = self.axis_sliders[axis]
            slider.slider.blockSignals(True)
            slider.set_value(val)
            slider.slider.blockSignals(False)

    def _set_limits(self, limits):

        for axis in ["x", "y", "z"]:
            lower, upper = limits.get(axis, (-180, 180))
            self.axis_sliders[axis].set_range(lower, upper)

    def _clamp_angles(self, angles, limits):

        result = []
        for axis, val in zip(["x", "y", "z"], angles):
            lower, upper = limits.get(axis, (-180, 180))
            result.append(max(lower, min(upper, int(val))))
        return result


class BoneItem(QWidget):

    def __init__(self, name, angles, on_select):

        super().__init__()

        self.name = name

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.button = QPushButton(name)
        self.button.clicked.connect(lambda checked=False: on_select(name))

        x, y, z = angles
        self.value = QLabel(f"X(:{x}  Y:{y}  Z:{z}")
        self.value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value.setFixedWidth(140)

        layout.addWidget(self.button, 1)
        layout.addWidget(self.value, 0)

        self.setLayout(layout)

    def set_angles(self, angles):

        x, y, z = angles
        self.value.setText(f"X:{x}  Y:{y}  Z:{z}")


class SkeletonPlot(QWidget):

    def __init__(self):

        super().__init__()

        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title("Skeleton")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.grid(True)
        self.ax.view_init(elev=20, azim=-60)

        self.lines = {}
        self.axis_lines = {}
        self.axis_colors = ("#ff3b30", "#34c759", "#007aff")
        self.points = self.ax.scatter([], [], [], s=10, color="#34c759")
        self.bones = {}
        self.bone_order = []
        self.angles = {}
        self.axis_scale = 0.35
        self.axis_min = 0.1
        self.visible_bones = None

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def _set_axes_equal(self, points):

        if not points:
            return

        xs, ys, zs = zip(*points)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)

        max_range = max(
            max_x - min_x,
            max_y - min_y,
            max_z - min_z,
            1e-6
        )

        mid_x = (min_x + max_x) / 2.0
        mid_y = (min_y + max_y) / 2.0
        mid_z = (min_z + max_z) / 2.0

        half = max_range / 2.0
        self.ax.set_xlim(mid_x - half, mid_x + half)
        self.ax.set_ylim(mid_y - half, mid_y + half)
        self.ax.set_zlim(mid_z - half, mid_z + half)

    def _matmul(self, a, b):
        return [
            [
                a[0][0] * b[0][j] + a[0][1] * b[1][j] + a[0][2] * b[2][j]
                for j in range(3)
            ],
            [
                a[1][0] * b[0][j] + a[1][1] * b[1][j] + a[1][2] * b[2][j]
                for j in range(3)
            ],
            [
                a[2][0] * b[0][j] + a[2][1] * b[1][j] + a[2][2] * b[2][j]
                for j in range(3)
            ],
        ]

    def _matmul4(self, a, b):
        return [
            [
                a[i][0] * b[0][j] +
                a[i][1] * b[1][j] +
                a[i][2] * b[2][j] +
                a[i][3] * b[3][j]
                for j in range(4)
            ]
            for i in range(4)
        ]

    def _identity4(self):
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def _rot4_xyz(self, angles_deg):
        ax = math.radians(angles_deg[0])
        ay = math.radians(angles_deg[1])
        az = math.radians(angles_deg[2])
        r3 = self._euler_xyz(ax, ay, az)
        return [
            [r3[0][0], r3[0][1], r3[0][2], 0.0],
            [r3[1][0], r3[1][1], r3[1][2], 0.0],
            [r3[2][0], r3[2][1], r3[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def _transform_point(self, mat, point):
        x, y, z = point
        return (
            mat[0][0] * x + mat[0][1] * y + mat[0][2] * z + mat[0][3],
            mat[1][0] * x + mat[1][1] * y + mat[1][2] * z + mat[1][3],
            mat[2][0] * x + mat[2][1] * y + mat[2][2] * z + mat[2][3],
        )

    def _base_from_head_tail(self, head, tail):
        x_axis, y_axis, z_axis = self._basis_from_bone(head, tail)
        hx, hy, hz = head
        return [
            [x_axis[0], y_axis[0], z_axis[0], hx],
            [x_axis[1], y_axis[1], z_axis[1], hy],
            [x_axis[2], y_axis[2], z_axis[2], hz],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def _build_order(self):
        children = {name: [] for name in self.bones}
        roots = []
        for name, data in self.bones.items():
            parent = data.get("parent")
            if parent and parent in self.bones:
                children[parent].append(name)
            else:
                roots.append(name)

        order = []
        visited = set()

        def _dfs(node):
            if node in visited:
                return
            visited.add(node)
            order.append(node)
            for child in children.get(node, []):
                _dfs(child)

        for root in roots:
            _dfs(root)
        for name in self.bones:
            if name not in visited:
                _dfs(name)

        self.bone_order = order
    def _normalize(self, v, eps=1e-8):
        x, y, z = v
        n = (x * x + y * y + z * z) ** 0.5
        if n < eps:
            return (0.0, 0.0, 0.0)
        return (x / n, y / n, z / n)

    def _cross(self, a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def _basis_from_matrix(self, mat):
        if not mat or len(mat) < 3 or len(mat[0]) < 3:
            return None
        x_axis = (mat[0][0], mat[1][0], mat[2][0])
        y_axis = (mat[0][1], mat[1][1], mat[2][1])
        z_axis = (mat[0][2], mat[1][2], mat[2][2])
        return (
            self._normalize(x_axis),
            self._normalize(y_axis),
            self._normalize(z_axis),
        )

    def _basis_from_bone(self, head, tail):
        hx, hy, hz = head
        tx, ty, tz = tail
        y_axis = self._normalize((tx - hx, ty - hy, tz - hz))
        up = (0.0, 0.0, 1.0)
        if abs(y_axis[2]) > 0.99:
            up = (0.0, 1.0, 0.0)
        x_axis = self._normalize(self._cross(up, y_axis))
        z_axis = self._normalize(self._cross(y_axis, x_axis))
        return x_axis, y_axis, z_axis

    def _euler_xyz(self, ax, ay, az):
        cx, sx = math.cos(ax), math.sin(ax)
        cy, sy = math.cos(ay), math.sin(ay)
        cz, sz = math.cos(az), math.sin(az)
        rx = [
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ]
        ry = [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ]
        rz = [
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ]
        return self._matmul(rz, self._matmul(ry, rx))

    def update_angles(self, angles):
        self.angles = angles or {}
        self._redraw_scene()

    def set_visible_bones(self, bones):
        if bones:
            self.visible_bones = set(bones)
        else:
            self.visible_bones = None
        self._redraw_scene()

    def update_transforms(self, transforms):

        self.bones = {}

        for bone in transforms:
            name = bone.get("name")
            head = bone.get("head")
            tail = bone.get("tail")
            if not name or not head or not tail:
                continue
            if len(head) != 3 or len(tail) != 3:
                continue
            length = ((tail[0] - head[0]) ** 2 + (tail[1] - head[1]) ** 2 + (tail[2] - head[2]) ** 2) ** 0.5
            self.bones[name] = {
                "parent": bone.get("parent"),
                "head": head,
                "tail": tail,
                "matrix": bone.get("matrix"),
                "local_matrix": bone.get("local_matrix"),
                "length": length
            }

        self._build_order()
        self._redraw_scene()

    def _compute_pose(self):
        worlds = {}
        heads = {}
        tails = {}

        for name in self.bone_order:
            data = self.bones[name]
            angles = self.angles.get(name, [0.0, 0.0, 0.0])
            rot = self._rot4_xyz(angles)

            parent = data.get("parent")
            local = data.get("local_matrix")
            base = data.get("matrix")

            if parent and parent in worlds and local:
                world = self._matmul4(worlds[parent], self._matmul4(local, rot))
            else:
                if not base:
                    if local:
                        base = local
                    else:
                        base = self._base_from_head_tail(data["head"], data["tail"])
                world = self._matmul4(base, rot)

            worlds[name] = world
            heads[name] = self._transform_point(world, (0.0, 0.0, 0.0))
            tails[name] = self._transform_point(world, (0.0, data["length"], 0.0))

        return worlds, heads, tails

    def _redraw_scene(self):

        if not self.bones:
            for line in self.lines.values():
                line.set_data_3d([], [], [])
            for line_set in self.axis_lines.values():
                for line in line_set:
                    line.set_data_3d([], [], [])
            self.points._offsets3d = ([], [], [])
            self.canvas.draw_idle()
            return

        worlds, heads, tails = self._compute_pose()

        points = []
        seen = set()

        for name, head in heads.items():
            if self.visible_bones is not None and name not in self.visible_bones:
                continue
            tail = tails.get(name)
            if not tail:
                continue
            hx, hy, hz = head
            tx, ty, tz = tail
            points.append((hx, hy, hz))
            points.append((tx, ty, tz))
            line = self.lines.get(name)
            if line is None:
                line, = self.ax.plot([], [], [], linewidth=2, color="#007aff")
                self.lines[name] = line
            line.set_data_3d([hx, tx], [hy, ty], [hz, tz])
            seen.add(name)

        for name, line in list(self.lines.items()):
            if name not in seen:
                line.set_data_3d([], [], [])

        if points:
            xs, ys, zs = zip(*points)
            self.points._offsets3d = (xs, ys, zs)
            self._set_axes_equal(points)
        else:
            self.points._offsets3d = ([], [], [])

        axis_seen = set()
        for name, world in worlds.items():
            if self.visible_bones is not None and name not in self.visible_bones:
                continue
            head = heads.get(name)
            tail = tails.get(name)
            if not head or not tail:
                continue
            basis = self._basis_from_matrix(world)
            if basis is None:
                basis = self._basis_from_bone(head, tail)
            x_axis, y_axis, z_axis = basis

            hx, hy, hz = head
            length = ((tail[0] - hx) ** 2 + (tail[1] - hy) ** 2 + (tail[2] - hz) ** 2) ** 0.5
            axis_len = max(self.axis_min, length * self.axis_scale)

            line_set = self.axis_lines.get(name)
            if line_set is None:
                x_line, = self.ax.plot([], [], [], linewidth=1, color=self.axis_colors[0])
                y_line, = self.ax.plot([], [], [], linewidth=1, color=self.axis_colors[1])
                z_line, = self.ax.plot([], [], [], linewidth=1, color=self.axis_colors[2])
                line_set = (x_line, y_line, z_line)
                self.axis_lines[name] = line_set

            x_line, y_line, z_line = line_set
            x_line.set_data_3d(
                [hx, hx + axis_len * x_axis[0]],
                [hy, hy + axis_len * x_axis[1]],
                [hz, hz + axis_len * x_axis[2]],
            )
            y_line.set_data_3d(
                [hx, hx + axis_len * y_axis[0]],
                [hy, hy + axis_len * y_axis[1]],
                [hz, hz + axis_len * y_axis[2]],
            )
            z_line.set_data_3d(
                [hx, hx + axis_len * z_axis[0]],
                [hy, hy + axis_len * z_axis[1]],
                [hz, hz + axis_len * z_axis[2]],
            )
            axis_seen.add(name)

        for name, line_set in list(self.axis_lines.items()):
            if name not in axis_seen:
                for line in line_set:
                    line.set_data_3d([], [], [])

        self.canvas.draw_idle()


class Studio(QWidget):

    def _make_panel(self, title, widget, actions=None):

        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        label = QLabel(title)
        label.setStyleSheet("font-size: 13px; font-weight: 600;")
        header.addWidget(label)
        header.addStretch(1)
        if actions:
            for btn in actions:
                header.addWidget(btn)
        layout.addLayout(header)
        layout.addWidget(widget, 1)
        frame.setLayout(layout)
        return frame

    def __init__(self):

        super().__init__()

        self.client = BlenderClient(self.on_blender_message)
        self.bone_angles = {}
        self.bone_items = {}
        self.bone_order = []
        self.bone_limits = {}
        self.bone_tree = []
        self._connected_once = False
        self._pending_save_skeleton = False
        self._skeleton_loaded = False
        self._skeleton_mtime = None
        self._requested_bones_count = None
        self._save_timeout_timer = QTimer()
        self._save_timeout_timer.setInterval(2000)
        self._save_timeout_timer.setSingleShot(True)
        self._save_timeout_timer.timeout.connect(self._on_save_timeout)
        self.config = load_config()
        bones = self.config.get("bones", [])

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(12, 12, 12, 12)

        list_panel = QFrame()
        list_panel.setFrameShape(QFrame.StyledPanel)
        list_layout = QVBoxLayout()

        self.status_label = QLabel("Status: Waiting")
        self.status_label.setStyleSheet("font-size: 13px;")
        list_layout.addWidget(self.status_label)

        self.reset_btn = QPushButton("Reset All")
        self.reset_btn.clicked.connect(self.reset_all)
        list_layout.addWidget(self.reset_btn)

        self.save_skeleton_btn = QPushButton("Save Skeleton")
        self.save_skeleton_btn.clicked.connect(self.save_skeleton_from_blender)
        list_layout.addWidget(self.save_skeleton_btn)

        list_title = QLabel("Bones")
        list_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        list_layout.addWidget(list_title)

        list_scroll = QScrollArea()
        list_scroll.setWidgetResizable(True)

        list_container = QWidget()
        list_container_layout = QVBoxLayout()

        self.joint_panel = JointPanel(
            self.client,
            self.on_axis_change,
            self.get_angles,
            self.get_limits
        )
        self.skeleton_plot = SkeletonPlot()

        self.camera_label = QLabel("Camera")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(320, 240)
        self.camera_label.setStyleSheet("background: #111; color: #ddd;")

        hide_labels = [
            "LEFT_EYE_INNER",
            "LEFT_EYE_OUTER",
            "RIGHT_EYE_INNER",
            "RIGHT_EYE_OUTER",
            "MOUTH_LEFT",
            "MOUTH_RIGHT",
            "LEFT_EAR",
            "RIGHT_EAR",
        ]

        self.detector = MediaPipeDetector()
        humanoid_fig = Figure()
        pikachu_fig = Figure()
        self.humanoid_plotter = HumanoidPlotter(
            show_grid=True,
            show_global_axes=False,
            show_names=False,
            show_axes=False,
            hide_labels=hide_labels,
            figure=humanoid_fig,
        )
        self.humanoid_canvas = FigureCanvas(humanoid_fig)

        self.pikachu_plotter = PikachuPlotter(
            show_grid=True,
            show_global_axes=False,
            show_names=False,
            show_axes=False,
            hide_labels=hide_labels,
            config_path=os.path.join(MEDIA_DIR, "pikachu.yaml"),
            figure=pikachu_fig,
        )
        self.pikachu_canvas = FigureCanvas(pikachu_fig)

        self.humanoid_names_btn = QPushButton("N")
        self.humanoid_names_btn.setCheckable(True)
        self.humanoid_names_btn.setToolTip("Show names (-n)")
        self.humanoid_axes_btn = QPushButton("A")
        self.humanoid_axes_btn.setCheckable(True)
        self.humanoid_axes_btn.setToolTip("Show axes (-a)")

        self.pikachu_names_btn = QPushButton("N")
        self.pikachu_names_btn.setCheckable(True)
        self.pikachu_names_btn.setToolTip("Show names (-n)")
        self.pikachu_axes_btn = QPushButton("A")
        self.pikachu_axes_btn.setCheckable(True)
        self.pikachu_axes_btn.setToolTip("Show axes (-a)")

        for btn in [
            self.humanoid_names_btn,
            self.humanoid_axes_btn,
            self.pikachu_names_btn,
            self.pikachu_axes_btn,
        ]:
            btn.setFixedSize(24, 20)
            btn.setStyleSheet(
                "QPushButton { font-size: 11px; padding: 0 2px; }"
                "QPushButton:checked { background: #2d6cdf; color: white; }"
            )

        self.humanoid_names_btn.toggled.connect(self.humanoid_plotter.set_show_names)
        self.humanoid_axes_btn.toggled.connect(self.humanoid_plotter.set_show_axes)
        self.pikachu_names_btn.toggled.connect(self.pikachu_plotter.set_show_names)
        self.pikachu_axes_btn.toggled.connect(self.pikachu_plotter.set_show_axes)

        for bone_cfg in bones:
            name = bone_cfg.get("name", "").strip()
            if not name:
                continue
            angles = [
                bone_cfg.get("x", 0),
                bone_cfg.get("y", 0),
                bone_cfg.get("z", 0)
            ]
            ax = _parse_axis_spec(angles[0])
            ay = _parse_axis_spec(angles[1])
            az = _parse_axis_spec(angles[2])
            parsed_angles = [ax[0], ay[0], az[0]]
            self.bone_angles[name] = parsed_angles
            self.bone_limits[name] = {
                "x": (ax[1], ax[2]),
                "y": (ay[1], ay[2]),
                "z": (az[1], az[2])
            }
            item = BoneItem(name, parsed_angles, self.joint_panel.set_bone)
            self.bone_items[name] = item
            self.bone_order.append(name)
            list_container_layout.addWidget(item)

        self.skeleton_plot.set_visible_bones(self.bone_order)

        list_container_layout.addStretch(1)
        list_container.setLayout(list_container_layout)
        list_scroll.setWidget(list_container)

        list_layout.addWidget(list_scroll)
        list_panel.setLayout(list_layout)

        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.addWidget(list_panel)
        left_splitter.addWidget(self.joint_panel)
        left_splitter.setStretchFactor(0, 2)
        left_splitter.setStretchFactor(1, 1)

        grid_container = QFrame()
        grid_container.setFrameShape(QFrame.StyledPanel)
        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(6, 6, 6, 6)
        grid_layout.setSpacing(8)
        grid_layout.addWidget(self._make_panel("Camera", self.camera_label), 0, 0)
        grid_layout.addWidget(
            self._make_panel("Pose 3D", self.humanoid_canvas, [self.humanoid_axes_btn, self.humanoid_names_btn]),
            0,
            1
        )
        grid_layout.addWidget(
            self._make_panel("Pikachu 3D", self.pikachu_canvas, [self.pikachu_axes_btn, self.pikachu_names_btn]),
            1,
            0
        )
        grid_layout.addWidget(self._make_panel("Skeleton", self.skeleton_plot), 1, 1)
        grid_layout.setRowStretch(0, 1)
        grid_layout.setRowStretch(1, 1)
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)
        grid_container.setLayout(grid_layout)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(grid_container)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)
        main_splitter.setSizes([360, 920])

        main_layout.addWidget(main_splitter)
        self.setLayout(main_layout)

        self.setWindowTitle("Skeleton Studio")

        self.resize(1280, 720)

        self.status_timer = QTimer()
        self.status_timer.setInterval(500)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.camera_label.setText("Camera: open failed")
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.camera_timer = QTimer()
        self.camera_timer.setInterval(30)
        self.camera_timer.timeout.connect(self.update_camera)
        self.camera_timer.start()

        self.skeleton_watch_timer = QTimer()
        self.skeleton_watch_timer.setInterval(1000)
        self.skeleton_watch_timer.timeout.connect(self.check_skeleton_file)
        self.skeleton_watch_timer.start()

        self.check_skeleton_file()

        if self.bone_order:
            self.joint_panel.set_bone(self.bone_order[0])

    def update_status(self):

        if self.client.connected:
            self.status_label.setText("Status: Connected")
            if not self._connected_once:
                self._connected_once = True
        else:
            self.status_label.setText("Status: Waiting")
            self._connected_once = False

    def update_camera(self):

        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        results = self.detector.process(frame)
        landmarks = self.detector.extract_landmarks(results)

        if landmarks:
            self.humanoid_plotter.update(landmarks)
            self.pikachu_plotter.update(landmarks)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(52, 199, 89),
                    thickness=1,
                    circle_radius=2
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 122, 0),
                    thickness=1
                )
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.camera_label.setPixmap(
            pix.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def request_transforms(self):
        if self.client.connected:
            self.client.request_transforms()

    def request_pose(self):
        if self.client.connected:
            self.client.request_pose()

    def on_blender_message(self, msg):

        msg_type = msg.get("type")

        if msg_type == "bones":
            self.bone_tree = msg.get("data", [])
        elif msg_type == "pose":
            self._apply_pose(msg.get("data", {}))
        elif msg_type == "debug":
            print("[Blender]", msg.get("message"))
        elif msg_type == "transforms":
            transforms = msg.get("data", [])
            if not transforms:
                print("Transforms empty or missing.")
            else:
                preview = transforms[:3]
                print(f"Transforms received: {len(transforms)} bones. Preview: {preview}")
            root_names = [b.get("name") for b in transforms if b.get("parent") in (None, "")]
            if root_names:
                allowed = list(dict.fromkeys(self.bone_order + root_names))
                self.skeleton_plot.set_visible_bones(allowed)
            self.skeleton_plot.update_transforms(transforms)
            self.skeleton_plot.update_angles(self.bone_angles)
            if self._pending_save_skeleton and transforms:
                self._pending_save_skeleton = False
                if (
                    self._requested_bones_count is not None and
                    len(transforms) != self._requested_bones_count
                ):
                    if len(transforms) > self._requested_bones_count:
                        print(
                            "Info: received",
                            len(transforms),
                            "bones (includes ancestors of",
                            self._requested_bones_count,
                            ")"
                        )
                    else:
                        print(
                            "Warning: requested",
                            self._requested_bones_count,
                            "bones but received",
                            len(transforms)
                        )
                save_skeleton({"bones": transforms})
                self._skeleton_loaded = True
                self._skeleton_mtime = os.path.getmtime(SKELETON_PATH)
                self._requested_bones_count = None
            elif self._pending_save_skeleton and not transforms:
                print("Save Skeleton requested but transforms are empty.")

    def on_axis_change(self, bone, axis, val):

        angles = self.bone_angles.get(bone, [0, 0, 0])

        if axis == "x":
            angles[0] = val
        elif axis == "y":
            angles[1] = val
        elif axis == "z":
            angles[2] = val

        self.bone_angles[bone] = angles

        item = self.bone_items.get(bone)
        if item:
            item.set_angles(angles)
        self.skeleton_plot.update_angles(self.bone_angles)

    def get_angles(self, bone):

        return self.bone_angles.get(bone, [0, 0, 0])

    def get_limits(self, bone):

        return self.bone_limits.get(bone, {
            "x": (-180, 180),
            "y": (-180, 180),
            "z": (-180, 180)
        })

    def _apply_pose(self, pose):

        if not pose:
            return

        for name, angles in pose.items():
            if name not in self.bone_angles:
                continue
            updated = [int(round(a)) for a in angles]
            self.bone_angles[name] = updated
            item = self.bone_items.get(name)
            if item:
                item.set_angles(updated)

        if self.joint_panel.bone and self.joint_panel.bone in pose:
            angles = pose[self.joint_panel.bone]
            limits = self.get_limits(self.joint_panel.bone)
            clamped = self.joint_panel._clamp_angles(angles, limits)
            self.joint_panel._set_angles(clamped)

    def save_skeleton_from_blender(self):

        if not self.client.connected:
            print("Save Skeleton failed: not connected to Blender.")
            return
        if not self.bone_order:
            print("Save Skeleton failed: joint_config.yaml has no bones.")
            return
        print("Save Skeleton: requesting transforms from Blender...")
        self._pending_save_skeleton = True
        bones = self.bone_order
        self._requested_bones_count = len(bones)
        self.client.request_transforms(bones=bones)
        self._save_timeout_timer.start()

    def _on_save_timeout(self):
        if self._pending_save_skeleton:
            print("Save Skeleton timeout: no transforms received.")
            self._pending_save_skeleton = False
            self._requested_bones_count = None

    def _apply_skeleton_data(self, data):

        if not data:
            return
        bones = data.get("bones", [])
        if not bones:
            return
        by_name = {}
        for bone in bones:
            if not isinstance(bone, dict):
                continue
            matrix = bone.get("matrix")
            matrix = _parse_matrix_value(matrix)
            local_matrix = _parse_matrix_value(bone.get("local_matrix"))
            name = bone.get("name")
            if not name:
                continue
            by_name[name] = {
                "name": bone.get("name"),
                "parent": bone.get("parent"),
                "head": _parse_vec3(bone.get("head")),
                "tail": _parse_vec3(bone.get("tail")),
                "matrix": matrix,
                "local_matrix": local_matrix
            }
        normalized = list(by_name.values())
        root_names = [n for n, b in by_name.items() if not b.get("parent")]
        allowed = set(self.bone_order)
        allowed.update(root_names)
        self.skeleton_plot.set_visible_bones(sorted(allowed))
        self.skeleton_plot.update_transforms(normalized)
        self.skeleton_plot.update_angles(self.bone_angles)
        self._skeleton_loaded = True

    def check_skeleton_file(self):

        if not os.path.exists(SKELETON_PATH):
            return

        mtime = os.path.getmtime(SKELETON_PATH)
        if self._skeleton_mtime is not None and mtime == self._skeleton_mtime:
            return

        data = load_skeleton()
        if data:
            self._apply_skeleton_data(data)
            self._skeleton_mtime = mtime

    def reset_all(self):

        for name in self.bone_order:
            limits = self.get_limits(name)
            angles = [
                max(limits["x"][0], min(limits["x"][1], 0)),
                max(limits["y"][0], min(limits["y"][1], 0)),
                max(limits["z"][0], min(limits["z"][1], 0))
            ]
            self.bone_angles[name] = angles
            item = self.bone_items.get(name)
            if item:
                item.set_angles(angles)

            for axis in ["x", "y", "z"]:
                self.client.set_joint(name, axis, 0)

        if self.joint_panel.bone:
            self.joint_panel._set_angles(self.bone_angles[self.joint_panel.bone])
        self.skeleton_plot.update_angles(self.bone_angles)

    def closeEvent(self, event):
        if hasattr(self, "camera_timer"):
            self.camera_timer.stop()
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
        if hasattr(self, "humanoid_plotter") and self.humanoid_plotter is not None:
            self.humanoid_plotter.close()
        if hasattr(self, "pikachu_plotter") and self.pikachu_plotter is not None:
            self.pikachu_plotter.close()
        super().closeEvent(event)


if __name__ == "__main__":

    print("Qt script:", os.path.abspath(__file__))
    print("CONFIG_PATH:", os.path.abspath(CONFIG_PATH))
    print("SKELETON_PATH:", os.path.abspath(SKELETON_PATH))

    app = QApplication(sys.argv)

    w = Studio()

    w.show()

    sys.exit(app.exec())
