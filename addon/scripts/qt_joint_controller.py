import sys
import socket
import json
import os
from typing import Any

from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, QTimer


HOST = "127.0.0.1"
PORT = 9999
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "joint_config.yaml")

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

    def __init__(self):

        self.sock = None
        self.connected = False
        self._retry_timer = QTimer()
        self._retry_timer.setInterval(1000)
        self._retry_timer.timeout.connect(self._try_connect)

        self._try_connect()
        
    def send(self,data):

        if not self.connected or self.sock is None:
            return

        try:
            self.sock.sendall((json.dumps(data)+"\n").encode())
        except Exception:
            self._handle_disconnect()

    def set_joint(self,bone,axis,angle):

        self.send({
            "type":"set_joint",
            "bone":bone,
            "axis":axis,
            "angle":angle
        })

    def _try_connect(self):

        if self.connected:
            return

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(1.0)
            print("Connecting to Blender server...")
            self.sock.connect((HOST,PORT))
            self.sock.settimeout(None)
            self.connected = True
            self._retry_timer.stop()
            print("Connected!")
        except Exception as e:
            if not self._retry_timer.isActive():
                self._retry_timer.start()
            print("Connect failed:", e)
            self._handle_disconnect(retry=False)

    def _handle_disconnect(self, retry=True):

        self.connected = False

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
        self.value = QLabel(f"X(Pitch):{x}  Y(Yaw):{y}  Z(Roll):{z}")
        self.value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value.setFixedWidth(140)

        layout.addWidget(self.button, 1)
        layout.addWidget(self.value, 0)

        self.setLayout(layout)

    def set_angles(self, angles):

        x, y, z = angles
        self.value.setText(f"X:{x}  Y:{y}  Z:{z}")


class Studio(QWidget):

    def __init__(self):

        super().__init__()

        self.client = BlenderClient()
        self.bone_angles = {}
        self.bone_items = {}
        self.bone_order = []
        self.bone_limits = {}
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

        list_container_layout.addStretch(1)
        list_container.setLayout(list_container_layout)
        list_scroll.setWidget(list_container)

        list_layout.addWidget(list_scroll)
        list_panel.setLayout(list_layout)

        side_panel = QFrame()
        side_panel.setFrameShape(QFrame.StyledPanel)
        side_layout = QVBoxLayout()
        side_layout.addWidget(self.joint_panel)
        side_panel.setLayout(side_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(list_panel)
        splitter.addWidget(side_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([800, 400])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        self.setWindowTitle("Skeleton Studio")

        self.resize(1280, 720)

        self.status_timer = QTimer()
        self.status_timer.setInterval(500)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start()

        if self.bone_order:
            self.joint_panel.set_bone(self.bone_order[0])

    def update_status(self):

        if self.client.connected:
            self.status_label.setText("Status: Connected")
        else:
            self.status_label.setText("Status: Waiting")

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

    def get_angles(self, bone):

        return self.bone_angles.get(bone, [0, 0, 0])

    def get_limits(self, bone):

        return self.bone_limits.get(bone, {
            "x": (-180, 180),
            "y": (-180, 180),
            "z": (-180, 180)
        })

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


if __name__ == "__main__":

    app = QApplication(sys.argv)

    w = Studio()

    w.show()

    sys.exit(app.exec())
