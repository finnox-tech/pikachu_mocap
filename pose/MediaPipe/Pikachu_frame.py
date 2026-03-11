import math
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d  # noqa: F401
import mediapipe as mp

try:
    import yaml
    _HAVE_YAML = True
except Exception:
    yaml = None
    _HAVE_YAML = False


def _load_yaml_config(path):
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        if _HAVE_YAML:
            cfg = yaml.safe_load(data) or {}
            return cfg.get("bones", []) or []
    except Exception:
        pass

    bones = []
    current = None
    for raw in data.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            if current:
                bones.append(current)
            current = {}
            rest = line[1:].strip()
            if rest.startswith("name:"):
                current["name"] = rest.split(":", 1)[1].strip()
        elif ":" in line and current is not None:
            k, v = line.split(":", 1)
            current[k.strip()] = v.strip()
    if current:
        bones.append(current)
    return bones


class PikachuPlotter:
    def __init__(
        self,
        title="Pikachu 3D",
        show_grid=True,
        show_global_axes=False,
        show_names=False,
        show_axes=False,
        hide_labels=None,
        smooth_alpha=0.2,
        align_torso=True,
        scale=1.5,
        config_path=None,
        figure=None,
    ):
        self.mp_pose = mp.solutions.pose
        self.connections = list(self.mp_pose.POSE_CONNECTIONS)
        self.smooth_alpha = smooth_alpha
        self.align_torso = align_torso
        self.scale = scale
        self.show_names = show_names
        self.show_axes = show_axes
        self.show_global_axes = show_global_axes
        self.show_head_link = False
        self.hide_labels = {n.upper() for n in (hide_labels or [])}

        self.smoothed = None
        self.fixed_limit = None

        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "pikachu.yaml")
        self.config_path = config_path

        if figure is None:
            plt.ion()
            self.fig = plt.figure(title)
        else:
            self.fig = figure
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.view_init(elev=15, azim=-70)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z (Up)")
        self.ax.grid(show_grid)

        self.zoom_scale = 1.0
        self.zoom_min = 0.3
        self.zoom_max = 3.0

        self.points = self.ax.scatter([], [], [], s=10, color="#ffd60a")
        self.lines = []
        for _ in self.connections:
            line, = self.ax.plot([], [], [], linewidth=1, color="#ff9f0a")
            self.lines.append(line)

        self.landmark_names = [lm.name for lm in self.mp_pose.PoseLandmark]
        self.label_offset = (0.02, 0.02, 0.02)
        self.label_fontsize = 7
        self.labels = []
        if self.show_names:
            self.labels = [
                self.ax.text2D(0, 0, name, transform=self.ax.transAxes, fontsize=self.label_fontsize)
                for name in self.landmark_names
            ]

        self.axis_len = 0.15
        self.axis_colors = ("#ff3b30", "#34c759", "#007aff")
        self.triads = []
        if self.show_axes:
            for _ in self.landmark_names:
                x_line, = self.ax.plot([], [], [], linewidth=1, color=self.axis_colors[0])
                y_line, = self.ax.plot([], [], [], linewidth=1, color=self.axis_colors[1])
                z_line, = self.ax.plot([], [], [], linewidth=1, color=self.axis_colors[2])
                self.triads.append((x_line, y_line, z_line))

        self.base_text = None
        self.base_x = self.base_y = self.base_z = None
        if self.show_axes or self.show_names:
            self.base_text = self.ax.text2D(
                0,
                0,
                "base_link",
                transform=self.ax.transAxes,
                fontsize=self.label_fontsize + 1,
                color="#111111",
            )
            if self.show_axes:
                self.base_x, = self.ax.plot([], [], [], linewidth=2, color=self.axis_colors[0])
                self.base_y, = self.ax.plot([], [], [], linewidth=2, color=self.axis_colors[1])
                self.base_z, = self.ax.plot([], [], [], linewidth=2, color=self.axis_colors[2])

        self.global_x = self.global_y = self.global_z = None
        if self.show_global_axes:
            self.global_x, = self.ax.plot([], [], [], linewidth=2, color=self.axis_colors[0])
            self.global_y, = self.ax.plot([], [], [], linewidth=2, color=self.axis_colors[1])
            self.global_z, = self.ax.plot([], [], [], linewidth=2, color=self.axis_colors[2])

        self.head_text = None
        self.head_x = self.head_y = self.head_z = None
        if self.show_axes or self.show_names:
            self.head_text = self.ax.text2D(
                0,
                0,
                "head_link",
                transform=self.ax.transAxes,
                fontsize=self.label_fontsize + 1,
                color="#111111",
            )
            if self.show_axes:
                self.head_x, = self.ax.plot([], [], [], linewidth=2, color=self.axis_colors[0])
                self.head_y, = self.ax.plot([], [], [], linewidth=2, color=self.axis_colors[1])
                self.head_z, = self.ax.plot([], [], [], linewidth=2, color=self.axis_colors[2])

        self.left_shoulder = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
        self.right_shoulder = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        self.left_hip = self.mp_pose.PoseLandmark.LEFT_HIP.value
        self.right_hip = self.mp_pose.PoseLandmark.RIGHT_HIP.value
        self.nose = self.mp_pose.PoseLandmark.NOSE.value
        self.left_ear = self.mp_pose.PoseLandmark.LEFT_EAR.value
        self.right_ear = self.mp_pose.PoseLandmark.RIGHT_EAR.value

        self.parent_map, self.adjacency = self._build_parent_map(
            self.connections,
            [self.left_hip],
            len(self.mp_pose.PoseLandmark),
        )

        self.name_to_idx = {lm.name: lm.value for lm in self.mp_pose.PoseLandmark}
        self.length_scales = {}
        self.config_names = []
        self.last_pts = None
        self._load_lengths()

        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _mat4_mul(self, a, b):
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

    def _invert_rt(self, mat):
        r = [
            [mat[0][0], mat[0][1], mat[0][2]],
            [mat[1][0], mat[1][1], mat[1][2]],
            [mat[2][0], mat[2][1], mat[2][2]],
        ]
        t = [mat[0][3], mat[1][3], mat[2][3]]
        rt = [
            [r[0][0], r[1][0], r[2][0]],
            [r[0][1], r[1][1], r[2][1]],
            [r[0][2], r[1][2], r[2][2]],
        ]
        inv_t = [
            -(rt[0][0] * t[0] + rt[0][1] * t[1] + rt[0][2] * t[2]),
            -(rt[1][0] * t[0] + rt[1][1] * t[1] + rt[1][2] * t[2]),
            -(rt[2][0] * t[0] + rt[2][1] * t[1] + rt[2][2] * t[2]),
        ]
        return [
            [rt[0][0], rt[0][1], rt[0][2], inv_t[0]],
            [rt[1][0], rt[1][1], rt[1][2], inv_t[1]],
            [rt[2][0], rt[2][1], rt[2][2], inv_t[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def _bone_basis(self, head, tail):
        direction = self._vec_sub(tail, head)
        x_axis, length = self._normalize(direction)
        if length < 1e-6:
            x_axis = (1.0, 0.0, 0.0)
        up = (0.0, 0.0, 1.0)
        y_axis, y_len = self._normalize(self._cross(up, x_axis))
        if y_len < 1e-6:
            y_axis, y_len = self._normalize(self._cross((0.0, 1.0, 0.0), x_axis))
            if y_len < 1e-6:
                y_axis = (0.0, 1.0, 0.0)
        z_axis, _ = self._normalize(self._cross(x_axis, y_axis))
        return x_axis, y_axis, z_axis

    def export_skeleton(self):
        if self.last_pts is None:
            return []

        pts = self.last_pts
        total = len(pts)
        parents = self.parent_map
        names = [lm.name for lm in self.mp_pose.PoseLandmark]
        adjacency = self.adjacency

        order = []
        visited = set()

        def _visit(idx):
            if idx in visited:
                return
            parent = parents.get(idx)
            if parent is not None:
                _visit(parent)
            visited.add(idx)
            order.append(idx)

        for i in range(total):
            _visit(i)

        world = {}
        bones = []
        for idx in order:
            parent = parents.get(idx)
            head = pts[parent] if parent is not None else pts[idx]
            if parent is None:
                children = [n for n in adjacency.get(idx, []) if n != parent]
                if children:
                    tail = pts[children[0]]
                else:
                    tail = (head[0] + 0.1, head[1], head[2])
            else:
                tail = pts[idx]

            x_axis, y_axis, z_axis = self._bone_basis(head, tail)
            mat = [
                [x_axis[0], y_axis[0], z_axis[0], head[0]],
                [x_axis[1], y_axis[1], z_axis[1], head[1]],
                [x_axis[2], y_axis[2], z_axis[2], head[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
            if parent is None:
                local = mat
            else:
                local = self._mat4_mul(self._invert_rt(world[parent]), mat)
            world[idx] = mat

            bones.append({
                "name": names[idx],
                "parent": names[parent] if parent is not None else None,
                "head": [float(head[0]), float(head[1]), float(head[2])],
                "tail": [float(tail[0]), float(tail[1]), float(tail[2])],
                "matrix": mat,
                "local_matrix": local,
            })

        return bones

    def _ensure_labels(self):
        if not self.labels:
            self.labels = [
                self.ax.text2D(0, 0, name, transform=self.ax.transAxes, fontsize=self.label_fontsize)
                for name in self.landmark_names
            ]

    def _ensure_axes(self):
        if not self.triads:
            for _ in self.landmark_names:
                x_line, = self.ax.plot([], [], [], linewidth=1, color=self.axis_colors[0])
                y_line, = self.ax.plot([], [], [], linewidth=1, color=self.axis_colors[1])
                z_line, = self.ax.plot([], [], [], linewidth=1, color=self.axis_colors[2])
                self.triads.append((x_line, y_line, z_line))

    def _ensure_base(self):
        if self.base_text is None:
            self.base_text = self.ax.text2D(
                0,
                0,
                "base_link",
                transform=self.ax.transAxes,
                fontsize=self.label_fontsize + 1,
                color="#111111",
            )
        if self.base_x is None:
            self.base_x, = self.ax.plot([], [], [], linewidth=2, color=self.axis_colors[0])
            self.base_y, = self.ax.plot([], [], [], linewidth=2, color=self.axis_colors[1])
            self.base_z, = self.ax.plot([], [], [], linewidth=2, color=self.axis_colors[2])

    def set_show_names(self, enabled: bool):
        self.show_names = bool(enabled)
        if self.show_names:
            self._ensure_labels()
            self._ensure_base()
            for label in self.labels:
                label.set_visible(True)
            if self.base_text is not None:
                self.base_text.set_visible(True)
        else:
            for label in self.labels:
                label.set_visible(False)
            if self.base_text is not None and not self.show_axes:
                self.base_text.set_visible(False)
        self.fig.canvas.draw_idle()

    def set_show_axes(self, enabled: bool):
        self.show_axes = bool(enabled)
        if self.show_axes:
            self._ensure_axes()
            self._ensure_base()
            self._ensure_head_link()
            for x_line, y_line, z_line in self.triads:
                x_line.set_visible(True)
                y_line.set_visible(True)
                z_line.set_visible(True)
            if self.base_x is not None:
                self.base_x.set_visible(True)
                self.base_y.set_visible(True)
                self.base_z.set_visible(True)
            if self.base_text is not None:
                self.base_text.set_visible(True)
            if self.head_text is not None:
                self.head_text.set_visible(True)
            if self.head_x is not None:
                self.head_x.set_visible(True)
                self.head_y.set_visible(True)
                self.head_z.set_visible(True)
        else:
            for x_line, y_line, z_line in self.triads:
                x_line.set_visible(False)
                y_line.set_visible(False)
                z_line.set_visible(False)
                x_line.set_data_3d([], [], [])
                y_line.set_data_3d([], [], [])
                z_line.set_data_3d([], [], [])
            if self.base_x is not None:
                self.base_x.set_visible(False)
                self.base_y.set_visible(False)
                self.base_z.set_visible(False)
                self.base_x.set_data_3d([], [], [])
                self.base_y.set_data_3d([], [], [])
                self.base_z.set_data_3d([], [], [])
            if self.base_text is not None and not self.show_names:
                self.base_text.set_visible(False)
            if self.head_text is not None and not self.show_names:
                self.head_text.set_visible(False)
            if self.head_x is not None:
                self.head_x.set_visible(False)
                self.head_y.set_visible(False)
                self.head_z.set_visible(False)
                self.head_x.set_data_3d([], [], [])
                self.head_y.set_data_3d([], [], [])
                self.head_z.set_data_3d([], [], [])
        self.fig.canvas.draw_idle()

    def set_show_head_link(self, enabled: bool):
        self.show_head_link = bool(enabled)
        if self.show_head_link:
            self._ensure_head_link()
            if self.head_text is not None:
                self.head_text.set_visible(True)
            if self.head_x is not None:
                self.head_x.set_visible(True)
                self.head_y.set_visible(True)
                self.head_z.set_visible(True)
        else:
            if self.head_text is not None:
                self.head_text.set_visible(False)
            if self.head_x is not None:
                self.head_x.set_visible(False)
                self.head_y.set_visible(False)
                self.head_z.set_visible(False)
                self.head_x.set_data_3d([], [], [])
                self.head_y.set_data_3d([], [], [])
                self.head_z.set_data_3d([], [], [])
        self.fig.canvas.draw_idle()

    def _ensure_head_link(self):
        if self.head_text is None:
            self.head_text = self.ax.text2D(
                0,
                0,
                "head_link",
                transform=self.ax.transAxes,
                fontsize=self.label_fontsize + 1,
                color="#111111",
            )
        if self.head_x is None:
            self.head_x, = self.ax.plot([], [], [], linewidth=2, color=self.axis_colors[0])
            self.head_y, = self.ax.plot([], [], [], linewidth=2, color=self.axis_colors[1])
            self.head_z, = self.ax.plot([], [], [], linewidth=2, color=self.axis_colors[2])

    def _load_lengths(self):
        self.length_scales = {}
        bones = _load_yaml_config(self.config_path)
        self.config_names = []
        for item in bones:
            name = str(item.get("name", "")).strip().upper()
            if not name:
                continue
            self.config_names.append(name)
            idx = self.name_to_idx.get(name)
            if idx is None:
                continue
            try:
                length = float(item.get("length", 1.0))
            except Exception:
                length = 1.0
            self.length_scales[idx] = max(0.1, min(5.0, length))

    def reload_config(self):
        self._load_lengths()

    def get_config_names(self):
        return list(self.config_names)

    def close(self):
        plt.ioff()
        plt.close(self.fig)

    def _build_parent_map(self, connections_list, root_indices, total_count):
        adj = {i: set() for i in range(total_count)}
        for a, b in connections_list:
            adj[a].add(b)
            adj[b].add(a)

        parent = {}
        visited = set()

        def _bfs(start):
            queue = [start]
            visited.add(start)
            while queue:
                cur = queue.pop(0)
                for nb in adj[cur]:
                    if nb not in visited:
                        parent[nb] = cur
                        visited.add(nb)
                        queue.append(nb)

        for root in root_indices:
            if 0 <= root < total_count and root not in visited:
                _bfs(root)

        for i in range(total_count):
            if i not in visited:
                _bfs(i)

        return parent, adj

    def _project_to_axes(self, x, y, z):
        x2, y2, _ = proj3d.proj_transform(x, y, z, self.ax.get_proj())
        disp = self.ax.transData.transform((x2, y2))
        ax_coords = self.ax.transAxes.inverted().transform(disp)
        return ax_coords[0], ax_coords[1]

    def _apply_zoom(self):
        if self.fixed_limit is None:
            return
        lim = self.fixed_limit / self.zoom_scale
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(-lim, lim)
        if self.show_global_axes and self.global_x is not None:
            self.global_x.set_data_3d([-lim, lim], [0, 0], [0, 0])
            self.global_y.set_data_3d([0, 0], [-lim, lim], [0, 0])
            self.global_z.set_data_3d([0, 0], [0, 0], [-lim, lim])

    def _on_scroll(self, event):
        if event.button == "up":
            self.zoom_scale = min(self.zoom_max, self.zoom_scale * 1.1)
        elif event.button == "down":
            self.zoom_scale = max(self.zoom_min, self.zoom_scale / 1.1)
        self._apply_zoom()

    def _vec_add(self, a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def _vec_sub(self, a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def _vec_mul(self, a, s):
        return (a[0] * s, a[1] * s, a[2] * s)

    def _dot(self, a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def _cross(self, a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        )

    def _norm(self, a):
        return math.sqrt(self._dot(a, a))

    def _normalize(self, a, eps=1e-8):
        n = self._norm(a)
        if n < eps:
            return (0.0, 0.0, 0.0), 0.0
        return (a[0] / n, a[1] / n, a[2] / n), n

    def _rotate_points(self, points_in, axis, angle):
        axis, _ = self._normalize(axis)
        if axis == (0.0, 0.0, 0.0):
            return points_in
        c = math.cos(angle)
        s = math.sin(angle)
        rotated = []
        for p in points_in:
            v = p
            cross_kv = self._cross(axis, v)
            dot_kv = self._dot(axis, v)
            term1 = self._vec_mul(v, c)
            term2 = self._vec_mul(cross_kv, s)
            term3 = self._vec_mul(axis, dot_kv * (1 - c))
            rotated.append(self._vec_add(self._vec_add(term1, term2), term3))
        return rotated

    def _local_axes(self, idx, pts):
        parent = self.parent_map.get(idx)
        neighbors = list(self.adjacency.get(idx, []))
        children = [n for n in neighbors if n != parent]
        next_idx = children[0] if children else parent

        if next_idx is None:
            return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)

        direction = self._vec_sub(pts[next_idx], pts[idx])
        x_axis, x_len = self._normalize(direction)
        if x_len < 1e-6:
            x_axis = (1.0, 0.0, 0.0)

        y_axis, y_len = self._normalize(self._cross((0.0, 0.0, 1.0), x_axis))
        if y_len < 1e-6:
            y_axis, y_len = self._normalize(self._cross((0.0, 1.0, 0.0), x_axis))
            if y_len < 1e-6:
                y_axis = (0.0, 1.0, 0.0)

        z_axis, z_len = self._normalize(self._cross(x_axis, y_axis))
        if z_len < 1e-6:
            z_axis = (0.0, 0.0, 1.0)

        return x_axis, y_axis, z_axis

    def _to_points(self, landmarks):
        xs = [p.x * 2 - 1 for p in landmarks]
        zs = [-(p.y * 2 - 1) for p in landmarks]
        ys = [-p.z for p in landmarks]

        if self.smoothed is None:
            self.smoothed = list(zip(xs, ys, zs))
        else:
            new_smoothed = []
            for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
                sx, sy, sz = self.smoothed[i]
                sx = (1 - self.smooth_alpha) * sx + self.smooth_alpha * x
                sy = (1 - self.smooth_alpha) * sy + self.smooth_alpha * y
                sz = (1 - self.smooth_alpha) * sz + self.smooth_alpha * z
                new_smoothed.append((sx, sy, sz))
            self.smoothed = new_smoothed

        pts = [p for p in self.smoothed]

        if self.align_torso and len(pts) > max(self.left_shoulder, self.right_shoulder, self.left_hip, self.right_hip):
            mid_hip = self._vec_mul(
                self._vec_add(pts[self.left_hip], pts[self.right_hip]),
                0.5
            )
            mid_sh = self._vec_mul(
                self._vec_add(pts[self.left_shoulder], pts[self.right_shoulder]),
                0.5
            )
            torso = self._vec_sub(mid_sh, mid_hip)
            torso_dir, torso_len = self._normalize(torso)

            pts = [self._vec_sub(p, mid_hip) for p in pts]
            if torso_len > 1e-6:
                pts = [self._vec_mul(p, 1.0 / torso_len) for p in pts]

            target = (0.0, 0.0, 1.0)
            dot_v = self._dot(torso_dir, target)
            if dot_v < -0.999:
                pts = self._rotate_points(pts, (1.0, 0.0, 0.0), math.pi)
            elif dot_v > 0.999:
                pass
            else:
                axis = self._cross(torso_dir, target)
                angle = math.acos(max(-1.0, min(1.0, dot_v)))
                pts = self._rotate_points(pts, axis, angle)

            # rotate around Z so shoulder/hip plane is parallel to XZ
            lr = self._vec_sub(pts[self.right_shoulder], pts[self.left_shoulder])
            if self._norm(lr) < 1e-6:
                lr = self._vec_sub(pts[self.right_hip], pts[self.left_hip])
            lr = (lr[0], lr[1], 0.0)
            lr_dir, lr_len = self._normalize(lr)
            if lr_len > 1e-6:
                current = math.atan2(lr_dir[1], lr_dir[0])
                target_angle = 0.0
                rot = target_angle - current
                pts = self._rotate_points(pts, (0.0, 0.0, 1.0), rot)

        if self.scale != 1.0:
            pts = [self._vec_mul(p, self.scale) for p in pts]

        return pts

    def _remap_lengths(self, pts):
        new_pts = [None] * len(pts)
        root = self.left_hip
        new_pts[root] = pts[root]

        ordered = [root]
        for i in range(len(pts)):
            if i != root and i in self.parent_map:
                ordered.append(i)

        for idx in ordered:
            parent = self.parent_map.get(idx)
            if parent is None:
                continue
            if new_pts[parent] is None:
                new_pts[parent] = pts[parent]
            direction = self._vec_sub(pts[idx], pts[parent])
            direction, length = self._normalize(direction)
            scale = self.length_scales.get(idx, 1.0)
            new_pts[idx] = self._vec_add(new_pts[parent], self._vec_mul(direction, length * scale))

        for i in range(len(new_pts)):
            if new_pts[i] is None:
                new_pts[i] = pts[i]

        return new_pts

    def update(self, landmarks):
        if not landmarks:
            return

        pts = self._to_points(landmarks)
        pts = self._remap_lengths(pts)
        self.last_pts = pts

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]

        if self.fixed_limit is None:
            max_abs = max(max(abs(v) for v in xs), max(abs(v) for v in ys), max(abs(v) for v in zs))
            self.fixed_limit = max(0.5, max_abs * 1.1)
            self._apply_zoom()

        self.points._offsets3d = (xs, ys, zs)
        for (a, b), line in zip(self.connections, self.lines):
            line.set_data_3d([xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]])

        if self.show_names:
            for i, (x, y, z) in enumerate(pts):
                if self.landmark_names[i].upper() in self.hide_labels:
                    self.labels[i].set_visible(False)
                    continue
                self.labels[i].set_visible(True)
                lx, ly, lz = self.label_offset
                x2, y2 = self._project_to_axes(x + lx, y + ly, z + lz)
                self.labels[i].set_position((x2, y2))

        if self.show_axes:
            for i, (x, y, z) in enumerate(pts):
                if self.landmark_names[i].upper() in self.hide_labels:
                    x_line, y_line, z_line = self.triads[i]
                    x_line.set_visible(False)
                    y_line.set_visible(False)
                    z_line.set_visible(False)
                    continue
                x_line, y_line, z_line = self.triads[i]
                x_line.set_visible(True)
                y_line.set_visible(True)
                z_line.set_visible(True)
                ax_x, ax_y, ax_z = self._local_axes(i, pts)
                x_line.set_data_3d(
                    [x, x + self.axis_len * ax_x[0]],
                    [y, y + self.axis_len * ax_x[1]],
                    [z, z + self.axis_len * ax_x[2]],
                )
                y_line.set_data_3d(
                    [x, x + self.axis_len * ax_y[0]],
                    [y, y + self.axis_len * ax_y[1]],
                    [z, z + self.axis_len * ax_y[2]],
                )
                z_line.set_data_3d(
                    [x, x + self.axis_len * ax_z[0]],
                    [y, y + self.axis_len * ax_z[1]],
                    [z, z + self.axis_len * ax_z[2]],
                )

        if self.base_text is not None:
            base = (
                (pts[self.left_hip][0] + pts[self.right_hip][0] + pts[self.left_shoulder][0] + pts[self.right_shoulder][0]) / 4.0,
                (pts[self.left_hip][1] + pts[self.right_hip][1] + pts[self.left_shoulder][1] + pts[self.right_shoulder][1]) / 4.0,
                (pts[self.left_hip][2] + pts[self.right_hip][2] + pts[self.left_shoulder][2] + pts[self.right_shoulder][2]) / 4.0,
            )
            mid_hip = self._vec_mul(
                self._vec_add(pts[self.left_hip], pts[self.right_hip]),
                0.5
            )
            mid_sh = self._vec_mul(
                self._vec_add(pts[self.left_shoulder], pts[self.right_shoulder]),
                0.5
            )
            up_vec = self._vec_sub(mid_sh, mid_hip)
            up_dir, up_len = self._normalize(up_vec)
            if up_len < 1e-6:
                up_dir = (0.0, 0.0, 1.0)
            right_vec = self._vec_sub(pts[self.right_shoulder], pts[self.left_shoulder])
            right_dir, right_len = self._normalize(right_vec)
            if right_len < 1e-6:
                right_dir = (1.0, 0.0, 0.0)
            forward_dir = self._cross(up_dir, right_dir)
            forward_dir, forward_len = self._normalize(forward_dir)
            if forward_len < 1e-6:
                forward_dir = (0.0, 1.0, 0.0)
            right_dir = self._cross(forward_dir, up_dir)
            right_dir, right_len = self._normalize(right_dir)
            if right_len < 1e-6:
                right_dir = (1.0, 0.0, 0.0)
            
            # 修改Y轴方向朝负方向
            forward_dir = self._vec_mul(forward_dir, -1.0)
            
            bx, by, bz = base
            bx2, by2 = self._project_to_axes(
                bx + self.label_offset[0],
                by + self.label_offset[1],
                bz + self.label_offset[2],
            )
            self.base_text.set_position((bx2, by2))
            if self.show_axes and self.base_x is not None:
                scale = self.axis_len * 1.5
                self.base_x.set_data_3d(
                    [bx, bx + scale * right_dir[0]],
                    [by, by + scale * right_dir[1]],
                    [bz, bz + scale * right_dir[2]],
                )
                self.base_y.set_data_3d(
                    [bx, bx + scale * forward_dir[0]],
                    [by, by + scale * forward_dir[1]],
                    [bz, bz + scale * forward_dir[2]],
                )
                self.base_z.set_data_3d(
                    [bx, bx + scale * up_dir[0]],
                    [by, by + scale * up_dir[1]],
                    [bz, bz + scale * up_dir[2]],
                )

        # head_link: 位于LEFT_EAR和RIGHT_EAR的中点，Y轴指向NOSE，Z轴垂直于三点平面
        if self.head_text is not None:
            left_ear_pt = pts[self.left_ear]
            right_ear_pt = pts[self.right_ear]
            nose_pt = pts[self.nose]
            
            # ear中点
            ear_mid = self._vec_mul(self._vec_add(left_ear_pt, right_ear_pt), 0.5)
            
            # Y轴指向NOSE
            y_dir = self._vec_sub(nose_pt, ear_mid)
            y_dir, y_len = self._normalize(y_dir)
            if y_len < 1e-6:
                y_dir = (0.0, 0.0, 1.0)
            
            # 耳朵连线方向（X轴）
            ear_vec = self._vec_sub(right_ear_pt, left_ear_pt)
            ear_dir, ear_len = self._normalize(ear_vec)
            if ear_len < 1e-6:
                ear_dir = (1.0, 0.0, 0.0)
            
            # Z轴垂直于三点平面
            z_dir = self._cross(ear_dir, y_dir)
            z_dir, z_len = self._normalize(z_dir)
            if z_len < 1e-6:
                z_dir = (0.0, 0.0, 1.0)
            
            # X轴
            x_dir = self._cross(y_dir, z_dir)
            x_dir, x_len = self._normalize(x_dir)
            if x_len < 1e-6:
                x_dir = (1.0, 0.0, 0.0)
            
            hx, hy, hz = ear_mid
            hx2, hy2 = self._project_to_axes(
                hx + self.label_offset[0],
                hy + self.label_offset[1],
                hz + self.label_offset[2],
            )
            self.head_text.set_position((hx2, hy2))
            if self.show_axes and self.head_x is not None:
                scale = self.axis_len * 1.5
                self.head_x.set_data_3d(
                    [hx, hx + scale * x_dir[0]],
                    [hy, hy + scale * x_dir[1]],
                    [hz, hz + scale * x_dir[2]],
                )
                self.head_y.set_data_3d(
                    [hx, hx + scale * y_dir[0]],
                    [hy, hy + scale * y_dir[1]],
                    [hz, hz + scale * y_dir[2]],
                )
                self.head_z.set_data_3d(
                    [hx, hx + scale * z_dir[0]],
                    [hy, hy + scale * z_dir[1]],
                    [hz, hz + scale * z_dir[2]],
                )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
