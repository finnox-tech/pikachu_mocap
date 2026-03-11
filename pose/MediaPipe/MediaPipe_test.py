import argparse
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d  # noqa: F401
import math
try:
    from mediapipe.framework.formats import landmark_pb2
except Exception:
    landmark_pb2 = None

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--names", action="store_true", help="show landmark names")
parser.add_argument("-a", "--axes", action="store_true", help="show joint local axes")
parser.add_argument("-g", "--grid", action="store_true", help="show grid and global axes")
args = parser.parse_args()

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Failed to open camera index 0")

# Matplotlib 3D setup
plt.ion()
fig = plt.figure("Pose 3D")
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=15, azim=-70)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z (Up)")
ax.grid(args.grid)

zoom_scale = 1.0
zoom_min = 0.3
zoom_max = 3.0

connections = list(mp_pose.POSE_CONNECTIONS)
HIDE_LABELS = [
    "LEFT_EYE_INNER",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE_OUTER",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_EAR",
    "RIGHT_EAR",
]
HIDE_LABELS_SET = {name.upper() for name in HIDE_LABELS}
FILTERED_LANDMARKS = {lm.value for lm in mp_pose.PoseLandmark}
points = ax.scatter([], [], [], s=10, color="#34c759")
lines = []
for _ in connections:
    line, = ax.plot([], [], [], linewidth=1, color="#007aff")
    lines.append(line)

landmark_names = [lm.name for lm in mp_pose.PoseLandmark]
print("Pose landmarks:")
for i, name in enumerate(landmark_names):
    print(f"{i:02d}: {name}")
label_offset = (0.02, 0.02, 0.02)
label_fontsize = 7
labels = []
if args.names:
    labels = [
        ax.text2D(0, 0, name, transform=ax.transAxes, fontsize=label_fontsize)
        for name in landmark_names
    ]

axis_len = 0.05
axis_colors = ("#ff3b30", "#34c759", "#007aff")  # X, Y, Z
triads = []
if args.axes:
    for _ in landmark_names:
        x_line, = ax.plot([], [], [], linewidth=1, color=axis_colors[0])
        y_line, = ax.plot([], [], [], linewidth=1, color=axis_colors[1])
        z_line, = ax.plot([], [], [], linewidth=1, color=axis_colors[2])
        triads.append((x_line, y_line, z_line))

base_text = None
base_x = base_y = base_z = None
if args.axes or args.names:
    base_text = ax.text2D(0, 0, "base_link", transform=ax.transAxes, fontsize=label_fontsize + 1, color="#111111")
    if args.axes:
        base_x, = ax.plot([], [], [], linewidth=2, color=axis_colors[0])
        base_y, = ax.plot([], [], [], linewidth=2, color=axis_colors[1])
        base_z, = ax.plot([], [], [], linewidth=2, color=axis_colors[2])

global_x = global_y = global_z = None
if args.grid:
    global_x, = ax.plot([], [], [], linewidth=2, color=axis_colors[0])
    global_y, = ax.plot([], [], [], linewidth=2, color=axis_colors[1])
    global_z, = ax.plot([], [], [], linewidth=2, color=axis_colors[2])


def _project_to_axes(x, y, z):
    x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    disp = ax.transData.transform((x2, y2))
    ax_coords = ax.transAxes.inverted().transform(disp)
    return ax_coords[0], ax_coords[1]


def _apply_zoom():
    if fixed_limit is None:
        return
    lim = fixed_limit / zoom_scale
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    if args.grid and global_x is not None:
        global_x.set_data_3d([-lim, lim], [0, 0], [0, 0])
        global_y.set_data_3d([0, 0], [-lim, lim], [0, 0])
        global_z.set_data_3d([0, 0], [0, 0], [-lim, lim])


def _on_scroll(event):
    global zoom_scale
    if event.button == "up":
        zoom_scale = min(zoom_max, zoom_scale * 1.1)
    elif event.button == "down":
        zoom_scale = max(zoom_min, zoom_scale / 1.1)
    _apply_zoom()


fig.canvas.mpl_connect("scroll_event", _on_scroll)

frame_id = 0
alpha = 0.2
smoothed = None
fixed_limit = None
SKELETON_SCALE = 1.5

LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
ROOT_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value
]


def _build_parent_map(connections_list, root_indices, total_count):
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


parent_map, adjacency = _build_parent_map(
    connections,
    ROOT_LANDMARKS,
    len(landmark_names)
)


def _vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )


def _norm(a):
    return math.sqrt(_dot(a, a))


def _normalize(a, eps=1e-8):
    n = _norm(a)
    if n < eps:
        return (0.0, 0.0, 0.0), 0.0
    return (a[0] / n, a[1] / n, a[2] / n), n


def _rotate_points(points_in, axis, angle):
    axis, _ = _normalize(axis)
    if axis == (0.0, 0.0, 0.0):
        return points_in
    c = math.cos(angle)
    s = math.sin(angle)
    rotated = []
    for p in points_in:
        v = p
        cross_kv = _cross(axis, v)
        dot_kv = _dot(axis, v)
        term1 = _vec_mul(v, c)
        term2 = _vec_mul(cross_kv, s)
        term3 = _vec_mul(axis, dot_kv * (1 - c))
        rotated.append(_vec_add(_vec_add(term1, term2), term3))
    return rotated


GLOBAL_X = (1.0, 0.0, 0.0)
GLOBAL_Y = (0.0, 1.0, 0.0)
GLOBAL_Z = (0.0, 0.0, 1.0)


def _local_axes(idx, pts):
    parent = parent_map.get(idx)
    neighbors = list(adjacency.get(idx, []))
    children = [n for n in neighbors if n != parent]

    next_idx = children[0] if children else parent

    if next_idx is None:
        return GLOBAL_X, GLOBAL_Y, GLOBAL_Z

    direction = _vec_sub(pts[next_idx], pts[idx])
    x_axis, x_len = _normalize(direction)
    if x_len < 1e-6:
        x_axis = GLOBAL_X

    y_axis, y_len = _normalize(_cross(GLOBAL_Z, x_axis))
    if y_len < 1e-6:
        y_axis, y_len = _normalize(_cross(GLOBAL_Y, x_axis))
        if y_len < 1e-6:
            y_axis = GLOBAL_Y

    z_axis, z_len = _normalize(_cross(x_axis, y_axis))
    if z_len < 1e-6:
        z_axis = GLOBAL_Z

    return x_axis, y_axis, z_axis

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True

        if results.pose_landmarks:
            # build filtered landmarks list for drawing
            if FILTERED_LANDMARKS:
                connections_to_draw = [
                    c for c in mp_pose.POSE_CONNECTIONS
                    if c[0] in FILTERED_LANDMARKS and c[1] in FILTERED_LANDMARKS
                ]
            else:
                connections_to_draw = mp_pose.POSE_CONNECTIONS

            if FILTERED_LANDMARKS and landmark_pb2 is not None:
                fl = landmark_pb2.NormalizedLandmarkList()
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    if i in FILTERED_LANDMARKS:
                        fl.landmark.add().CopyFrom(lm)
                    else:
                        fl.landmark.add()
                draw_landmarks = fl

                mp_drawing.draw_landmarks(
                    frame,
                    draw_landmarks,
                    connections_to_draw,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(52, 199, 89),
                        thickness=1,
                        circle_radius=2
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 122, 0),
                        thickness=1
                    )
                )
            elif FILTERED_LANDMARKS:
                h, w = frame.shape[:2]
                for a, b in connections_to_draw:
                    pa = results.pose_landmarks.landmark[a]
                    pb = results.pose_landmarks.landmark[b]
                    axp, ayp = int(pa.x * w), int(pa.y * h)
                    bxp, byp = int(pb.x * w), int(pb.y * h)
                    cv2.line(frame, (axp, ayp), (bxp, byp), (255, 122, 0), 1)
                for i in FILTERED_LANDMARKS:
                    p = results.pose_landmarks.landmark[i]
                    px, py = int(p.x * w), int(p.y * h)
                    cv2.circle(frame, (px, py), 2, (52, 199, 89), -1)
            else:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    connections_to_draw,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(52, 199, 89),
                        thickness=1,
                        circle_radius=2
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 122, 0),
                        thickness=1
                    )
                )

            # 3D skeleton visualization (fast update)
            frame_id += 1
            if frame_id % 1 == 0:
                lm = results.pose_landmarks.landmark
                xs = [p.x * 2 - 1 for p in lm]
                zs = [-(p.y * 2 - 1) for p in lm]  # Z up
                ys = [-p.z for p in lm]

                if smoothed is None:
                    smoothed = list(zip(xs, ys, zs))
                else:
                    new_smoothed = []
                    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
                        sx, sy, sz = smoothed[i]
                        sx = (1 - alpha) * sx + alpha * x
                        sy = (1 - alpha) * sy + alpha * y
                        sz = (1 - alpha) * sz + alpha * z
                        new_smoothed.append((sx, sy, sz))
                    smoothed = new_smoothed

                xs = [p[0] for p in smoothed]
                ys = [p[1] for p in smoothed]
                zs = [p[2] for p in smoothed]

                # normalize + align torso to Z up
                pts = list(zip(xs, ys, zs))
                mid_hip = _vec_mul(
                    _vec_add(pts[LEFT_HIP], pts[RIGHT_HIP]),
                    0.5
                )
                mid_sh = _vec_mul(
                    _vec_add(pts[LEFT_SHOULDER], pts[RIGHT_SHOULDER]),
                    0.5
                )
                torso = _vec_sub(mid_sh, mid_hip)
                torso_dir, torso_len = _normalize(torso)

                # translate to hip center
                pts = [_vec_sub(p, mid_hip) for p in pts]

                # scale by torso length
                if torso_len > 1e-6:
                    pts = [_vec_mul(p, 1.0 / torso_len) for p in pts]

                # rotate torso to Z axis
                target = (0.0, 0.0, 1.0)
                dot_v = _dot(torso_dir, target)
                if dot_v < -0.999:
                    pts = _rotate_points(pts, (1.0, 0.0, 0.0), math.pi)
                elif dot_v > 0.999:
                    pass
                else:
                    axis = _cross(torso_dir, target)
                    angle = math.acos(max(-1.0, min(1.0, dot_v)))
                    pts = _rotate_points(pts, axis, angle)

                # rotate around Z so shoulder/hip plane is parallel to YZ
                lr = _vec_sub(pts[RIGHT_SHOULDER], pts[LEFT_SHOULDER])
                if _norm(lr) < 1e-6:
                    lr = _vec_sub(pts[RIGHT_HIP], pts[LEFT_HIP])
                lr = (lr[0], lr[1], 0.0)
                lr_dir, lr_len = _normalize(lr)
                if lr_len > 1e-6:
                    current = math.atan2(lr_dir[1], lr_dir[0])
                    target_angle = math.pi / 2.0  # align to +Y
                    rot = target_angle - current
                    pts = _rotate_points(pts, (0.0, 0.0, 1.0), rot)

                pts = [_vec_mul(p, SKELETON_SCALE) for p in pts]

                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                zs = [p[2] for p in pts]

                # fixed axes: 1.1x max extent from first valid frame
                if fixed_limit is None:
                    max_abs = max(
                        max(abs(v) for v in xs),
                        max(abs(v) for v in ys),
                        max(abs(v) for v in zs)
                    )
                    fixed_limit = max(0.5, max_abs * 1.1)
                    _apply_zoom()

                # update points and lines without clearing
                if FILTERED_LANDMARKS:
                    mask = [i in FILTERED_LANDMARKS for i in range(len(xs))]
                    fx = [v for v, keep in zip(xs, mask) if keep]
                    fy = [v for v, keep in zip(ys, mask) if keep]
                    fz = [v for v, keep in zip(zs, mask) if keep]
                    points._offsets3d = (fx, fy, fz)
                else:
                    points._offsets3d = (xs, ys, zs)

                for (a, b), line in zip(connections, lines):
                    if FILTERED_LANDMARKS and (a not in FILTERED_LANDMARKS or b not in FILTERED_LANDMARKS):
                        line.set_data_3d([], [], [])
                    else:
                        line.set_data_3d([xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]])

                # labels and local axis triads
                if args.names:
                    for i, (x, y, z) in enumerate(pts):
                        if landmark_names[i].upper() in HIDE_LABELS_SET:
                            labels[i].set_visible(False)
                            continue
                        labels[i].set_visible(True)
                        lx, ly, lz = label_offset
                        x2, y2 = _project_to_axes(x + lx, y + ly, z + lz)
                        labels[i].set_position((x2, y2))

                if args.axes:
                    for i, (x, y, z) in enumerate(pts):
                        x_line, y_line, z_line = triads[i]
                        if FILTERED_LANDMARKS and i not in FILTERED_LANDMARKS:
                            x_line.set_data_3d([], [], [])
                            y_line.set_data_3d([], [], [])
                            z_line.set_data_3d([], [], [])
                            continue
                        ax_x, ax_y, ax_z = _local_axes(i, pts)
                        x_line.set_data_3d(
                            [x, x + axis_len * ax_x[0]],
                            [y, y + axis_len * ax_x[1]],
                            [z, z + axis_len * ax_x[2]]
                        )
                        y_line.set_data_3d(
                            [x, x + axis_len * ax_y[0]],
                            [y, y + axis_len * ax_y[1]],
                            [z, z + axis_len * ax_y[2]]
                        )
                        z_line.set_data_3d(
                            [x, x + axis_len * ax_z[0]],
                            [y, y + axis_len * ax_z[1]],
                            [z, z + axis_len * ax_z[2]]
                        )

                # base_link at torso center, aligned with global axes
                base = (
                    (pts[LEFT_HIP][0] + pts[RIGHT_HIP][0] + pts[LEFT_SHOULDER][0] + pts[RIGHT_SHOULDER][0]) / 4.0,
                    (pts[LEFT_HIP][1] + pts[RIGHT_HIP][1] + pts[LEFT_SHOULDER][1] + pts[RIGHT_SHOULDER][1]) / 4.0,
                    (pts[LEFT_HIP][2] + pts[RIGHT_HIP][2] + pts[LEFT_SHOULDER][2] + pts[RIGHT_SHOULDER][2]) / 4.0
                )
                bx, by, bz = base
                if base_text is not None:
                    bx2, by2 = _project_to_axes(
                        bx + label_offset[0],
                        by + label_offset[1],
                        bz + label_offset[2]
                    )
                    base_text.set_position((bx2, by2))
                if args.axes:
                    base_x.set_data_3d([bx, bx + axis_len * 1.5], [by, by], [bz, bz])
                    base_y.set_data_3d([bx, bx], [by, by + axis_len * 1.5], [bz, bz])
                    base_z.set_data_3d([bx, bx], [by, by], [bz, bz + axis_len * 1.5])

                fig.canvas.draw_idle()
                fig.canvas.flush_events()

        cv2.imshow("pose", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close(fig)
