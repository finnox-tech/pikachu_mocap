import argparse
import cv2
import mediapipe as mp

from MediaPipe_detect import MediaPipeDetector
from Humanoid_frame import HumanoidPlotter
from Pikachu_frame import PikachuPlotter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--names", action="store_true", help="show landmark names")
    parser.add_argument("-a", "--axes", action="store_true", help="show joint local axes")
    parser.add_argument("-g", "--grid", action="store_true", help="show grid and global axes")
    args = parser.parse_args()

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

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera index 0")

    detector = MediaPipeDetector()
    plotter = HumanoidPlotter(
        show_grid=args.grid,
        show_global_axes=args.grid,
        show_names=args.names,
        show_axes=args.axes,
        hide_labels=HIDE_LABELS,
    )
    pikachu = PikachuPlotter(
        show_grid=args.grid,
        show_global_axes=args.grid,
        show_names=args.names,
        show_axes=args.axes,
        hide_labels=HIDE_LABELS,
    )

    print("Pikachu bones:")
    for i, name in enumerate(pikachu.get_config_names()):
        print(f"{i:02d}: {name}")

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    frame_count = 0
    PRINT_ANGLES = True
    PRINT_INTERVAL = 10

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            results = detector.process(frame)
            landmarks = detector.extract_landmarks(results)

            if landmarks:
                plotter.update(landmarks)
                pikachu.update(landmarks)
                frame_count += 1

                if PRINT_ANGLES and plotter.last_angles and frame_count % PRINT_INTERVAL == 0:
                    print(f"\nFrame {frame_count} joint angles (deg):")
                    for name, (ax, ay, az) in plotter.last_angles.items():
                        print(f"{name}: ({ax:.1f}, {ay:.1f}, {az:.1f})")

                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
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

            cv2.imshow("pose", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord('l'):
                pikachu.reload_config()
                print("Reloaded pikachu.yaml")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        plotter.close()
        pikachu.close()


if __name__ == "__main__":
    main()
