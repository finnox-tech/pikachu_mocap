import sys

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QSlider
)

from PySide6.QtCore import Qt

from robot_model import RobotModel
from robot_viewer import RobotViewer


class MainWindow(QWidget):

    def __init__(self, urdf_path):

        super().__init__()

        self.robot = RobotModel(urdf_path)

        layout = QHBoxLayout(self)

        self.viewer = RobotViewer(self.robot)

        layout.addWidget(self.viewer, 4)

        control_panel = QVBoxLayout()

        for name in self.robot.joint_names:

            label = QLabel(name)

            slider = QSlider(Qt.Horizontal)

            lower, upper = self.robot.joint_limits[name]

            slider.setMinimum(int(lower * 100))
            slider.setMaximum(int(upper * 100))
            slider.setValue(0)

            slider.valueChanged.connect(
                lambda v, n=name: self.set_joint(n, v)
            )

            control_panel.addWidget(label)
            control_panel.addWidget(slider)

        control_panel.addStretch()

        layout.addLayout(control_panel, 1)

    def set_joint(self, name, value):

        angle = value / 100.0

        self.robot.set_joint(name, angle)

        self.viewer.update_robot()


if __name__ == "__main__":

    app = QApplication(sys.argv)

    win = MainWindow("/home/finnox/Pikachu/Mocap/urdf/robot/Pikachu_V025/urdf/Pikachu_V025_flat_21dof.urdf")

    win.resize(1400, 900)

    win.show()

    sys.exit(app.exec())