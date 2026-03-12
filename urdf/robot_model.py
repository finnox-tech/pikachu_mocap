import os
import numpy as np
from urdfpy import URDF


class RobotModel:

    def __init__(self, urdf_path):

        # 保存URDF路径
        self.urdf_path = os.path.abspath(urdf_path)
        self.base_dir = os.path.dirname(self.urdf_path)

        self.robot = URDF.load(self.urdf_path)

        self.joint_names = []
        self.joint_limits = {}

        for j in self.robot.joints:

            if j.joint_type in ["revolute", "prismatic"]:

                self.joint_names.append(j.name)

                if j.limit:
                    self.joint_limits[j.name] = (
                        j.limit.lower,
                        j.limit.upper
                    )
                else:
                    self.joint_limits[j.name] = (-3.14, 3.14)

        self.q = {j: 0.0 for j in self.joint_names}

    def set_joint(self, name, value):
        self.q[name] = value

    def compute_fk(self):
        return self.robot.link_fk(cfg=self.q)