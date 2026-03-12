import os
import numpy as np
import trimesh
import meshcat
import meshcat.geometry as mg
import meshcat.transformations as mt

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QUrl


class RobotViewer(QWidget):

    def __init__(self, robot_model):

        super().__init__()

        self.robot = robot_model

        # 创建meshcat viewer
        self.viewer = meshcat.Visualizer()

        # 获取meshcat的URL
        self.meshcat_url = self.viewer.url()

        # 创建Qt布局
        layout = QVBoxLayout(self)

        # 创建QWebEngineView来加载meshcat网页
        self.web_view = QWebEngineView()
        self.web_view.setUrl(QUrl(self.meshcat_url))

        # 当网页加载完成后，设置默认相机视角
        self.web_view.loadFinished.connect(self._on_web_view_loaded)

        layout.addWidget(self.web_view)

        # 注意: meshcat.geometry没有Grid属性，所以这里不添加地面网格

        self.mesh_items = {}

        print(f"Meshcat viewer initialized. URL: {self.meshcat_url}")

        self.load_robot()

    def _on_web_view_loaded(self, success):
        """当meshcat网页加载完成后，设置默认相机视角"""
        if success:
            # 使用JavaScript调整OrbitControls的参数
            # 这样可以保持meshcat的交互功能
            js_code = """
            // 等待一小段时间确保viewer和controls都已初始化
            setTimeout(function() {
                // 尝试获取viewer对象
                if (typeof viewer !== 'undefined' && viewer && viewer.controls) {
                    // 设置更近的初始距离
                    viewer.controls.minDistance = 1.0;
                    viewer.controls.maxDistance = 10.0;
                    // 将相机移到更近的位置
                    viewer.camera.position.set(0.5, 0.3, -0.3);
                    viewer.camera.lookAt(0, 0, 0);
                    viewer.controls.update();
                }
            }, 500);
            """
            self.web_view.page().runJavaScript(js_code)

    def load_robot(self):

        for link in self.robot.robot.links:

            if not link.visuals:
                continue

            visual = link.visuals[0]

            if visual.geometry.mesh is None:
                continue

            mesh_path = visual.geometry.mesh.filename

            base_dir = self.robot.base_dir

            mesh_path = os.path.join(base_dir, mesh_path)
            mesh_path = os.path.abspath(mesh_path)

            if not os.path.exists(mesh_path):
                print("Mesh not found:", mesh_path)
                return

            mesh = trimesh.load(mesh_path)

            # 应用visual的origin变换
            if visual.origin is not None:
                mesh = mesh.apply_transform(visual.origin)

            # 创建meshcat的TriangularMeshGeometry
            mesh_geom = mg.TriangularMeshGeometry(mesh.vertices, mesh.faces)

            # 为每个link创建一个场景节点
            self.viewer["robot"][link.name].set_object(mesh_geom)

            # 存储mesh信息（meshcat不需要像pyqtgraph那样存储mesh_item）
            self.mesh_items[link.name] = {
                'mesh': mesh,
                'visual_origin': visual.origin
            }

        # 初始化时更新所有link的位置和姿态
        self.update_robot()

        # 设置robot节点的缩放，使机器人显示得更大
        # 使用缩放矩阵放大1.5倍
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = 1.5
        scale_matrix[1, 1] = 1.5
        scale_matrix[2, 2] = 1.5
        self.viewer["robot"].set_transform(scale_matrix)

    def update_robot(self):

        fk = self.robot.compute_fk()

        for link, T in fk.items():

            if link.name not in self.mesh_items:
                continue

            # meshcat使用4x4变换矩阵
            # T是urdfpy返回的变换矩阵，可以直接使用
            self.viewer["robot"][link.name].set_transform(T)