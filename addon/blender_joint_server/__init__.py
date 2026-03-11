bl_info = {
    "name": "Skeleton Server",
    "author": "yue yue",
    "version": (1,0,0),
    "blender": (4,0,0),
    "location": "View3D > Sidebar",
    "category": "Animation",
}

import bpy
import os
from math import degrees

from . import server
from . import rig_sync


# ==========================
# Log helpers
# ==========================

def _ensure_server_logs():
    if not hasattr(server, "log_messages"):
        server.log_messages = []
    if not hasattr(server, "add_log"):
        def _fallback_add_log(message):
            text = str(message)
            server.log_messages.append(text)
            if len(server.log_messages) > 50:
                del server.log_messages[:-50]
            if hasattr(server, "_mark_state_dirty"):
                server._mark_state_dirty()
        server.add_log = _fallback_add_log
    return server.log_messages


# ==========================
# Operator
# ==========================

class SKSERVER_OT_start(bpy.types.Operator):

    bl_idname = "skserver.start"
    bl_label = "Start Skeleton Server"

    def execute(self, context):

        server.start_server()

        if not bpy.app.timers.is_registered(rig_sync.blender_loop):
            bpy.app.timers.register(rig_sync.blender_loop, persistent=True)

        return {'FINISHED'}


class SKSERVER_OT_stop(bpy.types.Operator):

    bl_idname = "skserver.stop"
    bl_label = "Stop Skeleton Server"

    def execute(self, context):

        server.stop_server()

        if bpy.app.timers.is_registered(rig_sync.blender_loop):
            bpy.app.timers.unregister(rig_sync.blender_loop)

        return {'FINISHED'}


# ==========================
# Panel
# ==========================

class SKSERVER_PT_panel(bpy.types.Panel):

    bl_label = "Skeleton Server"

    bl_idname = "SKSERVER_PT_panel"

    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Skeleton"

    def draw(self, context):

        layout = self.layout

        layout.operator("skserver.start")
        layout.operator("skserver.stop")

        layout.separator()

        if server.server_running:
            layout.label(text="Server: Running", icon="CHECKMARK")
        else:
            layout.label(text="Server: Stopped", icon="CANCEL")

        if server.client_connected:
            layout.label(text="Client: Connected", icon="LINKED")
        else:
            layout.label(text="Client: Waiting", icon="UNLINKED")

        layout.separator()

        arm = rig_sync.get_armature()

        if arm:
            layout.label(text=f"Armature: {arm.name}")
        else:
            layout.label(text="Armature: None")

        if hasattr(rig_sync, "get_active_pose_bone"):
            pb = rig_sync.get_active_pose_bone(context)
        else:
            pb = getattr(context, "active_pose_bone", None)

        if pb:
            if hasattr(rig_sync, "get_bone_angles"):
                x, y, z = rig_sync.get_bone_angles(pb)
            else:
                e = pb.rotation_euler
                x, y, z = degrees(e.x), degrees(e.y), degrees(e.z)
            layout.label(text=f"Active Bone: {pb.name}")
            layout.label(text=f"X: {x:.1f}  Y: {y:.1f}  Z: {z:.1f}")
        else:
            layout.label(text="Active Bone: None")

        layout.separator()

        layout.separator()
        layout.label(text="Logs:")
        logs = _ensure_server_logs()[-8:]
        if not logs:
            layout.label(text="(no logs)")
        for line in logs:
            layout.label(text=str(line)[:80])

        layout.separator()
        layout.label(text="Paths:")
        layout.label(text=f"Addon: {os.path.abspath(__file__)[:80]}")
        if hasattr(server, "__file__"):
            layout.label(text=f"Server: {os.path.abspath(server.__file__)[:80]}")
        if hasattr(rig_sync, "__file__"):
            layout.label(text=f"RigSync: {os.path.abspath(rig_sync.__file__)[:80]}")


classes = [
    SKSERVER_OT_start,
    SKSERVER_OT_stop,
    SKSERVER_PT_panel
]


def register():

    for c in classes:
        bpy.utils.register_class(c)
    logs = _ensure_server_logs()
    server.add_log("Addon registered")
    server.add_log(f"Addon path: {os.path.abspath(__file__)}")
    if hasattr(server, "__file__"):
        server.add_log(f"Server path: {os.path.abspath(server.__file__)}")
    if hasattr(rig_sync, "__file__"):
        server.add_log(f"RigSync path: {os.path.abspath(rig_sync.__file__)}")


def unregister():

    for c in classes:
        bpy.utils.unregister_class(c)
