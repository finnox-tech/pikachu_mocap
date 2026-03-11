bl_info = {
    "name": "Skeleton Server",
    "author": "yue yue",
    "version": (1,0,0),
    "blender": (4,0,0),
    "location": "View3D > Sidebar",
    "category": "Animation",
}

import bpy
from math import degrees

from . import server
from . import rig_sync


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

        if arm:
            layout.label(text="Bones:")

            for b in arm.data.bones[:20]:

                layout.label(text=b.name)


classes = [
    SKSERVER_OT_start,
    SKSERVER_OT_stop,
    SKSERVER_PT_panel
]


def register():

    for c in classes:
        bpy.utils.register_class(c)


def unregister():

    for c in classes:
        bpy.utils.unregister_class(c)
