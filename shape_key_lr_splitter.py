bl_info = {
    "name": "Shape Key L/R Splitter",
    "author": "Shape Key L/R Splitter contributors",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "3D View > Sidebar > Shape Key Splitter",
    "description": "Split a shape key into left and right with an adjustable center transition",
    "category": "Object",
}

import bpy
from bpy.props import FloatProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup


def left_weight(x, center, width):
    """Return the positive-X weight, with a smooth, complementary transition."""
    if width <= 0.0:
        if x == center:
            return 0.5
        return 1.0 if x > center else 0.0
    t = max(0.0, min(1.0, (x - center) / width + 0.5))
    return t * t * (3.0 - 2.0 * t)


class SHAPEKEYSPLIT_PG_settings(PropertyGroup):
    transition_percent: FloatProperty(
        name="Transition Width",
        description="Full transition width as a percentage of the Basis mesh's local X width",
        default=20.0,
        min=0.0,
        max=100.0,
        subtype='PERCENTAGE',
    )
    center: FloatProperty(
        name="Center X",
        description="Center of the transition in the object's local X coordinates",
        default=0.0,
        subtype='DISTANCE',
    )


class SHAPEKEYSPLIT_OT_split(Operator):
    bl_idname = "shape_key.split_left_right"
    bl_label = "Split Active Shape Key"
    bl_description = "Create complementary left and right keys from the active shape key"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return bool(obj and obj.type == 'MESH' and obj.mode == 'OBJECT'
                    and obj.data.shape_keys and obj.active_shape_key_index > 0)

    def execute(self, context):
        obj = context.active_object
        keys = obj.data.shape_keys
        if not keys.use_relative:
            self.report({'ERROR'}, "Absolute shape keys are not supported. Use relative shape keys.")
            return {'CANCELLED'}
        source = obj.active_shape_key
        reference = source.relative_key
        basis = keys.key_blocks[0]
        if not basis.data:
            self.report({'ERROR'}, "The mesh has no vertices.")
            return {'CANCELLED'}

        settings = context.scene.shape_key_lr_splitter
        x_values = [point.co.x for point in basis.data]
        width = (max(x_values) - min(x_values)) * settings.transition_percent / 100.0
        weights = [left_weight(x, settings.center, width) for x in x_values]

        # Capture source/reference before adding keys; preserve non-Basis relative keys.
        coordinates = [(tuple(point.co), tuple(ref.co))
                       for point, ref in zip(source.data, reference.data)]
        created = []
        try:
            for suffix, positive_side in (("L", True), ("R", False)):
                result = obj.shape_key_add(name=f"{source.name}.{suffix}", from_mix=False)
                created.append(result)
                result.relative_key = reference
                result.vertex_group = source.vertex_group
                result.interpolation = source.interpolation
                result.slider_min = source.slider_min
                result.slider_max = source.slider_max
                result.value = 0.0
                for index, (source_co, reference_co) in enumerate(coordinates):
                    weight = weights[index] if positive_side else 1.0 - weights[index]
                    result.data[index].co = tuple(
                        reference_co[axis] + (source_co[axis] - reference_co[axis]) * weight
                        for axis in range(3)
                    )
        except Exception as exc:
            for result in reversed(created):
                obj.shape_key_remove(result)
            self.report({'ERROR'}, f"Could not split shape key: {exc}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Created {created[0].name} and {created[1].name}")
        return {'FINISHED'}


class SHAPEKEYSPLIT_PT_panel(Panel):
    bl_label = "Shape Key L/R Splitter"
    bl_idname = "SHAPEKEYSPLIT_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Shape Key Splitter"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            layout.label(text="Select a mesh object.", icon='INFO')
            return
        if obj.mode != 'OBJECT':
            layout.label(text="Switch to Object Mode.", icon='INFO')
            return
        keys = obj.data.shape_keys
        if not keys or obj.active_shape_key_index == 0:
            layout.label(text="Select a non-Basis shape key.", icon='INFO')
            return
        if not keys.use_relative:
            layout.label(text="Relative shape keys are required.", icon='ERROR')
            return

        layout.label(text=f"Active: {obj.active_shape_key.name}")
        settings = context.scene.shape_key_lr_splitter
        layout.prop(settings, "transition_percent", slider=True)
        layout.prop(settings, "center")
        layout.label(text="L: +X   |   R: -X")
        layout.operator("shape_key.split_left_right", icon='MOD_MIRROR')


classes = (
    SHAPEKEYSPLIT_PG_settings,
    SHAPEKEYSPLIT_OT_split,
    SHAPEKEYSPLIT_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.shape_key_lr_splitter = PointerProperty(type=SHAPEKEYSPLIT_PG_settings)


def unregister():
    del bpy.types.Scene.shape_key_lr_splitter
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
