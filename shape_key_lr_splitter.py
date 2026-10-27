bl_info = {
    "name": "Shape Key L/R Splitter",
    "author": "Shape Key L/R Splitter contributors",
    "version": (1, 2, 0),
    "blender": (3, 6, 0),
    "location": "3D View > Sidebar > Shape Key Splitter",
    "description": "Split a shape key into left and right with an adjustable center transition",
    "category": "Object",
}

import bpy
import gpu
import numpy as np
from array import array
from bpy.props import BoolProperty, EnumProperty, FloatProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup
from gpu_extras.batch import batch_for_shader


_draw_handle = None


def redraw_viewports(_self, context):
    """Refresh every 3D view as the preview settings are changed."""
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def basis_bounds(basis):
    """Return local-space bounds using the same Basis coordinates as the split."""
    coords = array('f', [0.0]) * (len(basis.data) * 3)
    basis.data.foreach_get('co', coords)
    axes = [coords[i::3] for i in range(3)]
    return tuple((min(axis), max(axis)) for axis in axes)


def preview_coordinates(obj, side, amount, center, width):
    """Calculate the pre-modifier mesh with the active key replaced by one side."""
    mesh = obj.data
    keys = mesh.shape_keys.key_blocks
    source = obj.active_shape_key
    count = len(mesh.vertices)

    def coordinates(key):
        result = np.empty(count * 3, dtype=np.float32)
        key.data.foreach_get('co', result)
        return result.reshape((count, 3))

    masks = {}

    def group_mask(name):
        group = obj.vertex_groups.get(name) if name else None
        if group is None:
            return 1.0
        if group.index not in masks:
            masks[group.index] = np.array([
                next((membership.weight for membership in vertex.groups
                      if membership.group == group.index), 0.0)
                for vertex in mesh.vertices
            ], dtype=np.float32)[:, None]
        return masks[group.index]

    basis = coordinates(keys[0])
    result = basis.copy()
    # Match the other active relative keys, while replacing the selected key.
    for key in list(keys)[1:]:
        if key == source or key.mute or key.value == 0.0:
            continue
        result += (coordinates(key) - coordinates(key.relative_key)) * key.value * group_mask(key.vertex_group)

    if width <= 0.0:
        weights = np.where(basis[:, 0] > center, 1.0,
                           np.where(basis[:, 0] < center, 0.0, 0.5))
    else:
        t = np.clip((basis[:, 0] - center) / width + 0.5, 0.0, 1.0)
        weights = t * t * (3.0 - 2.0 * t)
    if side == 'RIGHT':
        weights = 1.0 - weights
    result += ((coordinates(source) - coordinates(source.relative_key))
               * weights[:, None] * amount * group_mask(source.vertex_group))
    return result


def draw_transition_preview():
    context = bpy.context
    scene = context.scene
    obj = context.active_object
    space = context.space_data
    if (not scene or not space or space.type != 'VIEW_3D'
            or not space.overlay.show_overlays
            or not obj or obj.type != 'MESH' or obj.mode != 'OBJECT'
            or not obj.visible_get()):
        return
    keys = obj.data.shape_keys
    if (not keys or not keys.use_relative or obj.active_shape_key_index == 0
            or not keys.key_blocks[0].data):
        return

    bounds = basis_bounds(keys.key_blocks[0])
    settings = scene.shape_key_lr_splitter
    if not settings.show_preview and settings.preview_side == 'OFF':
        return
    width = (bounds[0][1] - bounds[0][0]) * settings.transition_percent / 100.0
    # Give flat meshes and the box edges enough room to remain visible.
    margin = max(max(high - low for low, high in bounds) * 0.05, 0.001)
    y0, y1 = bounds[1][0] - margin, bounds[1][1] + margin
    z0, z1 = bounds[2][0] - margin, bounds[2][1] + margin

    def rectangle(x):
        return [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)]

    def outline(points):
        return [points[i] for pair in ((0, 1), (1, 2), (2, 3), (3, 0)) for i in pair]

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    boundaries = [rectangle(settings.center - width / 2),
                  rectangle(settings.center + width / 2)]
    center = rectangle(settings.center)
    with gpu.matrix.push_pop():
        gpu.matrix.multiply_matrix(obj.matrix_world)
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        try:
            shader.bind()
            if settings.show_preview and width > 0.0:
                triangles = [point for rect in boundaries
                             for point in (rect[0], rect[1], rect[2], rect[0], rect[2], rect[3])]
                shader.uniform_float('color', (0.05, 0.8, 1.0, 0.07))
                batch_for_shader(shader, 'TRIS', {"pos": triangles}).draw(shader)
                shader.uniform_float('color', (0.05, 0.8, 1.0, 0.8))
                batch_for_shader(shader, 'LINES', {"pos": outline(boundaries[0]) + outline(boundaries[1])}).draw(shader)
            if settings.show_preview:
                shader.uniform_float('color', (1.0, 0.7, 0.1, 1.0))
                batch_for_shader(shader, 'LINES', {"pos": outline(center)}).draw(shader)
            if settings.preview_side != 'OFF':
                mesh = obj.data
                positions = preview_coordinates(obj, settings.preview_side,
                                                settings.preview_amount, settings.center, width)
                mesh.calc_loop_triangles()
                if mesh.loop_triangles:
                    triangles = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
                    mesh.loop_triangles.foreach_get('vertices', triangles)
                    shader.uniform_float('color', (0.8, 0.25, 1.0, 0.12))
                    batch_for_shader(shader, 'TRIS', {"pos": positions},
                                     indices=triangles.reshape((-1, 3))).draw(shader)
                if mesh.edges:
                    edges = np.empty(len(mesh.edges) * 2, dtype=np.int32)
                    mesh.edges.foreach_get('vertices', edges)
                    shader.uniform_float('color', (0.9, 0.4, 1.0, 0.9))
                    batch_for_shader(shader, 'LINES', {"pos": positions},
                                     indices=edges.reshape((-1, 2))).draw(shader)
        finally:
            gpu.state.blend_set('NONE')
            gpu.state.depth_test_set('NONE')


def left_weight(x, center, width):
    """Return the positive-X weight, with a smooth, complementary transition."""
    if width <= 0.0:
        if x == center:
            return 0.5
        return 1.0 if x > center else 0.0
    t = max(0.0, min(1.0, (x - center) / width + 0.5))
    return t * t * (3.0 - 2.0 * t)


class SHAPEKEYSPLIT_PG_settings(PropertyGroup):
    preview_side: EnumProperty(
        name="Side Preview",
        description="Preview one split side as a purple overlay without changing shape keys",
        items=(('OFF', "Off", "Hide the split-side preview"),
               ('LEFT', "L", "Preview the +X side"),
               ('RIGHT', "R", "Preview the -X side")),
        default='OFF',
        update=redraw_viewports,
    )
    preview_amount: FloatProperty(
        name="Preview Strength",
        description="Value applied to the previewed side",
        default=1.0, min=0.0, max=1.0,
        subtype='FACTOR', update=redraw_viewports,
    )
    show_preview: BoolProperty(
        name="Show Transition Preview",
        description="Show the center and transition boundaries in the 3D Viewport",
        default=True,
        update=redraw_viewports,
    )
    transition_percent: FloatProperty(
        name="Transition Width",
        description="Full transition width as a percentage of the Basis mesh's local X width",
        default=20.0,
        min=0.0,
        max=100.0,
        subtype='PERCENTAGE',
        update=redraw_viewports,
    )
    center: FloatProperty(
        name="Center X",
        description="Center of the transition in the object's local X coordinates",
        default=0.0,
        subtype='DISTANCE',
        update=redraw_viewports,
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
        layout.prop(settings, "show_preview", icon='HIDE_OFF')
        layout.prop(settings, "transition_percent", slider=True)
        layout.prop(settings, "center")
        layout.label(text="L: +X   |   R: -X")
        layout.prop(settings, "preview_side", expand=True)
        if settings.preview_side != 'OFF':
            layout.prop(settings, "preview_amount", slider=True)
            layout.label(text="Purple overlay: before modifiers", icon='INFO')
        layout.operator("shape_key.split_left_right", icon='MOD_MIRROR')


classes = (
    SHAPEKEYSPLIT_PG_settings,
    SHAPEKEYSPLIT_OT_split,
    SHAPEKEYSPLIT_PT_panel,
)


def register():
    global _draw_handle
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.shape_key_lr_splitter = PointerProperty(type=SHAPEKEYSPLIT_PG_settings)
    _draw_handle = bpy.types.SpaceView3D.draw_handler_add(draw_transition_preview, (), 'WINDOW', 'POST_VIEW')


def unregister():
    global _draw_handle
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        _draw_handle = None
    del bpy.types.Scene.shape_key_lr_splitter
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


if __name__ == "__main__":
    register()
