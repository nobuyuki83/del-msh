import numpy as np
import bpy

class MeshVtxAnimation:

    def __init__(self, tri2vtx, vtx2xyz):
        bpy.ops.wm.read_factory_settings(use_empty=True) # ensure new scene
        self.scene = bpy.context.scene
        self.scene.render.fps = 30
        self.scene.frame_start = 1

        # build mesh
        mesh = bpy.data.meshes.new("Torus_Mesh")
        mesh.from_pydata(vtx2xyz.tolist(), [], tri2vtx.tolist())
        mesh.validate()
        mesh.update()

        # build object
        self.obj  = bpy.data.objects.new("Torus", mesh)
        bpy.context.scene.collection.objects.link(self.obj)

        # Basis を必ず作る
        if self.obj.data.shape_keys is None:
            self.obj.shape_key_add(name="Basis", from_mix=False)

        sk = self.obj.data.shape_keys                  # bpy.types.Key
        sk.use_relative = False                   # Absolute モード
        # Basis の座標を t=0 に置換
        kb0 = sk.key_blocks[0]                    # Basis
        kb0.data.foreach_set("co", vtx2xyz.astype(np.float32).ravel())
        # Basis の frame は変更不可（read-only）

        # 等間隔にリタイム（Absolute の内部フレームを自動配置）
        # -> オブジェクトをアクティブにして context override で確実に実行
        vl = bpy.context.view_layer
        vl.objects.active = self.obj
        self.obj.select_set(True)
        with bpy.context.temp_override(object=self.obj, active_object=self.obj, selected_objects=[self.obj], view_layer=vl):
            bpy.ops.object.shape_key_retime()

        # eval_time をシーンのフレームにドライブ（これで再生/書き出し時に正しいキーが評価される）
        fcurve = sk.driver_add("eval_time")
        drv = fcurve.driver
        drv.type = 'SCRIPTED'
        var = drv.variables.new(); var.name = "f"; var.type = 'SINGLE_PROP'
        var.targets[0].id_type = 'SCENE'; var.targets[0].id = self.scene
        var.targets[0].data_path = "frame_current"
        drv.expression = "f"

        self.i_frame = 1

    def add_frame(self, vtx2xyz):
        kb = self.obj.shape_key_add(name=f"f{self.i_frame}", from_mix=False)
        self.i_frame += 1
        kb.data.foreach_set("co", vtx2xyz.astype(np.float32).ravel())

    def write(self, output_abc_path):
        self.scene.frame_end = self.i_frame

        # Alembic 書き出し（選択対象にする）
        for o in bpy.context.selected_objects:
            o.select_set(False)
        self.obj.select_set(True)
        bpy.context.view_layer.objects.active = self.obj

        kwargs = dict(
            filepath=bpy.path.abspath(output_abc_path),
            start=self.scene.frame_start,
            end=self.scene.frame_end,
            selected=True,
            uvs=False, normals=True, face_sets=False,
            visible_objects_only=True, renderable_only=True,
            evaluation_mode='RENDER',
        )
        op = bpy.ops.wm.alembic_export
        safe_kwargs = {k:v for k,v in kwargs.items() if k in op.get_rna_type().properties}
        op(**safe_kwargs)



