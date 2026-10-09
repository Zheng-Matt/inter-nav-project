import hashlib,json,sys
from pathlib import Path
from pxr import Gf,Usd,UsdGeom,UsdLux

root=Path(sys.argv[1])
source=Path('grutopia/assets/scenes/GRScenes-100/home_scenes/scenes/MV7J6NIKTKJZ2AABAAAAADA8_usd/start_result_navigation_preview.usda').resolve()
prim_path='/Root/Meshes/Animation/refrigerator/model_8958330cdaa9b7dcda067c99f5916ca3_0'
original=Usd.Stage.Open(str(source),Usd.Stage.LoadNone)
prim=original.GetPrimAtPath(prim_path)
if not prim: raise RuntimeError('Source refrigerator prim not found')
matrix=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
output=root/'minimal-real-fridge.usda'
if output.exists() or (root/'minimal-profile.json').exists():
    raise FileExistsError('Refuse to overwrite an existing smoke scene or profile')
stage=Usd.Stage.CreateNew(str(output))
UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage,.01)
main=UsdGeom.Xform.Define(stage,'/Root')
stage.SetDefaultPrim(main.GetPrim())
target=UsdGeom.Xform.Define(stage,'/Root/refrigerator')
target.GetPrim().GetReferences().AddReference(str(source),prim_path)
target.MakeMatrixXform().Set(matrix)
light=UsdLux.DomeLight.Define(stage,'/Root/light')
light.CreateIntensityAttr(1000)
stage.GetRootLayer().Save()
p=json.loads((root/'near-start-profile.json').read_text())
p.update(name='minimal_real_grscene_fridge',scene_asset_path=str(output.resolve()),static_obstacle_metadata_path=None,static_obstacle_categories=[])
(root/'minimal-profile.json').write_text(json.dumps(p,indent=2)+'\n')
record={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'referenced_prim':prim_path,'output':str(output),'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'transform':[list(row) for row in matrix],'limitations':'Single real scene asset, original geometry/material reference, neutral dome light, existing flat floor. Not full-house navigation.'}
(root/'minimal-scene-provenance.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
