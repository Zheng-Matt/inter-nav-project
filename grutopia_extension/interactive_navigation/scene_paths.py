"""Scene-directory helpers shared by navigation runners."""

from pathlib import Path


def configure_scene_mdl_paths(scene_asset_path):
    """Register a GRScenes Materials folder so Isaac can resolve MDL assets."""

    if not scene_asset_path or '://' in scene_asset_path:
        return
    scene_directory = Path(scene_asset_path).resolve().parent
    material_directory = scene_directory / 'Materials'
    if not material_directory.exists():
        return
    import carb

    setting = '/app/mdl/additionalUserPaths'
    settings = carb.settings.get_settings()
    paths = list(settings.get(setting) or [])
    material_path = str(material_directory.resolve())
    if material_path not in paths:
        paths.append(material_path)
        settings.set_string_array(setting, paths)
