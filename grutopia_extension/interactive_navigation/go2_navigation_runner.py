"""Isaac/GRUtopia runner for Unitree Go2 point navigation."""

import json
import math
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

from grutopia.core.config import Config, SimConfig
from grutopia.core.gym_env import Env
from grutopia.core.runtime import SimulatorRuntime
from grutopia_extension import import_extensions
from grutopia_extension.configs.robots.go2 import (
    DEFAULT_GO2_POLICY_PATH,
    DEFAULT_GO2_USD_PATH,
    GO2_JOINT_NAMES,
    Go2RobotCfg,
    go2_navigation_controller_cfgs,
)
from grutopia_extension.configs.sensors import PhysXLidarCfg
from grutopia_extension.configs.tasks import (
    SingleInferenceEpisodeCfg,
    SingleInferenceTaskCfg,
)
from grutopia_extension.interactive_navigation.scene_paths import (
    configure_scene_mdl_paths,
)
from grutopia_extension.interactive_navigation.mapping_runtime import (
    SemanticDetectionMode,
)
from grutopia_extension.interactive_navigation.navigation_recording import (
    NavigationRecordingSession,
)
from grutopia_extension.interactive_navigation.navigation_sensors import (
    GO2_FIRST_PERSON_OFFSET,
    GO2_FIRST_PERSON_PITCH_DEGREES,
    NavigationSensorRig,
)
from grutopia_extension.interactive_navigation.point_navigation import (
    PointNavigationComponent,
    PointNavigationConfig,
)
from grutopia_extension.interactive_navigation.point_navigation_profiles import (
    PointNavigationSceneProfile,
    load_profile_static_obstacles,
)


@dataclass(frozen=True)
class Go2NavigationRunConfig:
    gpu: int = 0
    headless: bool = True
    max_steps: int = 8000
    mapping_warmup_steps: int = 60
    log_every: int = 100
    record_dir: str = ''
    record_every: int = 20
    video_fps: float = 12.0
    map_output: str = ''
    goal_index: int = 0
    policy_path: str = DEFAULT_GO2_POLICY_PATH
    robot_usd_path: str = DEFAULT_GO2_USD_PATH
    generate_fallback_asset: bool = True
    # World z of the walking surface, fed into the policy's synthetic
    # height scan (e.g. 0.15 for the GRScene collision floor).
    ground_height: float = 0.0
    # Physics steps per render; raising it (e.g. 20 -> 10 Hz rendering at the
    # 200 Hz physics rate) trades camera freshness for large wall-clock savings.
    rendering_interval: int = 4
    # Keep physics state in Fabric instead of writing USD back every step.
    use_fabric: bool = False

    def __post_init__(self):
        if self.gpu < 0:
            raise ValueError('gpu cannot be negative')
        if self.max_steps <= 0 or self.mapping_warmup_steps < 0:
            raise ValueError(
                'max_steps must be positive and warmup cannot be negative'
            )
        if self.log_every <= 0 or self.record_every <= 0 or self.video_fps <= 0:
            raise ValueError('logging and recording intervals must be positive')
        if self.rendering_interval <= 0:
            raise ValueError('rendering_interval must be positive')
        if not self.policy_path:
            raise ValueError('policy_path cannot be empty')
        if not self.robot_usd_path:
            raise ValueError('robot_usd_path cannot be empty')


@dataclass(frozen=True)
class Go2SemanticExplorationRunConfig:
    target_query: str = 'refrigerator'
    gpu: int = 1
    perception_gpu: int = 5
    qwen_device: str = 'cuda:4'
    headless: bool = True
    max_steps: int = 12_000
    mapping_warmup_steps: int = 80
    log_every: int = 100
    record_dir: str = ''
    record_every: int = 20
    video_fps: float = 12.0
    map_output: str = ''
    policy_path: str = DEFAULT_GO2_POLICY_PATH
    robot_usd_path: str = DEFAULT_GO2_USD_PATH
    generate_fallback_asset: bool = True
    ground_height: float = 0.0
    # 'isaac' (ground-truth labels only), 'open_vocab' (VLM detections only),
    # or 'hybrid' (fused). See SemanticDetectionMode.
    semantic_detection_mode: str = 'hybrid'
    enable_qwen: bool = True
    qwen_model: str = 'Qwen/Qwen3-8B'
    qwen_python: str = os.environ.get('QWEN3_PYTHON', 'python'),
    rendering_interval: int = 4
    use_fabric: bool = False

    def __post_init__(self):
        if not self.target_query.strip():
            raise ValueError('target_query cannot be empty')
        if min(self.gpu, self.perception_gpu) < 0:
            raise ValueError('GPU indices cannot be negative')
        if self.max_steps <= 0 or self.mapping_warmup_steps < 0:
            raise ValueError('max_steps must be positive and warmup cannot be negative')
        if self.log_every <= 0 or self.record_every <= 0 or self.video_fps <= 0:
            raise ValueError('logging and recording intervals must be positive')
        if self.rendering_interval <= 0:
            raise ValueError('rendering_interval must be positive')
        object.__setattr__(
            self,
            'semantic_detection_mode',
            SemanticDetectionMode.parse(self.semantic_detection_mode).value,
        )


def build_go2_navigation_config(
    profile: PointNavigationSceneProfile,
    run: Go2NavigationRunConfig,
    objects: Sequence = (),
) -> Config:
    controllers = go2_navigation_controller_cfgs(
        run.policy_path,
        ground_height=getattr(run, 'ground_height', 0.0),
    )
    robot = Go2RobotCfg(
        position=profile.start,
        usd_path=run.robot_usd_path,
        generate_fallback_asset=run.generate_fallback_asset,
        controllers=list(controllers),
        sensors=[
            PhysXLidarCfg(
                name='lidar',
                prim_path='base/MappingLidar',
                translation=(0.24, 0.0, 0.10),
                fov=(360.0, 40.0),
            ),
        ],
    )
    return Config(
        simulator=SimConfig(
            physics_dt=1 / 200,
            # Keep the render step short: Isaac advances a full rendering_dt
            # of open-loop physics substeps whenever it renders, and the Go2
            # policy only tolerates the original 4-substep (1/50 s) window.
            # Render frequency is reduced via rendering_interval alone.
            rendering_dt=1 / 50,
            rendering_interval=run.rendering_interval,
            use_fabric=run.use_fabric,
        ),
        task_config=SingleInferenceTaskCfg(
            episodes=[
                SingleInferenceEpisodeCfg(
                    scene_asset_path=profile.scene_asset_path,
                    scene_scale=profile.scene_scale,
                    scene_position=profile.scene_position,
                    robots=[robot],
                    objects=list(objects),
                )
            ]
        ),
    )


def run_go2_point_navigation(
    profile: PointNavigationSceneProfile,
    run: Go2NavigationRunConfig,
    objects: Sequence = (),
) -> int:
    if run.goal_index < 0 or run.goal_index >= len(profile.goals):
        raise ValueError(f'goal_index {run.goal_index} is outside profile goals')
    runtime = SimulatorRuntime(
        config_class=build_go2_navigation_config(
            profile,
            run,
            objects=objects,
        ),
        headless=run.headless,
        active_gpu=run.gpu,
        physics_gpu=run.gpu,
    )
    configure_scene_mdl_paths(profile.scene_asset_path)
    env = component = recorder = None
    result_code = 2
    try:
        import_extensions(('controllers', 'objects', 'robots', 'sensors', 'tasks'))
        env = Env(runtime)
        robot_observation, _ = env.reset()
        if profile.apply_koostruct_material_fallbacks:
            from omni.isaac.core.utils.stage import get_current_stage

            from grutopia_extension.interactive_navigation.material_fallback import (
                apply_koostruct_material_fallbacks,
            )

            print(
                json.dumps(
                    {
                        'event': 'material_fallback',
                        **apply_koostruct_material_fallbacks(
                            get_current_stage()
                        ),
                    }
                ),
                flush=True,
            )

        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni.isaac.core.utils.semantics import add_update_semantics

        active_task = next(iter(env.runner.current_tasks.values()))
        active_robot = next(iter(active_task.robots.values()))
        add_update_semantics(
            get_prim_at_path(active_robot.config.prim_path),
            'go2',
        )
        component = PointNavigationComponent(
            PointNavigationConfig(
                goal=profile.goals[run.goal_index],
                mapping=profile.mapping,
                max_steps=run.max_steps,
                success_distance=profile.success_distance,
                fall_height=profile.fall_height,
                safe_base_height=profile.safe_base_height,
                replan_interval_steps=profile.replan_interval_steps,
                replan_lookahead_distance=profile.replan_lookahead_distance,
                use_scene_graph=profile.use_scene_graph,
                use_rgb_occupancy=profile.use_rgb_occupancy,
                rgb_clears_free_space=profile.rgb_clears_free_space,
                velocity_control=profile.velocity_control,
                max_forward_speed=profile.max_forward_speed,
                max_lateral_speed=profile.max_lateral_speed,
            )
        )
        component.seed_static_obstacles(load_profile_static_obstacles(profile))
        sensor_rig = NavigationSensorRig(
            offset=GO2_FIRST_PERSON_OFFSET,
            pitch_degrees=GO2_FIRST_PERSON_PITCH_DEGREES,
            horizontal_fov_degrees=96.0,
            rgb_interval=component.config.rgb_interval,
        )
        if run.record_dir:
            recorder = NavigationRecordingSession(
                run.record_dir,
                capture_interval=run.record_every,
                fps=run.video_fps,
                overview_position=profile.overview_position,
                overview_look_at=profile.overview_look_at,
                show_scene_graph=profile.use_scene_graph,
                follow_robot=profile.overview_follow_robot,
                third_person_offset=profile.third_person_offset,
                third_person_pitch_degrees=profile.third_person_pitch_degrees,
                third_person_fov_degrees=profile.third_person_fov_degrees,
            )
            recorder.prime(robot_observation)
        sensor_rig.prime(robot_observation)
        robot_observation, _, _, _, _ = env.step(component.warmup_action())

        for step in range(run.mapping_warmup_steps):
            sensor_rig.update(step, robot_observation)
            component.update(step, robot_observation)
            if recorder is not None:
                recorder.update_pose(robot_observation)
            robot_observation, _, _, _, _ = env.step(
                component.warmup_action()
            )

        for step in range(run.max_steps):
            force_camera = (
                recorder is not None and step % run.record_every == 0
            )
            sensor_rig.update(
                step,
                robot_observation,
                force_capture=force_camera,
            )
            component.update(step, robot_observation)
            if step % run.log_every == 0:
                print(
                    json.dumps(
                        component.progress_event(step, robot_observation)
                    ),
                    flush=True,
                )
            if recorder is not None:
                recorder.capture(step, robot_observation, component)
                recorder.update_pose(robot_observation)
            result = component.evaluate(step, robot_observation)
            if result.terminal:
                print(json.dumps(result.as_event()), flush=True)
                result_code = 0 if result.success else 2
                break
            robot_observation, _, terminated, _, _ = env.step(
                component.action(step, robot_observation)
            )
            if terminated:
                raise RuntimeError(
                    'task terminated before Go2 reached the navigation goal'
                )
    except Exception as error:
        traceback.print_exc()
        print(
            json.dumps(
                {
                    'event': 'navigation_error',
                    'robot': 'go2',
                    'type': type(error).__name__,
                    'message': str(error),
                }
            ),
            flush=True,
        )
    finally:
        if recorder is not None:
            recorder.close()
        if component is not None:
            component.save(run.map_output)
        if env is not None:
            env.close()
        else:
            runtime.simulation_app.close()
    return result_code


def run_go2_semantic_exploration(
    profile: PointNavigationSceneProfile,
    run: Go2SemanticExplorationRunConfig,
    objects: Sequence = (),
) -> int:
    """Explore an unknown map until a text-described semantic target is reached."""

    from grutopia_extension.interactive_navigation.open_vocabulary_perception import (
        OpenVocabularyPerception,
        OpenVocabularyPerceptionConfig,
    )
    from grutopia_extension.interactive_navigation.semantic_exploration import (
        Qwen3Scorer,
        Qwen3WorkerGenerator,
    )
    from grutopia_extension.interactive_navigation.semantic_exploration_component import (
        SemanticExplorationComponent,
        SemanticExplorationConfig,
    )

    runtime = SimulatorRuntime(
        config_class=build_go2_navigation_config(
            profile,
            Go2NavigationRunConfig(
                gpu=run.gpu,
                headless=run.headless,
                max_steps=run.max_steps,
                mapping_warmup_steps=run.mapping_warmup_steps,
                log_every=run.log_every,
                record_dir=run.record_dir,
                record_every=run.record_every,
                video_fps=run.video_fps,
                map_output=run.map_output,
                policy_path=run.policy_path,
                robot_usd_path=run.robot_usd_path,
                generate_fallback_asset=run.generate_fallback_asset,
                ground_height=run.ground_height,
                rendering_interval=run.rendering_interval,
                use_fabric=run.use_fabric,
            ),
            objects=objects,
        ),
        headless=run.headless,
        active_gpu=run.gpu,
        physics_gpu=run.gpu,
    )
    configure_scene_mdl_paths(profile.scene_asset_path)
    env = component = recorder = qwen_worker = None
    result_code = 2
    try:
        import_extensions(('controllers', 'objects', 'robots', 'sensors', 'tasks'))
        env = Env(runtime)
        robot_observation, _ = env.reset()
        if profile.apply_koostruct_material_fallbacks:
            from omni.isaac.core.utils.stage import get_current_stage

            from grutopia_extension.interactive_navigation.material_fallback import (
                apply_koostruct_material_fallbacks,
            )

            print(
                json.dumps(
                    {
                        'event': 'material_fallback',
                        **apply_koostruct_material_fallbacks(get_current_stage()),
                    }
                ),
                flush=True,
            )
        active_task = next(iter(env.runner.current_tasks.values()))
        active_robot = next(iter(active_task.robots.values()))
        _label_go2_and_household_semantics(profile, active_robot.config.prim_path)
        _apply_high_friction_material('/World/env_0/objects/grscene_go2_floor')

        detection_mode = SemanticDetectionMode.parse(run.semantic_detection_mode)
        perception = None
        if detection_mode.uses_open_vocabulary:
            perception = OpenVocabularyPerception(
                OpenVocabularyPerceptionConfig(
                    clip_device=f'cuda:{run.perception_gpu}',
                    query_interval=0.25,
                    request_timeout=8.0,
                )
            )
        scorer = None
        if run.enable_qwen:
            worker_script = (
                Path(__file__).resolve().parents[2]
                / 'grutopia'
                / 'demo'
                / 'qwen3_topology_worker.py'
            )
            qwen_worker = Qwen3WorkerGenerator(
                python_executable=run.qwen_python,
                worker_script=str(worker_script),
                model_name=run.qwen_model,
                device=run.qwen_device,
                startup_timeout=360.0,
            )
            try:
                qwen_worker.start()
                scorer = Qwen3Scorer(
                    generator=qwen_worker,
                    timeout_seconds=20.0,
                )
            except Exception as error:
                print(
                    json.dumps(
                        {
                            'event': 'qwen3_fallback',
                            'type': type(error).__name__,
                            'message': str(error),
                        }
                    ),
                    flush=True,
                )
                qwen_worker.close()
                qwen_worker = None
        component = SemanticExplorationComponent(
            SemanticExplorationConfig(
                target_query=run.target_query,
                mapping=profile.mapping,
                max_steps=run.max_steps,
                target_distance=profile.success_distance,
                fall_height=profile.fall_height,
                safe_base_height=profile.safe_base_height,
                max_forward_speed=profile.max_forward_speed,
                max_lateral_speed=profile.max_lateral_speed,
                semantic_detection_mode=detection_mode.value,
            ),
            perception=perception,
            scorer=scorer,
        )
        # Boxes are kept for collision statistics only: the exploration map
        # must start empty and grow purely from online sensing.
        component.mapping.seed_static_obstacles(
            load_profile_static_obstacles(profile),
            mark_occupancy=False,
        )
        sensor_rig = NavigationSensorRig(
            offset=GO2_FIRST_PERSON_OFFSET,
            pitch_degrees=GO2_FIRST_PERSON_PITCH_DEGREES,
            horizontal_fov_degrees=96.0,
            rgb_interval=component.mapping.rgb_interval,
        )
        if run.record_dir:
            recorder = NavigationRecordingSession(
                run.record_dir,
                capture_interval=run.record_every,
                fps=run.video_fps,
                overview_position=profile.overview_position,
                overview_look_at=profile.overview_look_at,
                show_scene_graph=True,
                follow_robot=profile.overview_follow_robot,
                third_person_offset=profile.third_person_offset,
                third_person_pitch_degrees=profile.third_person_pitch_degrees,
                third_person_fov_degrees=profile.third_person_fov_degrees,
            )
            recorder.prime(robot_observation)
        sensor_rig.prime(robot_observation)
        robot_observation, _, _, _, _ = env.step(component.warmup_action())

        # Standby phase: the robot stands still and the map stays empty so the
        # recording clearly shows that mapping starts only with the task.
        for step in range(run.mapping_warmup_steps):
            sensor_rig.update(step, robot_observation)
            if recorder is not None:
                recorder.capture(
                    step - run.mapping_warmup_steps,
                    robot_observation,
                    component,
                    state='standby_empty_map',
                )
                recorder.update_pose(robot_observation)
            robot_observation, _, _, _, _ = env.step(component.warmup_action())

        timing = None
        if os.environ.get('GO2_STEP_TIMING'):
            timing = {
                'sensor': 0.0,
                'mapping': 0.0,
                'recording': 0.0,
                'evaluate': 0.0,
                'action': 0.0,
                'env_step': 0.0,
                'steps': 0,
            }
        for step in range(run.max_steps):
            t0 = time.perf_counter() if timing is not None else 0.0
            sensor_rig.update(
                step,
                robot_observation,
                force_capture=recorder is not None and step % run.record_every == 0,
            )
            t1 = time.perf_counter() if timing is not None else 0.0
            component.update(step, robot_observation)
            t2 = time.perf_counter() if timing is not None else 0.0
            if step % run.log_every == 0:
                print(
                    json.dumps(component.progress_event(step, robot_observation)),
                    flush=True,
                )
            if recorder is not None:
                recorder.capture(
                    step,
                    robot_observation,
                    component,
                    state=component.state,
                )
                recorder.update_pose(robot_observation)
            t3 = time.perf_counter() if timing is not None else 0.0
            result = component.evaluate(step, robot_observation)
            if result.terminal:
                print(json.dumps(result.as_event()), flush=True)
                result_code = 0 if result.success else 2
                break
            t4 = time.perf_counter() if timing is not None else 0.0
            action = component.action(step, robot_observation)
            t5 = time.perf_counter() if timing is not None else 0.0
            robot_observation, _, terminated, _, _ = env.step(action)
            if timing is not None:
                t6 = time.perf_counter()
                timing['sensor'] += t1 - t0
                timing['mapping'] += t2 - t1
                timing['recording'] += t3 - t2
                timing['evaluate'] += t4 - t3
                timing['action'] += t5 - t4
                timing['env_step'] += t6 - t5
                timing['steps'] += 1
                if timing['steps'] % 100 == 0:
                    print(json.dumps({'event': 'step_timing', **{k: round(v, 2) for k, v in timing.items()}}), flush=True)
            if terminated:
                raise RuntimeError('task terminated during Go2 semantic exploration')
    except Exception as error:
        traceback.print_exc()
        print(
            json.dumps(
                {
                    'event': 'semantic_exploration_error',
                    'robot': 'go2',
                    'type': type(error).__name__,
                    'message': str(error),
                }
            ),
            flush=True,
        )
    finally:
        if recorder is not None:
            recorder.close()
        if component is not None:
            component.save(run.map_output)
        if qwen_worker is not None:
            qwen_worker.close()
        if env is not None:
            env.close()
        else:
            runtime.simulation_app.close()
    return result_code


def run_go2_constant_walk(
    profile: PointNavigationSceneProfile,
    run: Go2NavigationRunConfig,
    objects: Sequence = (),
    command: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    heading_hold: bool = False,
    heading_gain: float = 0.5,
) -> int:
    """Drive Go2 with a fixed body-frame velocity command and record gait.

    With ``heading_hold`` the yaw channel closes a proportional loop on the
    walk-start heading (Isaac Lab's heading-command deployment mode), so the
    velocity command tests straight-line tracking instead of accumulating the
    policy's open-loop yaw drift.
    """

    if len(command) != 3:
        raise ValueError('walk command must be (forward, lateral, yaw)')
    if heading_gain <= 0:
        raise ValueError('heading_gain must be positive')
    walk_action = {
        'move_by_speed': [float(command[0]), float(command[1]), float(command[2])],
    }
    stand_action = {'move_by_speed': [0.0, 0.0, 0.0]}
    runtime = SimulatorRuntime(
        config_class=build_go2_navigation_config(
            profile,
            run,
            objects=objects,
        ),
        headless=run.headless,
        active_gpu=run.gpu,
        physics_gpu=run.gpu,
    )
    configure_scene_mdl_paths(profile.scene_asset_path)
    env = component = recorder = None
    result_code = 2
    try:
        import_extensions(('controllers', 'objects', 'robots', 'sensors', 'tasks'))
        env = Env(runtime)
        robot_observation, _ = env.reset()

        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni.isaac.core.utils.semantics import add_update_semantics

        active_task = next(iter(env.runner.current_tasks.values()))
        active_robot = next(iter(active_task.robots.values()))
        add_update_semantics(
            get_prim_at_path(active_robot.config.prim_path),
            'go2',
        )
        # The helper is a no-op for missing prims, so both the programmatic
        # walk floor and the GRScene collision floor can be covered here.
        _apply_high_friction_material('/World/env_0/objects/walk_floor')
        _apply_high_friction_material('/World/env_0/objects/grscene_go2_floor')
        component = PointNavigationComponent(
            PointNavigationConfig(
                goal=profile.goals[0],
                mapping=profile.mapping,
                max_steps=run.max_steps,
                success_distance=profile.success_distance,
                fall_height=profile.fall_height,
                safe_base_height=profile.safe_base_height,
            )
        )
        sensor_rig = NavigationSensorRig(
            offset=GO2_FIRST_PERSON_OFFSET,
            pitch_degrees=GO2_FIRST_PERSON_PITCH_DEGREES,
            horizontal_fov_degrees=96.0,
            rgb_interval=component.config.rgb_interval,
        )
        if run.record_dir:
            recorder = NavigationRecordingSession(
                run.record_dir,
                capture_interval=run.record_every,
                fps=run.video_fps,
                overview_position=profile.overview_position,
                overview_look_at=profile.overview_look_at,
                show_scene_graph=False,
                follow_robot=True,
                third_person_offset=profile.third_person_offset,
                third_person_pitch_degrees=profile.third_person_pitch_degrees,
                third_person_fov_degrees=profile.third_person_fov_degrees,
            )
            recorder.prime(robot_observation)
        sensor_rig.prime(robot_observation)
        previous_position = [
            float(value) for value in robot_observation['position']
        ]
        robot_observation, _, _, _, _ = env.step(stand_action)

        for step in range(run.mapping_warmup_steps):
            sensor_rig.update(step, robot_observation)
            component.update(step, robot_observation)
            if recorder is not None:
                recorder.update_pose(robot_observation)
            previous_position = [
                float(value) for value in robot_observation['position']
            ]
            robot_observation, _, _, _, _ = env.step(stand_action)

        start_position = [float(value) for value in robot_observation['position']]
        walk_heights = []
        walk_track = []
        reference_yaw = float(
            robot_observation.get('controllers', {})
            .get('move_by_speed', {})
            .get('yaw', 0.0)
        )
        print(
            json.dumps(
                {
                    'event': 'walk_start',
                    'command': list(walk_action['move_by_speed']),
                    'warmup_steps': run.mapping_warmup_steps,
                    'position': start_position,
                    'heading_hold': bool(heading_hold),
                    'reference_yaw': reference_yaw,
                    'joint_names': list(GO2_JOINT_NAMES),
                }
            ),
            flush=True,
        )
        for step in range(run.max_steps):
            force_camera = (
                recorder is not None and step % run.record_every == 0
            )
            sensor_rig.update(
                step,
                robot_observation,
                force_capture=force_camera,
            )
            component.update(step, robot_observation)
            position = [float(value) for value in robot_observation['position']]
            walk_heights.append(position[2])
            controller_obs = robot_observation.get('controllers', {}).get(
                'move_by_speed',
                {},
            )
            current_yaw = float(controller_obs.get('yaw', 0.0))
            walk_track.append(
                {
                    'step': step,
                    'position': position,
                    'yaw': current_yaw,
                }
            )
            if heading_hold:
                heading_error = math.atan2(
                    math.sin(reference_yaw - current_yaw),
                    math.cos(reference_yaw - current_yaw),
                )
                walk_action['move_by_speed'][2] = float(
                    command[2]
                    + max(-0.5, min(0.5, heading_gain * heading_error))
                )
            if step % run.log_every == 0:
                dt = run.log_every / 200.0
                measured_speed = [
                    (position[index] - previous_position[index]) / dt
                    for index in range(3)
                ]
                print(
                    json.dumps(
                        {
                            'event': 'walk_progress',
                            'step': step,
                            'command': list(walk_action['move_by_speed']),
                            'position': position,
                            'measured_world_speed': measured_speed,
                            'base_height': position[2],
                            'yaw': float(controller_obs.get('yaw', 0.0)),
                            'body_lin_vel': [
                                float(value)
                                for value in controller_obs.get(
                                    'body_lin_vel',
                                    (0.0, 0.0, 0.0),
                                )
                            ],
                            'body_ang_vel': [
                                float(value)
                                for value in controller_obs.get(
                                    'body_ang_vel',
                                    (0.0, 0.0, 0.0),
                                )
                            ],
                        }
                    ),
                    flush=True,
                )
                previous_position = position
            if recorder is not None:
                recorder.capture(
                    step,
                    robot_observation,
                    component,
                    state='constant_walk',
                )
                recorder.update_pose(robot_observation)
            if position[2] < profile.fall_height:
                print(
                    json.dumps(
                        {
                            'event': 'walk_result',
                            'success': False,
                            'reason': 'fallen',
                            'step': step,
                            'position': position,
                            'command': list(walk_action['move_by_speed']),
                        }
                    ),
                    flush=True,
                )
                result_code = 2
                break
            robot_observation, _, terminated, _, _ = env.step(walk_action)
            if terminated:
                raise RuntimeError('task terminated during the constant-speed walk')
        else:
            position = [float(value) for value in robot_observation['position']]
            metrics = _walk_metrics(start_position, position, walk_heights, walk_track)
            success = (
                metrics['forward_distance'] >= 5.0
                and metrics['lateral_drift'] <= 1.20
                and metrics['mean_height'] >= 0.28
            )
            print(
                json.dumps(
                    {
                        'event': 'walk_result',
                        'success': success,
                        'reason': 'finished' if success else 'unstable_gait',
                        'step': run.max_steps - 1,
                        'position': position,
                        'command': list(walk_action['move_by_speed']),
                        **metrics,
                    }
                ),
                flush=True,
            )
            result_code = 0 if success else 2
        if run.record_dir and walk_track:
            trajectory_path = Path(run.record_dir) / 'walk_trajectory.json'
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        'command': list(walk_action['move_by_speed']),
                        'start_position': start_position,
                        'track': walk_track,
                    }
                ),
                encoding='utf-8',
            )
    except Exception as error:
        traceback.print_exc()
        print(
            json.dumps(
                {
                    'event': 'walk_error',
                    'type': type(error).__name__,
                    'message': str(error),
                }
            ),
            flush=True,
        )
    finally:
        if recorder is not None:
            recorder.close()
        if component is not None:
            component.save(run.map_output)
        if env is not None:
            env.close()
        else:
            runtime.simulation_app.close()
    return result_code


_HOUSEHOLD_SEMANTIC_LABELS = ('refrigerator', 'chair', 'plant')


def _label_go2_and_household_semantics(profile, robot_prim_path: str):
    from omni.isaac.core.utils.prims import get_prim_at_path
    from omni.isaac.core.utils.semantics import add_update_semantics
    from omni.isaac.core.utils.stage import get_current_stage

    add_update_semantics(get_prim_at_path(robot_prim_path), 'go2')
    scene_path = profile.scene_asset_path or ''
    if 'GRScenes' not in scene_path:
        return
    labeled = 0
    for prim in get_current_stage().Traverse():
        path = str(prim.GetPath()).casefold()
        for label in _HOUSEHOLD_SEMANTIC_LABELS:
            if f'/{label}' in path or path.endswith(label):
                add_update_semantics(prim, label)
                labeled += 1
                break
    print(
        json.dumps({'event': 'grscene_semantic_labels', 'labeled_prims': labeled}),
        flush=True,
    )


def _walk_metrics(start_position, end_position, heights, track=()):
    forward = float(end_position[0] - start_position[0])
    lateral = abs(float(end_position[1] - start_position[1]))
    metrics = {
        'forward_distance': forward,
        'lateral_drift': lateral,
        'mean_height': float(sum(heights) / len(heights)) if heights else 0.0,
        'min_height': float(min(heights)) if heights else 0.0,
    }
    if track:
        start_y = float(start_position[1])
        offsets = [float(point['position'][1]) - start_y for point in track]
        metrics['max_lateral_deviation'] = max(abs(offset) for offset in offsets)
        metrics['lateral_rms'] = float(
            math.sqrt(sum(offset * offset for offset in offsets) / len(offsets))
        )
        yaws = [float(point['yaw']) for point in track]
        metrics['yaw_range'] = float(max(yaws) - min(yaws))
    return metrics


def _apply_high_friction_material(prim_path: str) -> None:
    from omni.isaac.core.materials import PhysicsMaterial
    from omni.isaac.core.prims import GeometryPrim
    from omni.isaac.core.utils.prims import is_prim_path_valid

    if not is_prim_path_valid(prim_path):
        return
    material = PhysicsMaterial(
        prim_path='/World/PhysicsMaterials/go2_walk_floor',
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )
    GeometryPrim(prim_path=prim_path).apply_physics_material(material)
