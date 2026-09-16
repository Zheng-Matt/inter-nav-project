"""Runtime glue from GRUtopia sensor observations to the fused map."""

import json
from dataclasses import asdict
from math import atan2, cos, pi, sin
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from grutopia_extension.interactive_navigation.mapping import FusedMap, MappingConfig, PlanningError, SemanticDetection
from grutopia_extension.interactive_navigation.state_machine import (
    ControllerCommand,
    InteractionState,
    StateMachineDecision,
)

_NON_OBJECT_SEMANTIC_LABELS = {
    'ceiling',
    'floor',
    'g1',
    'go2',
    'ground',
    'other',
    'robot',
    'unitree_go2',
    'wall',
}


class MapNavigationRuntime:
    """Fuse live sensors and replace path commands with A* map plans."""

    def __init__(
        self,
        mapping_config: MappingConfig = MappingConfig(),
        lidar_interval: int = 8,
        rgb_interval: int = 24,
        use_planner: bool = True,
        use_scene_graph: bool = True,
        safe_base_height: float = 0.95,
        replan_interval_steps: Optional[int] = None,
        allow_goal_door_traversal: bool = True,
        use_semantic_occupancy: bool = True,
        use_rgb_occupancy: bool = True,
        rgb_clears_free_space: bool = True,
        replan_only_if_blocked: bool = False,
        replan_lookahead_distance: Optional[float] = None,
        semantic_target: str = '',
        open_vocabulary_perception=None,
        use_semantic_voronoi: bool = False,
        semantic_voronoi_config=None,
        topology_update_interval: int = 40,
        prefer_voronoi_paths: bool = False,
    ):
        if lidar_interval <= 0 or rgb_interval <= 0:
            raise ValueError('sensor update intervals must be positive')
        if replan_lookahead_distance is not None and replan_lookahead_distance <= 0:
            raise ValueError('replan_lookahead_distance must be positive when provided')
        if topology_update_interval <= 0:
            raise ValueError('topology_update_interval must be positive')
        self.map = FusedMap(mapping_config)
        self.lidar_interval = lidar_interval
        self.rgb_interval = rgb_interval
        self.use_planner = use_planner
        self.use_scene_graph = use_scene_graph
        self.safe_base_height = safe_base_height
        self.replan_interval_steps = replan_interval_steps
        self.allow_goal_door_traversal = allow_goal_door_traversal
        self.use_semantic_occupancy = use_semantic_occupancy
        self.use_rgb_occupancy = use_rgb_occupancy
        self.rgb_clears_free_space = rgb_clears_free_space
        self.replan_only_if_blocked = replan_only_if_blocked
        self.replan_lookahead_distance = replan_lookahead_distance
        self.semantic_target = str(semantic_target).strip()
        self.open_vocabulary_perception = open_vocabulary_perception
        self.open_vocabulary_frames = 0
        self.open_vocabulary_failures = 0
        self.topology_update_interval = int(topology_update_interval)
        self.semantic_voronoi = None
        self.voronoi_planner = None
        self.voronoi_plan_count = 0
        self.voronoi_plan_fallbacks = 0
        if use_semantic_voronoi:
            from grutopia_extension.interactive_navigation.semantic_voronoi import (
                SemanticVoronoiGraph,
                VoronoiPathPlanner,
            )

            self.semantic_voronoi = SemanticVoronoiGraph(
                self.map.occupancy,
                semantic_voronoi_config,
            )
            if prefer_voronoi_paths:
                # Waypoints are sampled directly from the skeleton path at a
                # short spacing, so the robot walks along the max-clearance
                # medial axis instead of driving straight chords towards the
                # goal. Line-of-sight cuts only check the inflated mask,
                # which under-protects thin obstacles such as standing
                # people (observed 0.26 m near-miss with a 2.5 m cap).
                self.voronoi_planner = VoronoiPathPlanner(
                    self.semantic_voronoi,
                    waypoint_spacing=0.30,
                )
        self._last_lidar_physics_step = -1
        self._planned_state: Optional[InteractionState] = None
        self._planned_path = None
        self._waypoint_index = 0
        self._planned_paths = {}
        self._plan_history = []
        self.planning_failures = 0
        self.replan_count = 0
        self._progress_position = None
        self._progress_step = None
        self._last_plan_step = None
        self._last_safe_position = None
        self._static_obstacle_boxes = []
        self._static_obstacle_violations = 0
        self._violated_static_labels = set()
        self.navigation_headings = {
            InteractionState.NAVIGATE_TO_OBSTACLE: 0.0,
            InteractionState.NAVIGATE_TO_OBJECT: 0.0,
            InteractionState.NAVIGATE_TO_DOOR: 0.0,
        }
        self.final_approach_distance = 0.45
        self.intermediate_waypoint_tolerance = 0.28
        self.calibrated_waypoint_tolerance = 0.08
        self.object_heading_alignment_distance = 0.03
        # Pure-pursuit tracking of planner paths: the carrot slides along the
        # dense skeleton waypoints so velocity commands stay continuous.
        self.velocity_lookahead_distance = 0.55
        self.final_goal_slow_radius = 0.50
        # Blocked-path checks run every step (bounded by this cooldown) so a
        # freshly mapped obstacle triggers a replan within a few steps.
        self.blocked_replan_cooldown_steps = 10
        self._last_blocked_replan_step = None
        # Standing person footprint (0.6 x 0.37 m body circumscribed) stamped
        # into occupancy from camera scene-graph nodes.
        self.person_obstacle_radius = 0.35
        self.person_obstacle_min_observations = 10

    def seed_static_obstacles(self, obstacles: Iterable[dict], mark_occupancy: bool = True):
        """Install immutable metadata obstacles used by the point-navigation stack.

        With ``mark_occupancy=False`` the boxes are only used for collision
        statistics, keeping the online map free of prior knowledge (as needed
        by unknown-environment exploration runs).
        """

        known_labels = {label for label, _, _ in self._static_obstacle_boxes}
        for obstacle in obstacles:
            label = str(obstacle['label'])
            if label in known_labels:
                continue
            minimum = np.asarray(obstacle['minimum_xy'], dtype=np.float64)
            maximum = np.asarray(obstacle['maximum_xy'], dtype=np.float64)
            if mark_occupancy:
                self.map.occupancy.mark_box_occupied(minimum, maximum)
            self._static_obstacle_boxes.append((label, minimum, maximum))
            known_labels.add(label)

    def update(self, step: int, robot_observation: dict):
        if step % self.topology_update_interval == 0:
            # Low-frequency maintenance before new evidence arrives: cells that
            # stayed weak and isolated through the whole interval are noise,
            # while anything hit repeatedly has passed the confirmation level.
            self.map.occupancy.prune_isolated_occupancy()
        sensors = robot_observation.get('sensors', {})
        lidar = sensors.get('lidar', {})
        if step % self.lidar_interval == 0:
            physics_step = int(lidar.get('physics_step', -1))
            points = _points(lidar.get('pointcloud'))
            if physics_step != self._last_lidar_physics_step and len(points) > 0:
                origin = _vector3(lidar.get('position'))
                self.map.update_lidar(
                    origin=origin,
                    points=points,
                    step=step,
                    max_range=float(lidar.get('max_range', 8.0)),
                )
                self._last_lidar_physics_step = physics_step

        camera = sensors.get('camera', {})
        if step % self.rgb_interval == 0:
            points, colors, point_image = _camera_cloud(camera)
            if len(points) > 0:
                detections = ()
                if self.use_scene_graph:
                    detections = self._camera_semantic_detections(
                        camera,
                        point_image,
                        step,
                    )
                camera_position = camera.get('position')
                origin = None if camera_position is None else _vector3(camera_position)
                self.map.update_rgb(
                    points,
                    colors,
                    detections,
                    step=step,
                    origin=origin,
                    update_occupancy=self.use_rgb_occupancy,
                    update_semantic_occupancy=self.use_semantic_occupancy,
                    clear_free_space=self.rgb_clears_free_space,
                )

        position = robot_observation.get('position')
        if position is not None:
            position = np.asarray(position, dtype=np.float64)
            self._update_static_obstacle_violations(position)
            self.map.occupancy.mark_free(position[:2], self.map.config.robot_radius)
            cell = self.map.occupancy.world_to_cell(position[:2])
            if (
                position[2] >= self.safe_base_height
                and self.map.config.safe_recovery_y_limits[0]
                <= position[1]
                <= self.map.config.safe_recovery_y_limits[1]
                and cell is not None
                and self.map.occupancy.observed[cell]
                and not self.map.occupancy.inflated_mask()[cell]
            ):
                self._last_safe_position = tuple(float(value) for value in position[:3])
        if step % self.topology_update_interval == 0:
            self._mark_person_obstacles()
        if self.semantic_voronoi is not None and step % self.topology_update_interval == 0:
            self.semantic_voronoi.update()
            self.semantic_voronoi.attach_semantics(
                self.map.scene_graph.object_nodes()
            )

    def _mark_person_obstacles(self):
        """Stamp well-observed person nodes into the occupancy grid.

        People are thin and often enter the lidar occupancy layer only at
        very close range, while the camera scene graph localizes them within
        centimetres long before that. Marking them explicitly lets the
        planner keep clearance from the start instead of replanning at the
        last moment. The observation threshold skips depth-median ghosts.
        """

        if not self.use_scene_graph:
            return
        for node in self.map.scene_graph.object_nodes():
            if node.label == 'person' and node.observations >= self.person_obstacle_min_observations:
                self.map.occupancy.mark_occupied(
                    node.position[:2],
                    self.person_obstacle_radius,
                )

    def _camera_semantic_detections(self, camera: dict, point_image, step: int):
        fallback = tuple(_semantic_detections(camera, point_image))
        if self.open_vocabulary_perception is None or not self.semantic_target:
            return fallback
        try:
            detections = self.open_vocabulary_perception.perceive(
                rgba=camera.get('rgba'),
                depth=camera.get('depth'),
                point_image=point_image,
                target=self.semantic_target,
                step=step,
            )
            semantic_detections = []
            for detection in detections:
                if isinstance(detection, SemanticDetection):
                    semantic_detections.append(detection)
                elif hasattr(detection, 'to_semantic_detection'):
                    semantic_detections.append(detection.to_semantic_detection())
            self.open_vocabulary_frames += 1
            # Open-vocabulary detections enrich the camera's ground-truth
            # semantics instead of replacing them: dropping the fallback hid
            # target classes (e.g. refrigerator) from the scene graph whenever
            # GroundingDINO returned any context detection, so lexical target
            # matching could never fire.
            return _deduplicate_semantic_detections(
                tuple(semantic_detections) + fallback
            )
        except Exception:
            self.open_vocabulary_failures += 1
            return fallback

    def action_for(
        self,
        decision: StateMachineDecision,
        robot_observation: dict,
        step: Optional[int] = None,
    ) -> dict:
        action = decision.as_action()
        command = decision.command
        if not self.use_planner:
            return action
        if (
            command is not None
            and decision.state == InteractionState.PUSH_OBSTACLE
            and command.name == 'move_by_speed'
        ):
            return self._aligned_push_action(command, robot_observation)
        if (
            command is not None
            and decision.state == InteractionState.RECOVER
            and command.name == 'recover'
            and self._last_safe_position is not None
            and self._planned_state is not None
        ):
            recover_height = float(command.data[0][2])
            recovery_position = self._last_safe_position
            if (
                self._planned_state == InteractionState.NAVIGATE_TO_GOAL
                and 'door' in self.map.traversable_semantic_labels
                and self._planned_path
            ):
                portal = self._planned_path[0]
                if self._last_safe_position[0] <= portal[0] + 0.20:
                    recovery_position = portal
            recover_target = (
                recovery_position[0],
                recovery_position[1],
                recover_height,
            )
            self._planned_state = None
            self._planned_path = None
            self._waypoint_index = 0
            return ControllerCommand(command.name, (recover_target, command.data[1])).as_action()
        if decision.state == InteractionState.PUSH_DOOR and 'move_by_speed' in action:
            speed_command = action['move_by_speed']
            action['move_by_speed'] = self._aligned_speed_data(
                forward_speed=float(speed_command[0]),
                orientation=robot_observation['orientation'],
            )
            return action
        if command is None or command.name != 'move_along_path':
            self._planned_state = None
            self._planned_path = None
            self._waypoint_index = 0
            self._progress_position = None
            self._progress_step = None
            self._last_plan_step = None
            return action

        if self.map.lidar_frames == 0:
            return ControllerCommand('move_by_speed', (0.0, 0.0, 0.0)).as_action()

        start = _vector3(robot_observation['position'])
        configured_path = command.data[0]
        if (
            decision.state
            in (
                InteractionState.NAVIGATE_TO_OBJECT,
                InteractionState.NAVIGATE_TO_CARRY_GOAL,
            )
            and len(configured_path) > 1
        ):
            return self._calibrated_path_action(
                decision.state,
                command.name,
                configured_path,
                start,
                step,
            )
        goal = tuple(float(value) for value in configured_path[-1])
        if (
            decision.state == InteractionState.NAVIGATE_TO_OBJECT
            and np.linalg.norm(np.asarray(start[:2]) - np.asarray(goal[:2]))
            <= self.object_heading_alignment_distance
            and abs(_yaw(robot_observation['orientation'])) > 0.12
        ):
            return ControllerCommand(
                'move_by_speed',
                tuple(self._aligned_speed_data(0.0, robot_observation['orientation'])),
            ).as_action()
        if self._planned_state != decision.state:
            self._planned_path = None
            self._progress_position = np.asarray(start[:2])
            self._progress_step = step
            self._last_plan_step = None
        elif (
            step is not None
            and self.replan_only_if_blocked
            and self._planned_path is not None
            and (
                self._last_blocked_replan_step is None
                or step - self._last_blocked_replan_step >= self.blocked_replan_cooldown_steps
            )
            and self._remaining_path_blocked(start)
        ):
            # Obstacles confirmed into the map at close range (e.g. a person
            # the lidar only resolves late) must interrupt the plan at once:
            # waiting for the periodic replan timer gave a blind window of
            # ~0.5 m at walking speed and led to a near-collision pass.
            self._last_blocked_replan_step = step
            self._planned_path = None
            self._waypoint_index = 0
            self.replan_count += 1
        elif (
            step is not None
            and self.replan_interval_steps is not None
            and self._last_plan_step is not None
            and step - self._last_plan_step >= self.replan_interval_steps
        ):
            if self.replan_only_if_blocked and not self._remaining_path_blocked(start):
                self._last_plan_step = step
            else:
                self._planned_path = None
                self._waypoint_index = 0
                self.replan_count += 1
        elif step is not None and self._progress_step is not None and step - self._progress_step >= 240:
            current_position = np.asarray(start[:2])
            if np.linalg.norm(current_position - self._progress_position) < 0.12:
                self._planned_path = None
                self._waypoint_index = 0
                self.replan_count += 1
            self._progress_position = current_position
            self._progress_step = step

        if self._planned_state != decision.state or self._planned_path is None:
            if (
                decision.state == InteractionState.NAVIGATE_TO_GOAL
                and self.use_scene_graph
                and self.allow_goal_door_traversal
            ):
                self.map.set_semantic_labels_traversable(
                    {
                        'door',
                        'door_frame_hinge',
                        'door_frame_latch',
                    }
                )
            try:
                self._planned_path = self._plan_with_final_heading(decision.state, start, goal)
            except PlanningError:
                self.planning_failures += 1
                return ControllerCommand('move_by_speed', (0.0, 0.0, 0.0)).as_action()
            self._planned_state = decision.state
            self._waypoint_index = 0
            self._last_plan_step = step
            self._planned_paths[decision.state.value] = self._planned_path
            self._plan_history.append(
                {
                    'step': step,
                    'state': decision.state.value,
                    'path': [list(point) for point in self._planned_path],
                }
            )

        while self._waypoint_index < len(self._planned_path) - 1:
            waypoint = np.asarray(self._planned_path[self._waypoint_index][:2], dtype=np.float64)
            if np.linalg.norm(waypoint - np.asarray(start[:2])) >= self.intermediate_waypoint_tolerance:
                break
            self._waypoint_index += 1
        action[command.name] = [self._planned_path[self._waypoint_index :]]
        return action

    def _calibrated_path_action(self, state, command_name, configured_path, start, step):
        path = tuple(
            tuple(float(value) for value in waypoint)
            for waypoint in configured_path
        )
        if self._planned_state != state or self._planned_path != path:
            self._planned_state = state
            self._planned_path = path
            self._waypoint_index = 0
            self._last_plan_step = step
            self._planned_paths[state.value] = path
            self._plan_history.append(
                {
                    'step': step,
                    'state': state.value,
                    'source': 'calibrated',
                    'path': [list(point) for point in path],
                }
            )
            self.map.plan_count += 1
        while self._waypoint_index < len(path) - 1:
            waypoint = np.asarray(path[self._waypoint_index][:2], dtype=np.float64)
            if (
                np.linalg.norm(waypoint - np.asarray(start[:2]))
                >= self.calibrated_waypoint_tolerance
            ):
                break
            self._waypoint_index += 1
        return ControllerCommand(
            command_name,
            (path[self._waypoint_index :],),
        ).as_action()

    def _remaining_path_blocked(self, start) -> bool:
        if not self._planned_path:
            return True
        blocked = self.map.occupancy.inflated_mask()
        points = [np.asarray(start[:2], dtype=np.float64)]
        points.extend(
            np.asarray(point[:2], dtype=np.float64)
            for point in self._planned_path[self._waypoint_index :]
        )
        spacing = self.map.config.grid_resolution * 0.5
        remaining = self.replan_lookahead_distance
        for left, right in zip(points, points[1:]):
            distance = float(np.linalg.norm(right - left))
            checked_distance = distance if remaining is None else min(distance, remaining)
            if checked_distance <= 0:
                return False
            endpoint = left if distance == 0 else left + (right - left) * (checked_distance / distance)
            samples = max(1, int(np.ceil(checked_distance / spacing)))
            for ratio in np.linspace(0.0, 1.0, samples + 1)[1:]:
                cell = self.map.occupancy.world_to_cell(left + (endpoint - left) * ratio)
                if cell is None or blocked[cell]:
                    return True
            if remaining is not None:
                remaining -= checked_distance
                if remaining <= 0:
                    return False
        return False

    def velocity_action_for(
        self,
        decision: StateMachineDecision,
        robot_observation: dict,
        step: Optional[int] = None,
        max_forward_speed: float = 0.75,
        max_lateral_speed: float = 0.35,
    ) -> dict:
        fallback_action = self.action_for(decision, robot_observation, step=step)
        if 'move_by_speed' in fallback_action or not self._planned_path:
            return fallback_action
        position = np.asarray(robot_observation['position'][:2], dtype=np.float64)
        # Calibrated approach states must hit every configured waypoint with
        # centimetre tolerance, so they keep targeting the raw waypoint. All
        # other states track a carrot interpolated on the planned path, which
        # keeps the commanded heading (and thus the gait) continuous while
        # dense skeleton waypoints are consumed.
        calibrated = decision.state in (
            InteractionState.NAVIGATE_TO_OBJECT,
            InteractionState.NAVIGATE_TO_CARRY_GOAL,
        )
        if calibrated:
            target = np.asarray(self._planned_path[self._waypoint_index][:2], dtype=np.float64)
        else:
            target = self._lookahead_target(position)
        error_world = target - position
        distance = float(np.linalg.norm(error_world))
        if (
            decision.state == InteractionState.NAVIGATE_TO_OBJECT
            and self._waypoint_index == len(self._planned_path) - 1
            and distance <= 0.35
        ):
            return self._pose_target_action(
                target,
                robot_observation,
                desired_yaw=self.navigation_headings[InteractionState.NAVIGATE_TO_OBJECT],
            )
        yaw = _yaw(robot_observation['orientation'])
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        forward_error = cos_yaw * error_world[0] + sin_yaw * error_world[1]
        lateral_error = -sin_yaw * error_world[0] + cos_yaw * error_world[1]
        desired_yaw = atan2(error_world[1], error_world[0])
        heading_error = (desired_yaw - yaw + pi) % (2.0 * pi) - pi
        rotation_speed = float(np.clip(1.8 * heading_error, -1.2, 1.2))
        if calibrated:
            forward_speed = float(np.clip(0.9 * forward_error, 0.0, max_forward_speed))
            lateral_speed = float(np.clip(1.2 * lateral_error, -max_lateral_speed, max_lateral_speed))
            if abs(heading_error) > 0.8:
                if (
                    decision.state == InteractionState.NAVIGATE_TO_CARRY_GOAL
                    and abs(heading_error) > 2.0
                ):
                    forward_speed = -0.18
                else:
                    forward_speed = 0.0
                lateral_speed = 0.0
            if distance < 0.30:
                forward_speed *= max(0.25, distance / 0.30)
                lateral_speed *= max(0.25, distance / 0.30)
            return ControllerCommand(
                'move_by_speed',
                (forward_speed, lateral_speed, rotation_speed),
            ).as_action()
        # Smooth speed law: cos^2 has a flat top, so small heading noise does
        # not modulate the forward speed; it fades to zero (turn in place)
        # as the error approaches 90 degrees without any hard threshold.
        heading_factor = max(0.0, cos(heading_error)) ** 2
        final_goal = np.asarray(self._planned_path[-1][:2], dtype=np.float64)
        final_distance = float(np.linalg.norm(final_goal - position))
        goal_factor = min(1.0, final_distance / self.final_goal_slow_radius)
        forward_speed = float(max_forward_speed * heading_factor * goal_factor)
        lateral_speed = float(
            np.clip(1.2 * lateral_error, -max_lateral_speed, max_lateral_speed) * heading_factor
        )
        return ControllerCommand(
            'move_by_speed',
            (forward_speed, lateral_speed, rotation_speed),
        ).as_action()

    def _lookahead_target(self, position: np.ndarray) -> np.ndarray:
        """Carrot point a fixed arc length ahead on the remaining planned path.

        Walking the polyline robot -> active waypoint -> ... and interpolating
        keeps the target sliding continuously as waypoints are passed, instead
        of jumping 0.3 m sideways whenever the active waypoint switches.
        """

        remaining = float(self.velocity_lookahead_distance)
        current = np.asarray(position, dtype=np.float64)
        for waypoint in self._planned_path[self._waypoint_index :]:
            point = np.asarray(waypoint[:2], dtype=np.float64)
            segment = point - current
            length = float(np.linalg.norm(segment))
            if length > remaining and length > 0.0:
                return current + segment * (remaining / length)
            remaining -= length
            current = point
        return current

    @staticmethod
    def _pose_target_action(target, robot_observation, desired_yaw: float) -> dict:
        position = np.asarray(robot_observation['position'][:2], dtype=np.float64)
        error_world = np.asarray(target[:2], dtype=np.float64) - position
        distance = float(np.linalg.norm(error_world))
        yaw = _yaw(robot_observation['orientation'])
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        forward_error = cos_yaw * error_world[0] + sin_yaw * error_world[1]
        lateral_error = -sin_yaw * error_world[0] + cos_yaw * error_world[1]
        heading_error = (desired_yaw - yaw + pi) % (2.0 * pi) - pi
        if abs(heading_error) < 0.25:
            translation_scale = 1.0
        elif abs(heading_error) < 0.60:
            translation_scale = 0.4
        else:
            translation_scale = 0.0
        translation = np.asarray(
            (
                np.clip(0.8 * forward_error, -0.12, 0.12),
                np.clip(0.8 * lateral_error, -0.08, 0.08),
            ),
            dtype=np.float64,
        )
        translation_norm = float(np.linalg.norm(translation))
        if distance > 0.04 and 0.0 < translation_norm < 0.08:
            translation *= 0.08 / translation_norm
        translation *= translation_scale
        return ControllerCommand(
            'move_by_speed',
            (
                float(translation[0]),
                float(translation[1]),
                float(np.clip(2.0 * heading_error, -0.8, 0.8)),
            ),
        ).as_action()

    def point_navigation_action(
        self,
        goal,
        robot_observation: dict,
        step: Optional[int] = None,
        velocity_control: bool = False,
        max_forward_speed: float = 0.75,
        max_lateral_speed: float = 0.35,
    ) -> dict:
        """Plan and track a point-navigation goal without exposing interaction states.

        The interaction demo still uses :meth:`action_for` directly.  This
        adapter keeps pure point-navigation callers independent from the
        deterministic interaction state machine while preserving the same
        tested planning implementation.
        """

        goal = _vector3(goal)
        decision = StateMachineDecision(
            state=InteractionState.NAVIGATE_TO_GOAL,
            command=ControllerCommand('move_along_path', ((goal,),)),
        )
        if velocity_control:
            return self.velocity_action_for(
                decision,
                robot_observation,
                step=step,
                max_forward_speed=max_forward_speed,
                max_lateral_speed=max_lateral_speed,
            )
        return self.action_for(decision, robot_observation, step=step)

    def _plan_with_final_heading(self, state: InteractionState, start, goal):
        if (
            state == InteractionState.NAVIGATE_TO_GOAL
            and self.use_scene_graph
            and self.allow_goal_door_traversal
        ):
            portal_path = self._plan_through_open_door(goal)
            if portal_path is not None:
                return portal_path
        heading = self.navigation_headings.get(state)
        if heading is None:
            return self._map_plan(start, goal)
        approach = (
            goal[0] - self.final_approach_distance * cos(heading),
            goal[1] - self.final_approach_distance * sin(heading),
            goal[2],
        )
        if state == InteractionState.NAVIGATE_TO_DOOR:
            semantic_path = self._plan_around_pickup_pedestal(start, approach)
            if semantic_path is not None:
                path = list(semantic_path)
                path.append(goal)
                return tuple(path)
        approach_path = self._map_plan(start, approach)
        path = list(approach_path)
        if not path or np.linalg.norm(np.asarray(path[-1][:2]) - np.asarray(approach[:2])) > 1e-6:
            path.append(approach)
        path.append(goal)
        return tuple(path)

    def _plan_around_pickup_pedestal(self, start, approach):
        start_xy = np.asarray(start[:2], dtype=np.float64)
        candidates = [
            node
            for node in self.map.scene_graph.object_nodes()
            if node.label == 'pedestal'
            and node.observations >= 2
            and np.linalg.norm(np.asarray(node.position[:2]) - start_xy) < 3.0
        ]
        if not candidates:
            return None
        pedestal = max(candidates, key=lambda node: node.observations)
        egress = (
            pedestal.position[0] - 0.45,
            pedestal.position[1] + 0.55,
            approach[2],
        )
        clearance = (
            pedestal.position[0] + 0.45,
            pedestal.position[1] + 1.20,
            approach[2],
        )
        try:
            final_leg = self._map_plan(clearance, approach)
        except PlanningError:
            return None
        return (egress, clearance, *final_leg)

    def _plan_through_open_door(self, goal):
        nodes = self.map.scene_graph.object_nodes()
        hinges = [
            node
            for node in nodes
            if node.label == 'door_frame_hinge' and node.observations >= 2 and node.position[2] > 0.3
        ]
        latches = [
            node
            for node in nodes
            if node.label == 'door_frame_latch' and node.observations >= 2 and node.position[2] > 0.3
        ]
        if not hinges or not latches:
            return None
        hinge = max(hinges, key=lambda node: node.observations)
        latch = max(latches, key=lambda node: node.observations)
        center_x = (hinge.position[0] + latch.position[0]) / 2.0
        center_y = (hinge.position[1] + latch.position[1]) / 2.0
        self.map.occupancy.mark_free((center_x, center_y), radius=0.75)
        portal = (center_x + 0.70, center_y, goal[2])
        try:
            final_leg = self._map_plan(portal, goal)
        except PlanningError:
            return None
        path = [portal]
        path.extend(final_leg)
        return self._densify_waypoints(path, max_spacing=0.45)

    def _map_plan(self, start, goal):
        if self.voronoi_planner is not None:
            try:
                path = self.voronoi_planner.plan(start, goal)
            except PlanningError:
                self.voronoi_plan_fallbacks += 1
            else:
                self.voronoi_plan_count += 1
                self.map.plan_count += 1
                return path
        return self.map.plan(
            start,
            goal,
            reinforce_semantics=self.use_semantic_occupancy,
        )

    @staticmethod
    def _densify_waypoints(path, max_spacing: float):
        if len(path) < 2:
            return tuple(path)
        dense_path = [tuple(path[0])]
        for endpoint in path[1:]:
            start = np.asarray(dense_path[-1], dtype=np.float64)
            endpoint_array = np.asarray(endpoint, dtype=np.float64)
            distance = float(np.linalg.norm(endpoint_array[:2] - start[:2]))
            segments = max(1, int(np.ceil(distance / max_spacing)))
            for index in range(1, segments + 1):
                point = start + (endpoint_array - start) * (index / segments)
                dense_path.append(tuple(float(value) for value in point))
        return tuple(dense_path)

    @staticmethod
    def _aligned_push_action(command: ControllerCommand, robot_observation: dict):
        aligned_data = MapNavigationRuntime._aligned_speed_data(
            forward_speed=float(command.data[0]),
            orientation=robot_observation['orientation'],
        )
        return ControllerCommand(command.name, tuple(aligned_data)).as_action()

    @staticmethod
    def _aligned_speed_data(forward_speed: float, orientation):
        yaw = _yaw(orientation)
        heading_error = (0.0 - yaw + pi) % (2.0 * pi) - pi
        forward_speed = forward_speed if abs(heading_error) < 0.12 else 0.0
        rotation_speed = float(np.clip(2.5 * heading_error, -1.2, 1.2))
        return [forward_speed, 0.0, rotation_speed]

    def statistics(self) -> dict:
        if self.use_scene_graph:
            stats = self.map.statistics()
        else:
            stats = {
                'lidar_frames': self.map.lidar_frames,
                'rgb_frames': self.map.rgb_frames,
                'lidar_points': self.map.lidar_points,
                'rgb_points': self.map.rgb_points,
                'voxel_count': self.map.voxels.voxel_count,
                'lidar_voxel_count': self.map.voxels.lidar_voxel_count,
                'colored_voxel_count': self.map.voxels.colored_voxel_count,
                'occupied_cells': int(self.map.occupancy.occupied_mask().sum()),
                'observed_cells': int(self.map.occupancy.observed.sum()),
                'plans': self.map.plan_count,
            }
        stats.update(
            {
                'planning_failures': self.planning_failures,
                'replans': self.replan_count,
                'planned_states': sorted(self._planned_paths),
                'planned_waypoints': {
                    state: len(path) for state, path in sorted(self._planned_paths.items())
                },
                'active_waypoint_index': self._waypoint_index,
                'active_waypoint': (
                    None
                    if not self._planned_path
                    else list(self._planned_path[min(self._waypoint_index, len(self._planned_path) - 1)])
                ),
                'last_safe_position': self._last_safe_position,
                'static_obstacle_boxes': len(self._static_obstacle_boxes),
                'static_occupied_cells': int(self.map.occupancy.static_occupied.sum()),
                'trajectory_static_obstacle_violations': self._static_obstacle_violations,
                'violated_static_obstacle_labels': sorted(self._violated_static_labels),
                'semantic_target': self.semantic_target,
                'open_vocabulary_frames': self.open_vocabulary_frames,
                'open_vocabulary_failures': self.open_vocabulary_failures,
                'voronoi_plans': self.voronoi_plan_count,
                'voronoi_plan_fallbacks': self.voronoi_plan_fallbacks,
            }
        )
        if self.semantic_voronoi is not None:
            stats['semantic_voronoi'] = self.semantic_voronoi.statistics()
        return stats

    def _update_static_obstacle_violations(self, position):
        xy = np.asarray(position[:2], dtype=np.float64)
        # Interaction standoffs may intentionally sit just outside a support
        # surface's AABB. Count only an actual center-point penetration here;
        # planning still uses the full static-obstacle inflation radius.
        padding = 0.0
        violated = False
        for label, minimum, maximum in self._static_obstacle_boxes:
            if np.all(xy >= minimum - padding) and np.all(xy <= maximum + padding):
                self._violated_static_labels.add(label)
                violated = True
        if violated:
            self._static_obstacle_violations += 1

    def save(self, output_prefix: str):
        if not output_prefix:
            return
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        points = self.map.voxels.points()
        colors = self.map.voxels.colors()
        skeleton = np.zeros_like(self.map.occupancy.observed, dtype=bool)
        if self.semantic_voronoi is not None:
            for cell in self.semantic_voronoi.update(force=True).skeleton_cells:
                skeleton[cell] = True
        np.savez_compressed(
            str(prefix) + '.npz',
            points=points,
            colors=colors,
            occupancy_log_odds=self.map.occupancy.log_odds,
            occupancy_observed=self.map.occupancy.observed,
            occupancy_inflated=self.map.occupancy.inflated_mask(),
            occupancy_static=self.map.occupancy.static_occupied,
            semantic_voronoi_skeleton=skeleton,
        )
        payload = {
            'mapping_config': asdict(self.map.config),
            'statistics': self.statistics(),
            'planned_paths': {
                state: [list(point) for point in path] for state, path in sorted(self._planned_paths.items())
            },
            'plan_history': self._plan_history,
        }
        if self.use_scene_graph:
            graph = self.map.snapshot()
            payload['scene_graph'] = {
                'nodes': [asdict(node) for node in graph.nodes],
                'edges': [asdict(edge) for edge in graph.edges],
            }
        if self.semantic_voronoi is not None:
            payload['semantic_voronoi'] = self.semantic_voronoi.to_dict()
        with Path(str(prefix) + '.json').open('w', encoding='utf-8') as output_file:
            json.dump(payload, output_file, indent=2)


def _camera_cloud(camera: dict):
    points = _points(camera.get('pointcloud'))
    rgba = camera.get('rgba')
    depth = camera.get('depth')
    if len(points) == 0 or rgba is None or depth is None:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            None,
        )

    rgba = np.asarray(rgba)
    depth = np.asarray(depth)
    if rgba.ndim != 3 or depth.ndim != 2 or rgba.shape[:2] != depth.shape:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            None,
        )
    mask = np.isfinite(depth) & (depth > 0.01) & (depth < 10000.0)
    colors = rgba[..., :3][mask]
    if len(points) == depth.size:
        points = points.reshape((*depth.shape, 3))[mask]
    count = min(len(points), len(colors))
    points = points[:count]
    colors = colors[:count]
    point_image = np.full((*depth.shape, 3), np.nan, dtype=np.float32)
    valid_indices = np.argwhere(mask)
    valid_indices = valid_indices[:count]
    point_image[valid_indices[:, 0], valid_indices[:, 1]] = points
    return points, colors, point_image


def _semantic_detections(camera: dict, point_image) -> list:
    if point_image is None:
        return []
    bounding_boxes = camera.get('bounding_box_2d_tight')
    rgba = camera.get('rgba')
    if not isinstance(bounding_boxes, dict) or rgba is None:
        return []
    data = bounding_boxes.get('data', [])
    label_lookup = bounding_boxes.get('info', {}).get('idToLabels', {})
    height, width = point_image.shape[:2]
    detections = []
    for row in data:
        values = tuple(row.tolist()) if hasattr(row, 'tolist') else tuple(row)
        if len(values) < 5:
            continue
        semantic_id, x_min, y_min, x_max, y_max = values[:5]
        label_data = label_lookup.get(str(int(semantic_id)), label_lookup.get(str(semantic_id), {}))
        label = label_data.get('class') if isinstance(label_data, dict) else None
        label = _semantic_category(label)
        if not label or label in _NON_OBJECT_SEMANTIC_LABELS:
            continue
        if x_max <= 0 or y_max <= 0 or x_min >= width or y_min >= height:
            continue
        x_min = max(0, min(width - 1, int(x_min)))
        x_max = max(x_min + 1, min(width, int(x_max)))
        y_min = max(0, min(height - 1, int(y_min)))
        y_max = max(y_min + 1, min(height, int(y_max)))
        if (x_max - x_min) * (y_max - y_min) < width * height * 0.001:
            continue
        if len(values) > 5 and float(values[5]) > 0.85:
            continue
        region = point_image[y_min:y_max, x_min:x_max]
        valid = np.isfinite(region).all(axis=2)
        if not valid.any():
            continue
        position = np.median(region[valid], axis=0)
        color_region = np.asarray(rgba)[y_min:y_max, x_min:x_max, :3]
        color = np.median(color_region[valid], axis=0)
        detections.append(
            SemanticDetection(
                label=str(label),
                position=tuple(float(value) for value in position),
                color=tuple(int(value) for value in np.clip(color, 0, 255)),
            )
        )
    return detections


def _deduplicate_semantic_detections(detections) -> tuple:
    """Fuse duplicate observations produced by multiple sources in one frame."""

    merged = []
    for detection in detections:
        match_index = None
        incoming_embedding = detection.embedding
        for index, candidate in enumerate(merged):
            distance = float(
                np.linalg.norm(
                    np.asarray(candidate.position[:2], dtype=np.float64)
                    - np.asarray(detection.position[:2], dtype=np.float64)
                )
            )
            if distance >= 0.75:
                continue
            same_label = candidate.label.strip().casefold() == detection.label.strip().casefold()
            similar_embedding = False
            if candidate.embedding is not None and incoming_embedding is not None:
                left = np.asarray(candidate.embedding, dtype=np.float64)
                right = np.asarray(incoming_embedding, dtype=np.float64)
                if left.shape == right.shape and left.size:
                    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
                    similar_embedding = denominator > 1e-12 and float(np.dot(left, right) / denominator) >= 0.86
            if same_label or similar_embedding:
                match_index = index
                break
        if match_index is None:
            merged.append(detection)
            continue

        previous = merged[match_index]
        # Prefer the geometrically denser observation, while retaining
        # complementary appearance information from either source.
        geometry = detection if detection.point_count > previous.point_count else previous
        appearance = detection if detection.embedding is not None else previous
        color_source = detection if detection.color is not None else previous
        label_source = detection if detection.confidence > previous.confidence else previous
        merged[match_index] = SemanticDetection(
            label=label_source.label,
            position=geometry.position,
            color=color_source.color,
            confidence=max(float(previous.confidence), float(detection.confidence)),
            embedding=appearance.embedding,
            point_count=max(int(previous.point_count), int(detection.point_count)),
            step=max(int(previous.step), int(detection.step)),
        )
    return tuple(merged)


def _semantic_category(label) -> str:
    if not label:
        return ''
    return str(label).strip().lower().split('/', 1)[0]


def _points(value) -> np.ndarray:
    if value is None:
        return np.empty((0, 3), dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return array.reshape(-1, 3)


def _vector3(value):
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if len(array) < 3:
        raise ValueError('expected a three-dimensional vector')
    return tuple(float(component) for component in array[:3])


def _yaw(quaternion) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
