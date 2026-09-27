"""Go2-independent orchestration for semantic target exploration."""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from grutopia_extension.interactive_navigation.mapping import (
    MappingConfig,
    SceneGraphNode,
    canonical_semantic_label,
)
from grutopia_extension.interactive_navigation.mapping_runtime import MapNavigationRuntime, _yaw
from grutopia_extension.interactive_navigation.semantic_exploration import (
    AdaptiveExplorationPlanner,
    ExplorationConfig,
    ExplorationDecision,
)
from grutopia_extension.interactive_navigation.semantic_voronoi import SemanticVoronoiConfig


class SemanticExplorationStatus(str, Enum):
    RUNNING = 'running'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'


@dataclass(frozen=True)
class SemanticExplorationConfig:
    target_query: str
    mapping: MappingConfig = field(default_factory=MappingConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    voronoi: SemanticVoronoiConfig = field(default_factory=SemanticVoronoiConfig)
    max_steps: int = 12_000
    target_distance: float = 0.70
    frontier_reached_distance: float = 0.45
    fall_height: float = 0.12
    safe_base_height: float = 0.22
    target_min_observations: int = 2
    target_confirmation_frames: int = 2
    target_confirmation_max_gap_steps: int = 96
    target_confirmation_valid_steps: int = 960
    target_confirmation_observation_distance: float = 5.0
    target_confirmation_min_observation_distance: float = 1.0
    target_confirmation_min_box_area: float = 0.0025
    target_confirmation_min_box_span: float = 0.025
    target_confirmation_edge_margin: float = 2.0
    target_confirmation_timeout_steps: int = 240
    target_retry_cooldown_steps: int = 480
    target_position_replan_distance: float = 0.35
    target_navigation_max_attempts: int = 3
    target_navigation_stall_steps: int = 480
    target_navigation_min_progress: float = 0.20
    target_approach_separation: float = 0.30
    # Kept for compatibility with existing callers; similarity alone no longer
    # qualifies a node as the requested target.
    target_embedding_threshold: float = 0.24
    frontier_selection_interval: int = 160
    frontier_navigation_stall_steps: int = 480
    frontier_retry_cooldown_steps: int = 480
    exploration_stationary_steps: int = 1440
    exploration_no_frontier_steps: int = 960
    # Incremental Voronoi updates keep frequent topology refreshes cheap; a
    # short interval also keeps changed-cell windows small and splice-able.
    topology_update_interval: int = 40
    max_forward_speed: float = 0.80
    max_lateral_speed: float = 0.20
    semantic_source: str = 'mixed'

    def __post_init__(self):
        if not self.target_query.strip():
            raise ValueError('target_query cannot be empty')
        if self.max_steps < 0:
            raise ValueError('max_steps cannot be negative; 0 is unlimited')
        if self.semantic_source not in ('mixed', 'model'):
            raise ValueError('semantic_source must be mixed or model')
        for name in (
            'target_min_observations',
            'target_confirmation_frames',
            'target_confirmation_max_gap_steps',
            'target_confirmation_valid_steps',
            'target_confirmation_timeout_steps',
            'target_retry_cooldown_steps',
            'target_navigation_max_attempts',
            'target_navigation_stall_steps',
            'frontier_selection_interval',
            'frontier_navigation_stall_steps',
            'frontier_retry_cooldown_steps',
            'exploration_stationary_steps',
            'exploration_no_frontier_steps',
            'topology_update_interval',
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        for name in ('target_distance', 'frontier_reached_distance', 'safe_base_height',
                     'target_position_replan_distance', 'target_navigation_min_progress',
                     'target_approach_separation', 'target_confirmation_observation_distance',
                     'target_confirmation_edge_margin'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        for name in ('target_confirmation_min_box_area', 'target_confirmation_min_box_span'):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in (0, 1]')
        if not 0 <= self.target_confirmation_min_observation_distance < self.target_confirmation_observation_distance:
            raise ValueError('confirmation minimum distance must be below maximum distance and nonnegative')


@dataclass(frozen=True)
class SemanticExplorationResult:
    status: SemanticExplorationStatus
    step: int
    target_query: str
    position: Tuple[float, float, float]
    target_position: Optional[Tuple[float, float, float]]
    failure_reason: Optional[str]
    statistics: dict

    @property
    def terminal(self) -> bool:
        return self.status != SemanticExplorationStatus.RUNNING

    @property
    def success(self) -> bool:
        return self.status == SemanticExplorationStatus.SUCCEEDED

    def as_event(self) -> dict:
        return {
            'event': 'semantic_exploration_result',
            'success': self.success,
            'step': self.step,
            'target_query': self.target_query,
            'position': list(self.position),
            'target_position': None if self.target_position is None else list(self.target_position),
            'failure_reason': self.failure_reason,
            'statistics': self.statistics,
        }


class SemanticExplorationComponent:
    """Fuse observations, select semantic/frontier goals, and drive point navigation."""

    def __init__(self, config: SemanticExplorationConfig, perception=None, scorer=None):
        self.config = config
        self.perception = perception
        self.mapping = MapNavigationRuntime(
            mapping_config=config.mapping,
            safe_base_height=config.safe_base_height,
            replan_interval_steps=120,
            replan_only_if_blocked=True,
            replan_lookahead_distance=2.0,
            allow_goal_door_traversal=False,
            use_scene_graph=True,
            use_semantic_occupancy=False,
            # Occupancy geometry is lidar-only: the flattened top-down scan
            # keeps furniture mapped without camera-point assistance.
            use_rgb_occupancy=False,
            rgb_clears_free_space=False,
            semantic_target=config.target_query,
            open_vocabulary_perception=perception,
            semantic_source=config.semantic_source,
            use_semantic_voronoi=True,
            semantic_voronoi_config=config.voronoi,
            topology_update_interval=config.topology_update_interval,
            prefer_voronoi_paths=True,
        )
        self.planner = AdaptiveExplorationPlanner(config.exploration, scorer=scorer)
        self.current_goal: Optional[Tuple[float, float, float]] = None
        self.current_frontier_id: Optional[str] = None
        self.target_node: Optional[SceneGraphNode] = None
        self.target_navigation_position: Optional[Tuple[float, float, float]] = None
        self.last_decision: Optional[ExplorationDecision] = None
        self.decision_history = []
        self._last_selection_step = -config.frontier_selection_interval
        self._target_lexical = False
        self._trajectory = []
        self._step = 0
        self._target_selected_step = -1
        self._target_position_anchor = None
        self._reset_target_confirmation()
        self._target_retry_after = {}
        # Geometric rejection is separate from the existing arrival-confirmation
        # cooldown. More semantic votes alone cannot clear an unreachable target.
        self._unreachable_targets = {}
        self._last_unreachable_review_step = -config.frontier_selection_interval
        self._reset_target_navigation()
        self.target_events = []
        self.frontier_events = []
        self._frontier_retry_after = {}
        self._exploration_progress_position = None
        self._exploration_progress_step = 0
        self._no_frontier_since = None
        self._exploration_failure_reason = None
        self._sync_runtime_state()

    @staticmethod
    def warmup_action() -> dict:
        return {'move_by_speed': [0.0, 0.0, 0.0]}

    @property
    def state(self) -> str:
        if self.target_node is not None:
            return 'navigate_to_semantic_target'
        if self.last_decision is None:
            return 'initialize_exploration'
        return f'explore_{self.last_decision.mode.value}'

    def update(self, step: int, robot_observation: dict):
        self._step = step
        self.mapping.update(step, robot_observation)
        position = _position(robot_observation)
        self._trajectory.append(position)
        self._review_unreachable_targets(step, position)
        previous_target = self.target_node
        # Revalidate the current scene graph instead of keeping a stale target
        # lock after its category changes or the node disappears.
        self.target_node = self._best_target_node()
        previous_id = None if previous_target is None else previous_target.node_id
        target_id = None if self.target_node is None else self.target_node.node_id
        if previous_id != target_id:
            self.mapping.invalidate_navigation_plan('semantic_target_changed', step)
            self._target_selected_step = step
            self._target_position_anchor = None
            self._reset_target_confirmation()
            self._reset_target_navigation()
            self._record_target_event(step, 'selected', previous_node=previous_id)
            self.target_navigation_position = None
            self.current_goal = None
            self.current_frontier_id = None
            if self.target_node is None:
                # Resume frontier exploration immediately after rejecting a target.
                self._last_selection_step = step - self.config.frontier_selection_interval
        if self.target_node is not None:
            # Frontier watchdogs measure exploration only. Existing semantic
            # navigation and arrival-confirmation guards remain responsible here.
            self._exploration_progress_position = None
            self._no_frontier_since = None
            if self._target_position_anchor is not None and _distance_xy(
                self.target_node.position, self._target_position_anchor
            ) >= self.config.target_position_replan_distance:
                self.mapping.invalidate_navigation_plan('semantic_target_moved', step)
                self.target_navigation_position = None
                self._reset_target_confirmation()
                self._reset_target_navigation()
            if self.target_navigation_position is None:
                self.target_navigation_position = self._target_approach_position(
                    self.target_node,
                    position,
                )
                self._target_position_anchor = self.target_node.position
                if self.target_navigation_position is None:
                    self._exclude_unreachable_target(step, position, 'no_reachable_approach')
                    # Re-select on the next update instead of planning a frontier
                    # route for one tick while another semantic candidate exists.
                    self._sync_runtime_state()
                    return
                else:
                    self._reset_navigation_progress(step, position)
            if self.target_node is not None:
                self.current_goal = self.target_navigation_position
                self.current_frontier_id = None
                self._update_target_confirmation(step, position)
            if self.target_node is not None:
                self._sync_runtime_state()
                return

        if self.current_goal is not None and self.current_frontier_id is not None:
            distance = _distance_xy(position, self.current_goal)
            if distance <= self.config.frontier_reached_distance:
                self.planner.record_arrival(self.current_frontier_id)
                self.current_goal = None
                self.current_frontier_id = None
                self._last_selection_step = step - self.config.frontier_selection_interval
            else:
                self.planner.update_progress(position)

        needs_selection = (
            self.last_decision is None
            or step - self._last_selection_step >= self.config.frontier_selection_interval
        )
        if needs_selection:
            self._select_frontier(step, position)
        self._update_exploration_exit(step, position)
        self._sync_runtime_state()

    def _reset_target_confirmation(self):
        self._target_arrival_step = None
        self._target_confirmation_count = 0
        self._target_confirmation_last_step = None
        self._target_confirmation_valid_until = None
        self._target_confirmation_node_id = None
        self._target_confirmation_position = None
        self._target_confirmation_best_distance = None
        # Changes of target/position/approach require evidence after that change.
        self._target_confirmation_after_step = self._step

    def _reset_target_navigation(self):
        self._target_navigation_attempts = 0
        self._failed_approach_positions = []
        self._reset_navigation_progress(self._step, None)

    def _reset_navigation_progress(self, step, position):
        self._navigation_progress_step = step
        self._navigation_progress_position = position
        self._navigation_progress_plan = None
        self._navigation_best_remaining = None

    def _navigation_stalled(self, step, position, stall_steps=None):
        remaining = self.mapping.remaining_navigation_distance(position)
        if remaining is None:
            # Frontier goals can have no usable path without a new exception.
            return (stall_steps is not None
                    and step - self._navigation_progress_step >= stall_steps)
        plan_number = self.mapping.map.plan_count
        threshold = self.config.target_navigation_min_progress
        if plan_number != self._navigation_progress_plan:
            # A replan changes route length; only physical movement earns a new
            # time budget here. Replanning repeatedly while stationary must expire.
            if (self._navigation_progress_position is not None
                    and _distance_xy(position, self._navigation_progress_position) >= threshold):
                self._navigation_progress_step = step
                self._navigation_progress_position = position
            self._navigation_best_remaining = remaining
            self._navigation_progress_plan = plan_number
        elif remaining <= self._navigation_best_remaining - threshold:
            self._navigation_progress_step = step
            self._navigation_progress_position = position
            self._navigation_best_remaining = remaining
        limit = self.config.target_navigation_stall_steps if stall_steps is None else stall_steps
        return step - self._navigation_progress_step >= limit

    def _record_frontier_event(self, step, event, **details):
        entry = {'step': step, 'event': event,
                 'frontier_id': self.current_frontier_id,
                 'goal': self.current_goal, **details}
        self.frontier_events.append(entry)
        print('[FRONTIER_GUARD] ' + json.dumps(entry), flush=True)

    def _frontier_attempt_failed(self, step, position, reason):
        frontier_id = self.current_frontier_id
        self.planner.record_failure(frontier_id)
        self._frontier_retry_after[frontier_id] = step + self.config.frontier_retry_cooldown_steps
        self._record_frontier_event(
            step, 'navigation_attempt_failed', reason=reason, position=position,
            failures=self.planner.failure_counts[frontier_id],
            blacklisted=frontier_id in self.planner.blacklist,
            retry_after=self._frontier_retry_after[frontier_id],
            planning_error=self.mapping.last_planning_error if reason == 'planning_failed' else None,
        )
        self.mapping.invalidate_navigation_plan('frontier_' + reason, step)
        self.current_goal = None
        self.current_frontier_id = None
        # Force reselection next update; the failed frontier is excluded even
        # when its geometric score still wins after applying the failure penalty.
        self._last_selection_step = step - self.config.frontier_selection_interval
        self._sync_runtime_state()

    def _update_exploration_exit(self, step, position):
        if self._exploration_failure_reason is not None:
            return
        # This clock is independent of frontier identity and route replanning:
        # cycling candidates cannot keep a physically stuck robot alive forever.
        if (self._exploration_progress_position is None
                or _distance_xy(position, self._exploration_progress_position)
                >= self.config.target_navigation_min_progress):
            self._exploration_progress_position = position
            self._exploration_progress_step = step
        if self.current_goal is None:
            if self._no_frontier_since is None:
                self._no_frontier_since = step
                self._record_frontier_event(step, 'scan_without_frontier', position=position)
        else:
            self._no_frontier_since = None
        reason = None
        if (self._no_frontier_since is not None
                and step - self._no_frontier_since >= self.config.exploration_no_frontier_steps):
            reason = 'no_available_frontiers'
        elif step - self._exploration_progress_step >= self.config.exploration_stationary_steps:
            reason = 'exploration_stalled'
        if reason is not None:
            self._exploration_failure_reason = reason
            self._record_frontier_event(
                step, 'exploration_failed', reason=reason, position=position,
                last_motion_step=self._exploration_progress_step,
                no_frontier_since=self._no_frontier_since,
            )
            self.mapping.invalidate_navigation_plan(reason, step)
            self.current_goal = None
            self.current_frontier_id = None

    def _navigation_attempt_failed(self, step, position, reason):
        failed_goal = self.target_navigation_position
        if failed_goal is not None:
            self._failed_approach_positions.append(failed_goal)
        self._target_navigation_attempts += 1
        self._record_target_event(
            step, 'navigation_attempt_failed', reason=reason,
            attempts=self._target_navigation_attempts,
            failed_goal=failed_goal,
            planning_error=self.mapping.last_planning_error if reason == 'planning_failed' else None,
        )
        self.mapping.invalidate_navigation_plan(reason, step)
        self.target_navigation_position = None
        self.current_goal = None
        self._reset_target_confirmation()
        if self._target_navigation_attempts >= self.config.target_navigation_max_attempts:
            self._exclude_unreachable_target(step, position, reason)
        self._sync_runtime_state()

    def _exclude_unreachable_target(self, step, position, reason):
        node = self.target_node
        approaches = self._target_approach_candidates(node, position, force=True)
        self._unreachable_targets[node.node_id] = {
            'step': step, 'reason': reason, 'node_position': tuple(node.position),
            'approaches': approaches, 'attempts': self._target_navigation_attempts,
        }
        self._record_target_event(
            step, 'navigation_unreachable', reason=reason,
            attempts=self._target_navigation_attempts,
            failed_goals=list(self._failed_approach_positions),
            reachable_approaches=len(approaches),
        )
        self.mapping.invalidate_navigation_plan('target_unreachable', step)
        self.target_node = None
        self._target_lexical = False
        self.target_navigation_position = None
        self._target_position_anchor = None
        self.current_goal = None
        self.current_frontier_id = None
        self._reset_target_confirmation()
        self._reset_target_navigation()
        self._last_selection_step = step - self.config.frontier_selection_interval

    def _review_unreachable_targets(self, step, position):
        if (not self._unreachable_targets
                or step - self._last_unreachable_review_step < self.config.frontier_selection_interval):
            return
        self._last_unreachable_review_step = step
        nodes = {node.node_id: node for node in self.mapping.map.scene_graph.object_nodes()}
        for node_id, rejection in list(self._unreachable_targets.items()):
            node = nodes.get(node_id)
            if node is None:
                continue
            approaches = self._target_approach_candidates(node, position)
            moved = _distance_xy(node.position, rejection['node_position']) >= self.config.target_position_replan_distance
            new_approach = bool(approaches)
            if approaches and rejection['approaches']:
                points = np.asarray(approaches, dtype=np.float64)[:, :2]
                old = np.asarray(rejection['approaches'], dtype=np.float64)[:, :2]
                distances_squared = ((points[:, None, :] - old[None, :, :]) ** 2).sum(axis=2)
                new_approach = bool(np.any(
                    distances_squared.min(axis=1) >= self.config.target_approach_separation ** 2
                ))
            if approaches and (moved or new_approach):
                del self._unreachable_targets[node_id]
                self._record_target_event(step, 'navigation_candidate_reopened',
                                          candidate_node=node_id,
                                          reason='position_changed' if moved else 'new_reachable_approach')

    def _record_target_event(self, step: int, event: str, **details):
        entry = {'step': step, 'event': event,
                 'target_node': None if self.target_node is None else self.target_node.node_id,
                 'target_position': None if self.target_node is None else list(self.target_node.position),
                 **details}
        self.target_events.append(entry)
        print('[TARGET_GUARD] ' + json.dumps(entry), flush=True)

    def _target_confirmation_valid(self, step: int) -> bool:
        return (
            self.target_node is not None
            and self._target_confirmation_node_id == self.target_node.node_id
            and self._target_confirmation_count >= self.config.target_confirmation_frames
            and self._target_confirmation_valid_until is not None
            and step <= self._target_confirmation_valid_until
        )

    def _confirmation_observation(self, step: int, position):
        distance = _distance_xy(position, self.target_node.position)
        if distance > self.config.target_confirmation_observation_distance:
            return None, 'outside_observation_range'
        if distance < self.config.target_confirmation_min_observation_distance:
            return None, 'inside_minimum_observation_range'
        if self.config.semantic_source != 'model':
            # Preserve the explicitly selected mixed/simulator source behavior.
            return {'source': 'scene_graph', 'object_distance': distance}, None
        label = _normalize_label(self.target_node.label)
        nodes = [node for node in self.mapping.map.scene_graph.object_nodes()
                 if _normalize_label(node.label) == label]
        matches = []
        for observation in self.mapping.model_confirmation_observations:
            if observation['step'] != step or _normalize_label(observation['label']) != label:
                continue
            nearest = min(nodes, key=lambda node: _distance_xy(node.position, observation['position']))
            # Use the scene graph's same-label, nearest-XY association radius.
            # A different instance of the same category cannot confirm this node.
            if (nearest.node_id == self.target_node.node_id
                    and _distance_xy(nearest.position, observation['position']) < 0.75):
                matches.append(observation)
        if not matches:
            return None, 'no_fresh_model_box_for_node'
        # The graph also gives the strongest same-frame observation precedence.
        observation = max(matches, key=lambda item: item['confidence'])
        height, width = observation['image_shape']
        x1, y1, x2, y2 = observation['bbox']
        if not np.isfinite([x1, y1, x2, y2]).all() or width <= 0 or height <= 0:
            return None, 'invalid_box'
        margin = self.config.target_confirmation_edge_margin
        touches_edge = x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin
        box_width, box_height = x2 - x1, y2 - y1
        if (box_width / width < self.config.target_confirmation_min_box_span
                or box_height / height < self.config.target_confirmation_min_box_span
                or box_width * box_height / (width * height) < self.config.target_confirmation_min_box_area):
            return None, 'small_box'
        # Mere contact with an image edge is not grounds for rejection. Keep
        # the existing conservative shape test for long, clipped fragments.
        if touches_edge and (
            (box_height / box_width >= 4.0 and box_height / height >= 0.8
             and (y1 <= margin or y2 >= height - margin))
            or (box_width / box_height >= 4.0 and box_width / width >= 0.8
                and (x1 <= margin or x2 >= width - margin))
        ):
            return None, 'clipped_edge_strip'
        return {'source': 'model', 'object_distance': distance,
                'bbox': list(observation['bbox']), 'box_touches_edge': bool(touches_edge)}, None

    def _retain_confirmation_on_progress(self, step: int, position):
        if (not self._target_confirmation_valid(step)
                or self._target_confirmation_position is None
                or self._target_confirmation_best_distance is None):
            return
        # Compare against the best distance to a fixed evidence-time position.
        # Oscillation, node-centroid drift and replanning cannot renew evidence.
        distance = _distance_xy(position, self._target_confirmation_position)
        if distance <= self._target_confirmation_best_distance - self.config.target_navigation_min_progress:
            self._target_confirmation_best_distance = distance
            self._target_confirmation_valid_until = step + self.config.target_confirmation_valid_steps
            self._record_target_event(step, 'confirmation_progress_retained',
                                      object_distance=distance,
                                      valid_until=self._target_confirmation_valid_until)

    def _update_target_confirmation(self, step: int, position):
        # Keep completed visual evidence while approaching. Arrival and visual
        # confirmation have separate clocks, so a clipped close-up cannot erase
        # valid evidence gathered earlier along this approach.
        self._retain_confirmation_on_progress(step, position)
        if (self._target_confirmation_valid_until is not None
                and step > self._target_confirmation_valid_until):
            self._record_target_event(step, 'confirmation_expired',
                                      valid_until=self._target_confirmation_valid_until)
            self._target_confirmation_count = 0
            self._target_confirmation_last_step = None
            self._target_confirmation_valid_until = None
            self._target_confirmation_node_id = None
            self._target_confirmation_position = None
            self._target_confirmation_best_distance = None
        last_step = self._target_confirmation_last_step
        if (not self._target_confirmation_valid(step) and last_step is not None
                and step - last_step > self.config.target_confirmation_max_gap_steps):
            self._target_confirmation_count = 0
            self._target_confirmation_last_step = None
            self._target_confirmation_node_id = None
        arrived = _distance_xy(position, self.target_navigation_position) <= self.config.target_distance
        observed_step = self.target_node.last_seen_step
        if (
            observed_step == step
            and observed_step > max(self._target_selected_step, self._target_confirmation_after_step)
            and observed_step != self._target_confirmation_last_step
        ):
            evidence, reason = self._confirmation_observation(step, position)
            if evidence is None:
                self._record_target_event(
                    step, 'confirmation_observation_ignored', reason=reason,
                    object_distance=_distance_xy(position, self.target_node.position),
                    observation_min_distance=self.config.target_confirmation_min_observation_distance,
                    observation_max_distance=self.config.target_confirmation_observation_distance,
                )
            else:
                was_valid = self._target_confirmation_valid(step)
                self._target_confirmation_count = min(
                    self.config.target_confirmation_frames, self._target_confirmation_count + 1,
                )
                self._target_confirmation_last_step = observed_step
                self._target_confirmation_node_id = self.target_node.node_id
                if self._target_confirmation_count >= self.config.target_confirmation_frames:
                    self._target_confirmation_valid_until = step + self.config.target_confirmation_valid_steps
                    self._target_confirmation_position = tuple(self.target_node.position)
                    self._target_confirmation_best_distance = _distance_xy(position, self.target_node.position)
                self._record_target_event(
                    step, 'fresh_confirmation', frames=self._target_confirmation_count,
                    phase='arrival' if arrived else 'approach',
                    valid_until=self._target_confirmation_valid_until, **evidence,
                )
                if not was_valid and self._target_confirmation_valid(step):
                    self._record_target_event(step, 'visual_confirmation_ready',
                                              valid_until=self._target_confirmation_valid_until)
        if not arrived:
            # Leaving the arrival radius restarts only the arrival wait, not
            # the visual evidence already earned for the current target.
            self._target_arrival_step = None
            return
        if self._target_arrival_step is None:
            self._target_arrival_step = step
            self._record_target_event(step, 'arrival_pending',
                                      visual_confirmed=self._target_confirmation_valid(step))
        if (
            not self._target_confirmation_valid(step)
            and step - self._target_arrival_step >= self.config.target_confirmation_timeout_steps
        ):
            retry_step = step + self.config.target_retry_cooldown_steps
            self._target_retry_after[self.target_node.node_id] = retry_step
            self._record_target_event(step, 'confirmation_timeout', retry_after=retry_step)
            self.mapping.invalidate_navigation_plan('target_confirmation_timeout', step)
            self.target_node = None
            self.target_navigation_position = None
            self._target_position_anchor = None
            self.current_goal = None
            self.current_frontier_id = None
            self._reset_target_confirmation()
            self._last_selection_step = step - self.config.frontier_selection_interval

    def _select_frontier(self, step: int, position):
        previous_frontier = self.current_frontier_id
        previous_goal = self.current_goal
        topology = self.mapping.semantic_voronoi
        snapshot = None if topology is None else topology.cached_snapshot()
        if snapshot is None:
            self.current_goal = None
            self.current_frontier_id = None
            return
        self._frontier_retry_after = {
            frontier_id: retry_after for frontier_id, retry_after in self._frontier_retry_after.items()
            if step < retry_after
        }
        decision, goal_xy = self.planner.select_goal(
            robot_position=position,
            semantic_voronoi=snapshot,
            occupancy=self.mapping.map.occupancy,
            target_query=self.config.target_query,
            excluded_frontier_ids=self._frontier_retry_after,
        )
        self.last_decision = decision
        self._last_selection_step = step
        self.current_frontier_id = decision.selected_frontier_id
        self.current_goal = (
            None
            if goal_xy is None
            else (float(goal_xy[0]), float(goal_xy[1]), float(position[2]))
        )
        if (previous_frontier != self.current_frontier_id
                or (previous_goal is None) != (self.current_goal is None)
                or (previous_goal is not None and self.current_goal is not None
                    and _distance_xy(previous_goal, self.current_goal)
                    >= self.config.target_navigation_min_progress)):
            self.mapping.invalidate_navigation_plan('frontier_goal_changed', step)
            self._reset_navigation_progress(step, position)
            self._record_frontier_event(step, 'selected', previous_frontier=previous_frontier)
        # Selecting the same frontier every 160 steps must not reset its stall timer.
        self.decision_history.append(
            {
                'step': step,
                'mode': decision.mode.value,
                'frontier_id': decision.selected_frontier_id,
                'reason': decision.reason,
                'used_model': decision.used_model,
                'fell_back': decision.fell_back,
                'goal': None if self.current_goal is None else list(self.current_goal),
                'candidates': [
                    {
                        'id': item.frontier_id,
                        'score': item.score,
                        'geometric_score': item.geometric_score,
                        'semantic_score': item.semantic_score,
                        'path_distance': item.path_distance,
                        'information_gain': item.information_gain,
                    }
                    for item in decision.candidates
                ],
            }
        )
        self._sync_runtime_state()

    def action(self, step: int, robot_observation: dict) -> dict:
        if self._exploration_failure_reason is not None:
            return self.warmup_action()
        if self.current_goal is None:
            # A slow in-place scan exposes new RGB/LiDAR evidence before a
            # frontier exists and is also the safe fallback after plan failure.
            return {'move_by_speed': [0.0, 0.0, 0.35]}
        if self.target_node is not None and self._target_arrival_step is not None:
            # Use scheduled RGB observations while facing the candidate; no extra
            # inference request and no forward motion during arrival verification.
            delta = np.asarray(self.target_node.position[:2]) - np.asarray(_position(robot_observation)[:2])
            heading = np.arctan2(delta[1], delta[0]) - _yaw(robot_observation['orientation'])
            heading = (heading + np.pi) % (2.0 * np.pi) - np.pi
            return {'move_by_speed': [0.0, 0.0, float(np.clip(1.5 * heading, -0.5, 0.5))]}
        # Velocity control with a sliding lookahead target: the discrete
        # move_along_path controller re-targets every dense skeleton waypoint
        # and collapses its forward speed on each heading jump, which showed
        # up as a stuttering gait once waypoints were spaced 0.3 m apart.
        failures_before = self.mapping.planning_failures
        action = self.mapping.point_navigation_action(
            self.current_goal,
            robot_observation,
            step=step,
            velocity_control=True,
            max_forward_speed=self.config.max_forward_speed,
            max_lateral_speed=self.config.max_lateral_speed,
        )
        if self.target_node is not None:
            position = _position(robot_observation)
            if self.mapping.planning_failures > failures_before:
                self._navigation_attempt_failed(step, position, 'planning_failed')
                return self.warmup_action()
            if self._navigation_stalled(step, position):
                self._navigation_attempt_failed(step, position, 'no_navigation_progress')
                return self.warmup_action()
        elif self.current_frontier_id is not None:
            position = _position(robot_observation)
            if self.mapping.planning_failures > failures_before:
                self._frontier_attempt_failed(step, position, 'planning_failed')
                return self.warmup_action()
            if self._navigation_stalled(step, position, self.config.frontier_navigation_stall_steps):
                self._frontier_attempt_failed(step, position, 'no_navigation_progress')
                return self.warmup_action()
        return action

    def evaluate(self, step: int, robot_observation: dict) -> SemanticExplorationResult:
        position = _position(robot_observation)
        status = SemanticExplorationStatus.RUNNING
        failure_reason = None
        if (
            self._is_confirmed_target(self.target_node)
            and self._target_confirmation_valid(step)
            and self.target_navigation_position is not None
            and _distance_xy(position, self.target_navigation_position) <= self.config.target_distance
        ):
            status = SemanticExplorationStatus.SUCCEEDED
        elif position[2] < self.config.fall_height:
            status = SemanticExplorationStatus.FAILED
            failure_reason = 'robot_fell'
        elif self._exploration_failure_reason is not None:
            status = SemanticExplorationStatus.FAILED
            failure_reason = self._exploration_failure_reason
        elif self.config.max_steps > 0 and step + 1 >= self.config.max_steps:
            status = SemanticExplorationStatus.FAILED
            failure_reason = 'global_step_limit'
        terminal = status != SemanticExplorationStatus.RUNNING
        return SemanticExplorationResult(
            status=status,
            step=step,
            target_query=self.config.target_query,
            position=position,
            target_position=(
                None if self.target_node is None else tuple(float(value) for value in self.target_node.position)
            ),
            failure_reason=failure_reason,
            # Full statistics walk the ever-growing scene graph and voxel map;
            # computing them on every running step dominated the whole loop, so
            # they are only materialized for terminal results.
            statistics=self.statistics() if terminal else {},
        )

    def progress_event(self, step: int, robot_observation: dict) -> dict:
        position = _position(robot_observation)
        return {
            'event': 'semantic_exploration_progress',
            'step': step,
            'state': self.state,
            'position': list(position),
            'target_query': self.config.target_query,
            'target_found': self.target_node is not None,
            'target_position': (
                None if self.target_node is None else list(self.target_node.position)
            ),
            'active_goal': None if self.current_goal is None else list(self.current_goal),
            'statistics': self.statistics(),
        }

    def statistics(self) -> dict:
        stats = dict(self.mapping.statistics())
        stats.update(
            {
                'exploration_state': self.state,
                'target_query': self.config.target_query,
                'target_found': self.target_node is not None,
                'target_node': None if self.target_node is None else self.target_node.node_id,
                'target_confirmation_frames': self._target_confirmation_count,
                'target_confirmation_required': self.config.target_confirmation_frames,
                'target_confirmation_last_step': self._target_confirmation_last_step,
                'target_visual_confirmed': self._target_confirmation_valid(self._step),
                'target_confirmation_valid_until': self._target_confirmation_valid_until,
                'target_confirmation_after_step': self._target_confirmation_after_step,
                'target_confirmation_observation_distance': self.config.target_confirmation_observation_distance,
                'target_confirmation_min_observation_distance': self.config.target_confirmation_min_observation_distance,
                'target_confirmation_best_distance': self._target_confirmation_best_distance,
                'target_confirmation_valid_steps': self.config.target_confirmation_valid_steps,
                'target_selected_step': self._target_selected_step,
                'target_arrival_step': self._target_arrival_step,
                'target_retry_after': dict(self._target_retry_after),
                'target_navigation_attempts': self._target_navigation_attempts,
                'target_navigation_max_attempts': self.config.target_navigation_max_attempts,
                'target_navigation_last_progress_step': self._navigation_progress_step,
                'unreachable_targets': {
                    node_id: {key: value for key, value in record.items() if key != 'approaches'}
                    for node_id, record in self._unreachable_targets.items()
                },
                'target_match': (
                    None
                    if self.target_node is None
                    else ('lexical' if self._target_lexical else 'embedding')
                ),
                'active_frontier': self.current_frontier_id,
                'active_goal': None if self.current_goal is None else list(self.current_goal),
                'exploration_decisions': len(self.decision_history),
                'frontier_visits': dict(self.planner.visit_counts),
                'frontier_failures': dict(self.planner.failure_counts),
                'frontier_blacklist': sorted(self.planner.blacklist),
                'frontier_retry_after': dict(self._frontier_retry_after),
                'exploration_last_motion_step': self._exploration_progress_step,
                'exploration_no_frontier_since': self._no_frontier_since,
                'exploration_failure_reason': self._exploration_failure_reason,
                'trajectory_points': len(self._trajectory),
            }
        )
        return stats

    def save(self, output_prefix: str):
        self.mapping.save(output_prefix)
        if not output_prefix:
            return
        metadata_path = Path(str(output_prefix) + '.json')
        with metadata_path.open('r', encoding='utf-8') as input_file:
            payload = json.load(input_file)
        payload['semantic_exploration'] = {
            'target_query': self.config.target_query,
            'statistics': self.statistics(),
            'decisions': self.decision_history,
            'target_events': self.target_events,
            'frontier_events': self.frontier_events,
            'trajectory': [list(point) for point in self._trajectory],
        }
        with metadata_path.open('w', encoding='utf-8') as output_file:
            json.dump(payload, output_file, indent=2)

    def _candidate_target_nodes(self) -> list:
        return [
            node
            for node in self.mapping.map.scene_graph.object_nodes()
            if node.observations >= self.config.target_min_observations
            and self._step >= self._target_retry_after.get(node.node_id, -1)
            and node.node_id not in self._unreachable_targets
        ]

    def _lexical_target_node(self, nodes=None) -> Optional[SceneGraphNode]:
        if nodes is None:
            nodes = self._candidate_target_nodes()
        lexical = [
            node
            for node in nodes
            if self._is_confirmed_target(node)
        ]
        if not lexical:
            return None
        # Evidence first: confidence saturates at 1.0 for both real objects
        # and long-range depth ghosts, so observation count discriminates.
        best = max(lexical, key=lambda node: (node.observations, node.confidence))
        # A DINO proposal-score tie-break is not evidence of a better VL category.
        current = next((node for node in lexical if self.target_node is not None
                        and node.node_id == self.target_node.node_id), None)
        if current is not None and current.observations >= best.observations:
            return current
        return best

    def _is_confirmed_target(self, node: Optional[SceneGraphNode]) -> bool:
        """Require category evidence; a CLIP nearest neighbour is not confirmation."""
        return (
            node is not None
            and node.kind == 'object'
            and node.observations >= self.config.target_min_observations
            and _normalize_label(node.label) == _normalize_label(self.config.target_query)
        )

    def _best_target_node(self) -> Optional[SceneGraphNode]:
        node = self._lexical_target_node()
        self._target_lexical = node is not None
        return node

    def _target_approach_position(self, node: SceneGraphNode, robot_position):
        candidates = self._target_approach_candidates(node, robot_position, force=True)
        return next((point for point in candidates
                     if all(_distance_xy(point, failed) >= self.config.target_approach_separation
                            for failed in self._failed_approach_positions)), None)

    def _target_approach_candidates(self, node: SceneGraphNode, robot_position, force=False):
        target = np.asarray(node.position[:2], dtype=np.float64)
        robot = np.asarray(robot_position[:2], dtype=np.float64)
        direction = robot - target
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            direction = np.array((-1.0, 0.0), dtype=np.float64)
        else:
            direction /= norm
        desired = target + direction * max(
            self.config.frontier_reached_distance,
            self.config.mapping.robot_radius + 0.20,
        )
        occupancy = self.mapping.map.occupancy
        desired_cell = occupancy.world_to_cell(desired)
        reachable = self.mapping.reachable_mask(robot_position, self._step, force=force)
        candidates = []
        if desired_cell is not None:
            search_radius = max(
                2,
                int(np.ceil(self.config.target_distance / occupancy.config.grid_resolution)),
            )
            for row in range(desired_cell[0] - search_radius, desired_cell[0] + search_radius + 1):
                for col in range(desired_cell[1] - search_radius, desired_cell[1] + search_radius + 1):
                    cell = (row, col)
                    if occupancy.in_bounds(cell) and occupancy.observed[cell] and reachable[cell]:
                        xy = occupancy.cell_to_world(cell)
                        candidates.append((float(xy[0]), float(xy[1]), float(robot_position[2])))
        candidates.sort(key=lambda point: float(np.linalg.norm(np.asarray(point[:2]) - desired)))
        return candidates

    def _sync_runtime_state(self):
        self.mapping.exploration_state = self.state
        self.mapping.exploration_target_query = self.config.target_query
        self.mapping.exploration_goal = self.current_goal
        self.mapping.exploration_decision = self.last_decision


def _position(robot_observation: dict) -> Tuple[float, float, float]:
    position = np.asarray(robot_observation['position'], dtype=np.float64).reshape(-1)
    if len(position) < 3:
        raise ValueError('robot position must contain three coordinates')
    return tuple(float(value) for value in position[:3])


def _distance_xy(left, right) -> float:
    return float(np.linalg.norm(np.asarray(left[:2], dtype=np.float64) - np.asarray(right[:2], dtype=np.float64)))


def _normalize_label(value: str) -> str:
    return canonical_semantic_label(value)


__all__ = [
    'SemanticExplorationComponent',
    'SemanticExplorationConfig',
    'SemanticExplorationResult',
    'SemanticExplorationStatus',
]
