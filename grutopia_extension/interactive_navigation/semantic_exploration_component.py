"""Go2-independent orchestration for semantic target exploration."""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from grutopia_extension.interactive_navigation.mapping import MappingConfig, SceneGraphNode
from grutopia_extension.interactive_navigation.mapping_runtime import (
    MapNavigationRuntime,
    SemanticDetectionMode,
)
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
    # `isaac` = simulator ground-truth labels only, `open_vocab` = VLM
    # detections only, `hybrid` = fused. See SemanticDetectionMode.
    semantic_detection_mode: str = 'hybrid'
    open_vocabulary_startup_error: Optional[str] = None
    target_min_observations: int = 2
    target_embedding_threshold: float = 0.24
    require_lexical_confirmation: bool = True
    target_confirmation_min_label_observations: int = 8
    target_confirmation_min_label_fraction: float = 0.60
    frontier_selection_interval: int = 160
    # Incremental Voronoi updates keep frequent topology refreshes cheap; a
    # short interval also keeps changed-cell windows small and splice-able.
    topology_update_interval: int = 40
    max_forward_speed: float = 0.80
    max_lateral_speed: float = 0.20

    def __post_init__(self):
        if not self.target_query.strip():
            raise ValueError('target_query cannot be empty')
        object.__setattr__(
            self,
            'semantic_detection_mode',
            SemanticDetectionMode.parse(self.semantic_detection_mode).value,
        )
        for name in (
            'max_steps',
            'target_min_observations',
            'target_confirmation_min_label_observations',
            'frontier_selection_interval',
            'topology_update_interval',
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        for name in ('target_distance', 'frontier_reached_distance', 'safe_base_height'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if not 0.0 < self.target_confirmation_min_label_fraction <= 1.0:
            raise ValueError('target_confirmation_min_label_fraction must be in (0, 1]')


@dataclass(frozen=True)
class SemanticExplorationResult:
    status: SemanticExplorationStatus
    step: int
    target_query: str
    position: Tuple[float, float, float]
    target_position: Optional[Tuple[float, float, float]]
    failure_reason: Optional[str]
    statistics: dict
    approach_goal: Optional[Tuple[float, float, float]] = None
    goal_distance: Optional[float] = None
    arrival_threshold: float = 0.70

    @property
    def terminal(self) -> bool:
        return self.status != SemanticExplorationStatus.RUNNING

    @property
    def success(self) -> bool:
        return self.status == SemanticExplorationStatus.SUCCEEDED

    def as_event(self) -> dict:
        """Keep terminal stdout small; the map and run summary hold details."""
        return {
            'event': 'semantic_exploration_result',
            'status': self.status.value,
            'success': self.success,
            'step': self.step,
            'target_query': self.target_query,
            'position': [round(value, 3) for value in self.position],
            'goal_distance_m': None if self.goal_distance is None else round(self.goal_distance, 3),
            'arrival_threshold_m': self.arrival_threshold,
            'reason': self.failure_reason,
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
            semantic_detection_mode=config.semantic_detection_mode,
            open_vocabulary_startup_error=config.open_vocabulary_startup_error,
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
        self._target_embedding = None
        self._target_embedding_attempted = False
        self._target_lexical = False
        self._rejected_embedding_targets = set()
        self._trajectory = []
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

    def _node_label_evidence(self, node: SceneGraphNode) -> Tuple[int, int, int]:
        counts = self.mapping.map.scene_graph.label_counts(node.node_id)
        if not counts:
            counts = {node.label: node.observations}
        query = _normalize_label(self.config.target_query)
        matching_counts = [
            count for label, count in counts.items()
            if query in _normalize_label(label) or _normalize_label(label) in query
        ]
        return sum(matching_counts), sum(counts.values()), max(matching_counts, default=0)

    def _target_label_support(self) -> Tuple[int, int]:
        if self.target_node is None:
            return 0, 0
        matching, total, _ = self._node_label_evidence(self.target_node)
        return matching, total

    @property
    def target_confirmed(self) -> bool:
        if self.target_node is None:
            return False
        if not self.config.require_lexical_confirmation:
            return True
        matching, total, repeated_label = self._node_label_evidence(self.target_node)
        return (
            self._target_lexical
            and matching >= self.config.target_confirmation_min_label_observations
            and total > 0
            and (
                matching / total >= self.config.target_confirmation_min_label_fraction
                # A stable refrigerator can also be called "door" in most
                # frames. Repeated identical target labels are stronger
                # evidence than their fraction of all node observations.
                or repeated_label >= self.config.target_confirmation_min_label_observations
            )
        )

    def update(self, step: int, robot_observation: dict):
        self.mapping.update(step, robot_observation)
        position = _position(robot_observation)
        self._trajectory.append(position)
        if self.target_node is None:
            self.target_node = self._best_target_node()
        else:
            self.target_node = next(
                (
                    node
                    for node in self.mapping.map.scene_graph.object_nodes()
                    if node.node_id == self.target_node.node_id
                ),
                self.target_node,
            )
            lexical = self._lexical_target_node()
            if lexical is None:
                self._target_lexical = False
            elif lexical.node_id == self.target_node.node_id:
                self._target_lexical = True
            else:
                # An embedding-only lock is provisional. For two lexical
                # nodes, prefer evidence for the requested label before total
                # observations; generic "door" frames must not hold the lock
                # against a better-observed refrigerator.
                if not self._target_lexical or (
                    (self._node_label_evidence(lexical)[0], lexical.observations, lexical.confidence)
                    > (
                        self._node_label_evidence(self.target_node)[0],
                        self.target_node.observations,
                        self.target_node.confidence,
                    )
                ):
                    self.target_node = lexical
                    self._target_lexical = True
                    self.target_navigation_position = None
        if (
            self.config.require_lexical_confirmation
            and self.target_node is not None
            and not self.target_confirmed
            and self.target_navigation_position is not None
            and _distance_xy(position, self.target_navigation_position) <= self.config.target_distance
        ):
            # An embedding match is a place to investigate, not proof that
            # the named object was found. Continue exploring after reaching it.
            self._rejected_embedding_targets.add(self.target_node.node_id)
            self.target_node = None
            self.target_navigation_position = None
            self.current_goal = None
            self.current_frontier_id = None
            self._last_selection_step = step - self.config.frontier_selection_interval
        if self.target_node is not None:
            if self.target_navigation_position is None:
                self.target_navigation_position = self._target_approach_position(
                    self.target_node,
                    position,
                )
            self.current_goal = self.target_navigation_position
            self.current_frontier_id = None
            self._sync_runtime_state()
            return

        if self.current_goal is not None and self.current_frontier_id is not None:
            distance = _distance_xy(position, self.current_goal)
            if distance <= self.config.frontier_reached_distance:
                self.planner.record_arrival(self.current_frontier_id)
                self.current_goal = None
                self.current_frontier_id = None
            else:
                self.planner.update_progress(position)

        needs_selection = (
            self.last_decision is None
            or step - self._last_selection_step >= self.config.frontier_selection_interval
        )
        if needs_selection:
            self._select_frontier(step, position)
        self._sync_runtime_state()

    def _select_frontier(self, step: int, position):
        topology = self.mapping.semantic_voronoi
        snapshot = None if topology is None else topology.cached_snapshot()
        if snapshot is None:
            self.current_goal = None
            return
        decision, goal_xy = self.planner.select_goal(
            robot_position=position,
            semantic_voronoi=snapshot,
            occupancy=self.mapping.map.occupancy,
            target_query=self.config.target_query,
        )
        self.last_decision = decision
        self._last_selection_step = step
        self.current_frontier_id = decision.selected_frontier_id
        self.current_goal = (
            None
            if goal_xy is None
            else (float(goal_xy[0]), float(goal_xy[1]), float(position[2]))
        )
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
        if self.current_goal is None:
            # A slow in-place scan exposes new RGB/LiDAR evidence before a
            # frontier exists and is also the safe fallback after plan failure.
            return {'move_by_speed': [0.0, 0.0, 0.35]}
        # Velocity control with a sliding lookahead target: the discrete
        # move_along_path controller re-targets every dense skeleton waypoint
        # and collapses its forward speed on each heading jump, which showed
        # up as a stuttering gait once waypoints were spaced 0.3 m apart.
        return self.mapping.point_navigation_action(
            self.current_goal,
            robot_observation,
            step=step,
            velocity_control=True,
            max_forward_speed=self.config.max_forward_speed,
            max_lateral_speed=self.config.max_lateral_speed,
        )

    def evaluate(self, step: int, robot_observation: dict) -> SemanticExplorationResult:
        position = _position(robot_observation)
        approach_goal = self.target_navigation_position
        goal_distance = None if approach_goal is None else _distance_xy(position, approach_goal)
        status = SemanticExplorationStatus.RUNNING
        failure_reason = None
        if position[2] < self.config.fall_height:
            status = SemanticExplorationStatus.FAILED
            failure_reason = 'robot_fell'
        elif (
            self.target_node is not None
            and self.target_confirmed
            and goal_distance is not None
            and goal_distance <= self.config.target_distance
        ):
            status = SemanticExplorationStatus.SUCCEEDED
        elif step + 1 >= self.config.max_steps:
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
            # Full statistics are materialized only for terminal results.
            statistics=self.statistics() if terminal else {},
            approach_goal=approach_goal,
            goal_distance=goal_distance,
            arrival_threshold=self.config.target_distance,
        )

    def progress_event(self, step: int, robot_observation: dict) -> dict:
        position = _position(robot_observation)
        goal_distance = None if self.current_goal is None else _distance_xy(position, self.current_goal)
        return {
            'event': 'semantic_exploration_progress',
            'step': step,
            'state': self.state,
            'position': [round(value, 2) for value in position],
            'target_found': self.target_node is not None,
            'target_confirmed': self.target_confirmed,
            'goal_distance_m': None if goal_distance is None else round(goal_distance, 2),
        }

    def statistics(self) -> dict:
        stats = dict(self.mapping.statistics())
        stats.update(
            {
                'exploration_state': self.state,
                'target_query': self.config.target_query,
                'target_found': self.target_node is not None,
                'target_confirmed': self.target_confirmed,
                'target_label_support': self._target_label_support()[0],
                'rejected_provisional_targets': len(self._rejected_embedding_targets),
                'target_node': None if self.target_node is None else self.target_node.node_id,
                'target_match': (
                    None
                    if self.target_node is None
                    else ('lexical' if self._target_lexical else 'embedding')
                ),
                'target_match_method': (
                    None
                    if self.target_node is None
                    else ('lexical' if self._target_lexical else 'embedding')
                ),
                'target_sources': (
                    [] if self.target_node is None else list(self.target_node.sources)
                ),
                'active_frontier': self.current_frontier_id,
                'active_goal': None if self.current_goal is None else list(self.current_goal),
                'exploration_decisions': len(self.decision_history),
                'frontier_visits': dict(self.planner.visit_counts),
                'frontier_failures': dict(self.planner.failure_counts),
                'frontier_blacklist': sorted(self.planner.blacklist),
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
            'trajectory': [list(point) for point in self._trajectory],
        }
        with metadata_path.open('w', encoding='utf-8') as output_file:
            json.dump(payload, output_file, indent=2)

    def _candidate_target_nodes(self) -> list:
        return [
            node
            for node in self.mapping.map.scene_graph.object_nodes()
            if node.observations >= self.config.target_min_observations
        ]

    def _lexical_target_node(self, nodes=None) -> Optional[SceneGraphNode]:
        if nodes is None:
            nodes = self._candidate_target_nodes()
        normalized_target = _normalize_label(self.config.target_query)
        lexical = []
        for node in nodes:
            representative_matches = (
                normalized_target in _normalize_label(node.label)
                or _normalize_label(node.label) in normalized_target
            )
            matching, _, repeated_label = self._node_label_evidence(node)
            if representative_matches or (
                matching >= self.config.target_confirmation_min_label_observations
                and repeated_label >= self.config.target_confirmation_min_label_observations
            ):
                lexical.append(node)
        if not lexical:
            return None
        # Rank target-specific evidence before generic "door" observations.
        return max(
            lexical,
            key=lambda node: (
                self._node_label_evidence(node)[0],
                node.observations,
                node.confidence,
            ),
        )

    def _best_target_node(self) -> Optional[SceneGraphNode]:
        nodes = self._candidate_target_nodes()
        if not nodes:
            return None
        lexical = self._lexical_target_node(nodes)
        if lexical is not None:
            self._target_lexical = True
            return lexical
        target_embedding = self._text_embedding()
        if target_embedding is None:
            return None
        scored = []
        for node in nodes:
            if node.node_id in self._rejected_embedding_targets or node.embedding is None:
                continue
            embedding = np.asarray(node.embedding, dtype=np.float32)
            if embedding.shape != target_embedding.shape:
                continue
            scored.append((float(np.dot(target_embedding, embedding)), node))
        if not scored:
            return None
        similarity, node = max(scored, key=lambda item: item[0])
        if similarity < self.config.target_embedding_threshold:
            return None
        self._target_lexical = False
        return node

    def _text_embedding(self):
        if self._target_embedding_attempted:
            return self._target_embedding
        self._target_embedding_attempted = True
        if self.perception is None or not hasattr(self.perception, 'embed_text'):
            return None
        try:
            self._target_embedding = np.asarray(
                self.perception.embed_text(self.config.target_query),
                dtype=np.float32,
            )
        except Exception:
            self._target_embedding = None
        return self._target_embedding

    def _target_approach_position(self, node: SceneGraphNode, robot_position):
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
        blocked = occupancy.inflated_mask()
        if desired_cell is not None:
            candidates = []
            search_radius = max(
                2,
                int(np.ceil(self.config.target_distance / occupancy.config.grid_resolution)),
            )
            for radius in range(search_radius + 1):
                for row in range(desired_cell[0] - radius, desired_cell[0] + radius + 1):
                    for col in range(desired_cell[1] - radius, desired_cell[1] + radius + 1):
                        cell = (row, col)
                        if (
                            occupancy.in_bounds(cell)
                            and occupancy.observed[cell]
                            and not blocked[cell]
                        ):
                            candidates.append(cell)
                if candidates:
                    break
            if candidates:
                best = min(
                    candidates,
                    key=lambda cell: np.linalg.norm(
                        np.asarray(occupancy.cell_to_world(cell)) - desired
                    ),
                )
                desired = np.asarray(occupancy.cell_to_world(best), dtype=np.float64)
        return (float(desired[0]), float(desired[1]), float(robot_position[2]))

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
    return ' '.join(str(value).casefold().replace('_', ' ').replace('-', ' ').split())


__all__ = [
    'SemanticExplorationComponent',
    'SemanticExplorationConfig',
    'SemanticExplorationResult',
    'SemanticExplorationStatus',
]
