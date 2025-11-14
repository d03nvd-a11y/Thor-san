"""
Task Planner

High-level task planning for robotic manipulation.
Generates action sequences to achieve goals.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """Types of robot actions."""
    MOVE_TO = "move_to"
    GRASP = "grasp"
    PLACE = "place"
    RELEASE = "release"
    OBSERVE = "observe"
    WAIT = "wait"


@dataclass
class Action:
    """Single action in a plan."""
    action_type: ActionType
    target_position: Optional[np.ndarray] = None
    target_object_id: Optional[int] = None
    parameters: Optional[Dict] = None
    estimated_duration: float = 0.0  # seconds


@dataclass
class Plan:
    """Sequence of actions to achieve a goal."""
    goal: str
    actions: List[Action]
    total_duration: float
    confidence: float  # Planning confidence [0, 1]


class TaskPlanner:
    """
    High-level task planner for manipulation.

    Generates action sequences based on scene understanding and goals.
    """

    def __init__(
        self,
        planning_horizon: int = 10,
        replan_threshold: float = 0.5
    ):
        """
        Initialize task planner.

        Args:
            planning_horizon: Maximum number of actions in plan
            replan_threshold: Confidence threshold for replanning
        """
        self.planning_horizon = planning_horizon
        self.replan_threshold = replan_threshold

    def plan_pick_and_place(
        self,
        object_position: np.ndarray,
        target_position: np.ndarray,
        object_id: Optional[int] = None,
        approach_height: float = 0.1
    ) -> Plan:
        """
        Plan pick-and-place task.

        Args:
            object_position: Object location (x, y, z)
            target_position: Place location (x, y, z)
            object_id: Optional object identifier
            approach_height: Height above object for approach (meters)

        Returns:
            Action plan
        """
        actions = []

        # 1. Move above object
        approach_pos = object_position + np.array([0, 0, approach_height])
        actions.append(Action(
            action_type=ActionType.MOVE_TO,
            target_position=approach_pos,
            estimated_duration=2.0
        ))

        # 2. Move to object
        actions.append(Action(
            action_type=ActionType.MOVE_TO,
            target_position=object_position,
            estimated_duration=1.0
        ))

        # 3. Grasp object
        actions.append(Action(
            action_type=ActionType.GRASP,
            target_object_id=object_id,
            estimated_duration=1.0
        ))

        # 4. Lift object
        lift_pos = object_position + np.array([0, 0, approach_height])
        actions.append(Action(
            action_type=ActionType.MOVE_TO,
            target_position=lift_pos,
            estimated_duration=1.0
        ))

        # 5. Move above target
        target_approach = target_position + np.array([0, 0, approach_height])
        actions.append(Action(
            action_type=ActionType.MOVE_TO,
            target_position=target_approach,
            estimated_duration=2.0
        ))

        # 6. Move to target
        actions.append(Action(
            action_type=ActionType.MOVE_TO,
            target_position=target_position,
            estimated_duration=1.0
        ))

        # 7. Release object
        actions.append(Action(
            action_type=ActionType.RELEASE,
            estimated_duration=0.5
        ))

        # 8. Retract
        retract_pos = target_position + np.array([0, 0, approach_height])
        actions.append(Action(
            action_type=ActionType.MOVE_TO,
            target_position=retract_pos,
            estimated_duration=1.0
        ))

        total_duration = sum(a.estimated_duration for a in actions)

        plan = Plan(
            goal=f"Pick object at {object_position} and place at {target_position}",
            actions=actions,
            total_duration=total_duration,
            confidence=0.9
        )

        return plan

    def plan_observe_object(
        self,
        object_position: np.ndarray,
        observation_distance: float = 0.3,
        num_views: int = 4
    ) -> Plan:
        """
        Plan multi-view observation of object.

        Args:
            object_position: Object location (x, y, z)
            observation_distance: Distance from object (meters)
            num_views: Number of viewpoints

        Returns:
            Action plan
        """
        actions = []

        # Generate viewpoints around object
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

        for angle in angles:
            # Compute viewpoint position
            viewpoint = object_position + np.array([
                observation_distance * np.cos(angle),
                observation_distance * np.sin(angle),
                0.1  # Slightly above object
            ])

            # Move to viewpoint
            actions.append(Action(
                action_type=ActionType.MOVE_TO,
                target_position=viewpoint,
                estimated_duration=1.5
            ))

            # Observe
            actions.append(Action(
                action_type=ActionType.OBSERVE,
                target_position=object_position,
                parameters={'duration': 1.0},
                estimated_duration=1.0
            ))

        total_duration = sum(a.estimated_duration for a in actions)

        plan = Plan(
            goal=f"Observe object at {object_position} from {num_views} viewpoints",
            actions=actions,
            total_duration=total_duration,
            confidence=0.95
        )

        return plan

    def plan_clear_workspace(
        self,
        object_positions: List[np.ndarray],
        clear_zone_center: np.ndarray,
        clear_radius: float = 0.2
    ) -> Plan:
        """
        Plan to clear objects from workspace.

        Args:
            object_positions: List of object positions
            clear_zone_center: Center of area to clear
            clear_radius: Radius of clear zone (meters)

        Returns:
            Action plan
        """
        actions = []

        # Find objects in clear zone
        objects_to_move = []
        for i, pos in enumerate(object_positions):
            distance = np.linalg.norm(pos[:2] - clear_zone_center[:2])
            if distance < clear_radius:
                objects_to_move.append((i, pos))

        # Plan to move each object outside clear zone
        for obj_id, obj_pos in objects_to_move:
            # Find placement location outside clear zone
            direction = obj_pos[:2] - clear_zone_center[:2]
            direction = direction / (np.linalg.norm(direction) + 1e-6)

            target_pos = clear_zone_center.copy()
            target_pos[:2] += direction * (clear_radius + 0.1)

            # Add pick-and-place actions
            pick_place_plan = self.plan_pick_and_place(obj_pos, target_pos, obj_id)
            actions.extend(pick_place_plan.actions)

        total_duration = sum(a.estimated_duration for a in actions)

        plan = Plan(
            goal=f"Clear {len(objects_to_move)} objects from workspace",
            actions=actions,
            total_duration=total_duration,
            confidence=0.85
        )

        return plan

    def plan_organize_objects(
        self,
        object_positions: List[np.ndarray],
        object_classes: List[str],
        target_locations: Dict[str, np.ndarray]
    ) -> Plan:
        """
        Plan to organize objects by class.

        Args:
            object_positions: List of object positions
            object_classes: List of object class names
            target_locations: Dictionary mapping class -> target location

        Returns:
            Action plan
        """
        actions = []

        # Group objects by class
        for i, (pos, obj_class) in enumerate(zip(object_positions, object_classes)):
            if obj_class in target_locations:
                target_pos = target_locations[obj_class]

                # Add some offset to avoid stacking exactly
                offset = np.array([
                    np.random.uniform(-0.05, 0.05),
                    np.random.uniform(-0.05, 0.05),
                    0
                ])
                target_pos_offset = target_pos + offset

                # Plan pick and place
                pick_place_plan = self.plan_pick_and_place(
                    pos,
                    target_pos_offset,
                    object_id=i
                )
                actions.extend(pick_place_plan.actions)

        total_duration = sum(a.estimated_duration for a in actions)

        plan = Plan(
            goal="Organize objects by class",
            actions=actions,
            total_duration=total_duration,
            confidence=0.8
        )

        return plan

    def estimate_action_feasibility(
        self,
        action: Action,
        current_state: Dict
    ) -> float:
        """
        Estimate feasibility of action given current state.

        Args:
            action: Action to evaluate
            current_state: Current robot and environment state

        Returns:
            Feasibility score [0, 1]
        """
        # Simplified feasibility check
        # Real implementation would check:
        # - Reachability
        # - Collision-free path exists
        # - Gripper can grasp object
        # - Stability after placement

        if action.target_position is None:
            return 1.0

        # Check if position is in workspace
        robot_pos = current_state.get('robot_position', np.array([0, 0, 0]))
        max_reach = current_state.get('max_reach', 0.8)

        distance = np.linalg.norm(action.target_position - robot_pos)

        if distance > max_reach:
            return 0.0

        # Linear falloff with distance
        feasibility = 1.0 - (distance / max_reach) * 0.5

        return np.clip(feasibility, 0.0, 1.0)

    def replan_if_needed(
        self,
        current_plan: Plan,
        execution_feedback: Dict
    ) -> Optional[Plan]:
        """
        Check if replanning is needed based on execution feedback.

        Args:
            current_plan: Currently executing plan
            execution_feedback: Feedback from execution

        Returns:
            New plan if replanning needed, None otherwise
        """
        confidence = execution_feedback.get('confidence', 1.0)

        if confidence < self.replan_threshold:
            # Replanning needed
            print(f"[INFO] Replanning due to low confidence: {confidence:.2f}")
            return None  # Return signal to replan

        return None

    def visualize_plan(self, plan: Plan) -> str:
        """
        Create text visualization of plan.

        Returns:
            String representation of plan
        """
        lines = [
            f"=== PLAN ===",
            f"Goal: {plan.goal}",
            f"Actions: {len(plan.actions)}",
            f"Total Duration: {plan.total_duration:.1f}s",
            f"Confidence: {plan.confidence:.2f}",
            "",
            "Steps:"
        ]

        for i, action in enumerate(plan.actions, 1):
            action_str = f"{i}. {action.action_type.value}"

            if action.target_position is not None:
                pos = action.target_position
                action_str += f" -> ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"

            if action.target_object_id is not None:
                action_str += f" [obj_id={action.target_object_id}]"

            action_str += f" ({action.estimated_duration:.1f}s)"

            lines.append(action_str)

        return "\n".join(lines)
