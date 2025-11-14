"""
Scene Graph

Represents spatial relationships between objects in the environment.
Enables reasoning about object interactions and spatial queries.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class SpatialRelation(Enum):
    """Types of spatial relationships between objects."""
    ON_TOP_OF = "on_top_of"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    NEAR = "near"
    FAR = "far"
    INSIDE = "inside"
    CONTAINS = "contains"
    TOUCHING = "touching"


@dataclass
class SceneNode:
    """Node in scene graph representing an object."""
    node_id: int
    object_id: int  # Reference to object in ObjectDatabase
    class_name: str
    position: np.ndarray  # (x, y, z)
    bbox_min: np.ndarray  # (x, y, z)
    bbox_max: np.ndarray  # (x, y, z)


@dataclass
class SceneEdge:
    """Edge in scene graph representing a spatial relationship."""
    source_id: int
    target_id: int
    relation: SpatialRelation
    confidence: float = 1.0


class SceneGraph:
    """
    Scene graph for spatial reasoning.

    Represents objects as nodes and their spatial relationships as edges.
    """

    def __init__(self):
        """Initialize scene graph."""
        self.nodes: Dict[int, SceneNode] = {}  # node_id -> SceneNode
        self.edges: List[SceneEdge] = []
        self.next_node_id = 1

    def add_node(
        self,
        object_id: int,
        class_name: str,
        position: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray
    ) -> int:
        """
        Add node to scene graph.

        Args:
            object_id: Reference to ObjectDatabase
            class_name: Object class
            position: 3D position (x, y, z)
            bbox_min: Bounding box minimum
            bbox_max: Bounding box maximum

        Returns:
            node_id
        """
        node = SceneNode(
            node_id=self.next_node_id,
            object_id=object_id,
            class_name=class_name,
            position=position,
            bbox_min=bbox_min,
            bbox_max=bbox_max
        )

        self.nodes[node.node_id] = node
        self.next_node_id += 1

        return node.node_id

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        relation: SpatialRelation,
        confidence: float = 1.0
    ):
        """
        Add edge (spatial relationship) to scene graph.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation: Type of spatial relationship
            confidence: Confidence in relationship [0, 1]
        """
        edge = SceneEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            confidence=confidence
        )
        self.edges.append(edge)

    def compute_all_relations(self):
        """
        Compute spatial relationships between all pairs of nodes.

        Automatically detects and adds relationships based on geometric analysis.
        """
        self.edges.clear()

        node_list = list(self.nodes.values())

        for i, node1 in enumerate(node_list):
            for node2 in node_list[i + 1:]:
                relations = self._compute_pairwise_relations(node1, node2)

                for relation, confidence in relations:
                    self.add_edge(node1.node_id, node2.node_id, relation, confidence)

    def _compute_pairwise_relations(
        self,
        node1: SceneNode,
        node2: SceneNode
    ) -> List[Tuple[SpatialRelation, float]]:
        """
        Compute spatial relationships between two nodes.

        Returns:
            List of (relation, confidence) tuples
        """
        relations = []

        pos1 = node1.position
        pos2 = node2.position
        bbox1_min, bbox1_max = node1.bbox_min, node1.bbox_max
        bbox2_min, bbox2_max = node2.bbox_min, node2.bbox_max

        # Distance
        distance = np.linalg.norm(pos2 - pos1)

        # Near/far threshold (can be tuned)
        near_threshold = 0.5  # meters

        if distance < near_threshold:
            relations.append((SpatialRelation.NEAR, 1.0))
        else:
            relations.append((SpatialRelation.FAR, 1.0))

        # Vertical relationships (on top of / below)
        vertical_threshold = 0.05  # meters

        if bbox1_max[2] < bbox2_min[2] - vertical_threshold:
            # node1 is below node2
            relations.append((SpatialRelation.BELOW, 0.9))
        elif bbox2_max[2] < bbox1_min[2] - vertical_threshold:
            # node1 is on top of node2
            relations.append((SpatialRelation.ON_TOP_OF, 0.9))

        # Horizontal relationships (left/right)
        if pos2[0] > pos1[0] + 0.1:
            relations.append((SpatialRelation.RIGHT_OF, 0.8))
        elif pos2[0] < pos1[0] - 0.1:
            relations.append((SpatialRelation.LEFT_OF, 0.8))

        # Front/behind (y-axis)
        if pos2[1] > pos1[1] + 0.1:
            relations.append((SpatialRelation.IN_FRONT_OF, 0.8))
        elif pos2[1] < pos1[1] - 0.1:
            relations.append((SpatialRelation.BEHIND, 0.8))

        # Touching (close bounding boxes)
        touching_threshold = 0.02  # 2cm

        if self._boxes_close(bbox1_min, bbox1_max, bbox2_min, bbox2_max, touching_threshold):
            relations.append((SpatialRelation.TOUCHING, 0.7))

        # Containment
        if self._box_contains(bbox1_min, bbox1_max, bbox2_min, bbox2_max):
            relations.append((SpatialRelation.CONTAINS, 0.9))
        elif self._box_contains(bbox2_min, bbox2_max, bbox1_min, bbox1_max):
            relations.append((SpatialRelation.INSIDE, 0.9))

        return relations

    def _boxes_close(
        self,
        bbox1_min: np.ndarray,
        bbox1_max: np.ndarray,
        bbox2_min: np.ndarray,
        bbox2_max: np.ndarray,
        threshold: float
    ) -> bool:
        """Check if two bounding boxes are close to each other."""
        # Minimum distance between boxes
        dist = 0.0

        for i in range(3):
            if bbox1_max[i] < bbox2_min[i]:
                dist += (bbox2_min[i] - bbox1_max[i]) ** 2
            elif bbox2_max[i] < bbox1_min[i]:
                dist += (bbox1_min[i] - bbox2_max[i]) ** 2

        return np.sqrt(dist) < threshold

    def _box_contains(
        self,
        outer_min: np.ndarray,
        outer_max: np.ndarray,
        inner_min: np.ndarray,
        inner_max: np.ndarray
    ) -> bool:
        """Check if outer box contains inner box."""
        return np.all(inner_min >= outer_min) and np.all(inner_max <= outer_max)

    def query_relations(
        self,
        source_id: int,
        relation: Optional[SpatialRelation] = None
    ) -> List[SceneEdge]:
        """
        Query relationships from a specific node.

        Args:
            source_id: Source node ID
            relation: Optional filter for specific relation type

        Returns:
            List of matching edges
        """
        edges = [e for e in self.edges if e.source_id == source_id]

        if relation is not None:
            edges = [e for e in edges if e.relation == relation]

        return edges

    def find_objects_with_relation(
        self,
        source_id: int,
        relation: SpatialRelation
    ) -> List[SceneNode]:
        """
        Find all objects with specific relation to source object.

        Args:
            source_id: Source node ID
            relation: Spatial relationship to query

        Returns:
            List of related nodes
        """
        edges = self.query_relations(source_id, relation)
        related_nodes = [self.nodes[e.target_id] for e in edges if e.target_id in self.nodes]
        return related_nodes

    def get_objects_on_surface(self, surface_node_id: int) -> List[SceneNode]:
        """Get all objects on top of a surface."""
        return self.find_objects_with_relation(surface_node_id, SpatialRelation.ON_TOP_OF)

    def get_nearby_objects(
        self,
        node_id: int,
        max_distance: float = 0.5
    ) -> List[Tuple[SceneNode, float]]:
        """
        Get objects near a specific node.

        Args:
            node_id: Reference node ID
            max_distance: Maximum distance threshold (meters)

        Returns:
            List of (node, distance) tuples
        """
        if node_id not in self.nodes:
            return []

        ref_node = self.nodes[node_id]
        nearby = []

        for node in self.nodes.values():
            if node.node_id == node_id:
                continue

            distance = np.linalg.norm(node.position - ref_node.position)

            if distance <= max_distance:
                nearby.append((node, distance))

        # Sort by distance
        nearby.sort(key=lambda x: x[1])

        return nearby

    def clear(self):
        """Clear scene graph."""
        self.nodes.clear()
        self.edges.clear()
        self.next_node_id = 1

    def get_statistics(self) -> dict:
        """Get scene graph statistics."""
        relation_counts = {}
        for edge in self.edges:
            rel = edge.relation.value
            relation_counts[rel] = relation_counts.get(rel, 0) + 1

        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'relation_counts': relation_counts
        }

    def export_graph(self) -> dict:
        """
        Export scene graph as dictionary.

        Returns:
            Dictionary representation of graph
        """
        nodes_dict = {
            node_id: {
                'object_id': node.object_id,
                'class_name': node.class_name,
                'position': node.position.tolist(),
                'bbox_min': node.bbox_min.tolist(),
                'bbox_max': node.bbox_max.tolist()
            }
            for node_id, node in self.nodes.items()
        }

        edges_list = [
            {
                'source_id': edge.source_id,
                'target_id': edge.target_id,
                'relation': edge.relation.value,
                'confidence': edge.confidence
            }
            for edge in self.edges
        ]

        return {
            'nodes': nodes_dict,
            'edges': edges_list
        }
