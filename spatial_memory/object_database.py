"""
Object Database

Persistent storage of detected objects with 3D poses and properties.
Uses SQLite for efficient querying and persistence.
"""

import sqlite3
import json
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import os


@dataclass
class ObjectRecord:
    """Record of a detected object in 3D space."""
    object_id: Optional[int] = None  # Auto-generated
    class_name: str = ""
    confidence: float = 0.0

    # 3D pose
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x, y, z) in meters
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # Quaternion (x, y, z, w)

    # Bounding box (world coordinates)
    bbox_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Metadata
    first_seen: str = ""
    last_seen: str = ""
    num_observations: int = 0

    # Optional properties
    color: Optional[Tuple[int, int, int]] = None  # RGB
    size: Optional[Tuple[float, float, float]] = None  # (width, height, depth)
    properties: Optional[Dict] = None  # Additional properties as JSON


class ObjectDatabase:
    """
    Persistent object database using SQLite.

    Stores and queries detected objects in 3D space.
    """

    def __init__(self, database_path: str = "data/maps/object_database.db"):
        """
        Initialize object database.

        Args:
            database_path: Path to SQLite database file
        """
        self.database_path = database_path

        # Create directory if needed
        os.makedirs(os.path.dirname(database_path), exist_ok=True)

        # Connect to database
        self.conn = sqlite3.connect(database_path)
        self.cursor = self.conn.cursor()

        # Create tables
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS objects (
                object_id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_name TEXT NOT NULL,
                confidence REAL,

                pos_x REAL,
                pos_y REAL,
                pos_z REAL,

                orient_x REAL,
                orient_y REAL,
                orient_z REAL,
                orient_w REAL,

                bbox_min_x REAL,
                bbox_min_y REAL,
                bbox_min_z REAL,
                bbox_max_x REAL,
                bbox_max_y REAL,
                bbox_max_z REAL,

                first_seen TEXT,
                last_seen TEXT,
                num_observations INTEGER,

                color_r INTEGER,
                color_g INTEGER,
                color_b INTEGER,

                size_w REAL,
                size_h REAL,
                size_d REAL,

                properties TEXT
            )
        ''')

        # Create index on class_name for faster queries
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_class_name ON objects(class_name)
        ''')

        self.conn.commit()

    def add_object(self, obj: ObjectRecord) -> int:
        """
        Add object to database.

        Args:
            obj: ObjectRecord

        Returns:
            object_id of inserted object
        """
        now = datetime.now().isoformat()

        if not obj.first_seen:
            obj.first_seen = now
        obj.last_seen = now

        # Serialize properties
        properties_json = json.dumps(obj.properties) if obj.properties else None

        # Insert
        self.cursor.execute('''
            INSERT INTO objects (
                class_name, confidence,
                pos_x, pos_y, pos_z,
                orient_x, orient_y, orient_z, orient_w,
                bbox_min_x, bbox_min_y, bbox_min_z,
                bbox_max_x, bbox_max_y, bbox_max_z,
                first_seen, last_seen, num_observations,
                color_r, color_g, color_b,
                size_w, size_h, size_d,
                properties
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            obj.class_name, obj.confidence,
            obj.position[0], obj.position[1], obj.position[2],
            obj.orientation[0], obj.orientation[1], obj.orientation[2], obj.orientation[3],
            obj.bbox_min[0], obj.bbox_min[1], obj.bbox_min[2],
            obj.bbox_max[0], obj.bbox_max[1], obj.bbox_max[2],
            obj.first_seen, obj.last_seen, obj.num_observations,
            obj.color[0] if obj.color else None,
            obj.color[1] if obj.color else None,
            obj.color[2] if obj.color else None,
            obj.size[0] if obj.size else None,
            obj.size[1] if obj.size else None,
            obj.size[2] if obj.size else None,
            properties_json
        ))

        self.conn.commit()

        return self.cursor.lastrowid

    def update_object(self, obj: ObjectRecord):
        """
        Update existing object in database.

        Args:
            obj: ObjectRecord with valid object_id
        """
        if obj.object_id is None:
            raise ValueError("object_id must be set for update")

        obj.last_seen = datetime.now().isoformat()

        properties_json = json.dumps(obj.properties) if obj.properties else None

        self.cursor.execute('''
            UPDATE objects SET
                class_name = ?,
                confidence = ?,
                pos_x = ?, pos_y = ?, pos_z = ?,
                orient_x = ?, orient_y = ?, orient_z = ?, orient_w = ?,
                bbox_min_x = ?, bbox_min_y = ?, bbox_min_z = ?,
                bbox_max_x = ?, bbox_max_y = ?, bbox_max_z = ?,
                last_seen = ?,
                num_observations = ?,
                color_r = ?, color_g = ?, color_b = ?,
                size_w = ?, size_h = ?, size_d = ?,
                properties = ?
            WHERE object_id = ?
        ''', (
            obj.class_name, obj.confidence,
            obj.position[0], obj.position[1], obj.position[2],
            obj.orientation[0], obj.orientation[1], obj.orientation[2], obj.orientation[3],
            obj.bbox_min[0], obj.bbox_min[1], obj.bbox_min[2],
            obj.bbox_max[0], obj.bbox_max[1], obj.bbox_max[2],
            obj.last_seen, obj.num_observations,
            obj.color[0] if obj.color else None,
            obj.color[1] if obj.color else None,
            obj.color[2] if obj.color else None,
            obj.size[0] if obj.size else None,
            obj.size[1] if obj.size else None,
            obj.size[2] if obj.size else None,
            properties_json,
            obj.object_id
        ))

        self.conn.commit()

    def get_object(self, object_id: int) -> Optional[ObjectRecord]:
        """Get object by ID."""
        self.cursor.execute('SELECT * FROM objects WHERE object_id = ?', (object_id,))
        row = self.cursor.fetchone()

        if row:
            return self._row_to_object(row)
        return None

    def get_objects_by_class(self, class_name: str) -> List[ObjectRecord]:
        """Get all objects of specific class."""
        self.cursor.execute('SELECT * FROM objects WHERE class_name = ?', (class_name,))
        rows = self.cursor.fetchall()

        return [self._row_to_object(row) for row in rows]

    def get_objects_in_region(
        self,
        min_bound: Tuple[float, float, float],
        max_bound: Tuple[float, float, float]
    ) -> List[ObjectRecord]:
        """
        Get objects within spatial region.

        Args:
            min_bound: (x_min, y_min, z_min)
            max_bound: (x_max, y_max, z_max)

        Returns:
            List of objects in region
        """
        self.cursor.execute('''
            SELECT * FROM objects WHERE
                pos_x >= ? AND pos_x <= ? AND
                pos_y >= ? AND pos_y <= ? AND
                pos_z >= ? AND pos_z <= ?
        ''', (
            min_bound[0], max_bound[0],
            min_bound[1], max_bound[1],
            min_bound[2], max_bound[2]
        ))

        rows = self.cursor.fetchall()
        return [self._row_to_object(row) for row in rows]

    def get_nearest_object(
        self,
        position: Tuple[float, float, float],
        class_name: Optional[str] = None
    ) -> Optional[ObjectRecord]:
        """
        Get nearest object to position.

        Args:
            position: (x, y, z)
            class_name: Optional class filter

        Returns:
            Nearest object or None
        """
        if class_name:
            self.cursor.execute('SELECT * FROM objects WHERE class_name = ?', (class_name,))
        else:
            self.cursor.execute('SELECT * FROM objects')

        rows = self.cursor.fetchall()

        if not rows:
            return None

        # Find nearest
        min_distance = float('inf')
        nearest = None

        for row in rows:
            obj = self._row_to_object(row)
            distance = np.linalg.norm(np.array(obj.position) - np.array(position))

            if distance < min_distance:
                min_distance = distance
                nearest = obj

        return nearest

    def delete_object(self, object_id: int):
        """Delete object from database."""
        self.cursor.execute('DELETE FROM objects WHERE object_id = ?', (object_id,))
        self.conn.commit()

    def clear_all(self):
        """Delete all objects from database."""
        self.cursor.execute('DELETE FROM objects')
        self.conn.commit()

    def get_all_objects(self) -> List[ObjectRecord]:
        """Get all objects in database."""
        self.cursor.execute('SELECT * FROM objects')
        rows = self.cursor.fetchall()
        return [self._row_to_object(row) for row in rows]

    def get_statistics(self) -> dict:
        """Get database statistics."""
        self.cursor.execute('SELECT COUNT(*) FROM objects')
        total_objects = self.cursor.fetchone()[0]

        self.cursor.execute('SELECT class_name, COUNT(*) FROM objects GROUP BY class_name')
        class_counts = dict(self.cursor.fetchall())

        return {
            'total_objects': total_objects,
            'class_counts': class_counts,
            'database_path': self.database_path
        }

    def _row_to_object(self, row) -> ObjectRecord:
        """Convert database row to ObjectRecord."""
        properties = json.loads(row[26]) if row[26] else None

        return ObjectRecord(
            object_id=row[0],
            class_name=row[1],
            confidence=row[2],
            position=(row[3], row[4], row[5]),
            orientation=(row[6], row[7], row[8], row[9]),
            bbox_min=(row[10], row[11], row[12]),
            bbox_max=(row[13], row[14], row[15]),
            first_seen=row[16],
            last_seen=row[17],
            num_observations=row[18],
            color=(row[19], row[20], row[21]) if row[19] is not None else None,
            size=(row[22], row[23], row[24]) if row[22] is not None else None,
            properties=properties
        )

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
