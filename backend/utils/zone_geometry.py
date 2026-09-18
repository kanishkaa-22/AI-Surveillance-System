"""
Shared geometry checks used by both zone types.

LINE zone (2 points) -> side_of_line(): which side of the line a point is on.
  Used to detect CROSSING (entry/exit through a doorway).

RECTANGLE / polygon zone (4+ points) -> point_in_polygon(): is a point
  currently inside the shape. Used to detect PRESENCE (someone standing in
  a restricted area), not crossing.
"""

import cv2
import numpy as np


def side_of_line(point, line_p1, line_p2):
    x, y = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def point_in_polygon(point, polygon_points):
    """
    True if `point` (x, y) is inside the shape defined by polygon_points
    (a list of (x, y) corners, e.g. your 4 clicked rectangle corners).
    Works for any simple polygon, not just rectangles.
    """
    contour = np.array(polygon_points, dtype=np.int32)
    result = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
    return result >= 0