"""Shared vision primitives (M7): every vision subsystem (OCR, diagram
parsing, equipment detection, gauge reading, defect detection) locates its
output with the same BoundingBox shape, so the Vision Workspace can overlay
all of them consistently and the Vision Knowledge Graph can store all of
them as the same BoundingBox node type."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[int, int]:
        return (self.x_min + self.x_max) // 2, (self.y_min + self.y_max) // 2
