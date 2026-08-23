"""
Crop overlay widget for image cropping functionality.
"""

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QPainter, QPen, QColor


class CropOverlay(QWidget):
    """Overlay widget that displays and manages crop rectangle."""

    HANDLE_SIZE = 10
    HANDLE_CORNER = 0
    HANDLE_EDGE = 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

        # Crop rectangle (in widget coordinates)
        self.crop_rect = QRect()

        # Crop rectangle as ratios (0.0-1.0) relative to image size
        # Stored as (x_ratio, y_ratio, width_ratio, height_ratio)
        self.crop_ratios = None

        # Crop mode: None, "rectangle", or fixed sizes like "1024x1024"
        self.crop_mode = None

        # For fixed-size modes: whether the box represents a constant number
        # of real image pixels ("image", zoom-invariant — what you want for
        # producing consistently-sized extracts) or a constant number of
        # on-screen pixels ("screen" — the box shrinks/grows on screen as
        # you zoom, tracking a variable amount of real image content).
        self.align_mode = "image"

        # True pixel dimensions of the source image (not the possibly-zoomed
        # displayed pixmap) — needed to make "image" alignment zoom-invariant.
        self.native_size = None

        # Current image size for ratio calculations
        self.image_size = None

        # Dragging state
        self.dragging = False
        self.drag_start = QPoint()
        self.drag_handle = (
            None  # (type, position) - type is HANDLE_CORNER or HANDLE_EDGE
        )

        # Original rect for resize operations
        self.original_rect = QRect()

    def set_crop_mode(self, mode, image_size, native_size=None, align_mode="image"):
        """Set crop mode and initialize crop rectangle.

        native_size/align_mode only matter for fixed-size modes (e.g.
        "1024x1024"): with align_mode="image" the box is resized on every
        call to represent exactly that many *native* image pixels regardless
        of the current zoom level; with "screen" it stays a constant number
        of display pixels instead (so the real crop size varies with zoom).
        """
        # Reset stored ratios when the mode or alignment changes so we always
        # initialise fresh geometry rather than restoring a stale shape.
        if mode != self.crop_mode or align_mode != self.align_mode:
            self.crop_ratios = None

        self.crop_mode = mode
        self.align_mode = align_mode
        self.native_size = native_size
        self.image_size = image_size

        if mode is None:
            self.crop_rect = QRect()
            self.crop_ratios = None
            self.hide()
            return

        self.show()

        if mode in ("rectangle", "square"):
            # If we have existing ratios (same mode, window resize) restore them.
            if self.crop_ratios is not None:
                self._apply_ratios_to_rect()
            else:
                if mode == "rectangle":
                    width = int(image_size.width() * 0.3)
                    height = int(image_size.height() * 0.3)
                else:  # square
                    side = int(min(image_size.width(), image_size.height()) * 0.3)
                    width = height = side
                x = (image_size.width() - width) // 2
                y = (image_size.height() - height) // 2
                self.crop_rect = QRect(x, y, width, height)
                self._update_ratios_from_rect()
        else:
            # Fixed size mode like "1024x1024". Size is always recomputed
            # fresh (never restored from stale ratios) so it stays accurate
            # to the requested dimensions under align_mode="image"; only the
            # box's center position is preserved across zoom changes.
            parts = mode.split("x")
            req_w, req_h = int(parts[0]), int(parts[1])

            if align_mode == "image" and native_size and not native_size.isEmpty():
                scale_x = image_size.width() / native_size.width()
                scale_y = image_size.height() / native_size.height()
                width = max(1, round(req_w * scale_x))
                height = max(1, round(req_h * scale_y))
            else:
                width = req_w
                height = req_h

            width = min(width, image_size.width())
            height = min(height, image_size.height())

            if self.crop_ratios is not None:
                cx = (
                    self.crop_ratios[0] + self.crop_ratios[2] / 2
                ) * image_size.width()
                cy = (
                    self.crop_ratios[1] + self.crop_ratios[3] / 2
                ) * image_size.height()
            else:
                cx = image_size.width() / 2
                cy = image_size.height() / 2

            x = int(cx - width / 2)
            y = int(cy - height / 2)
            x = max(0, min(x, image_size.width() - width))
            y = max(0, min(y, image_size.height() - height))
            self.crop_rect = QRect(x, y, width, height)
            self._update_ratios_from_rect()

        self.update()

    def _update_ratios_from_rect(self):
        """Update stored ratios from current crop_rect."""
        if self.image_size and not self.image_size.isEmpty():
            self.crop_ratios = (
                self.crop_rect.x() / self.image_size.width(),
                self.crop_rect.y() / self.image_size.height(),
                self.crop_rect.width() / self.image_size.width(),
                self.crop_rect.height() / self.image_size.height(),
            )

    def _apply_ratios_to_rect(self):
        """Apply stored ratios to create crop_rect for current image size."""
        if self.crop_ratios and self.image_size and not self.image_size.isEmpty():
            x = int(self.crop_ratios[0] * self.image_size.width())
            y = int(self.crop_ratios[1] * self.image_size.height())
            width = int(self.crop_ratios[2] * self.image_size.width())
            height = int(self.crop_ratios[3] * self.image_size.height())
            self.crop_rect = QRect(x, y, width, height)

    def get_crop_rect(self):
        """Get the current crop rectangle."""
        return QRect(self.crop_rect)

    def paintEvent(self, event):
        """Draw the crop overlay."""
        if self.crop_mode is None or self.crop_rect.isNull():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw semi-transparent dark overlay around crop area (but not inside)
        # Top rectangle
        if self.crop_rect.top() > 0:
            painter.fillRect(
                0, 0, self.width(), self.crop_rect.top(), QColor(0, 0, 0, 128)
            )

        # Bottom rectangle
        if self.crop_rect.bottom() < self.height():
            painter.fillRect(
                0,
                self.crop_rect.bottom(),
                self.width(),
                self.height() - self.crop_rect.bottom(),
                QColor(0, 0, 0, 128),
            )

        # Left rectangle
        painter.fillRect(
            0,
            self.crop_rect.top(),
            self.crop_rect.left(),
            self.crop_rect.height(),
            QColor(0, 0, 0, 128),
        )

        # Right rectangle
        if self.crop_rect.right() < self.width():
            painter.fillRect(
                self.crop_rect.right(),
                self.crop_rect.top(),
                self.width() - self.crop_rect.right(),
                self.crop_rect.height(),
                QColor(0, 0, 0, 128),
            )

        # Draw red border around crop area
        pen = QPen(QColor(255, 0, 0), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)  # No fill
        painter.drawRect(self.crop_rect)

        # Draw resize handles for free-form modes
        if self.crop_mode in ("rectangle", "square"):
            self.draw_handles(painter)

    def draw_handles(self, painter):
        """Draw resize handles. Square mode gets corners only; rectangle gets all eight."""
        painter.setBrush(QColor(255, 0, 0))
        size = self.HANDLE_SIZE

        corners = [
            self.crop_rect.topLeft(),
            self.crop_rect.topRight(),
            self.crop_rect.bottomLeft(),
            self.crop_rect.bottomRight(),
        ]
        for corner in corners:
            painter.drawRect(corner.x() - size // 2, corner.y() - size // 2, size, size)

        if self.crop_mode == "rectangle":
            edges = [
                QPoint(self.crop_rect.center().x(), self.crop_rect.top()),
                QPoint(self.crop_rect.center().x(), self.crop_rect.bottom()),
                QPoint(self.crop_rect.left(), self.crop_rect.center().y()),
                QPoint(self.crop_rect.right(), self.crop_rect.center().y()),
            ]
            for edge in edges:
                painter.drawRect(edge.x() - size // 2, edge.y() - size // 2, size, size)

    def mousePressEvent(self, event):
        """Handle mouse press for dragging and resizing."""
        if event.button() != Qt.MouseButton.LeftButton:
            return

        pos = event.position().toPoint()

        # Check if clicking on a handle (free-form modes)
        if self.crop_mode in ("rectangle", "square"):
            handle = self.get_handle_at_pos(pos)
            if handle:
                self.dragging = True
                self.drag_handle = handle
                self.drag_start = pos
                self.original_rect = QRect(self.crop_rect)
                return

        # Check if clicking inside crop rect (for moving)
        if self.crop_rect.contains(pos):
            self.dragging = True
            self.drag_handle = None
            self.drag_start = pos
            self.original_rect = QRect(self.crop_rect)
            self.setCursor(Qt.CursorShape.SizeAllCursor)

    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging and resizing."""
        pos = event.position().toPoint()

        if self.dragging:
            delta = pos - self.drag_start

            if self.drag_handle:
                if self.crop_mode == "square":
                    self.resize_crop_rect_square(delta)
                else:
                    self.resize_crop_rect(delta)
            else:
                self.move_crop_rect(delta)

            self.update()
        else:
            if self.crop_mode in ("rectangle", "square"):
                handle = self.get_handle_at_pos(pos)
                if handle:
                    self.set_cursor_for_handle(handle)
                elif self.crop_rect.contains(pos):
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)
            else:
                if self.crop_rect.contains(pos):
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.drag_handle = None
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def get_handle_at_pos(self, pos):
        """Check if position is on a resize handle, return (type, position) or None."""
        size = self.HANDLE_SIZE * 1.5  # Slightly larger hit area

        # Check corners
        corners = [
            (self.crop_rect.topLeft(), "top_left"),
            (self.crop_rect.topRight(), "top_right"),
            (self.crop_rect.bottomLeft(), "bottom_left"),
            (self.crop_rect.bottomRight(), "bottom_right"),
        ]

        for corner_pos, corner_name in corners:
            if QRect(
                corner_pos.x() - size // 2, corner_pos.y() - size // 2, size, size
            ).contains(pos):
                return (self.HANDLE_CORNER, corner_name)

        # Edge handles only for rectangle mode (square uses corners only)
        if self.crop_mode == "rectangle":
            edges = [
                (QPoint(self.crop_rect.center().x(), self.crop_rect.top()), "top"),
                (
                    QPoint(self.crop_rect.center().x(), self.crop_rect.bottom()),
                    "bottom",
                ),
                (QPoint(self.crop_rect.left(), self.crop_rect.center().y()), "left"),
                (QPoint(self.crop_rect.right(), self.crop_rect.center().y()), "right"),
            ]
            for edge_pos, edge_name in edges:
                if QRect(
                    edge_pos.x() - size // 2, edge_pos.y() - size // 2, size, size
                ).contains(pos):
                    return (self.HANDLE_EDGE, edge_name)

        return None

    def set_cursor_for_handle(self, handle):
        """Set appropriate cursor for resize handle."""
        if handle is None:
            return

        handle_type, position = handle

        if handle_type == self.HANDLE_CORNER:
            if position in ("top_left", "bottom_right"):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            else:
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif handle_type == self.HANDLE_EDGE:
            if position in ("top", "bottom"):
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            else:
                self.setCursor(Qt.CursorShape.SizeHorCursor)

    def move_crop_rect(self, delta):
        """Move crop rectangle, keeping it within bounds."""
        new_rect = QRect(self.original_rect)
        new_rect.translate(delta)

        # Keep within widget bounds
        if new_rect.left() < 0:
            new_rect.moveLeft(0)
        if new_rect.top() < 0:
            new_rect.moveTop(0)
        if new_rect.right() > self.width():
            new_rect.moveRight(self.width())
        if new_rect.bottom() > self.height():
            new_rect.moveBottom(self.height())

        self.crop_rect = new_rect
        self._update_ratios_from_rect()

    def resize_crop_rect(self, delta):
        """Resize crop rectangle based on handle being dragged."""
        handle_type, position = self.drag_handle
        new_rect = QRect(self.original_rect)

        if handle_type == self.HANDLE_CORNER:
            if "top" in position:
                new_rect.setTop(new_rect.top() + delta.y())
            if "bottom" in position:
                new_rect.setBottom(new_rect.bottom() + delta.y())
            if "left" in position:
                new_rect.setLeft(new_rect.left() + delta.x())
            if "right" in position:
                new_rect.setRight(new_rect.right() + delta.x())
        elif handle_type == self.HANDLE_EDGE:
            if position == "top":
                new_rect.setTop(new_rect.top() + delta.y())
            elif position == "bottom":
                new_rect.setBottom(new_rect.bottom() + delta.y())
            elif position == "left":
                new_rect.setLeft(new_rect.left() + delta.x())
            elif position == "right":
                new_rect.setRight(new_rect.right() + delta.x())

        # Ensure minimum size
        min_size = 20
        if new_rect.width() < min_size:
            if "left" in position:
                new_rect.setLeft(new_rect.right() - min_size)
            else:
                new_rect.setRight(new_rect.left() + min_size)

        if new_rect.height() < min_size:
            if "top" in position:
                new_rect.setTop(new_rect.bottom() - min_size)
            else:
                new_rect.setBottom(new_rect.top() + min_size)

        # Keep within bounds
        if new_rect.left() < 0:
            new_rect.setLeft(0)
        if new_rect.top() < 0:
            new_rect.setTop(0)
        if new_rect.right() > self.width():
            new_rect.setRight(self.width())
        if new_rect.bottom() > self.height():
            new_rect.setBottom(self.height())

        self.crop_rect = new_rect
        self._update_ratios_from_rect()

    def resize_crop_rect_square(self, delta):
        """Resize crop rectangle maintaining a 1:1 aspect ratio.

        Only corner handles are used.  The fixed corner (diagonally opposite
        to the one being dragged) stays anchored.  The dominant axis of the
        drag (whichever of dx/dy has the larger absolute value) determines the
        new side length so the box feels natural to drag.
        """
        _, position = self.drag_handle
        r = self.original_rect
        dx, dy = delta.x(), delta.y()
        min_side = 20

        if position == "bottom_right":
            # Fixed corner: top-left.
            d = dx if abs(dx) >= abs(dy) else dy
            side = max(min_side, r.width() + d)
            side = min(side, self.width() - r.left(), self.height() - r.top())
            new_rect = QRect(r.left(), r.top(), side, side)

        elif position == "top_left":
            # Fixed corner: bottom-right.
            d = dx if abs(dx) >= abs(dy) else dy
            side = max(min_side, r.width() - d)
            side = min(side, r.right(), r.bottom())
            new_rect = QRect(r.right() - side, r.bottom() - side, side, side)

        elif position == "bottom_left":
            # Fixed corner: top-right. Left edge moves (−dx grows), bottom moves (+dy grows).
            d = -dx if abs(dx) >= abs(dy) else dy
            side = max(min_side, r.width() + d)
            side = min(side, r.right(), self.height() - r.top())
            new_rect = QRect(r.right() - side, r.top(), side, side)

        elif position == "top_right":
            # Fixed corner: bottom-left. Right edge moves (+dx grows), top moves (−dy grows).
            d = dx if abs(dx) >= abs(dy) else -dy
            side = max(min_side, r.width() + d)
            side = min(side, self.width() - r.left(), r.bottom())
            new_rect = QRect(r.left(), r.bottom() - side, side, side)

        else:
            return

        self.crop_rect = new_rect
        self._update_ratios_from_rect()
