"""
YOLO overlay widget for selection box and annotation boxes.
"""

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRect, QPoint, QSize, Signal
from PySide6.QtGui import QPainter, QPen, QColor


class YoloOverlay(QWidget):
    """Overlay widget for YOLO labeling with selection box and annotation boxes."""

    # Size mode constants
    SIZE_MODE_IMAGE = "image"  # Size scales with image zoom (ratios)
    SIZE_MODE_SCREEN = "screen"  # Size stays fixed on screen (pixels)

    # Interaction modes
    MODE_SELECT = "select"  # Select/move selection box or annotations
    MODE_DRAW = "draw"  # Draw new annotation boxes

    # Handle types for resizing
    HANDLE_NONE = None
    HANDLE_TL = "top_left"
    HANDLE_TR = "top_right"
    HANDLE_BL = "bottom_left"
    HANDLE_BR = "bottom_right"
    HANDLE_T = "top"
    HANDLE_B = "bottom"
    HANDLE_L = "left"
    HANDLE_R = "right"

    # Signal emitted when selection box position changes
    selection_changed = Signal()
    # Signal emitted when annotations change
    annotations_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

        # Size mode
        self.size_mode = self.SIZE_MODE_IMAGE

        # Whether selection is locked (set) - drawing only allowed when locked
        self.selection_locked = False

        # Interaction mode
        self.interaction_mode = self.MODE_SELECT

        # For IMAGE mode: selection box as ratios (0.0-1.0) relative to image size
        # Stored as (x_ratio, y_ratio, width_ratio, height_ratio)
        self.selection_ratios = None

        # For SCREEN mode: selection position as ratios, but size as fixed pixels
        # Position stored as (x_ratio, y_ratio) - center point relative to image
        self.selection_position_ratios = None  # (center_x_ratio, center_y_ratio)
        self.selection_fixed_size = None  # QSize in screen pixels

        # Current displayed image size
        self.image_size = None

        # Offset of image within the label (due to centering)
        self.image_offset = QPoint(0, 0)

        # Dragging state for selection box
        self.dragging_selection = False
        self.drag_start = QPoint()
        self.drag_start_ratios = None
        self.drag_start_position_ratios = None

        # Annotation boxes
        # List of dicts: {'class_id': int, 'color': QColor, 'name': str, 'ratios': (x, y, w, h)}
        self.annotations = []

        # Currently selected annotation index (for editing)
        self.selected_annotation = None

        # Drawing state for new annotations
        self.drawing_annotation = False
        self.draw_start = QPoint()
        self.draw_current = QPoint()
        self.current_draw_label = None  # {'class_id', 'color', 'name'}

        # Dragging/resizing annotation state
        self.dragging_annotation = False
        self.resizing_annotation = False
        self.resize_handle = self.HANDLE_NONE
        self.annotation_drag_start = QPoint()
        self.annotation_original_ratios = None

        # Handle size for hit detection
        self.handle_size = 10

    def set_size_mode(self, mode):
        """Set the size mode (image or screen)."""
        self.size_mode = mode

    def set_selection_locked(self, locked):
        """Set whether the selection is locked (drawing only allowed when locked)."""
        self.selection_locked = locked

    def set_draw_label(self, label_info):
        """Set the current label for drawing annotations.

        Args:
            label_info: dict with 'class_id', 'color', 'name' or None to disable drawing
        """
        self.current_draw_label = label_info
        if label_info:
            self.interaction_mode = self.MODE_DRAW
        else:
            self.interaction_mode = self.MODE_SELECT

    def set_selection_size(self, display_size, image_size, image_offset):
        """Set the selection box size.

        Args:
            display_size: QSize of selection box in display coordinates
            image_size: Current displayed image size (pixmap size)
            image_offset: QPoint offset of image within the label
        """
        self.image_size = image_size
        self.image_offset = image_offset

        if display_size is None or image_size is None or image_size.isEmpty():
            self.selection_ratios = None
            self.selection_position_ratios = None
            self.selection_fixed_size = None
            self.hide()
            return

        self.show()

        if self.size_mode == self.SIZE_MODE_IMAGE:
            # Image mode: store everything as ratios
            if self.selection_ratios is None:
                # Initialize centered in image
                width_ratio = min(display_size.width() / image_size.width(), 1.0)
                height_ratio = min(display_size.height() / image_size.height(), 1.0)
                x_ratio = (1.0 - width_ratio) / 2
                y_ratio = (1.0 - height_ratio) / 2
                self.selection_ratios = (x_ratio, y_ratio, width_ratio, height_ratio)
        else:
            # Screen mode: store fixed size in pixels, position as ratios
            self.selection_fixed_size = QSize(display_size)
            if self.selection_position_ratios is None:
                # Initialize centered
                self.selection_position_ratios = (0.5, 0.5)

        self.update()

    def _get_selection_rect_display(self):
        """Get selection rect in display (widget) coordinates."""
        if self.image_size is None:
            return QRect()

        if self.size_mode == self.SIZE_MODE_IMAGE:
            # Image mode: derive from ratios
            if self.selection_ratios is None:
                return QRect()

            x_ratio, y_ratio, w_ratio, h_ratio = self.selection_ratios

            # Convert ratios to display coordinates, adding image offset
            x = int(x_ratio * self.image_size.width()) + self.image_offset.x()
            y = int(y_ratio * self.image_size.height()) + self.image_offset.y()
            w = int(w_ratio * self.image_size.width())
            h = int(h_ratio * self.image_size.height())

            return QRect(x, y, w, h)
        else:
            # Screen mode: fixed size, position from ratios
            if (
                self.selection_position_ratios is None
                or self.selection_fixed_size is None
            ):
                return QRect()

            center_x_ratio, center_y_ratio = self.selection_position_ratios
            w = self.selection_fixed_size.width()
            h = self.selection_fixed_size.height()

            # Center position in display coordinates
            center_x = (
                int(center_x_ratio * self.image_size.width()) + self.image_offset.x()
            )
            center_y = (
                int(center_y_ratio * self.image_size.height()) + self.image_offset.y()
            )

            # Top-left from center
            x = center_x - w // 2
            y = center_y - h // 2

            return QRect(x, y, w, h)

    def _get_image_rect_display(self):
        """Get the image rectangle in display (widget) coordinates."""
        if self.image_size is None:
            return QRect()
        return QRect(
            self.image_offset.x(),
            self.image_offset.y(),
            self.image_size.width(),
            self.image_size.height(),
        )

    def get_selection_rect(self):
        """Get the current selection rectangle in display coordinates."""
        return self._get_selection_rect_display()

    def get_selection_ratios(self):
        """Get selection position as ratios relative to image."""
        return self.selection_ratios

    def clear_selection(self):
        """Clear the selection box."""
        self.selection_ratios = None
        self.selection_position_ratios = None
        self.selection_fixed_size = None
        self.hide()

    def clear_annotations(self):
        """Clear all annotation boxes."""
        self.annotations = []
        self.selected_annotation = None
        self.update()

    def add_annotation(self, class_id, color, name, ratios):
        """Add a new annotation box.

        Args:
            class_id: Integer class ID
            color: QColor for the box
            name: String label name
            ratios: Tuple (x, y, w, h) as ratios relative to selection box
        """
        self.annotations.append(
            {
                "class_id": class_id,
                "color": QColor(color) if isinstance(color, str) else color,
                "name": name,
                "ratios": ratios,
            }
        )
        self.selected_annotation = len(self.annotations) - 1
        self.update()
        self.annotations_changed.emit()

    def delete_selected_annotation(self):
        """Delete the currently selected annotation."""
        if self.selected_annotation is not None and 0 <= self.selected_annotation < len(
            self.annotations
        ):
            del self.annotations[self.selected_annotation]
            self.selected_annotation = None
            self.update()
            self.annotations_changed.emit()
            return True
        return False

    def delete_annotations_by_class(self, class_id):
        """Delete all annotations with the given class ID."""
        original_count = len(self.annotations)
        self.annotations = [
            ann for ann in self.annotations if ann["class_id"] != class_id
        ]

        # Clear selection if selected annotation was deleted
        if self.selected_annotation is not None:
            if self.selected_annotation >= len(self.annotations):
                self.selected_annotation = None

        if len(self.annotations) < original_count:
            self.update()
            self.annotations_changed.emit()
            return True
        return False

    def get_annotations(self):
        """Get all annotations."""
        return self.annotations.copy()

    def set_annotations(self, annotations):
        """Set annotations from saved data."""
        self.annotations = []
        for ann in annotations:
            self.annotations.append(
                {
                    "class_id": ann["class_id"],
                    "color": QColor(ann["color"])
                    if isinstance(ann["color"], str)
                    else ann["color"],
                    "name": ann.get("name", ""),
                    "ratios": tuple(ann["ratios"]),
                }
            )
        self.selected_annotation = None
        self.update()

    def _get_annotation_rect_display(self, annotation):
        """Get annotation rect in display coordinates."""
        selection_rect = self._get_selection_rect_display()
        if selection_rect.isNull():
            return QRect()

        ratios = annotation.get("ratios", (0, 0, 0, 0))
        x = selection_rect.x() + int(ratios[0] * selection_rect.width())
        y = selection_rect.y() + int(ratios[1] * selection_rect.height())
        w = int(ratios[2] * selection_rect.width())
        h = int(ratios[3] * selection_rect.height())

        return QRect(x, y, w, h)

    def _display_to_selection_ratios(self, rect):
        """Convert display rect to ratios relative to selection box."""
        selection_rect = self._get_selection_rect_display()
        if (
            selection_rect.isNull()
            or selection_rect.width() == 0
            or selection_rect.height() == 0
        ):
            return (0, 0, 0, 0)

        x_ratio = (rect.x() - selection_rect.x()) / selection_rect.width()
        y_ratio = (rect.y() - selection_rect.y()) / selection_rect.height()
        w_ratio = rect.width() / selection_rect.width()
        h_ratio = rect.height() / selection_rect.height()

        return (x_ratio, y_ratio, w_ratio, h_ratio)

    def _get_handle_at_pos(self, pos, rect):
        """Check if position is on a resize handle of the given rect."""
        hs = self.handle_size

        # Corner handles
        if QRect(rect.left() - hs // 2, rect.top() - hs // 2, hs, hs).contains(pos):
            return self.HANDLE_TL
        if QRect(rect.right() - hs // 2, rect.top() - hs // 2, hs, hs).contains(pos):
            return self.HANDLE_TR
        if QRect(rect.left() - hs // 2, rect.bottom() - hs // 2, hs, hs).contains(pos):
            return self.HANDLE_BL
        if QRect(rect.right() - hs // 2, rect.bottom() - hs // 2, hs, hs).contains(pos):
            return self.HANDLE_BR

        # Edge handles
        mid_x = rect.center().x()
        mid_y = rect.center().y()
        if QRect(mid_x - hs // 2, rect.top() - hs // 2, hs, hs).contains(pos):
            return self.HANDLE_T
        if QRect(mid_x - hs // 2, rect.bottom() - hs // 2, hs, hs).contains(pos):
            return self.HANDLE_B
        if QRect(rect.left() - hs // 2, mid_y - hs // 2, hs, hs).contains(pos):
            return self.HANDLE_L
        if QRect(rect.right() - hs // 2, mid_y - hs // 2, hs, hs).contains(pos):
            return self.HANDLE_R

        return self.HANDLE_NONE

    def _get_cursor_for_handle(self, handle):
        """Get the appropriate cursor for a resize handle."""
        if handle in (self.HANDLE_TL, self.HANDLE_BR):
            return Qt.CursorShape.SizeFDiagCursor
        elif handle in (self.HANDLE_TR, self.HANDLE_BL):
            return Qt.CursorShape.SizeBDiagCursor
        elif handle in (self.HANDLE_T, self.HANDLE_B):
            return Qt.CursorShape.SizeVerCursor
        elif handle in (self.HANDLE_L, self.HANDLE_R):
            return Qt.CursorShape.SizeHorCursor
        return Qt.CursorShape.ArrowCursor

    def paintEvent(self, event):
        """Draw the selection box and annotation boxes."""
        has_selection = (
            self.size_mode == self.SIZE_MODE_IMAGE and self.selection_ratios is not None
        ) or (
            self.size_mode == self.SIZE_MODE_SCREEN
            and self.selection_position_ratios is not None
        )

        if not has_selection and not self.annotations and not self.drawing_annotation:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw selection box if present
        if has_selection:
            self._draw_selection_box(painter)

        # Draw annotation boxes
        selection_rect = self._get_selection_rect_display()
        for i, annotation in enumerate(self.annotations):
            self._draw_annotation_box(
                painter, annotation, selection_rect, i == self.selected_annotation
            )

        # Draw annotation being created
        if self.drawing_annotation and self.current_draw_label:
            self._draw_drawing_box(painter, selection_rect)

    def _draw_selection_box(self, painter):
        """Draw the red selection box with semi-transparent overlay outside."""
        selection_rect = self._get_selection_rect_display()
        if selection_rect.isNull():
            return

        image_rect = self._get_image_rect_display()

        # Draw semi-transparent dark overlay around selection area (within image bounds)
        overlay_color = QColor(0, 0, 0, 100)

        # Only draw overlay within the image area
        if not image_rect.isNull():
            # Top rectangle (within image)
            if selection_rect.top() > image_rect.top():
                painter.fillRect(
                    image_rect.left(),
                    image_rect.top(),
                    image_rect.width(),
                    selection_rect.top() - image_rect.top(),
                    overlay_color,
                )

            # Bottom rectangle (within image)
            if selection_rect.bottom() < image_rect.bottom():
                painter.fillRect(
                    image_rect.left(),
                    selection_rect.bottom(),
                    image_rect.width(),
                    image_rect.bottom() - selection_rect.bottom(),
                    overlay_color,
                )

            # Left rectangle (within image, between top and bottom overlays)
            if selection_rect.left() > image_rect.left():
                painter.fillRect(
                    image_rect.left(),
                    selection_rect.top(),
                    selection_rect.left() - image_rect.left(),
                    selection_rect.height(),
                    overlay_color,
                )

            # Right rectangle (within image, between top and bottom overlays)
            if selection_rect.right() < image_rect.right():
                painter.fillRect(
                    selection_rect.right(),
                    selection_rect.top(),
                    image_rect.right() - selection_rect.right(),
                    selection_rect.height(),
                    overlay_color,
                )

        # Draw red border around selection area
        pen = QPen(QColor(255, 0, 0), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(selection_rect)

        # Draw corner indicators (small squares at corners)
        corner_size = 8
        painter.setBrush(QColor(255, 0, 0))
        corners = [
            selection_rect.topLeft(),
            selection_rect.topRight(),
            selection_rect.bottomLeft(),
            selection_rect.bottomRight(),
        ]
        for corner in corners:
            painter.drawRect(
                corner.x() - corner_size // 2,
                corner.y() - corner_size // 2,
                corner_size,
                corner_size,
            )

    def _draw_annotation_box(self, painter, annotation, selection_rect, is_selected):
        """Draw a single annotation box."""
        if selection_rect.isNull():
            return

        box_rect = self._get_annotation_rect_display(annotation)
        color = annotation.get("color", QColor(0, 255, 0))
        name = annotation.get("name", "")

        # Draw filled rectangle with transparency
        fill_color = QColor(color)
        fill_color.setAlpha(50)
        painter.fillRect(box_rect, fill_color)

        # Draw border
        pen_width = 3 if is_selected else 2
        pen = QPen(color, pen_width)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(box_rect)

        # Draw label text in top-left corner
        if name:
            # Background for text
            font = painter.font()
            font.setPointSize(10)
            painter.setFont(font)
            text_rect = painter.fontMetrics().boundingRect(name)
            text_bg = QRect(
                box_rect.x() + 2,
                box_rect.y() + 2,
                text_rect.width() + 6,
                text_rect.height() + 4,
            )
            painter.fillRect(text_bg, color)

            # Text
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                box_rect.x() + 5, box_rect.y() + 2 + text_rect.height(), name
            )

        # Draw resize handles if selected
        if is_selected:
            self._draw_annotation_handles(painter, box_rect, color)

    def _draw_annotation_handles(self, painter, rect, color):
        """Draw resize handles for selected annotation box."""
        handle_size = 8
        painter.setBrush(color)
        painter.setPen(QPen(Qt.GlobalColor.white, 1))

        # Corner handles
        corners = [
            rect.topLeft(),
            rect.topRight(),
            rect.bottomLeft(),
            rect.bottomRight(),
        ]
        for corner in corners:
            painter.drawRect(
                corner.x() - handle_size // 2,
                corner.y() - handle_size // 2,
                handle_size,
                handle_size,
            )

        # Edge handles
        mid_x = rect.center().x()
        mid_y = rect.center().y()
        edges = [
            QPoint(mid_x, rect.top()),
            QPoint(mid_x, rect.bottom()),
            QPoint(rect.left(), mid_y),
            QPoint(rect.right(), mid_y),
        ]
        for edge in edges:
            painter.drawRect(
                edge.x() - handle_size // 2,
                edge.y() - handle_size // 2,
                handle_size,
                handle_size,
            )

    def _draw_drawing_box(self, painter, selection_rect):
        """Draw the box currently being drawn."""
        if not self.current_draw_label:
            return

        color = self.current_draw_label.get("color", QColor(0, 255, 0))
        name = self.current_draw_label.get("name", "")

        # Calculate rect from draw points
        x1, y1 = self.draw_start.x(), self.draw_start.y()
        x2, y2 = self.draw_current.x(), self.draw_current.y()

        # Constrain to selection box
        x1 = max(selection_rect.left(), min(selection_rect.right(), x1))
        y1 = max(selection_rect.top(), min(selection_rect.bottom(), y1))
        x2 = max(selection_rect.left(), min(selection_rect.right(), x2))
        y2 = max(selection_rect.top(), min(selection_rect.bottom(), y2))

        box_rect = QRect(
            QPoint(min(x1, x2), min(y1, y2)), QPoint(max(x1, x2), max(y1, y2))
        )

        # Draw with dashed line to indicate it's being created
        fill_color = QColor(color)
        fill_color.setAlpha(30)
        painter.fillRect(box_rect, fill_color)

        pen = QPen(color, 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(box_rect)

        # Draw label
        if name and box_rect.width() > 20 and box_rect.height() > 20:
            font = painter.font()
            font.setPointSize(10)
            painter.setFont(font)
            text_rect = painter.fontMetrics().boundingRect(name)
            text_bg = QRect(
                box_rect.x() + 2,
                box_rect.y() + 2,
                text_rect.width() + 6,
                text_rect.height() + 4,
            )
            painter.fillRect(text_bg, color)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                box_rect.x() + 5, box_rect.y() + 2 + text_rect.height(), name
            )

    def mousePressEvent(self, event):
        """Handle mouse press for various interactions."""
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return

        pos = event.position().toPoint()
        selection_rect = self._get_selection_rect_display()

        # ALWAYS check for existing annotations first (for selection/moving/resizing)
        # Check if clicking on a selected annotation's handle
        if self.selected_annotation is not None and 0 <= self.selected_annotation < len(
            self.annotations
        ):
            ann_rect = self._get_annotation_rect_display(
                self.annotations[self.selected_annotation]
            )
            handle = self._get_handle_at_pos(pos, ann_rect)
            if handle != self.HANDLE_NONE:
                self.resizing_annotation = True
                self.resize_handle = handle
                self.annotation_drag_start = pos
                self.annotation_original_ratios = self.annotations[
                    self.selected_annotation
                ]["ratios"]
                self.setCursor(self._get_cursor_for_handle(handle))
                event.accept()
                return

        # Check if clicking inside an annotation (for selection/moving)
        for i in range(
            len(self.annotations) - 1, -1, -1
        ):  # Check top-to-bottom (latest first)
            ann_rect = self._get_annotation_rect_display(self.annotations[i])
            if ann_rect.contains(pos):
                self.selected_annotation = i
                self.dragging_annotation = True
                self.annotation_drag_start = pos
                self.annotation_original_ratios = self.annotations[i]["ratios"]
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                self.update()
                event.accept()
                return

        # Clicking outside annotations deselects any selected annotation
        if self.selected_annotation is not None:
            self.selected_annotation = None
            self.update()

        # Check if in draw mode and clicking inside selection rect (only if not on an annotation)
        # Drawing is only allowed when selection is locked (set)
        if (
            self.interaction_mode == self.MODE_DRAW
            and self.current_draw_label
            and self.selection_locked
        ):
            if not selection_rect.isNull() and selection_rect.contains(pos):
                self.drawing_annotation = True
                self.draw_start = pos
                self.draw_current = pos
                self.setCursor(Qt.CursorShape.CrossCursor)
                event.accept()
                return

        # Check if clicking inside selection rect (for moving selection box)
        if not selection_rect.isNull() and selection_rect.contains(pos):
            self.dragging_selection = True
            self.drag_start = pos
            if self.size_mode == self.SIZE_MODE_IMAGE:
                self.drag_start_ratios = self.selection_ratios
            else:
                self.drag_start_position_ratios = self.selection_position_ratios
            self.setCursor(Qt.CursorShape.SizeAllCursor)
            event.accept()
            return

        # Not on anything - let parent handle it (for panning)
        event.ignore()

    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging and drawing."""
        pos = event.position().toPoint()
        selection_rect = self._get_selection_rect_display()

        # Handle drawing annotation
        if self.drawing_annotation:
            self.draw_current = pos
            self.update()
            event.accept()
            return

        # Handle resizing annotation
        if self.resizing_annotation and self.selected_annotation is not None:
            self._resize_annotation(pos)
            self.update()
            event.accept()
            return

        # Handle dragging annotation
        if self.dragging_annotation and self.selected_annotation is not None:
            self._move_annotation(pos)
            self.update()
            event.accept()
            return

        # Handle dragging selection box
        if self.dragging_selection:
            self._move_selection(pos)
            self.update()
            self.selection_changed.emit()
            event.accept()
            return

        # Update cursor based on position
        cursor = Qt.CursorShape.ArrowCursor

        # Check annotation handles first
        if self.selected_annotation is not None and 0 <= self.selected_annotation < len(
            self.annotations
        ):
            ann_rect = self._get_annotation_rect_display(
                self.annotations[self.selected_annotation]
            )
            handle = self._get_handle_at_pos(pos, ann_rect)
            if handle != self.HANDLE_NONE:
                cursor = self._get_cursor_for_handle(handle)

        # Check if over an annotation
        if cursor == Qt.CursorShape.ArrowCursor:
            for ann in self.annotations:
                ann_rect = self._get_annotation_rect_display(ann)
                if ann_rect.contains(pos):
                    cursor = Qt.CursorShape.SizeAllCursor
                    break

        # Check if in draw mode inside selection
        if (
            cursor == Qt.CursorShape.ArrowCursor
            and self.interaction_mode == self.MODE_DRAW
        ):
            if not selection_rect.isNull() and selection_rect.contains(pos):
                cursor = Qt.CursorShape.CrossCursor

        # Check if over selection box
        if cursor == Qt.CursorShape.ArrowCursor:
            if not selection_rect.isNull() and selection_rect.contains(pos):
                cursor = Qt.CursorShape.SizeAllCursor

        self.setCursor(cursor)
        event.ignore()

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Finish drawing annotation
            if self.drawing_annotation:
                self._finish_drawing_annotation()
                self.drawing_annotation = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
                event.accept()
                return

            # Finish resizing annotation
            if self.resizing_annotation:
                self.resizing_annotation = False
                self.resize_handle = self.HANDLE_NONE
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self.annotations_changed.emit()
                event.accept()
                return

            # Finish dragging annotation
            if self.dragging_annotation:
                self.dragging_annotation = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self.annotations_changed.emit()
                event.accept()
                return

            # Finish dragging selection
            if self.dragging_selection:
                self.dragging_selection = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self.selection_changed.emit()
                event.accept()
                return

        event.ignore()

    def _finish_drawing_annotation(self):
        """Finish drawing a new annotation and add it."""
        if not self.current_draw_label:
            return

        selection_rect = self._get_selection_rect_display()
        if selection_rect.isNull():
            return

        # Calculate rect from draw points
        x1, y1 = self.draw_start.x(), self.draw_start.y()
        x2, y2 = self.draw_current.x(), self.draw_current.y()

        # Constrain to selection box
        x1 = max(selection_rect.left(), min(selection_rect.right(), x1))
        y1 = max(selection_rect.top(), min(selection_rect.bottom(), y1))
        x2 = max(selection_rect.left(), min(selection_rect.right(), x2))
        y2 = max(selection_rect.top(), min(selection_rect.bottom(), y2))

        box_rect = QRect(
            QPoint(min(x1, x2), min(y1, y2)), QPoint(max(x1, x2), max(y1, y2))
        )

        # Minimum size check
        if box_rect.width() < 10 or box_rect.height() < 10:
            self.update()
            return

        # Convert to ratios relative to selection box
        ratios = self._display_to_selection_ratios(box_rect)

        # Add annotation
        self.add_annotation(
            self.current_draw_label["class_id"],
            self.current_draw_label["color"],
            self.current_draw_label["name"],
            ratios,
        )

    def _move_annotation(self, pos):
        """Move the selected annotation based on mouse position."""
        if self.selected_annotation is None or self.annotation_original_ratios is None:
            return

        selection_rect = self._get_selection_rect_display()
        if (
            selection_rect.isNull()
            or selection_rect.width() == 0
            or selection_rect.height() == 0
        ):
            return

        # Calculate delta in selection ratios
        delta_x = (pos.x() - self.annotation_drag_start.x()) / selection_rect.width()
        delta_y = (pos.y() - self.annotation_drag_start.y()) / selection_rect.height()

        orig = self.annotation_original_ratios
        new_x = orig[0] + delta_x
        new_y = orig[1] + delta_y

        # Clamp to keep within selection box
        new_x = max(0.0, min(1.0 - orig[2], new_x))
        new_y = max(0.0, min(1.0 - orig[3], new_y))

        self.annotations[self.selected_annotation]["ratios"] = (
            new_x,
            new_y,
            orig[2],
            orig[3],
        )

    def _resize_annotation(self, pos):
        """Resize the selected annotation based on mouse position and handle."""
        if self.selected_annotation is None or self.annotation_original_ratios is None:
            return

        selection_rect = self._get_selection_rect_display()
        if (
            selection_rect.isNull()
            or selection_rect.width() == 0
            or selection_rect.height() == 0
        ):
            return

        orig = self.annotation_original_ratios
        orig_rect = QRect(
            int(orig[0] * selection_rect.width()),
            int(orig[1] * selection_rect.height()),
            int(orig[2] * selection_rect.width()),
            int(orig[3] * selection_rect.height()),
        )

        # Calculate delta
        delta_x = pos.x() - self.annotation_drag_start.x()
        delta_y = pos.y() - self.annotation_drag_start.y()

        new_rect = QRect(orig_rect)

        # Apply delta based on handle
        if self.resize_handle in (self.HANDLE_TL, self.HANDLE_L, self.HANDLE_BL):
            new_rect.setLeft(orig_rect.left() + delta_x)
        if self.resize_handle in (self.HANDLE_TR, self.HANDLE_R, self.HANDLE_BR):
            new_rect.setRight(orig_rect.right() + delta_x)
        if self.resize_handle in (self.HANDLE_TL, self.HANDLE_T, self.HANDLE_TR):
            new_rect.setTop(orig_rect.top() + delta_y)
        if self.resize_handle in (self.HANDLE_BL, self.HANDLE_B, self.HANDLE_BR):
            new_rect.setBottom(orig_rect.bottom() + delta_y)

        # Ensure minimum size
        min_size = 10
        if new_rect.width() < min_size:
            if self.resize_handle in (self.HANDLE_TL, self.HANDLE_L, self.HANDLE_BL):
                new_rect.setLeft(new_rect.right() - min_size)
            else:
                new_rect.setRight(new_rect.left() + min_size)
        if new_rect.height() < min_size:
            if self.resize_handle in (self.HANDLE_TL, self.HANDLE_T, self.HANDLE_TR):
                new_rect.setTop(new_rect.bottom() - min_size)
            else:
                new_rect.setBottom(new_rect.top() + min_size)

        # Clamp to selection box bounds
        new_rect.setLeft(max(0, new_rect.left()))
        new_rect.setTop(max(0, new_rect.top()))
        new_rect.setRight(min(selection_rect.width(), new_rect.right()))
        new_rect.setBottom(min(selection_rect.height(), new_rect.bottom()))

        # Convert back to ratios
        new_ratios = (
            new_rect.x() / selection_rect.width(),
            new_rect.y() / selection_rect.height(),
            new_rect.width() / selection_rect.width(),
            new_rect.height() / selection_rect.height(),
        )

        self.annotations[self.selected_annotation]["ratios"] = new_ratios

    def _move_selection(self, pos):
        """Move selection box based on mouse position.

        Bounds depend on whether selection is larger or smaller than image:
        - Selection smaller than image: keep selection inside image
        - Selection larger than image: keep image inside selection (allow free positioning)
        """
        if self.image_size is None:
            return

        # Calculate delta in display coordinates
        delta = pos - self.drag_start

        # Convert delta to ratio delta
        delta_x_ratio = delta.x() / self.image_size.width()
        delta_y_ratio = delta.y() / self.image_size.height()

        if self.size_mode == self.SIZE_MODE_IMAGE:
            if self.drag_start_ratios is None:
                return

            # Apply delta to original ratios
            x_ratio = self.drag_start_ratios[0] + delta_x_ratio
            y_ratio = self.drag_start_ratios[1] + delta_y_ratio
            w_ratio = self.drag_start_ratios[2]
            h_ratio = self.drag_start_ratios[3]

            # Clamp to keep within image bounds (0.0 to 1.0 - size)
            x_ratio = max(0.0, min(1.0 - w_ratio, x_ratio))
            y_ratio = max(0.0, min(1.0 - h_ratio, y_ratio))

            self.selection_ratios = (x_ratio, y_ratio, w_ratio, h_ratio)
        else:
            # Screen mode: move center position
            if (
                self.drag_start_position_ratios is None
                or self.selection_fixed_size is None
            ):
                return

            # Apply delta to center position
            center_x_ratio = self.drag_start_position_ratios[0] + delta_x_ratio
            center_y_ratio = self.drag_start_position_ratios[1] + delta_y_ratio

            # Calculate half-size as ratios
            half_w_ratio = (
                self.selection_fixed_size.width() / 2
            ) / self.image_size.width()
            half_h_ratio = (
                self.selection_fixed_size.height() / 2
            ) / self.image_size.height()

            # Check if selection is larger than image in each dimension
            selection_wider = (
                self.selection_fixed_size.width() >= self.image_size.width()
            )
            selection_taller = (
                self.selection_fixed_size.height() >= self.image_size.height()
            )

            if selection_wider:
                # Selection is wider than image: image must stay inside selection
                min_x = 1.0 - half_w_ratio
                max_x = half_w_ratio
                center_x_ratio = max(min_x, min(max_x, center_x_ratio))
            else:
                # Selection is narrower than image: selection must stay inside image
                center_x_ratio = max(
                    half_w_ratio, min(1.0 - half_w_ratio, center_x_ratio)
                )

            if selection_taller:
                # Selection is taller than image: image must stay inside selection
                min_y = 1.0 - half_h_ratio
                max_y = half_h_ratio
                center_y_ratio = max(min_y, min(max_y, center_y_ratio))
            else:
                # Selection is shorter than image: selection must stay inside image
                center_y_ratio = max(
                    half_h_ratio, min(1.0 - half_h_ratio, center_y_ratio)
                )

            self.selection_position_ratios = (center_x_ratio, center_y_ratio)
