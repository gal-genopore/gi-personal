
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QToolBar, QPushButton,
                             QLabel, QSlider, QWidget, QHBoxLayout, QVBoxLayout,
                             QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene,
                             QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsRectItem,
                             QGraphicsPolygonItem, QGraphicsTextItem, QGraphicsPathItem)
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import (QPixmap, QPainter, QPen, QBrush, QColor, QPolygonF, QImage,
                         QFont, QTransform, QPainterPath)
from PIL import Image, ImageQt, ImageDraw
import math
import os
import logging
try:
    import numpy as np
except ImportError:
    print("NumPy is required. Please install it using 'pip install numpy'.")
    sys.exit(1)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
                    datefmt='%S')
logger = logging.getLogger(__name__)

# --- Constants ---
DIE_NAMES = [
    "0:No_NP", "1:MX_2x2", "2:MX_3x3", "3:MX_4x4", "4:32x16", " :Test1",
    "5:Oval", "6:2x2", "7:3x3", "8:4x4", "9:100x4", " :Test2",
    "A:D130", "B:D140", "C:D150", "D:D160", "E:D170", " :Test3"
]
NUM_COLS = 3
NUM_ROWS = 6
SCP_SIZE = 4


class CustomGraphicsView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.is_panning = False
        self.pan_start_pos = QPointF()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            factor = 1.1
        else:
            factor = 0.9
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = True
            self.pan_start_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_panning:
            delta = event.pos() - self.pan_start_pos
            self.pan_start_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class ImageAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wafer Annotation Tool (PyQt6)")
        self.setGeometry(100, 100, 1200, 800)

        # Image and annotation data
        self.original_image = None
        self.original_image_path = None
        self.mask_layer = None
        self.annotation_layer = None
        self.circle_geom = {'center': None, 'radius': None}
        self.rectangle_geom = None
        self.super_control_points = {}
        self.interpolated_points = {}
        self.die_info_cache = {}
        self.die_origin_shift = (0, 0)
        self.Max_C = 0
        self.Max_R = 0

        # State variables
        self.mode = None
        self.circle_points = []
        self.rect_start_pos = None
        self.active_scp = None
        self.last_mask_pos = None
        self.brush_size = 20

        # Graphics items
        self.image_item = None
        self.mask_item = None
        self.annotation_item = None
        self.temp_items = []
        self.committed_grid_items = []
        self.live_ffd_grid_items = []
        self.scp_items = {}

        self.create_ui()
        self.load_image(initial=True)

    def create_ui(self):
        # Toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        toolbar.addAction("Load Image", lambda: self.load_image(initial=False))
        toolbar.addAction("Save Image", self.save_image)
        toolbar.addSeparator()

        self.mode_label = QLabel("Mode: Idle")
        toolbar.addWidget(self.mode_label)

        toolbar.addAction("Circle (3 pts)", lambda: self.set_mode('circle'))
        toolbar.addAction("Die Dimension", lambda: self.set_mode('rectangle'))
        toolbar.addAction("Mask", lambda: self.set_mode('mask'))

        self.ffd_mode_button = QPushButton("Edit Grid")
        self.ffd_mode_button.setCheckable(True)
        self.ffd_mode_button.toggled.connect(self.toggle_ffd_mode)
        toolbar.addWidget(self.ffd_mode_button)

        self.apply_ffd_button = QPushButton("APPLY GRID")
        self.apply_ffd_button.setEnabled(False)
        self.apply_ffd_button.clicked.connect(self._commit_ffd_changes)
        toolbar.addWidget(self.apply_ffd_button)

        toolbar.addAction("Set Die Origin", lambda: self.set_mode('set_naming_origin'))
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("Brush Size:"))
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(5, 150)
        self.brush_slider.setValue(20)
        self.brush_slider.valueChanged.connect(self.update_brush_size)
        toolbar.addWidget(self.brush_slider)

        toolbar.addAction("Clear Mask", self.clear_mask)
        toolbar.addSeparator()
        toolbar.addAction("Generate Report", self.generate_report)

        # Central widget and scene
        self.scene = QGraphicsScene()
        self.view = CustomGraphicsView(self.scene)
        self.setCentralWidget(self.view)
        self.scene.mousePressEvent = self.on_scene_mouse_press
        self.scene.mouseMoveEvent = self.on_scene_mouse_move
        self.scene.mouseReleaseEvent = self.on_scene_mouse_release

    def load_image(self, initial=False):
        if not initial:
            filepath, _ = QFileDialog.getOpenFileName(self, "Select Image File", "",
                                                      "Image Files (*.png *.jpg *.jpeg *.tif *.tiff)")
            if not filepath:
                return
        else:
            filepath = None

        try:
            if filepath:
                self.original_image = QImage(filepath)
                self.original_image_path = filepath
            else:
                self.original_image = QImage(800, 600, QImage.Format.Format_RGB32)
                self.original_image.fill(QColor('darkgrey'))
                self.original_image_path = "Placeholder_Image"

            self.scene.clear()
            self.image_item = QGraphicsPixmapItem(QPixmap.fromImage(self.original_image))
            self.scene.addItem(self.image_item)
            self.view.setSceneRect(self.image_item.boundingRect())

            W, H = self.original_image.width(), self.original_image.height()
            self.mask_layer = QImage(W, H, QImage.Format.Format_ARGB32)
            self.mask_layer.fill(Qt.GlobalColor.transparent)
            self.mask_item = QGraphicsPixmapItem(QPixmap.fromImage(self.mask_layer))
            self.scene.addItem(self.mask_item)

            self.annotation_layer = QImage(W, H, QImage.Format.Format_ARGB32)
            self.annotation_layer.fill(Qt.GlobalColor.transparent)
            self.annotation_item = QGraphicsPixmapItem(QPixmap.fromImage(self.annotation_layer))
            self.scene.addItem(self.annotation_item)

            # Reset state
            self.circle_geom = {'center': None, 'radius': None}
            self.rectangle_geom = None
            self.super_control_points = {}
            self.die_info_cache = {}
            self.die_origin_shift = (0, 0)
            self.view.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.set_mode(None)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {e}")

    def set_mode(self, mode):
        if self.mode == 'ffd_grid' and mode != 'ffd_grid':
            self._commit_ffd_changes()
            self.ffd_mode_button.setChecked(False)

        self.mode = mode
        self.circle_points = []
        self.rect_start_pos = None
        self.clear_temp_items()
        self.active_scp = None

        mode_text = {
            'circle': 'Circle - Click 3 points',
            'rectangle': 'Die Dimension - Click and drag',
            'mask': 'Mask - Paint over image',
            'ffd_grid': 'FFD Mesh - Drag blue control points',
            'set_naming_origin': 'Set Naming Origin - Click a die',
            None: 'Idle'
        }.get(mode, 'Idle')
        self.mode_label.setText(f"Mode: {mode_text}")

        if mode == 'ffd_grid':
            if not self.super_control_points:
                if not self._initialize_ffd_mesh():
                    self.set_mode(None) # Revert if initialization fails
                    self.ffd_mode_button.setChecked(False)
            self._draw_live_ffd_grid()
        else:
            self._clear_live_ffd_grid()


    def toggle_ffd_mode(self, checked):
        if checked:
            self.set_mode('ffd_grid')
        else:
            self.set_mode(None)

    def update_brush_size(self, value):
        self.brush_size = value

    def on_scene_mouse_press(self, event):
        pos = event.scenePos()

        if self.mode == 'circle':
            self.circle_points.append(pos)
            self.temp_items.append(self.scene.addEllipse(pos.x()-5, pos.y()-5, 10, 10,
                                                        QPen(QColor('white')), QBrush(QColor('red'))))
            if len(self.circle_points) == 3:
                self.draw_circle_from_points()
                self.circle_points = []
                self.clear_temp_items()

        elif self.mode == 'rectangle':
            self.rect_start_pos = pos
            rect_item = QGraphicsRectItem(QRectF(pos, pos))
            rect_item.setPen(QPen(QColor('blue'), 2))
            self.temp_items.append(self.scene.addItem(rect_item))

        elif self.mode == 'mask':
            if not self.circle_geom['radius']:
                QMessageBox.warning(self, "Mask Error", "Please define a circle first.")
                return
            self.last_mask_pos = pos
            self.paint_mask_stroke(pos, pos)

        elif self.mode == 'ffd_grid':
            self.active_scp = None
            for key, item in self.scp_items.items():
                if item.isUnderMouse():
                    self.active_scp = key
                    item.setBrush(QBrush(QColor('yellow')))
                    break

        elif self.mode == 'set_naming_origin':
            clicked_die = self._find_clicked_die(pos)
            if clicked_die:
                self.die_origin_shift = clicked_die
                logger.info(f"Die naming origin set to {clicked_die}")
                self._draw_committed_mesh()
                self.set_mode(None)

    def on_scene_mouse_move(self, event):
        pos = event.scenePos()

        if self.mode == 'rectangle' and self.rect_start_pos:
            rect_item = self.temp_items[0]
            rect_item.setRect(QRectF(self.rect_start_pos, pos).normalized())

        elif self.mode == 'mask' and self.last_mask_pos:
            self.paint_mask_stroke(self.last_mask_pos, pos)
            self.last_mask_pos = pos

        elif self.mode == 'ffd_grid' and self.active_scp:
            self.super_control_points[self.active_scp] = pos
            self.scp_items[self.active_scp].setPos(pos)
            self._draw_live_ffd_grid()
            self.apply_ffd_button.setEnabled(True)

    def on_scene_mouse_release(self, event):
        pos = event.scenePos()

        if self.mode == 'rectangle' and self.rect_start_pos:
            self.rectangle_geom = QRectF(self.rect_start_pos, pos).normalized()
            self.rect_start_pos = None
            self.clear_temp_items()
            self._rebuild_annotation_layer()
            self.super_control_points = {} # Invalidate old grid
            self.die_info_cache = {}

        elif self.mode == 'mask':
            self.last_mask_pos = None

        elif self.mode == 'ffd_grid' and self.active_scp:
            self.scp_items[self.active_scp].setBrush(QBrush(QColor('blue')))
            self.active_scp = None

    def _rebuild_annotation_layer(self):
        if not self.original_image: return
        self.annotation_layer.fill(Qt.GlobalColor.transparent)
        painter = QPainter(self.annotation_layer)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.circle_geom['radius']:
            center = self.circle_geom['center']
            radius = self.circle_geom['radius']
            painter.setPen(QPen(QColor(255, 0, 0, 100), 3))
            painter.drawEllipse(center, radius, radius)

        if self.rectangle_geom:
            painter.setPen(QPen(QColor(0, 0, 255, 100), 3))
            painter.drawRect(self.rectangle_geom)

        painter.end()
        self.annotation_item.setPixmap(QPixmap.fromImage(self.annotation_layer))


    def _calculate_all_interpolated_points(self):
        if not self.super_control_points:
            self.interpolated_points = {}
            return

        self.interpolated_points = {}
        num_patches_c = SCP_SIZE - 1
        num_patches_r = SCP_SIZE - 1

        for C in range(self.Max_C + 1):
            for R in range(self.Max_R + 1):
                u = C / self.Max_C if self.Max_C > 0 else 0
                v = R / self.Max_R if self.Max_R > 0 else 0

                C_s_float = u * num_patches_c
                R_s_float = v * num_patches_r

                C_s = min(num_patches_c - 1, int(C_s_float))
                R_s = min(num_patches_r - 1, int(R_s_float))

                u_local = C_s_float - C_s
                v_local = R_s_float - R_s

                P00 = self.super_control_points.get((C_s, R_s), QPointF(0, 0))
                P10 = self.super_control_points.get((C_s + 1, R_s), QPointF(0, 0))
                P01 = self.super_control_points.get((C_s, R_s + 1), QPointF(0, 0))
                P11 = self.super_control_points.get((C_s + 1, R_s + 1), QPointF(0, 0))

                P_u_top = (1 - u_local) * P00 + u_local * P10
                P_u_bottom = (1 - u_local) * P01 + u_local * P11
                interp_point = (1 - v_local) * P_u_top + v_local * P_u_bottom
                self.interpolated_points[(C, R)] = interp_point

    def _commit_ffd_changes(self):
        if not self.super_control_points: return

        self._calculate_all_interpolated_points()
        self.die_info_cache = {}
        for C in range(self.Max_C):
            for R in range(self.Max_R):
                polygon = self._get_die_polygon_by_index(C, R)
                if polygon:
                    center = polygon.boundingRect().center()
                    self.die_info_cache[(C, R)] = {'center': center, 'polygon': polygon}

        logger.info(f"FFD mesh committed. {len(self.die_info_cache)} dies cached.")
        self.apply_ffd_button.setEnabled(False)
        self._draw_committed_mesh()
        if self.mode == 'ffd_grid':
            self._draw_live_ffd_grid()

    def _clear_committed_mesh(self):
        for item in self.committed_grid_items:
            self.scene.removeItem(item)
        self.committed_grid_items = []

    def _draw_committed_mesh(self):
        self._clear_committed_mesh()
        if not self.die_info_cache: return

        radius_sq = self.circle_geom['radius']**2 if self.circle_geom['radius'] else float('inf')
        center = self.circle_geom['center'] if self.circle_geom['center'] else QPointF(0,0)

        for (C, R), info in self.die_info_cache.items():
            dist_sq = (info['center'].x() - center.x())**2 + (info['center'].y() - center.y())**2
            if dist_sq < radius_sq:
                # Draw polygon
                poly_item = QGraphicsPolygonItem(info['polygon'])
                poly_item.setPen(QPen(QColor('#FF69B4'), 1))
                self.scene.addItem(poly_item)
                self.committed_grid_items.append(poly_item)

                # Draw text
                die_name = self._get_die_name(C, R)
                is_masked = self._is_die_masked(info['center'])
                text_color = QColor('#0000FF') if is_masked else QColor('#00FFFF')

                text_item = QGraphicsTextItem(die_name)
                text_item.setDefaultTextColor(text_color)
                text_item.setFont(QFont("Arial", 8))
                text_rect = text_item.boundingRect()
                text_item.setPos(info['center'].x() - text_rect.width() / 2,
                                 info['center'].y() - text_rect.height() / 2)
                self.scene.addItem(text_item)
                self.committed_grid_items.append(text_item)

    def _clear_live_ffd_grid(self):
        for item in self.live_ffd_grid_items:
            self.scene.removeItem(item)
        self.live_ffd_grid_items = []
        for key, item in self.scp_items.items():
            self.scene.removeItem(item)
        self.scp_items = {}


    def _draw_live_ffd_grid(self):
        self._clear_live_ffd_grid()
        if not self.super_control_points: return

        self._calculate_all_interpolated_points()

        # Draw interpolated grid lines
        path = QPainterPath()
        for C in range(self.Max_C + 1):
            p1 = self.interpolated_points.get((C, 0))
            if not p1: continue
            path.moveTo(p1)
            for R in range(1, self.Max_R + 1):
                p2 = self.interpolated_points.get((C, R))
                if p2: path.lineTo(p2)
        for R in range(self.Max_R + 1):
            p1 = self.interpolated_points.get((0, R))
            if not p1: continue
            path.moveTo(p1)
            for C in range(1, self.Max_C + 1):
                p2 = self.interpolated_points.get((C, R))
                if p2: path.lineTo(p2)

        path_item = QGraphicsPathItem(path)
        path_item.setPen(QPen(QColor('red'), 1))
        self.scene.addItem(path_item)
        self.live_ffd_grid_items.append(path_item)

        # Draw SCPs
        for (C_s, R_s), pos in self.super_control_points.items():
            r = 7
            scp_item = QGraphicsEllipseItem(-r, -r, 2*r, 2*r)
            scp_item.setPos(pos)
            scp_item.setBrush(QBrush(QColor('blue')))
            scp_item.setPen(QPen(QColor('white'), 2))
            scp_item.setZValue(10) # ensure on top
            self.scene.addItem(scp_item)
            self.scp_items[(C_s, R_s)] = scp_item

    def _get_die_name(self, C, R):
        C_shift, R_shift = self.die_origin_shift
        C_new = C - C_shift
        R_new = R - R_shift

        Die_C = C_new % NUM_COLS
        Die_R_raw = R_new % NUM_ROWS
        Die_R = NUM_ROWS - 1 - Die_R_raw

        die_index = Die_C * NUM_ROWS + Die_R
        return DIE_NAMES[die_index % len(DIE_NAMES)]

    def draw_circle_from_points(self):
        p1, p2, p3 = self.circle_points
        D = 2 * (p1.x() * (p2.y() - p3.y()) + p2.x() * (p3.y() - p1.y()) + p3.x() * (p1.y() - p2.y()))
        if abs(D) < 1e-6:
            QMessageBox.warning(self, "Error", "Points are collinear.")
            return

        p1_sq = p1.x()**2 + p1.y()**2
        p2_sq = p2.x()**2 + p2.y()**2
        p3_sq = p3.x()**2 + p3.y()**2

        ux = (p1_sq * (p2.y() - p3.y()) + p2_sq * (p3.y() - p1.y()) + p3_sq * (p1.y() - p2.y())) / D
        uy = (p1_sq * (p3.x() - p2.x()) + p2_sq * (p1.x() - p3.x()) + p3_sq * (p2.x() - p1.x())) / D
        center = QPointF(ux, uy)
        radius = math.sqrt((p1.x() - ux)**2 + (p1.y() - uy)**2)

        self.circle_geom = {'center': center, 'radius': radius}
        self._rebuild_annotation_layer()

    def paint_mask_stroke(self, p1, p2):
        painter = QPainter(self.mask_layer)
        pen = QPen(QColor(0, 255, 0, 128), self.brush_size, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenCapStyle.RoundJoin)
        painter.setPen(pen)

        # Apply circle stencil
        path = QPainterPath()
        if self.circle_geom['radius']:
            center = self.circle_geom['center']
            radius = self.circle_geom['radius']
            path.addEllipse(center, radius, radius)
            painter.setClipPath(path)

        painter.drawLine(p1, p2)
        painter.end()
        self.mask_item.setPixmap(QPixmap.fromImage(self.mask_layer))
        self._draw_committed_mesh() # Redraw to update text color

    def clear_mask(self):
        if self.mask_layer:
            self.mask_layer.fill(Qt.GlobalColor.transparent)
            self.mask_item.setPixmap(QPixmap.fromImage(self.mask_layer))
            self._draw_committed_mesh()

    def clear_temp_items(self):
        for item in self.temp_items:
            self.scene.removeItem(item)
        self.temp_items = []

    def _initialize_ffd_mesh(self):
        if not self.original_image: return False
        if not self.circle_geom['radius']:
            QMessageBox.warning(self, "Init Error", "Define the Red Circle first.")
            return False
        if not self.rectangle_geom:
            QMessageBox.warning(self, "Init Error", "Define the Blue Die Dimension Rectangle first.")
            return False

        W_img, H_img = self.original_image.width(), self.original_image.height()
        R_circ = self.circle_geom['radius']
        ux, uy = self.circle_geom['center'].x(), self.circle_geom['center'].y()

        X_offset = max(0.0, ux - R_circ)
        Y_offset = max(0.0, uy - R_circ)
        W_FFD_max = W_img - X_offset - max(0.0, W_img - (ux + R_circ))
        H_FFD_max = H_img - Y_offset - max(0.0, H_img - (uy + R_circ))

        W_die = self.rectangle_geom.width()
        H_die = self.rectangle_geom.height()
        if W_die == 0 or H_die == 0: return False

        self.Max_C = math.ceil(W_FFD_max / W_die)
        self.Max_R = math.ceil(H_FFD_max / H_die)
        if self.Max_C == 0: self.Max_C = 1
        if self.Max_R == 0: self.Max_R = 1

        self.super_control_points = {}
        for C_s in range(SCP_SIZE):
            for R_s in range(SCP_SIZE):
                u = C_s / (SCP_SIZE - 1.0)
                v = R_s / (SCP_SIZE - 1.0)
                x = X_offset + u * W_FFD_max
                y = Y_offset + v * H_FFD_max
                self.super_control_points[(C_s, R_s)] = QPointF(x, y)

        self._commit_ffd_changes()
        return True

    def _get_die_polygon_by_index(self, C, R):
        p1 = self.interpolated_points.get((C, R))
        p2 = self.interpolated_points.get((C + 1, R))
        p3 = self.interpolated_points.get((C + 1, R + 1))
        p4 = self.interpolated_points.get((C, R + 1))
        if p1 and p2 and p3 and p4:
            return QPolygonF([p1, p2, p3, p4])
        return None

    def _is_die_masked(self, pos):
        if not self.mask_layer: return False
        x, y = int(pos.x()), int(pos.y())
        if self.mask_layer.valid(x, y):
            return QColor(self.mask_layer.pixel(x, y)).alpha() > 120
        return False

    def _find_clicked_die(self, pos):
        min_dist_sq = float('inf')
        closest_die = None
        for (C, R), info in self.die_info_cache.items():
            dist_sq = (pos.x() - info['center'].x())**2 + (pos.y() - info['center'].y())**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_die = (C, R)

        # Consider a click valid if it's within a reasonable distance of the center
        if closest_die and self.rectangle_geom:
             if min_dist_sq < (self.rectangle_geom.width()**2 + self.rectangle_geom.height()**2):
                 return closest_die
        return None

    def generate_report(self):
        if self.apply_ffd_button.isEnabled():
            QMessageBox.warning(self, "Pending Changes", "Please click 'APPLY GRID' first.")
            return
        if not all([self.die_info_cache, self.rectangle_geom, self.circle_geom['radius']]):
            QMessageBox.critical(self, "Error", "Define Circle, Die, and Grid before reporting.")
            return

        # ... (Report generation logic adapted from Tkinter version) ...
        report_lines = ["--- Die Count Report (PyQt6) ---"]
        # This part is a direct translation of the logic.
        # It's verbose but ensures feature parity.

        W_die, H_die = self.rectangle_geom.width(), self.rectangle_geom.height()
        Area_Die_Nominal = W_die * H_die

        R_circ = self.circle_geom['radius']
        Area_Circle = math.pi * R_circ**2

        # Convert mask to numpy array to count pixels
        mask_ptr = self.mask_layer.constBits()
        mask_ptr.setsize(self.mask_layer.sizeInBytes())
        mask_arr = np.frombuffer(mask_ptr, dtype=np.uint8).reshape((self.mask_layer.height(), self.mask_layer.width(), 4))
        mask_alpha = mask_arr[:, :, 3]

        # Create circle stencil
        y, x = np.ogrid[:self.mask_layer.height(), :self.mask_layer.width()]
        center = self.circle_geom['center']
        dist_from_center_sq = (x - center.x())**2 + (y - center.y())**2
        circle_mask = dist_from_center_sq <= R_circ**2

        Area_Mask_inside_Circle = np.sum((mask_alpha >= 120) & circle_mask)
        Area_Clean = Area_Circle - Area_Mask_inside_Circle
        Ratio_Estimation = Area_Clean / Area_Die_Nominal if Area_Die_Nominal > 0 else 0

        die_counts_clean = {name: 0 for name in DIE_NAMES}
        total_dies_in_circle = 0
        total_clean_dies = 0
        total_masked_dies = 0

        radius_sq = R_circ**2
        for (C, R), info in self.die_info_cache.items():
            dist_sq = (info['center'].x() - center.x())**2 + (info['center'].y() - center.y())**2
            if dist_sq < radius_sq:
                total_dies_in_circle += 1
                if not self._is_die_masked(info['center']):
                    total_clean_dies += 1
                    die_name = self._get_die_name(C, R)
                    die_counts_clean[die_name] += 1
                else:
                    total_masked_dies += 1

        report_lines.append(f"Image: {os.path.basename(self.original_image_path)}")
        report_lines.append("\n--- Area Ratio Estimation ---")
        report_lines.append(f"Circle Area: {Area_Circle:,.2f} px²")
        report_lines.append(f"Mask Area inside Circle: {Area_Mask_inside_Circle:,.0f} px²")
        report_lines.append(f"Clean Area inside Circle: {Area_Clean:,.2f} px²")
        report_lines.append(f"Nominal Die Area: {Area_Die_Nominal:,.2f} px²")
        report_lines.append(f"Estimated Total Dies: {Ratio_Estimation:.0f}")
        report_lines.append("\n--- Grid Die Counts ---")
        report_lines.append(f"Total Dies in Circle: {total_dies_in_circle}")
        report_lines.append(f"Masked Dies: {total_masked_dies}")
        report_lines.append(f"Clean Dies: {total_clean_dies}")
        report_lines.append("\n--- Detailed Clean Die Counts ---")
        for name, count in die_counts_clean.items():
            report_lines.append(f"{name}: {count}")

        report_content = "\n".join(report_lines)

        if self.original_image_path and self.original_image_path != "Placeholder_Image":
            base, _ = os.path.splitext(self.original_image_path)
            report_path = f"{base}_Report.txt"
        else:
            report_path = "Report.txt"

        try:
            with open(report_path, 'w') as f:
                f.write(report_content)
            QMessageBox.information(self, "Report Generated", f"Report saved to:\n{report_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save report: {e}")


    def save_image(self):
        if not self.original_image: return

        filepath, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                  "PNG Image (*.png);;JPEG Image (*.jpg)")
        if not filepath: return

        # Create a final image for saving
        output_image = self.original_image.copy()
        painter = QPainter(output_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw mask, then annotations, then the grid on top.
        painter.drawImage(0, 0, self.mask_layer)
        painter.drawImage(0, 0, self.annotation_layer)
        self._draw_committed_mesh_for_saving(painter)

        painter.end()

        output_image.save(filepath)
        QMessageBox.information(self, "Success", f"Image saved to:\n{filepath}")

    def _draw_committed_mesh_for_saving(self, painter):
        if not self.die_info_cache: return

        radius_sq = self.circle_geom['radius']**2 if self.circle_geom['radius'] else float('inf')
        center = self.circle_geom['center'] if self.circle_geom['center'] else QPointF(0,0)

        for (C, R), info in self.die_info_cache.items():
            dist_sq = (info['center'].x() - center.x())**2 + (info['center'].y() - center.y())**2
            if dist_sq < radius_sq:
                painter.setPen(QPen(QColor('#FF69B4'), 1))
                painter.drawPolygon(info['polygon'])

                die_name = self._get_die_name(C, R)
                is_masked = self._is_die_masked(info['center'])
                text_color = QColor('#0000FF') if is_masked else QColor('#00FFFF')

                painter.setPen(text_color)
                font = QFont("Arial", 8)
                painter.setFont(font)

                # Calculate the text's bounding box to center it properly
                fm = painter.fontMetrics()
                text_rect = fm.boundingRect(die_name)

                # Calculate the top-left position for the text to be centered
                center_point = info['center']
                top_left_x = center_point.x() - text_rect.width() / 2
                top_left_y = center_point.y() - text_rect.height() / 2

                # Adjust for the fact that drawText uses a baseline origin, not a top-left corner
                draw_point = QPointF(top_left_x - text_rect.x(), top_left_y - text_rect.y())
                painter.drawText(draw_point, die_name)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ImageAnnotator()
    main_window.show()
    sys.exit(app.exec())
