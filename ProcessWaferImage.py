import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import math
import os
import logging
try:
    import numpy as np
except ImportError:
    print("NumPy is required for area calculation and linear algebra. Please install it using 'pip install numpy'.")
    logging.error("NumPy import failed.")
    exit()

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
                    datefmt='%S')
logger = logging.getLogger(__name__)

# --- Die Names and FFD Patch Size ---
DIE_NAMES = [
    "0:No_NP", "1:MX_2x2", "2:MX_3x3", "3:MX_4x4", "4:32x16", " :Test1",
    "5:Oval", "6:2x2", "7:3x3", "8:4x4", "9:100x4", " :Test2",
    "A:D130", "B:D140", "C:D150", "D:D160", "E:D170", " :Test3"
]
NUM_COLS = 3
NUM_ROWS = 6

SCP_SIZE = 4
# --- END Die Names and FFD Size ---

# Global font object (loaded once)
try:
    GLOBAL_FONT = ImageFont.truetype("arial.ttf", 14)
except Exception:
    GLOBAL_FONT = ImageFont.load_default()
    logger.warning("Using default font. Arial not found.")

class ImageAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Wafer Annotation Tool")
        self.root.geometry("800x600")

        logger.info("Initializing ImageAnnotator.")

        # Image state
        self.original_image = None
        self.original_image_path = None
        self.photo = None
        self.canvas_image = None

        # Zoom and pan
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_panning = False

        # Annotation variables
        self.mode = None
        self.circle_points = []
        self.rect_start = None

        self.mask_paint_layer = None
        self.annotation_layer = None
        self.brush_size = 20

        # Stored Geometry Variables
        self.circle_geom = {'center': None, 'radius': None}
        self.rectangle_geom = None

        # Interpolated Mesh Variables
        self.super_control_points = {}
        self.active_scp = None
        self.Max_C = 0
        self.Max_R = 0
        self.interpolated_points = {}

        # **Committed Die Data Cache** (The result of a fast "APPLY")
        self.die_info_cache = {}

        # New variable to store the C, R coordinates of the die selected as the new (0, 0)
        self.die_origin_shift = (0, 0)

        self.circle_stencil = None
        self.live_mask_draw = None
        self.temp_items = []
        self.committed_grid_items = []
        self.last_mask_pos = None

        self.mask_dirty = False
        self.combined_image = None
        self.resize_job_id = None
        self.RESIZE_DEBOUNCE_MS = 33

        self.create_ui()
        self.load_image(initial=True)

        self.canvas.bind('<Configure>', self.on_canvas_configure)


    # --- UI Creation and Setup ---
    def create_ui(self):
        # Top toolbar
        toolbar = tk.Frame(self.root, relief=tk.RAISED, borderwidth=2)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(toolbar, text="Load Image", command=lambda: self.load_image(initial=False)).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="Save Image", command=self.save_image, bg='lightgreen').pack(side=tk.LEFT, padx=10, pady=2)

        tk.Label(toolbar, text=" | Mode:").pack(side=tk.LEFT, padx=5)
        tk.Button(toolbar, text="Circle (3 pts)", command=lambda: self.set_mode('circle')).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Die Dimension", command=lambda: self.set_mode('rectangle')).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Mask", command=lambda: self.set_mode('mask')).pack(side=tk.LEFT, padx=2)

        self.ffd_mode_button = tk.Button(toolbar, text="Edit Grid)", command=lambda: self.toggle_ffd_mode())
        self.ffd_mode_button.pack(side=tk.LEFT, padx=2)

        self.apply_ffd_button = tk.Button(toolbar, text="APPLY GRID", command=self._commit_ffd_changes, state=tk.DISABLED, bg='orange')
        self.apply_ffd_button.pack(side=tk.LEFT, padx=5)

        tk.Button(toolbar, text="Set Die", command=lambda: self.set_mode('set_naming_origin')).pack(side=tk.LEFT, padx=5)

        tk.Label(toolbar, text=" | Brush Size:").pack(side=tk.LEFT, padx=5)
        self.brush_slider = tk.Scale(toolbar, from_=5, to=150, orient=tk.HORIZONTAL,
                                     command=self.update_brush_size, length=150)
        self.brush_slider.set(20)
        self.brush_slider.pack(side=tk.LEFT, padx=2)

        tk.Button(toolbar, text="Clear Mask", command=self.clear_mask).pack(side=tk.LEFT, padx=2)

        tk.Button(toolbar, text="Generate Report",
                  command=self.count_valid_dies_and_generate_report,
                  bg='lightgreen').pack(side=tk.LEFT, padx=10, pady=2)

        # Status label
        self.status_label = tk.Label(toolbar, text="Mode: Idle (Pan with Middle Click)", fg="blue")
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Canvas with scrollbars
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind events
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)

        # Panning Bindings
        self.canvas.bind('<ButtonPress-2>', self.on_middle_mouse_down)
        self.canvas.bind('<B2-Motion>', self.on_middle_mouse_drag)
        self.canvas.bind('<ButtonRelease-2>', self.on_middle_mouse_up)

        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)
        self.canvas.bind('<Button-4>', self.on_mouse_wheel)
        self.canvas.bind('<Button-5>', self.on_mouse_wheel)
        self.canvas.bind('<Motion>', self.on_mouse_move)

    # --- Image Loading ---
    def load_image(self, initial=False):
        """Loads an image file, initializes layers, and resets geometry."""
        if not initial:
            filepath = filedialog.askopenfilename(
                title="Select Image File",
                filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff")]
            )
            if not filepath:
                return
        elif self.original_image_path is None:
            W, H = 800, 600
            self.original_image = Image.new('RGB', (W, H), 'darkgrey')
            self.original_image_path = "Placeholder_Image"
            logger.info("Loaded placeholder image.")

        if not initial and filepath:
            try:
                self.original_image = Image.open(filepath).convert("RGB")
                self.original_image_path = filepath
                logger.info(f"Image loaded: {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
                return

        if self.original_image is not None:
            W, H = self.original_image.size
            self.mask_paint_layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
            self.live_mask_draw = ImageDraw.Draw(self.mask_paint_layer)
            self.annotation_layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))

            self.circle_geom = {'center': None, 'radius': None}
            self.rectangle_geom = None
            self.circle_stencil = None
            self.super_control_points = {}
            self.die_info_cache = {}
            self.die_origin_shift = (0, 0)
            self.zoom_level = 1.0
            self.pan_x = 0
            self.pan_y = 0

            self._rebuild_annotation_layer()
            self.schedule_image_resize()
            self.set_mode(None)

    def toggle_ffd_mode(self):
        """Toggles FFD editing mode."""
        if self.mode == 'ffd_grid':
            self.set_mode(None) # set_mode will handle the commit via its guard
        else:
            self.set_mode('ffd_grid')

    def _commit_ffd_changes(self):
        """Commits the current SCP positions by recalculating the geometric cache."""
        if not self.super_control_points:
            messagebox.showwarning("Commit Error", "FFD grid not initialized.")
            return

        logger.info("Starting fast FFD mesh commit process...")

        self._calculate_all_interpolated_points()

        self.die_info_cache = {}
        for C in range(self.Max_C):
            for R in range(self.Max_R):
                polygon = self._get_die_polygon_by_index(C, R)
                if polygon is None:
                    continue

                P_LL, P_LR, P_UR, P_UL = polygon
                center_x = (P_LL[0] + P_LR[0] + P_UR[0] + P_UL[0]) / 4.0
                center_y = (P_LL[1] + P_LR[1] + P_UR[1] + P_UL[1]) / 4.0

                self.die_info_cache[(C, R)] = {
                    'center': (center_x, center_y),
                    'polygon': polygon
                }

        logger.info(f"FFD mesh geometric data successfully COMMITTED. {len(self.die_info_cache)} dies.")

        self._rebuild_annotation_layer()

        self.apply_ffd_button.config(state=tk.DISABLED)
        self.schedule_image_resize()

        if self.mode == 'ffd_grid':
             self._draw_live_ffd_grid()


    def set_mode(self, mode):
        if self.mode == 'ffd_grid' and mode != 'ffd_grid':
             self._commit_ffd_changes()

        self.mode = mode
        self.circle_points = []
        self.rect_start = None
        self.clear_temp_items()
        self.active_scp = None

        status_c, status_r = self.die_origin_shift

        mode_text = {
            'circle': 'Circle - Click 3 points',
            'rectangle': 'Die Dimension - Click and drag',
            'mask': 'Mask - Paint over image (Square Brush)',
            'ffd_grid': f'FFD Mesh - Drag the {SCP_SIZE}x{SCP_SIZE} blue Super-Control Points (SCPs). **Press APPLY FFD Changes (Fast) when done.**',
            'set_naming_origin': f'Set Naming Origin - Click the die you want to name Die ({status_c}, {status_r})',
            None: 'Idle (Pan with Middle Click)'
        }

        status_message = mode_text.get(mode, 'Idle (Pan with Middle Click)')

        if mode == 'ffd_grid':
            self.ffd_mode_button.config(relief=tk.SUNKEN)

            if not self.super_control_points:
                 if self._initialize_ffd_mesh():
                      self._draw_live_ffd_grid()
                      self.status_label.config(text=status_message)
                 else:
                      self.status_label.config(text="Mode: FFD Mesh - First define the Blue Die Dimension Rectangle.")
            else:
                 self._draw_live_ffd_grid()
                 self.status_label.config(text=status_message)

            logger.info(f"Mode set to: {mode}. SCPs: {len(self.super_control_points)}.")
        else:
            self.ffd_mode_button.config(relief=tk.RAISED)
            self.status_label.config(text=f"Mode: {status_message}")
            logger.info(f"Mode set to: {mode}")

    def _rebuild_annotation_layer(self):
        if self.original_image is None:
            return

        W, H = self.original_image.size
        self.annotation_layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(self.annotation_layer)

        if self.circle_geom['radius'] is not None:
            R = self.circle_geom['radius']
            ux, uy = self.circle_geom['center']
            bbox = [ux - R, uy - R, ux + R, uy + R]
            draw.ellipse(bbox, outline=(255, 0, 0, 50), width=3)

        if self.rectangle_geom is not None:
             draw.rectangle(self.rectangle_geom, outline=(0, 0, 255, 50), width=3)

        self.mask_dirty = True
        self.update_combined_image()


    # --- FFD Interpolation and Drawing ---

    def _calculate_all_interpolated_points(self):
        if self.original_image is None or not self.super_control_points:
            self.interpolated_points = {}
            return

        W_img, H_img = self.original_image.size
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

                u_local = max(0.0, min(1.0, u_local))
                v_local = max(0.0, min(1.0, v_local))

                P00 = self.super_control_points.get((C_s, R_s), (0, 0))
                P10 = self.super_control_points.get((C_s + 1, R_s), (W_img, 0))
                P01 = self.super_control_points.get((C_s, R_s + 1), (0, H_img))
                P11 = self.super_control_points.get((C_s + 1, R_s + 1), (W_img, H_img))

                P_u_top = np.array(P00) * (1 - u_local) + np.array(P10) * u_local
                P_u_bottom = np.array(P01) * (1 - u_local) + np.array(P11) * u_local

                P_interp = P_u_top * (1 - v_local) + P_u_bottom * v_local

                self.interpolated_points[(C, R)] = (P_interp[0], P_interp[1])

    def _draw_committed_mesh_on_canvas(self):
        """
        Draws the permanent (pink) high-resolution grid and text labels onto the Tkinter canvas
        using fast vector drawing, based on the `die_info_cache`.
        """
        self.canvas.delete("committed_grid")
        self.committed_grid_items = []

        if not self.die_info_cache:
             return

        if self.circle_geom['radius'] is None:
             ux, uy, radius_sq = 0, 0, float('inf')
        else:
             ux, uy = self.circle_geom['center']
             radius_sq = self.circle_geom['radius'] ** 2

        PINK_COLOR = '#FF69B4'
        TEXT_COLOR_CLEAN = '#00FFFF'
        TEXT_COLOR_MASKED = '#0000FF'
        LINE_WIDTH = 1

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        for (C, R), info in self.die_info_cache.items():
            polygon = info['polygon']
            center_x, center_y = info['center']

            dist_sq = (center_x - ux)**2 + (center_y - uy)**2

            if dist_sq < radius_sq:
                # 1. Check visibility
                s_poly = [self.image_to_screen_coords(p[0], p[1]) for p in polygon]

                screen_min_x = min(p[0] for p in s_poly)
                screen_max_x = max(p[0] for p in s_poly)
                screen_min_y = min(p[1] for p in s_poly)
                screen_max_y = max(p[1] for p in s_poly)

                if screen_max_x < 0 or screen_min_x > canvas_w or screen_max_y < 0 or screen_min_y > canvas_h:
                     continue

                # 2. Draw the Die Boundary (Polygon)
                line_coords = []
                for p in s_poly:
                    line_coords.extend(p)
                line_coords.extend(s_poly[0])

                item = self.canvas.create_polygon(line_coords, outline=PINK_COLOR, fill='', width=LINE_WIDTH, tags="committed_grid")
                self.committed_grid_items.append(item)

                # 3. Draw the Die Name with Dynamic Indexing
                die_name = self._get_die_name(C, R)

                is_masked = self._is_die_masked(center_x, center_y)
                TEXT_COLOR = TEXT_COLOR_MASKED if is_masked else TEXT_COLOR_CLEAN

                screen_center_x, screen_center_y = self.image_to_screen_coords(center_x, center_y)

                item = self.canvas.create_text(screen_center_x, screen_center_y,
                                               text=die_name,
                                               fill=TEXT_COLOR,
                                               font=("Arial", 8),
                                               anchor=tk.CENTER,
                                               tags="committed_grid")
                self.committed_grid_items.append(item)

        self.canvas.tag_raise("all")

    def _draw_live_ffd_grid(self):
        """Draws the temporary (red) high-resolution interpolated grid and SCPs."""
        self.clear_temp_items()

        self._calculate_all_interpolated_points()

        LIVE_LINE_COLOR = 'red'
        LIVE_LINE_WIDTH = 1

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        # 2. Draw the live interpolated grid lines (Die boundaries)
        for C in range(self.Max_C):
            for R in range(self.Max_R):
                polygon = self._get_die_polygon_by_index(C, R)
                if polygon is None:
                    continue

                s_poly = [self.image_to_screen_coords(p[0], p[1]) for p in polygon]

                screen_min_x = min(p[0] for p in s_poly)
                screen_max_x = max(p[0] for p in s_poly)
                screen_min_y = min(p[1] for p in s_poly)
                screen_max_y = max(p[1] for p in s_poly)

                if screen_max_x < 0 or screen_min_x > canvas_w or screen_max_y < 0 or screen_min_y > canvas_h:
                     continue

                line_coords = []
                for p in s_poly:
                    line_coords.extend(p)
                line_coords.extend(s_poly[0])

                item = self.canvas.create_polygon(line_coords, outline=LIVE_LINE_COLOR, fill='', width=LIVE_LINE_WIDTH, tags="temp_grid_line")
                self.temp_items.append(item)

        # 3. Draw the User-Editable Super Control Points (SCPs)
        for (C_s, R_s), point in self.super_control_points.items():
            r = 7
            screen_x, screen_y = self.image_to_screen_coords(point[0], point[1])

            color = 'yellow' if self.active_scp == (C_s, R_s) else 'white'

            item = self.canvas.create_oval(screen_x-r, screen_y-r, screen_x+r, screen_y+r,
                                           fill='blue', outline=color, width=2, tags=f"P_s_{C_s},{R_s}")
            self.temp_items.append(item)

        self.canvas.tag_raise("all")

    # --- Mouse Event Handlers ---
    def on_mouse_down(self, event):
        if self.original_image is None or self.is_panning:
            return
        self.clear_temp_items()

        if self.mode == 'circle':
            img_x, img_y = self.screen_to_image_coords(event.x, event.y)
            self.circle_points.append((int(img_x), int(img_y)))
            r = 5
            item = self.canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r, fill='red', outline='white', width=2)
            self.temp_items.append(item)
            if len(self.circle_points) == 3:
                self.draw_circle_from_points()
                self.circle_points = []

        elif self.mode == 'rectangle':
            self.rect_start = self.screen_to_image_coords(event.x, event.y)

        elif self.mode == 'mask':
            if self.circle_stencil is None:
                messagebox.showwarning("Mask Error", "Please define a circle before painting a mask.")
                return
            img_x, img_y = self.screen_to_image_coords(event.x, event.y)
            self.last_mask_pos = (img_x, img_y)
            self.paint_mask_stroke(int(img_x), int(img_y), int(img_x), int(img_y))
            self.draw_live_brush_stroke(self.image_to_screen_coords(img_x, img_y), self.image_to_screen_coords(img_x, img_y))

        elif self.mode == 'ffd_grid':
            if not self.super_control_points:
                messagebox.showerror("Error", "Please define the Blue Die Dimension Rectangle first.")
                return

            img_x, img_y = self.screen_to_image_coords(event.x, event.y)
            click_point = (img_x, img_y)

            r_screen = 10
            r_image = r_screen / self.zoom_level

            drag_point_found = False
            for (C_s, R_s), anchor in self.super_control_points.items():
                dist_sq = (click_point[0] - anchor[0])**2 + (click_point[1] - anchor[1])**2
                if dist_sq < r_image**2:
                    self.active_scp = (C_s, R_s)
                    drag_point_found = True
                    self._draw_live_ffd_grid()
                    break

            if not drag_point_found:
                self.active_scp = None

        elif self.mode == 'set_naming_origin' and self.die_info_cache:
            img_x, img_y = self.screen_to_image_coords(event.x, event.y)

            clicked_die_CR = self._find_clicked_die(img_x, img_y)

            if clicked_die_CR:
                C_clicked, R_clicked = clicked_die_CR

                self.die_origin_shift = (C_clicked, R_clicked)

                logger.info(f"Die Naming Origin set to Die ({C_clicked}, {R_clicked}).")
                self.status_label.config(text=f"Die Naming Origin set. New Naming Origin is Die ({C_clicked}, {R_clicked}).")

                self.set_mode(None)
                self.schedule_image_resize()
            else:
                self.status_label.config(text="Mode: Set Naming Origin - Click closer to a die center.")
                self.set_mode(None)


    def on_mouse_drag(self, event):
        if self.original_image is None or self.is_panning:
            return

        if self.mode == 'rectangle' and self.rect_start:
            self.clear_temp_items()
            x1, y1 = self.image_to_screen_coords(*self.rect_start)
            item = self.canvas.create_rectangle(x1, y1, event.x, event.y, outline='blue', width=2)
            self.temp_items.append(item)

        elif self.mode == 'mask' and self.last_mask_pos:
            if self.circle_stencil is None:
                return
            img_x, img_y = self.screen_to_image_coords(event.x, event.y)
            self.paint_mask_stroke(int(self.last_mask_pos[0]), int(self.last_mask_pos[1]),
                                   int(img_x), int(img_y))
            self.draw_live_brush_stroke(self.image_to_screen_coords(*self.last_mask_pos), (event.x, event.y))
            self.last_mask_pos = (img_x, img_y)

        elif self.mode == 'ffd_grid' and self.active_scp is not None:
            img_x, img_y = self.screen_to_image_coords(event.x, event.y)
            W_img, H_img = self.original_image.size
            img_x = max(0, min(W_img, img_x))
            img_y = max(0, min(H_img, img_y))
            C_s, R_s = self.active_scp
            self.super_control_points[(C_s, R_s)] = (img_x, img_y)
            self._draw_live_ffd_grid()
            self.apply_ffd_button.config(state=tk.NORMAL)


    def on_mouse_up(self, event):
        if self.original_image is None or self.is_panning:
            return

        if self.mode == 'rectangle' and self.rect_start:
            img_x, img_y = self.screen_to_image_coords(event.x, event.y)
            x1 = int(min(self.rect_start[0], img_x))
            y1 = int(min(self.rect_start[1], img_y))
            x2 = int(max(self.rect_start[0], img_x))
            y2 = int(max(self.rect_start[1], img_y))
            self.rectangle_geom = (x1, y1, x2, y2)
            logger.info(f"Die Dimension Rectangle finalized: {self.rectangle_geom}")
            self.super_control_points = {}
            self.die_info_cache = {}
            self._rebuild_annotation_layer()
            self.rect_start = None
            self.clear_temp_items()

        elif self.mode == 'mask' and self.last_mask_pos:
            self.last_mask_pos = None
            self.clear_temp_items()
            if self.mask_dirty:
                self.update_combined_image()
                self.schedule_image_resize()
                self.mask_dirty = False

        elif self.mode == 'ffd_grid':
            self.active_scp = None
            pass

    # --- count_valid_dies_and_generate_report ---
    def count_valid_dies_and_generate_report(self):
        if self.mode == 'ffd_grid' and self.apply_ffd_button.cget('state') == tk.NORMAL:
             messagebox.showwarning("Pending Changes", "Please click 'APPLY GRID' to save the current grid before generating the report.")
             return

        if not self.die_info_cache:
            messagebox.showerror("Error", "FFD grid geometric data is missing. Ensure the grid is defined and APPLY changes.")
            return
        if self.rectangle_geom is None:
             messagebox.showerror("Error", "Die Dimension Rectangle is missing. Define it before reporting.")
             return
        if self.circle_geom['radius'] is None:
             messagebox.showerror("Error", "Circle Area is missing. Define it before reporting.")
             return


        x1, y1, x2, y2 = self.rectangle_geom
        W_die_nominal, H_die_nominal = x2 - x1, y2 - y1
        Area_Die_Nominal = W_die_nominal * H_die_nominal

        R_circ = self.circle_geom['radius']
        Area_Circle = math.pi * R_circ**2

        Area_Mask_inside_Circle = self.calculate_mask_area_inside_circle()
        Area_Clean = Area_Circle - Area_Mask_inside_Circle
        Ratio_Estimation = Area_Clean / Area_Die_Nominal

        die_counts_clean = {name: 0 for name in DIE_NAMES}
        total_dies_in_circle = 0
        total_clean_dies = 0
        total_masked_dies = 0

        ux, uy = self.circle_geom['center']
        radius_sq = self.circle_geom['radius'] ** 2

        for (C, R), info in self.die_info_cache.items():
            center_x, center_y = info['center']

            dist_sq = (center_x - ux)**2 + (center_y - uy)**2

            if dist_sq < radius_sq:
                total_dies_in_circle += 1
                is_painted_green = self._is_die_masked(center_x, center_y)

                if not is_painted_green:
                    total_clean_dies += 1

                    die_name = self._get_die_name(C, R)

                    die_counts_clean[die_name] += 1
                else:
                    total_masked_dies += 1

        report_lines = []
        report_lines.append("--- Die Count Report ---")
        report_lines.append(f"Original Image: {os.path.basename(self.original_image_path)}")
        report_lines.append("------------------------------------------------------")
        report_lines.append("\n--- Area Ratio Estimation ---")
        report_lines.append(f"Circle Area: {Area_Circle:,.2f} px²")
        report_lines.append(f"Mask Area (No dies, inside Circle): {Area_Mask_inside_Circle:,.0f} px²")
        report_lines.append(f"Non-masked Area (Dies, inside Circle): {Area_Clean:,.2f} px²")
        report_lines.append(f"Nominal Die Area (User selected): {Area_Die_Nominal:,.2f} px²")
        report_lines.append(f"Estimated Total Dies: {Ratio_Estimation:.0f} dies")
        report_lines.append("------------------------------------------------------")
        report_lines.append("\n--- Grid Die Counts ---")
        report_lines.append(f"Total Dies Defined by Mesh in Circle: {total_dies_in_circle}")
        report_lines.append(f"Total Dies in Masked Area (Removed dies): {total_masked_dies}")
        report_lines.append(f"Total Dies in Non-masked Area (Avilaable dies): {total_clean_dies}")
        report_lines.append("------------------------------------------------------")
        report_lines.append("\n--- Detailed Die Counts (Clean/Unpainted Dies Only) ---")
        report_lines.append("Die Type\t\tCount")
        report_lines.append("-" * 49)
        for name, count in die_counts_clean.items():
            report_lines.append(f"{name}:\t{count}")
        report_content = "\n".join(report_lines)

        try:
            if self.original_image_path:
                img_dir = os.path.dirname(self.original_image_path)
                base_name = os.path.splitext(os.path.basename(self.original_image_path))[0]
                report_filename = os.path.join(img_dir, f"{base_name}_Report.txt")
            else:
                report_filename = "Report.txt"

            with open(report_filename, 'w') as f:
                f.write(report_content)

            messagebox.showinfo("Report Exported",
                                f"Full die count report successfully generated and saved to:\n{report_filename}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save report: {e}")

    # --- Display Update ---
    def update_display(self):
        if self.combined_image is None:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        x_img_start = -self.pan_x / self.zoom_level
        y_img_start = -self.pan_y / self.zoom_level

        w_img_crop = canvas_w / self.zoom_level
        h_img_crop = canvas_h / self.zoom_level

        x0 = max(0, int(x_img_start))
        y0 = max(0, int(y_img_start))
        x1 = min(self.combined_image.width, int(x_img_start + w_img_crop))
        y1 = min(self.combined_image.height, int(y_img_start + h_img_crop))

        if x1 <= x0 or y1 <= y0:
            if self.canvas_image:
                self.canvas.delete(self.canvas_image)
            self.canvas_image = None
            return

        cropped_img = self.combined_image.crop((x0, y0, x1, y1))
        actual_crop_width = x1 - x0
        actual_crop_height = y1 - y0
        new_size_w = int(actual_crop_width * self.zoom_level)
        new_size_h = int(actual_crop_height * self.zoom_level)

        if new_size_w > 0 and new_size_h > 0:
            display_img = cropped_img.resize((new_size_w, new_size_h), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(display_img)
        else:
            return

        new_pan_x = x0 * self.zoom_level + self.pan_x
        new_pan_y = y0 * self.zoom_level + self.pan_y

        if self.canvas_image:
            self.canvas.itemconfig(self.canvas_image, image=self.photo)
            self.canvas.coords(self.canvas_image, new_pan_x, new_pan_y)
        else:
            self.canvas_image = self.canvas.create_image(new_pan_x, new_pan_y,
                                                         anchor=tk.NW, image=self.photo)

        self.canvas.lower(self.canvas_image)

        if self.mode != 'ffd_grid':
            self._draw_committed_mesh_on_canvas()
        else:
            self.canvas.delete("committed_grid")
            self.committed_grid_items = []

        self.clear_temp_items()

        if self.mode == 'ffd_grid' and self.super_control_points:
            self._draw_live_ffd_grid()
        elif self.mode == 'circle' and len(self.circle_points) > 0:
             for point in self.circle_points:
                screen_x, screen_y = self.image_to_screen_coords(point[0], point[1])
                r = 5
                item = self.canvas.create_oval(screen_x-r, screen_y-r, screen_x+r, screen_y+r, fill='red', outline='white', width=2)
                self.temp_items.append(item)

        self.canvas.tag_raise("all")
        self.mask_dirty = False


# --- Helper Methods ---
    def update_brush_size(self, value):
        """Updates the brush size instance variable based on the slider value."""
        self.brush_size = int(float(value))
        logger.debug(f"Brush size updated to: {self.brush_size}")

    def _is_die_masked(self, x, y):
        """Checks the alpha channel of the mask_paint_layer at the given image coordinates (x, y)."""
        if self.mask_paint_layer is None:
            return False

        W, H = self.mask_paint_layer.size
        ix = max(0, min(W - 1, int(x)))
        iy = max(0, min(H - 1, int(y)))

        try:
            pixel = self.mask_paint_layer.getpixel((ix, iy))
            alpha_value = pixel[3]
            return alpha_value >= 120
        except IndexError:
            return False

    def _find_clicked_die(self, img_x, img_y):
        """Finds the (C, R) of the die whose center is closest to the click."""
        min_dist_sq = float('inf')
        closest_die_CR = None

        TOLERANCE_SQ = 2500

        for (C, R), info in self.die_info_cache.items():
            center_x, center_y = info['center']
            dist_sq = (img_x - center_x)**2 + (img_y - center_y)**2

            if dist_sq < min_dist_sq and dist_sq < TOLERANCE_SQ:
                min_dist_sq = dist_sq
                closest_die_CR = (C, R)

        return closest_die_CR

    def _get_interpolated_point(self, C, R):
        return self.interpolated_points.get((C, R), (0, 0))

    def _get_die_polygon_by_index(self, C, R):
        if C < 0 or C >= self.Max_C or R < 0 or R >= self.Max_R:
            return None
        P_UL = self._get_interpolated_point(C, R)
        P_UR = self._get_interpolated_point(C + 1, R)
        P_LR = self._get_interpolated_point(C + 1, R + 1)
        P_LL = self._get_interpolated_point(C, R + 1)
        die_polygon = [P_LL, P_LR, P_UR, P_UL]
        return die_polygon

    def _get_die_name(self, C, R):
        C_shift, R_shift = self.die_origin_shift
        C_new = C - C_shift
        R_new = R - R_shift

        Die_C = C_new % NUM_COLS

        # Die_R: Reverse the vertical direction of the repeating pattern
        Die_R_raw = R_new % NUM_ROWS
        Die_R = NUM_ROWS - 1 - Die_R_raw

        die_index = Die_C * NUM_ROWS + Die_R
        die_name = DIE_NAMES[die_index % len(DIE_NAMES)]

        return die_name

    def screen_to_image_coords(self, x, y):
        img_x = (x - self.pan_x) / self.zoom_level
        img_y = (y - self.pan_y) / self.zoom_level
        return img_x, img_y

    def image_to_screen_coords(self, x, y):
        screen_x = x * self.zoom_level + self.pan_x
        screen_y = y * self.zoom_level + self.pan_y
        return screen_x, screen_y

    def clear_temp_items(self):
        for item in self.temp_items:
            self.canvas.delete(item)
        self.temp_items = []

    def clear_mask(self):
        if self.original_image:
             W, H = self.original_image.size
             self.mask_paint_layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
             self.live_mask_draw = ImageDraw.Draw(self.mask_paint_layer)
             logger.info("Mask layer cleared.")
             self._rebuild_annotation_layer()

    def calculate_mask_area_inside_circle(self):
        if self.mask_paint_layer is None:
            return 0
        mask_alpha = np.array(self.mask_paint_layer.getchannel('A'))
        W, H = self.original_image.size

        x_min, x_max, y_min, y_max = 0, W, 0, H
        if self.circle_geom['radius'] is not None:
             ux, uy = self.circle_geom['center']
             R = self.circle_geom['radius']
             x_min = max(0, int(ux - R))
             x_max = min(W, int(ux + R))
             y_min = max(0, int(uy - R))
             y_max = min(H, int(uy + R))

        mask_pixels = (mask_alpha[y_min:y_max, x_min:x_max] >= 120)
        mask_area_pixels = np.sum(mask_pixels)
        return mask_area_pixels

    def draw_circle_from_points(self):
        p1, p2, p3 = self.circle_points
        ax, ay = p1
        bx, by = p2
        cx, cy = p3

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-10:
            messagebox.showerror("Error", "Points are collinear. Cannot define a circle.")
            self.clear_temp_items()
            return

        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
        radius = math.sqrt((ax - ux)**2 + (ay - uy)**2)

        self.circle_geom['center'] = (ux, uy)
        self.circle_geom['radius'] = radius

        bbox = [ux - radius, uy - radius, ux + radius, uy + radius]
        W, H = self.original_image.size
        self.circle_stencil = Image.new('1', (W, H), 0)
        stencil_draw = ImageDraw.Draw(self.circle_stencil)
        stencil_draw.ellipse(bbox, fill=1)

        self._rebuild_annotation_layer()
        self.clear_temp_items()

    def paint_mask_stroke(self, x1, y1, x2, y2):
        if self.circle_stencil is None:
            return
        mask_color = (0, 255, 0, 128)
        self.live_mask_draw.line([(x1, y1), (x2, y2)], fill=mask_color, width=self.brush_size, joint='bevel')
        r = self.brush_size / 2
        bbox_cap = [x2 - r, y2 - r, x2 + r, y2 + r]
        self.live_mask_draw.rectangle(bbox_cap, fill=mask_color)

        mask_alpha = np.array(self.mask_paint_layer.getchannel('A'))
        stencil_array = np.array(self.circle_stencil)

        if mask_alpha.shape == stencil_array.shape:
            clipped_alpha = mask_alpha * stencil_array
            self.mask_paint_layer.putalpha(Image.fromarray(clipped_alpha.astype(np.uint8)))
            self.live_mask_draw = ImageDraw.Draw(self.mask_paint_layer)

        self.mask_dirty = True

    def draw_live_brush_stroke(self, start_screen_pos, end_screen_pos):
        r = self.brush_size * self.zoom_level / 2
        temp_color = 'yellow'

        item = self.canvas.create_line(start_screen_pos[0], start_screen_pos[1], end_screen_pos[0], end_screen_pos[1], fill=temp_color, width=r*2, capstyle=tk.BUTT)
        self.temp_items.append(item)
        item = self.canvas.create_rectangle(end_screen_pos[0]-r, end_screen_pos[1]-r, end_screen_pos[0]+r, end_screen_pos[1]+r, fill=temp_color, outline='')
        self.temp_items.append(item)

    def on_middle_mouse_down(self, event):
        if self.original_image is None:
            return
        self.is_panning = True
        self.pan_start_x = event.x - self.pan_x
        self.pan_start_y = event.y - self.pan_y

    def on_middle_mouse_drag(self, event):
        if self.original_image is None or not self.is_panning:
            return
        old_pan_x, old_pan_y = self.pan_x, self.pan_y
        self.pan_x = event.x - self.pan_start_x
        self.pan_y = event.y - self.pan_start_y
        dx, dy = self.pan_x - old_pan_x, self.pan_y - old_pan_y

        if self.canvas_image:
            self.canvas.move(self.canvas_image, dx, dy)
        for item in self.temp_items:
            self.canvas.move(item, dx, dy)
        for item in self.committed_grid_items:
            self.canvas.move(item, dx, dy)

        self.schedule_image_resize()

    def on_middle_mouse_up(self, event):
        if self.original_image is None or not self.is_panning:
            return
        self.is_panning = False
        self.schedule_image_resize()

    def on_mouse_wheel(self, event):
        if self.original_image is None:
            return
        x, y = event.x, event.y
        if event.num == 5 or event.delta < 0:
            factor = 0.9
        else:
            factor = 1.1

        old_zoom = self.zoom_level
        self.zoom_level *= factor
        self.zoom_level = max(0.1, min(10.0, self.zoom_level))

        self.pan_x = x - (x - self.pan_x) * (self.zoom_level / old_zoom)
        self.pan_y = y - (y - self.pan_y) * (self.zoom_level / old_zoom)

        self.schedule_image_resize()

    def on_mouse_move(self, event):
        if self.original_image is None:
            return

        # Clear previous hover effects to prevent artifacts
        self.clear_temp_items()

        # Mask Brush Preview (maintains original functionality)
        if self.mode == 'mask':
            r = self.brush_size * self.zoom_level / 2
            item = self.canvas.create_rectangle(
                event.x - r, event.y - r, event.x + r, event.y + r,
                outline='yellow', width=2
            )
            self.temp_items.append(item)

        # Die Hover Highlight (new functionality)
        elif self.mode is None and self.die_info_cache:
            img_x, img_y = self.screen_to_image_coords(event.x, event.y)
            hovered_die_CR = self._find_clicked_die(img_x, img_y)

            if hovered_die_CR:
                self._draw_hover_highlight(hovered_die_CR)

    def _draw_hover_highlight(self, die_CR):
        """Draws a temporary highlight around a die on hover."""
        info = self.die_info_cache.get(die_CR)
        if not info:
            return

        polygon = info['polygon']
        s_poly = [self.image_to_screen_coords(p[0], p[1]) for p in polygon]

        line_coords = []
        for p in s_poly:
            line_coords.extend(p)
        line_coords.extend(s_poly[0])

        item = self.canvas.create_polygon(
            line_coords,
            outline='cyan',
            fill='',
            width=3,
            tags="temp_hover_highlight"
        )
        self.temp_items.append(item)

    def on_canvas_configure(self, event):
        self.schedule_image_resize()

    def update_combined_image(self):
        if self.original_image is None:
            return

        self.combined_image = self.original_image.copy().convert("RGBA")
        self.combined_image.paste(self.mask_paint_layer, (0, 0), self.mask_paint_layer)
        self.combined_image.paste(self.annotation_layer, (0, 0), self.annotation_layer)

    def schedule_image_resize(self):
        if self.resize_job_id:
            self.root.after_cancel(self.resize_job_id)
        self.resize_job_id = self.root.after(self.RESIZE_DEBOUNCE_MS, self.do_image_resize)

    def do_image_resize(self):
        self.update_display()
        self.resize_job_id = None

    def _initialize_ffd_mesh(self):
        if self.original_image is None:
            return False

        # --- REQUIREMENT CHECK: Circle must be defined for dynamic margin ---
        if self.circle_geom['radius'] is None or self.circle_geom['center'] is None:
            messagebox.showwarning("Initialization Error",
                                   "Please define the Red Circle of Interest first. FFD margin requires it.")
            return False

        if self.rectangle_geom is None:
            messagebox.showwarning("Initialization Error",
                                   "Please define the Blue Die Dimension Rectangle first.")
            return False

        # --- 1. Define Constants and Image Size ---
        W_img, H_img = self.original_image.size
        #SCP_SIZE = 4

        # --- 2. Calculate Dynamic Margin based on Circle ---
        R_circ = self.circle_geom['radius']
        ux, uy = self.circle_geom['center']

        # Calculate the four distances from the circle's edge to the image edge
        D_left = ux - R_circ
        D_right = W_img - (ux + R_circ)
        D_top = uy - R_circ
        D_bottom = H_img - (uy + R_circ)

        # The FFD grid's outer boundary is offset by D_x, D_y
        X_offset = max(0.0, D_left)
        W_FFD_max = W_img - max(0.0, D_left) - max(0.0, D_right)

        Y_offset = max(0.0, D_top)
        H_FFD_max = H_img - max(0.0, D_top) - max(0.0, D_bottom)

        # Fallback: Ensure minimum working area
        DEFAULT_MARGIN = 0.05
        if W_FFD_max < W_img * DEFAULT_MARGIN:
            W_FFD_max = W_img * (1.0 - 2*DEFAULT_MARGIN)
        if H_FFD_max < H_img * DEFAULT_MARGIN:
            H_FFD_max = H_img * (1.0 - 2*DEFAULT_MARGIN)

        # --- 3. Calculate Die Grid Size for Interpolated Mesh (based on FFD Area) ---
        x1, y1, x2, y2 = self.rectangle_geom
        W_die, H_die = x2 - x1, y2 - y1

        # Use W_FFD_max/H_FFD_max instead of W_img/H_img ***
        N_c = math.ceil(W_FFD_max / W_die)
        N_r = math.ceil(H_FFD_max / H_die)
        self.Max_C = N_c
        self.Max_R = N_r

        # Ensure a minimum grid size of 1x1 if die is very large (Max_C/R should be >= 1)
        if self.Max_C == 0:
            self.Max_C = 1
        if self.Max_R == 0:
            self.Max_R = 1

        # --- 4. Place SCPs within the Dynamic Working Area (W_FFD_max x H_FFD_max) ---
        self.super_control_points = {}
        self.initial_scp_points = {}

        for C_s in range(SCP_SIZE):
            for R_s in range(SCP_SIZE):

                # Normalized position (0.0 to 1.0) across the grid intervals
                u_norm_grid = C_s / (SCP_SIZE - 1.0)
                v_norm_grid = R_s / (SCP_SIZE - 1.0)

                # Map to the dynamic working area
                x_relative = u_norm_grid * W_FFD_max
                y_relative = v_norm_grid * H_FFD_max

                # Add the offset (D_left, D_top) to shift the entire grid inward
                x = x_relative + X_offset
                y = y_relative + Y_offset

                self.super_control_points[(C_s, R_s)] = (x, y)
                self.initial_scp_points[(C_s, R_s)] = (x, y)

        # --- 5. Commit Changes ---
        self._calculate_all_interpolated_points()
        self._commit_ffd_changes()
        return True

    def save_image(self):
        if self.combined_image is None:
            messagebox.showerror("Error", "No image loaded or image processing incomplete.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Save Annotated Image"
        )
        if filename:
            try:
                temp_annotation = self.annotation_layer.copy().convert("RGBA")
                temp_draw = ImageDraw.Draw(temp_annotation)

                PINK_COLOR = (255, 105, 180, 200)
                TEXT_COLOR_CLEAN = (0, 255, 255, 200)
                TEXT_COLOR_MASKED = (0, 0, 255, 150)
                LINE_THICKNESS = 1

                for (C, R), info in self.die_info_cache.items():
                    polygon = info['polygon']
                    center_x, center_y = info['center']

                    if self.circle_geom['radius'] is not None:
                        R_circ = self.circle_geom['radius']
                        ux, uy = self.circle_geom['center']
                        radius_sq = R_circ**2
                        dist_sq = (center_x - ux)**2 + (center_y - uy)**2
                        if dist_sq >= radius_sq:
                             continue

                    int_points = [(int(p[0]), int(p[1])) for p in polygon]
                    closed_die_line = int_points + [int_points[0]]
                    temp_draw.line(closed_die_line, fill=PINK_COLOR, width=LINE_THICKNESS)

                    die_name = self._get_die_name(C, R)

                    is_masked = self._is_die_masked(center_x, center_y)
                    TEXT_COLOR = TEXT_COLOR_MASKED if is_masked else TEXT_COLOR_CLEAN

                    # Use textbbox instead of textsize ---
                    # The bbox returns (left, top, right, bottom) of the text relative to the origin (0, 0)
                    bbox = temp_draw.textbbox((0, 0), die_name, font=GLOBAL_FONT)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]

                    # Calculate position for centering (anchor=tk.CENTER)
                    text_x = center_x - text_w / 2
                    text_y = center_y - text_h / 2

                    temp_draw.text((text_x, text_y), die_name, fill=TEXT_COLOR, font=GLOBAL_FONT)

                final_image = self.original_image.copy().convert("RGBA")
                final_image.paste(self.mask_paint_layer, (0, 0), self.mask_paint_layer)
                final_image.paste(temp_annotation, (0, 0), temp_annotation)

                final_image.save(filename)
                messagebox.showinfo("Success", f"Image saved successfully to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {e}")
                logger.error(f"Failed to save image: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnnotator(root)
    root.mainloop()
