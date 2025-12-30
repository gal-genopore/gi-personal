# (Full file - updated to group all drawable elements into a draw:g group inside content.xml
# and to avoid writing any unnecessary background rectangle)
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import math
import zipfile
import io
import os
import tempfile

# Define the standard size for the geometric glue point anchor
ANCHOR_RADIUS_BASE = 1.5 

class GraphicRenderer:
    """Abstract base for drawing to support both Canvas and SVG."""
    
    def draw_line(self, x1, y1, x2, y2, **kwargs):
        pass
    
    def draw_rect(self, x1, y1, x2, y2, **kwargs):
        pass
    
    def draw_polygon(self, points, **kwargs):
        pass
    
    def draw_arrow(self, x1, y1, x2, y2, **kwargs):
        pass
    
    def draw_t_stop(self, x, y, direction='up', **kwargs):
        pass
    
    def draw_zigzag(self, x, y, zig_width, height, horizontal=False, **kwargs):
        pass
    
    def draw_circle(self, x, y, r, **kwargs):
        pass
    
    def draw_text(self, x, y, text, font_size=10, **kwargs):
        pass


class CanvasRenderer(GraphicRenderer):
    """Draws directly to a Tkinter Canvas."""
    
    def __init__(self, canvas):
        self.c = canvas

    def draw_line(self, x1, y1, x2, y2, **kwargs):
        width = kwargs.get('width', 2)
        self.c.create_line(x1, y1, x2, y2, width=width, fill="black")

    def draw_rect(self, x1, y1, x2, y2, **kwargs):
        width = kwargs.get('width', 2)
        self.c.create_rectangle(x1, y1, x2, y2, outline="black", width=width)

    def draw_polygon(self, points, **kwargs):
        width = kwargs.get('width', 2)
        flat_points = [coord for pt in points for coord in pt]
        self.c.create_polygon(flat_points, outline="black", width=width, fill="")

    def draw_arrow(self, x1, y1, x2, y2, **kwargs):
        width = kwargs.get('width', 2)
        self.c.create_line(x1, y1, x2, y2, arrow=tk.LAST, width=width, fill="black")

    def draw_t_stop(self, x, y, direction='up', **kwargs):
        size = kwargs.get('size', 5)
        width = kwargs.get('width', 2)
        if direction == 'up':
            self.c.create_line(x, y, x, y-size, width=width, fill="black") # stem
            self.c.create_line(x-size, y-size, x+size, y-size, width=width, fill="black") # bar
        else:
            self.c.create_line(x, y, x, y+size, width=width, fill="black")
            self.c.create_line(x-size, y+size, x+size, y+size, width=width, fill="black")

    def draw_zigzag(self, x, y, zig_width, height, horizontal=False, **kwargs):
        pts = []
        steps = 6
        
        if horizontal:
            step_w = height / steps
            pts.append((x, y))
            for i in range(1, steps):
                y_offset = -zig_width/2 if i % 2 != 0 else zig_width/2
                pts.append((x + i * step_w, y + y_offset))
            pts.append((x + height, y))
        else:
            step_h = height / steps
            pts.append((x, y))
            for i in range(1, steps):
                x_offset = -zig_width/2 if i % 2 != 0 else zig_width/2
                pts.append((x + x_offset, y + i * step_h))
            pts.append((x, y + height))
        
        stroke_width = kwargs.get('width', 2)
        flat_pts = [coord for pt in pts for coord in pt]
        self.c.create_line(flat_pts, width=stroke_width, fill="black")

    def draw_circle(self, x, y, r, **kwargs):
        width = kwargs.get('width', 2)
        fill_color = kwargs.get('fill', "") 
        self.c.create_oval(x-r, y-r, x+r, y+r, outline="black", width=width, fill=fill_color)
    
    def draw_text(self, x, y, text, font_size=10, **kwargs):
        self.c.create_text(x, y, text=text, fill="black", font=("Arial", int(font_size)))


class SvgRenderer(GraphicRenderer):
    """Generates an SVG string."""
    
    def __init__(self):
        self.elements = []

    def draw_line(self, x1, y1, x2, y2, **kwargs):
        w = kwargs.get('width', 2)
        self.elements.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="{w}" />')

    def draw_rect(self, x1, y1, x2, y2, **kwargs):
        w_rect = abs(x2 - x1)
        h_rect = abs(y2 - y1)
        rx = min(x1, x2)
        ry = min(y1, y2)
        stroke = kwargs.get('width', 2)
        self.elements.append(f'<rect x="{rx}" y="{ry}" width="{w_rect}" height="{h_rect}" fill="none" stroke="black" stroke-width="{stroke}" />')

    def draw_polygon(self, points, **kwargs):
        pts_str = " ".join([f"{x},{y}" for x, y in points])
        stroke = kwargs.get('width', 2)
        self.elements.append(f'<polygon points="{pts_str}" fill="none" stroke="black" stroke-width="{stroke}" />')

    def draw_arrow(self, x1, y1, x2, y2, **kwargs):
        width = kwargs.get('width', 2)
        self.draw_line(x1, y1, x2, y2, width=width)
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_len = 5 * width
        angle1 = angle + math.pi / 6 + math.pi
        angle2 = angle - math.pi / 6 + math.pi
        
        ax1 = x2 + arrow_len * math.cos(angle1)
        ay1 = y2 + arrow_len * math.sin(angle1)
        ax2 = x2 + arrow_len * math.cos(angle2)
        ay2 = y2 + arrow_len * math.sin(angle2)
        
        self.elements.append(f'<polygon points="{x2},{y2} {ax1},{ay1} {ax2},{ay2}" fill="black" />')

    def draw_t_stop(self, x, y, direction='up', **kwargs):
        size = kwargs.get('size', 5)
        width = kwargs.get('width', 2)
        if direction == 'up':
            self.draw_line(x, y, x, y-size, width=width) # stem
            self.draw_line(x-size, y-size, x+size, y-size, width=width) # bar
        else:
            self.draw_line(x, y, x, y+size, width=width)
            self.draw_line(x-size, y+size, x+size, y+size, width=width)

    def draw_zigzag(self, x, y, zig_width, height, horizontal=False, **kwargs):
        points = ""
        steps = 6
        
        if horizontal:
            step_w = height / steps
            points += f"{x},{y} "
            for i in range(1, steps):
                y_offset = -zig_width/2 if i % 2 != 0 else zig_width/2
                points += f"{x + i * step_w},{y + y_offset} "
            points += f"{x + height},{y}"
        else:
            step_h = height / steps
            points = f"{x},{y} "
            for i in range(1, steps):
                x_offset = -zig_width/2 if i % 2 != 0 else zig_width/2
                points += f"{x + x_offset},{y + i * step_h} "
            points += f"{x},{y + height}"

        stroke = kwargs.get('width', 2)
        self.elements.append(f'<polyline points="{points}" fill="none" stroke="black" stroke-width="{stroke}" />')

    def draw_circle(self, x, y, r, **kwargs):
        stroke = kwargs.get('width', 2)
        fill_color = kwargs.get('fill', "none")
        self.elements.append(f'<circle cx="{x}" cy="{y}" r="{r}" stroke="black" stroke-width="{stroke}" fill="{fill_color}" />')

    def draw_text(self, x, y, text, font_size=10, **kwargs):
        self.elements.append(f'<text x="{x}" y="{y}" fill="black" font-family="Arial" font-size="{font_size}" text-anchor="middle">{text}</text>')

    def get_svg(self, width, height):
        header = '<?xml version="1.0" encoding="UTF-8"?>\n'
        header += f'<svg width="{width}px" height="{height}px" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" version="1.1">'
        footer = '</svg>'
        return header + "".join(self.elements) + footer


class PneumaticDesignerApp:
    """Main application for designing ISO 1219 pneumatic symbols."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("ISO 1219 Pneumatic Symbol Designer")
        self.root.geometry("1100x700")

        # Configuration Data
        self.num_ports = tk.IntVar(value=4)
        self.num_states = tk.IntVar(value=2)
        self.zoom_level = tk.DoubleVar(value=1.0)
        
        self.left_ops = {
            "Spring": tk.BooleanVar(),
            "Solenoid": tk.BooleanVar(),
            "Lever": tk.BooleanVar(),
            "Pilot": tk.BooleanVar(),
            "Detent": tk.BooleanVar()
        }
        
        self.right_ops = {
            "Spring": tk.BooleanVar(),
            "Solenoid": tk.BooleanVar(),
            "Lever": tk.BooleanVar(),
            "Pilot": tk.BooleanVar(),
            "Detent": tk.BooleanVar()
        }

        self.state_configs = [] # List of StringVars
        self.state_error_labels = [] # parallel list of error label widgets
        
        self._init_ui()

    def _init_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_panel = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        ttk.Label(control_panel, text="Number of Ports:").grid(row=0, column=0, sticky="w", pady=5)
        port_cb = ttk.Combobox(control_panel, textvariable=self.num_ports, values=[2, 3, 4, 5], state="readonly", width=5)
        port_cb.grid(row=0, column=1, sticky="e")
        port_cb.bind("<<ComboboxSelected>>", self.rebuild_state_inputs)

        ttk.Label(control_panel, text="Number of States:").grid(row=1, column=0, sticky="w", pady=5)
        state_cb = ttk.Combobox(control_panel, textvariable=self.num_states, values=[2, 3], state="readonly", width=5)
        state_cb.grid(row=1, column=1, sticky="e")
        state_cb.bind("<<ComboboxSelected>>", self.rebuild_state_inputs)

        ttk.Label(control_panel, text="Zoom Level:").grid(row=2, column=0, sticky="w", pady=5)
        zoom_scale = tk.Scale(control_panel, variable=self.zoom_level, from_=0.5, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, length=100, command=lambda v: self.refresh_preview())
        zoom_scale.grid(row=2, column=1, sticky="e")

        ttk.Separator(control_panel, orient='horizontal').grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)

        op_frame = ttk.Frame(control_panel)
        op_frame.grid(row=4, column=0, columnspan=2, sticky="ew")
        
        ttk.Label(op_frame, text="Left Operator").grid(row=0, column=0, sticky="w")
        ttk.Label(op_frame, text="Right Operator").grid(row=0, column=1, sticky="w")

        r_idx = 1
        for name in self.left_ops:
            ttk.Checkbutton(op_frame, text=name, variable=self.left_ops[name], command=self.refresh_preview).grid(row=r_idx, column=0, sticky="w")
            ttk.Checkbutton(op_frame, text=name, variable=self.right_ops[name], command=self.refresh_preview).grid(row=r_idx, column=1, sticky="w")
            r_idx += 1

        ttk.Separator(control_panel, orient='horizontal').grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)

        self.state_input_frame = ttk.LabelFrame(control_panel, text="Flow Paths (e.g. 1-2, 3-T)")
        self.state_input_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=5)
        
        help_lbl = ttk.Label(control_panel, text="Format: '1-2' (connect),\n'1-T' (block).\nSeparate with commas.", font=("Arial", 8), foreground="gray")
        help_lbl.grid(row=7, column=0, columnspan=2)

        btn_frame = ttk.Frame(control_panel)
        btn_frame.grid(row=8, column=0, columnspan=2, pady=20)
        
        ttk.Button(btn_frame, text="Save LibreDraw (SVG)", command=self.save_svg).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save LibreDraw (.odg)", command=self.save_odg).pack(side=tk.LEFT, padx=5)

        self.canvas_frame = ttk.LabelFrame(main_frame, text="Preview", padding="10")
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.rebuild_state_inputs()

    def rebuild_state_inputs(self, event=None):
        for widget in self.state_input_frame.winfo_children():
            widget.destroy()
        
        self.state_configs.clear()
        self.state_error_labels.clear()
        
        n_states = self.num_states.get()
        
        for i in range(n_states):
            lbl_text = f"Pos {i+1} (Left):" if i == 0 else (f"Pos {i+1} (Right):" if i == n_states-1 else f"Pos {i+1} (Center):")
            
            ttk.Label(self.state_input_frame, text=lbl_text).grid(row=i, column=0, sticky="w", pady=2)
            
            var = tk.StringVar()
            if self.num_ports.get() == 4:
                if i == 0: var.set("1-2, 4-3")
                else: var.set("1-4, 2-3")
            elif self.num_ports.get() == 5:
                if i == 0: var.set("1-2, 4-5")
                else: var.set("1-4, 2-3")
            elif self.num_ports.get() == 3:
                if i == 0: var.set("1-T, 2-3")
                else: var.set("1-2, 3-T")
            
            entry = ttk.Entry(self.state_input_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, sticky="e", pady=2)
            entry.bind("<Return>", lambda e: self.refresh_preview())
            entry.bind("<FocusOut>", lambda e: self.refresh_preview())
            
            err_lbl = tk.Label(self.state_input_frame, text="", fg="red", font=("Arial", 8))
            err_lbl.grid(row=i, column=2, sticky="w", padx=(6,0))
            
            self.state_configs.append(var)
            self.state_error_labels.append(err_lbl)
        
        self.refresh_preview()

    def is_valid_connection(self, token):
        token = token.strip()
        if not token:
            return True
        if '-' not in token:
            return False
        parts = token.split('-')
        if len(parts) != 2:
            return False
        left = parts[0].strip()
        right = parts[1].strip()
        try:
            left_v = int(left)
            if left_v < 1 or left_v > self.num_ports.get():
                return False
        except ValueError:
            return False
        if right.upper() == 'T':
            return True
        try:
            right_v = int(right)
            if right_v < 1 or right_v > self.num_ports.get():
                return False
        except ValueError:
            return False
        return True

    def validate_all_state_inputs(self):
        invalid_indices = []
        for idx, var in enumerate(self.state_configs):
            raw = var.get()
            tokens = [s.strip() for s in raw.split(',')]
            bad = False
            for t in tokens:
                if t == "":
                    continue
                if not self.is_valid_connection(t):
                    bad = True
                    break
            lbl = self.state_error_labels[idx]
            if bad:
                lbl.config(text="Invalid format")
                invalid_indices.append(idx)
            else:
                lbl.config(text="")
        return invalid_indices

    def get_port_coords(self, port_num, box_x, box_y, box_w, box_h):
        ports = self.num_ports.get()
        pos_x = 0.5
        pos_y = 1.0
        
        if ports == 2:
            if port_num == 1: pos_x, pos_y = 0.5, 1.0
            if port_num == 2: pos_x, pos_y = 0.5, 0.0
            
        elif ports == 3:
            if port_num == 1: pos_x, pos_y = 0.5, 1.0
            if port_num == 2: pos_x, pos_y = 0.5, 0.0
            if port_num == 3: pos_x, pos_y = 0.8, 1.0
            
        elif ports == 4:
            if port_num == 1: pos_x, pos_y = 0.3, 1.0
            if port_num == 3: pos_x, pos_y = 0.7, 1.0
            if port_num == 2: pos_x, pos_y = 0.3, 0.0
            if port_num == 4: pos_x, pos_y = 0.7, 0.0
            
        elif ports == 5:
            if port_num == 1: pos_x, pos_y = 0.5, 1.0
            if port_num == 3: pos_x, pos_y = 0.8, 1.0
            if port_num == 5: pos_x, pos_y = 0.2, 1.0
            if port_num == 2: pos_x, pos_y = 0.8, 0.0
            if port_num == 4: pos_x, pos_y = 0.2, 0.0

        return (box_x + pos_x * box_w, box_y + pos_y * box_h)

    def draw_symbol_logic(self, r, center_x, center_y, scale=1.0, collect_glue_points=None):
        BOX_SIZE = 60 * scale
        LINE_WIDTH = 1 * scale
        FONT_SIZE = 10 * scale
        
        n_states = self.num_states.get()
        total_w = BOX_SIZE * n_states
        start_x = center_x - (total_w / 2)
        top_y = center_y - (BOX_SIZE / 2)
        
        # 1. Draw Main Boxes
        for i in range(n_states):
            bx = start_x + i * BOX_SIZE
            r.draw_rect(bx, top_y, bx + BOX_SIZE, top_y + BOX_SIZE, width=LINE_WIDTH)
            
            raw_data = self.state_configs[i].get()
            connections = [s.strip() for s in raw_data.split(',') if s.strip()]
            
            for conn in connections:
                if '-' in conn:
                    parts = conn.split('-')
                    src = parts[0].strip()
                    dst = parts[1].strip()
                    
                    try:
                        p_src = int(src)
                        sx, sy = self.get_port_coords(p_src, bx, top_y, BOX_SIZE, BOX_SIZE)
                        
                        if dst.upper() == 'T':
                            direction = 'up' if sy > top_y + BOX_SIZE/2 else 'down'
                            ty = sy - (10*scale) if direction == 'up' else sy + (10*scale)
                            r.draw_line(sx, sy, sx, ty, width=LINE_WIDTH)
                            r.draw_t_stop(sx, ty, direction, size=5*scale, width=LINE_WIDTH)
                        else:
                            p_dst = int(dst)
                            ex, ey = self.get_port_coords(p_dst, bx, top_y, BOX_SIZE, BOX_SIZE)
                            r.draw_arrow(sx, sy, ex, ey, width=LINE_WIDTH)
                    except ValueError:
                        pass 

        # 2. Draw Exterior Ports
        default_state_idx = n_states - 1 if n_states == 2 else 1
        ref_box_x = start_x + default_state_idx * BOX_SIZE
        
        for p in range(1, self.num_ports.get() + 1):
            px, py = self.get_port_coords(p, ref_box_x, top_y, BOX_SIZE, BOX_SIZE)
            
            lbl_y = py + (15*scale) if py > center_y else py - (15*scale)
            r.draw_text(px, lbl_y, str(p), font_size=FONT_SIZE)
            
            ext_y = py + (10*scale) if py > center_y else py - (10*scale)
            r.draw_line(px, py, px, ext_y, width=LINE_WIDTH)

            ANCHOR_RADIUS = ANCHOR_RADIUS_BASE * scale
            r.draw_circle(px, ext_y, ANCHOR_RADIUS, width=0.5*scale, fill="black")
            
            if collect_glue_points is not None:
                collect_glue_points.append((px, ext_y))
            
        # 3. Draw Operators
        OP_LENGTH = 30 * scale
        S_15 = 15 * scale
        S_10 = 10 * scale
        S_5  = 5 * scale
        
        OP_HEIGHT = BOX_SIZE / 3
        ly_center = center_y
        OP_Y_TOP = ly_center - (OP_HEIGHT / 2)
        OP_Y_BOT = ly_center + (OP_HEIGHT / 2)
        
        lx = start_x
        l_offset = 0
        
        if self.left_ops["Spring"].get():
            spring_amp = OP_HEIGHT 
            spring_len = OP_LENGTH
            r.draw_zigzag(lx - spring_len - l_offset, ly_center, spring_amp, spring_len, horizontal=True, width=LINE_WIDTH)
            l_offset += OP_LENGTH
            
        if self.left_ops["Solenoid"].get():
            r.draw_rect(lx - OP_LENGTH - l_offset, OP_Y_TOP, lx - l_offset, OP_Y_BOT, width=LINE_WIDTH)
            r.draw_line(lx - OP_LENGTH - l_offset, OP_Y_TOP, lx - l_offset, OP_Y_BOT, width=LINE_WIDTH) 
            l_offset += OP_LENGTH
            
        if self.left_ops["Pilot"].get():
            r.draw_polygon([
                (lx - l_offset, ly_center), 
                (lx - l_offset - S_15, ly_center - S_10), 
                (lx - l_offset - S_15, ly_center + S_10)
            ], width=LINE_WIDTH)
            l_offset += S_15
            
        if self.left_ops["Detent"].get():
            dx1 = lx - l_offset - OP_LENGTH
            dx2 = lx - l_offset
            center_x_notch = dx1 + OP_LENGTH / 2
            notch_start_x = center_x_notch - S_5
            notch_end_x = center_x_notch + S_5
            notch_y_peak = OP_Y_TOP + S_5

            r.draw_line(dx1, OP_Y_TOP, dx1, OP_Y_BOT, width=LINE_WIDTH)
            r.draw_line(dx2, OP_Y_TOP, dx2, OP_Y_BOT, width=LINE_WIDTH)
            r.draw_line(dx1, OP_Y_BOT, dx2, OP_Y_BOT, width=LINE_WIDTH)
            r.draw_line(dx1, OP_Y_TOP, notch_start_x, OP_Y_TOP, width=LINE_WIDTH)
            r.draw_line(notch_end_x, OP_Y_TOP, dx2, OP_Y_TOP, width=LINE_WIDTH)
            r.draw_line(notch_start_x, OP_Y_TOP, center_x_notch, notch_y_peak, width=LINE_WIDTH)
            r.draw_line(center_x_notch, notch_y_peak, notch_end_x, OP_Y_TOP, width=LINE_WIDTH)
            
            l_offset += OP_LENGTH
            
        if self.left_ops["Lever"].get():
            w_top = 25 * scale
            w_bot = 10 * scale
            lever_h = BOX_SIZE / 3
            lever_top = ly_center - (lever_h / 2)
            lever_bot = ly_center + (lever_h / 2)
            
            p_wall_top = (lx - l_offset, lever_top)
            p_wall_bot = (lx - l_offset, lever_bot)
            p_outer_bot = (lx - l_offset - w_bot, lever_bot)
            p_outer_top = (lx - l_offset - w_top, lever_top)
            
            r.draw_polygon([p_wall_top, p_wall_bot, p_outer_bot, p_outer_top], width=LINE_WIDTH)
            
            vx = p_outer_top[0] - p_outer_bot[0]
            vy = p_outer_top[1] - p_outer_bot[1]
            
            v_len = math.sqrt(vx*vx + vy*vy)
            if v_len > 1e-6:
                nx, ny = vx/v_len, vy/v_len
                handle_len = 25 * scale
                hx, hy = p_outer_top[0] + nx * handle_len, p_outer_top[1] + ny * handle_len
                r.draw_line(p_outer_top[0], p_outer_top[1], hx, hy, width=LINE_WIDTH)
                r.draw_circle(hx, hy, S_5, width=LINE_WIDTH)
            l_offset += w_top

        rx = start_x + total_w
        r_offset = 0
        
        if self.right_ops["Spring"].get():
            spring_amp = OP_HEIGHT
            spring_len = OP_LENGTH
            r.draw_zigzag(rx + r_offset, ly_center, spring_amp, spring_len, horizontal=True, width=LINE_WIDTH)
            r_offset += OP_LENGTH
            
        if self.right_ops["Solenoid"].get():
            r.draw_rect(rx + r_offset, OP_Y_TOP, rx + r_offset + OP_LENGTH, OP_Y_BOT, width=LINE_WIDTH)
            r.draw_line(rx + r_offset, OP_Y_TOP, rx + r_offset + OP_LENGTH, OP_Y_BOT, width=LINE_WIDTH)
            r_offset += OP_LENGTH
            
        if self.right_ops["Pilot"].get():
            r.draw_polygon([
                (rx + r_offset, ly_center), 
                (rx + r_offset + S_15, ly_center - S_10), 
                (rx + r_offset + S_15, ly_center + S_10)
            ], width=LINE_WIDTH)
            r_offset += S_15
            
        if self.right_ops["Detent"].get():
            dx1 = rx + r_offset
            dx2 = rx + r_offset + OP_LENGTH
            center_x_notch = dx1 + OP_LENGTH / 2
            notch_start_x = center_x_notch - S_5
            notch_end_x = center_x_notch + S_5
            notch_y_peak = OP_Y_TOP + S_5

            r.draw_line(dx1, OP_Y_TOP, dx1, OP_Y_BOT, width=LINE_WIDTH)
            r.draw_line(dx2, OP_Y_TOP, dx2, OP_Y_BOT, width=LINE_WIDTH)
            r.draw_line(dx1, OP_Y_BOT, dx2, OP_Y_BOT, width=LINE_WIDTH)
            r.draw_line(dx1, OP_Y_TOP, notch_start_x, OP_Y_TOP, width=LINE_WIDTH)
            r.draw_line(notch_end_x, OP_Y_TOP, dx2, OP_Y_TOP, width=LINE_WIDTH)
            r.draw_line(notch_start_x, OP_Y_TOP, center_x_notch, notch_y_peak, width=LINE_WIDTH)
            r.draw_line(center_x_notch, notch_y_peak, notch_end_x, OP_Y_TOP, width=LINE_WIDTH)
            r_offset += OP_LENGTH
            
        if self.right_ops["Lever"].get():
            w_top = 25 * scale
            w_bot = 10 * scale
            lever_h = BOX_SIZE / 3
            lever_top = ly_center - (lever_h / 2)
            lever_bot = ly_center + (lever_h / 2)
            
            p_wall_top = (rx + r_offset, lever_top)
            p_wall_bot = (rx + r_offset, lever_bot)
            p_outer_bot = (rx + r_offset + w_bot, lever_bot)
            p_outer_top = (rx + r_offset + w_top, lever_top)
            
            r.draw_polygon([p_wall_top, p_wall_bot, p_outer_bot, p_outer_top], width=LINE_WIDTH)
            
            vx = p_outer_top[0] - p_outer_bot[0] 
            vy = p_outer_top[1] - p_outer_bot[1]
            
            v_len = math.sqrt(vx*vx + vy*vy)
            if v_len > 1e-6:
                nx, ny = vx/v_len, vy/v_len
                handle_len = 25 * scale
                hx, hy = p_outer_top[0] + nx * handle_len, p_outer_top[1] + ny * handle_len
                r.draw_line(p_outer_top[0], p_outer_top[1], hx, hy, width=LINE_WIDTH)
                r.draw_circle(hx, hy, S_5, width=LINE_WIDTH)
            r_offset += w_top

    def refresh_preview(self):
        self.validate_all_state_inputs()
        
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10: w, h = 400, 300
        
        scale = self.zoom_level.get()
        renderer = CanvasRenderer(self.canvas)
        self.draw_symbol_logic(renderer, w/2, h/2, scale=scale)

    def save_svg(self):
        invalid = self.validate_all_state_inputs()
        if invalid:
            proceed = messagebox.askyesno(
                "Invalid inputs",
                "Some state flow inputs are invalid. Fix them before exporting?\n\n"
                "Press 'No' to cancel save and edit inputs, or 'Yes' to continue exporting anyway."
            )
            if not proceed:
                return

        filename = filedialog.asksaveasfilename(defaultextension=".svg", 
                                                filetypes=[("SVG files", "*.svg"), ("All files", "*.*")],
                                                title="Save as SVG (LibreOffice Draw Compatible)")
        if not filename:
            return

        svg_r = SvgRenderer()
        glue_points = []
        self.draw_symbol_logic(svg_r, 300, 200, scale=1.0, collect_glue_points=glue_points) 
        
        content = svg_r.get_svg(600, 400)
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo(
                "Success",
                "File saved successfully!\n\n"
                "You can open this .svg file in LibreOffice Draw. The small black dots at the port ends are intended to serve as reliable snap/glue points when editing the diagram in Draw."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")

    def save_odg(self):
        invalid = self.validate_all_state_inputs()
        if invalid:
            proceed = messagebox.askyesno(
                "Invalid inputs",
                "Some state flow inputs are invalid. Fix them before exporting?\n\n"
                "Press 'No' to cancel save and edit inputs, or 'Yes' to continue exporting anyway."
            )
            if not proceed:
                return

        filename = filedialog.asksaveasfilename(defaultextension=".odg", 
                                                filetypes=[("LibreOffice Draw", "*.odg"), ("All files", "*.*")],
                                                title="Save as LibreOffice Draw (.odg)")
        if not filename:
            return

        svg_r = SvgRenderer()
        glue_points = []
        self.draw_symbol_logic(svg_r, 300, 200, scale=1.0, collect_glue_points=glue_points)
        svg_content = svg_r.get_svg(600, 400)

        try:
            self._write_minimal_odg(filename, svg_content, glue_points, svg_width=600, svg_height=400)
            messagebox.showinfo("Success", f"Saved {os.path.basename(filename)}\n\nYou can open it in LibreOffice Draw. Connection/glue points have been added based on the port endpoints.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save .odg file: {e}")

    def _write_minimal_odg(self, odg_path, svg_content, glue_points, svg_width=600, svg_height=400):
        """Create a minimal ODG package containing the SVG and a content.xml that references it.

        Changes made to address:
         - Grouping: All elements are wrapped inside a draw:g group so the whole symbol
           behaves as a single grouped object in LibreOffice Draw (connectors will snap to group).
         - Glue points: Remain inside the frame, and the frame is inside the group.
         - No background: we do not add any background rectangle. Embedded SVG is expected to be transparent.
         - Minimal required parts (mimetype first, manifest, content.xml, styles.xml, meta.xml, Pictures/).
        """
        manifest = f'''<?xml version="1.0" encoding="UTF-8"?>
<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0">
  <manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.graphics"/>
  <manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>
  <manifest:file-entry manifest:full-path="styles.xml" manifest:media-type="text/xml"/>
  <manifest:file-entry manifest:full-path="meta.xml" manifest:media-type="text/xml"/>
  <manifest:file-entry manifest:full-path="Pictures/diagram.svg" manifest:media-type="image/svg+xml"/>
</manifest:manifest>'''

        # Convert glue points (pixel coords) to percentage positions relative to the SVG dims.
        connection_points_xml = ""
        for idx, (gx, gy) in enumerate(glue_points, start=1):
            px = max(0.0, min(100.0, (gx / float(svg_width)) * 100.0))
            py = max(0.0, min(100.0, (gy / float(svg_height)) * 100.0))
            connection_points_xml += f'                <draw:connection-point draw:name="GP{idx}" draw:position-x="{px:.2f}%" draw:position-y="{py:.2f}%" />\n'

        # Group wrapper (draw:g) containing a draw:frame with the embedded SVG and the connection points.
        # Keeping connection points as children of the frame (so Draw recognizes them) while the frame itself
        # is a child of the group so the whole symbol is grouped.
        content_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
    xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
    xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0"
    xmlns:xlink="http://www.w3.org/1999/xlink"
    xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    office:version="1.2">
  <office:body>
    <office:drawing>
      <draw:page draw:name="page1">
        <!-- Group that represents the whole symbol. Using draw:g per ODF spec -->
        <draw:g draw:name="symbol-group" draw:z-index="0" svg:x="0cm" svg:y="0cm" svg:width="{svg_width}px" svg:height="{svg_height}px">
          <!-- Frame hosting the SVG picture. The frame is inside the group so the whole symbol is grouped. -->
          <draw:frame draw:name="diagram-frame" draw:z-index="0" svg:x="0cm" svg:y="0cm" svg:width="{svg_width}px" svg:height="{svg_height}px">
            <draw:image xlink:href="Pictures/diagram.svg" xlink:type="simple" xlink:show="embed" xlink:actuate="onLoad"/>
            <!-- Connection points are placed inside the frame so LibreOffice Draw recognizes them -->
            <draw:connection-points>
{connection_points_xml}            </draw:connection-points>
          </draw:frame>
        </draw:g>
      </draw:page>
    </office:drawing>
  </office:body>
</office:document-content>'''

        # Minimal styles.xml and meta.xml
        styles_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<office:document-styles
    xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    office:version="1.2">
  <office:styles/>
  <office:automatic-styles/>
  <office:master-styles/>
</office:document-styles>'''

        meta_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<office:document-meta
    xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    office:version="1.2">
  <office:meta>
    <meta:initial-creator>pneumatic-symbol-designer</meta:initial-creator>
  </office:meta>
</office:document-meta>'''

        # Write ODG (zip) with mimetype first and stored (no compression).
        with zipfile.ZipFile(odg_path, mode='w') as zf:
            zf.writestr("mimetype", "application/vnd.oasis.opendocument.graphics", compress_type=zipfile.ZIP_STORED)
            zf.writestr("META-INF/manifest.xml", manifest)
            zf.writestr("content.xml", content_xml)
            zf.writestr("styles.xml", styles_xml)
            zf.writestr("meta.xml", meta_xml)
            zf.writestr("Pictures/diagram.svg", svg_content)


if __name__ == "__main__":
    root = tk.Tk()
    app = PneumaticDesignerApp(root)
    root.after(100, app.refresh_preview)
    root.mainloop()