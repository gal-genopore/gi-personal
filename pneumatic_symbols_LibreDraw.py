import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import math
import zipfile
import os
import logging
from html import escape as escape_xml

# Constants
ANCHOR_RADIUS_BASE = 1.5
SVG_PADDING = 10
DPI_DEFAULT = 96.0

logger = logging.getLogger("pneumatic_symbols")


class GraphicRenderer:
    def draw_line(self, x1, y1, x2, y2, **kwargs): pass
    def draw_rect(self, x1, y1, x2, y2, **kwargs): pass
    def draw_polygon(self, points, **kwargs): pass
    def draw_polyline(self, points, **kwargs): pass
    def draw_arrow(self, x1, y1, x2, y2, **kwargs): pass
    def draw_t_stop(self, x, y, direction='up', **kwargs): pass
    def draw_zigzag(self, x, y, zig_width, height, horizontal=False, **kwargs): pass
    def draw_circle(self, x, y, r, **kwargs): pass
    def draw_text(self, x, y, text, font_size=10, **kwargs): pass


class CanvasRenderer(GraphicRenderer):
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
        flat = [coord for pt in points for coord in pt]
        self.c.create_polygon(flat, outline="black", width=width, fill="")
    def draw_polyline(self, points, **kwargs):
        width = kwargs.get('width', 2)
        flat = [coord for pt in points for coord in pt]
        self.c.create_polygon(flat, outline="black", width=width, fill="")
    def draw_circle(self, x, y, r, **kwargs):
        width = kwargs.get('width', 2)
        fill = kwargs.get('fill', "")
        self.c.create_oval(x-r, y-r, x+r, y+r, outline="black", width=width, fill='black')
    def draw_text(self, x, y, text, font_size=10, **kwargs):
        self.c.create_text(x, y, text=text, fill="black", font=("Arial", int(font_size)))
    def draw_zigzag(self, x, y, zig_dim, length, horizontal=False, **kwargs):
        width = kwargs.get('width', 2)
        num_steps = 5
        step_size = length / num_steps
        pts = []
        for i in range(num_steps + 1):
            pos = i * step_size
            offset = zig_dim/2 if i % 2 == 1 else -zig_dim/2
            if horizontal:
                pts.extend([x + pos, y + offset])
            else:
                pts.extend([x + offset, y + pos])
        self.c.create_line(pts, width=width, fill="black")
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

class SvgRenderer(GraphicRenderer):
    def __init__(self):
        self.elements = []
        self.min_x = float('inf'); self.min_y = float('inf')
        self.max_x = float('-inf'); self.max_y = float('-inf')
    def _update(self, x, y):
        if x < self.min_x: self.min_x = x
        if x > self.max_x: self.max_x = x
        if y < self.min_y: self.min_y = y
        if y > self.max_y: self.max_y = y
    def _update_bounds(self, x, y):
        if x < self.min_x: self.min_x = x
        if x > self.max_x: self.max_x = x
        if y < self.min_y: self.min_y = y
        if y > self.max_y: self.max_y = y
    def draw_rect(self, x1, y1, x2, y2, **kwargs):
        rx, ry = min(x1,x2), min(y1,y2)
        w, h = abs(x2-x1), abs(y2-y1)
        self.elements.append(f'<rect x="{rx}" y="{ry}" width="{w}" height="{h}" stroke="black" fill="none"/>')
        self._update(x1,y1); self._update(x2,y2)
    def draw_line(self, x1, y1, x2, y2, **kwargs):
        self.elements.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black"/>')
        self._update(x1,y1); self._update(x2,y2)
    def draw_circle(self, x, y, r, **kwargs):
        self.elements.append(f'<circle cx="{x}" cy="{y}" r="{r}" stroke="black" fill="black"/>')
        self._update(x-r,y-r); self._update(x+r,y+r)
    def draw_polygon(self, points, **kwargs):
        pts = " ".join([f"{x},{y}" for (x,y) in points])
        fill = kwargs.get('fill', 'none')
        self.elements.append(f'<polygon points="{pts}" stroke="black" fill="{fill}"/>')
        for x,y in points: self._update(x,y)
    def draw_polyline(self, points, **kwargs):
        pts = " ".join([f"{x},{y}" for (x,y) in points])
        self.elements.append(f'<polyline points="{pts}" stroke="black" fill="none"/>')
        for x,y in points: self._update(x,y)
    def draw_text(self, x, y, text, font_size=10, **kwargs):
        self.elements.append(f'<text x="{x}" y="{y}" font-size="{font_size}">{escape_xml(text)}</text>')
        self._update(x,y)
    def draw_zigzag(self, x, y, zig_dim, length, horizontal=False, **kwargs):
        num_steps = 5
        step_size = length / num_steps
        pts_list = []
        for i in range(num_steps + 1):
            pos = i * step_size
            offset = zig_dim/2 if i % 2 == 1 else -zig_dim/2
            curr_x = x + (pos if horizontal else offset)
            curr_y = y + (offset if horizontal else pos)
            pts_list.append(f"{curr_x},{curr_y}")
            self._update(curr_x, curr_y)
        pts_str = " ".join(pts_list)
        self.elements.append(f'<polyline points="{pts_str}" stroke="black" fill="none"/>')
    def draw_arrow(self, x1, y1, x2, y2, **kwargs):
        width = kwargs.get('width', 2)
        # Use underlying draw methods so bounds are updated automatically
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
        self._update_bounds(x2, y2)
        self._update_bounds(ax1, ay1)
        self._update_bounds(ax2, ay2)

    def draw_t_stop(self, x, y, direction='up', **kwargs):
        size = kwargs.get('size', 5)
        width = kwargs.get('width', 2)
        if direction == 'up':
            self.draw_line(x, y, x, y-size, width=width)
            self.draw_line(x-size, y-size, x+size, y-size, width=width)
        else:
            self.draw_line(x, y, x, y+size, width=width)
            self.draw_line(x-size, y+size, x+size, y+size, width=width)
    def get_bounds(self):
        if self.min_x == float('inf'):
            return (0,0,100,100)
        return (self.min_x, self.min_y, self.max_x, self.max_y)
    def get_svg(self, view_box=None):
        if view_box:
            vx, vy, vw, vh = view_box
            header = f'<?xml version="1.0" encoding="UTF-8"?><svg width="{vw}px" height="{vh}px" viewBox="{vx} {vy} {vw} {vh}" xmlns="http://www.w3.org/2000/svg">'
        else:
            header = '<?xml version="1.0" encoding="UTF-8"?><svg xmlns="http://www.w3.org/2000/svg">'
        return header + "".join(self.elements) + "</svg>"


class OdfRenderer(GraphicRenderer):
    def __init__(self, view_x_px, view_y_px, px_to_cm):
        self.view_x = view_x_px
        self.view_y = view_y_px
        self.px_to_cm = px_to_cm
        self.elements = []

    def _px_to_cm(self, px):
        return px * self.px_to_cm

    def _rel_px(self, x_px, y_px):
        return x_px - self.view_x, y_px - self.view_y

    def draw_rect(self, x1, y1, x2, y2, **kwargs):
        rx_px, ry_px = min(x1,x2), min(y1,y2)
        w_px, h_px = abs(x2-x1), abs(y2-y1)
        x_cm, y_cm = self._px_to_cm(rx_px - self.view_x), self._px_to_cm(ry_px - self.view_y)
        w_cm, h_cm = self._px_to_cm(w_px), self._px_to_cm(h_px)
        geom = '<draw:enhanced-geometry svg:viewBox="0 0 21600 21600" draw:type="rectangle" draw:enhanced-path="M 0 0 L 21600 0 21600 21600 0 21600 0 0 Z N"/>'
        el = (f'<draw:custom-shape draw:style-name="gr1" svg:width="{w_cm:.4f}cm" svg:height="{h_cm:.4f}cm" '
              f'svg:x="{x_cm:.4f}cm" svg:y="{y_cm:.4f}cm"><text:p/>{geom}</draw:custom-shape>')
        self.elements.append(el)

    def draw_line(self, x1, y1, x2, y2, **kwargs):
        x1_cm, y1_cm = self._px_to_cm(x1 - self.view_x), self._px_to_cm(y1 - self.view_y)
        x2_cm, y2_cm = self._px_to_cm(x2 - self.view_x), self._px_to_cm(y2 - self.view_y)
        el = f'<draw:line draw:style-name="gr1" svg:x1="{x1_cm:.4f}cm" svg:y1="{y1_cm:.4f}cm" svg:x2="{x2_cm:.4f}cm" svg:y2="{y2_cm:.4f}cm"><text:p/></draw:line>'
        self.elements.append(el)

    def draw_circle(self, cx, cy, r, **kwargs):
        tl_x_px, tl_y_px = cx - r, cy - r
        x_cm, y_cm = self._px_to_cm(tl_x_px - self.view_x), self._px_to_cm(tl_y_px - self.view_y)
        w_cm = h_cm = self._px_to_cm(2*r)
        geom = '<draw:enhanced-geometry svg:viewBox="0 0 21600 21600" draw:type="ellipse" draw:enhanced-path="U 10800 10800 10800 10800 0 360 Z N"/>'
        el = (f'<draw:custom-shape draw:style-name="filled_black" svg:width="{w_cm:.4f}cm" svg:height="{h_cm:.4f}cm" '
              f'svg:x="{x_cm:.4f}cm" svg:y="{y_cm:.4f}cm"><text:p/>{geom}</draw:custom-shape>')
        self.elements.append(el)

    def draw_polyline(self, points, **kwargs):
        rel_pts = [(x - self.view_x, y - self.view_y) for (x, y) in points]
        minx, miny = min(p[0] for p in rel_pts), min(p[1] for p in rel_pts)
        maxx, maxy = max(p[0] for p in rel_pts), max(p[1] for p in rel_pts)
        w, h = max(1.0, maxx - minx), max(1.0, maxy - miny)
        
        # Internal integer scaling for ODF points
        pts_attr = " ".join([f"{int((p[0]-minx)*(1000/w))},{int((p[1]-miny)*(1000/h))}" for p in rel_pts])
        x_cm, y_cm = self._px_to_cm(minx), self._px_to_cm(miny)
        w_cm, h_cm = self._px_to_cm(w), self._px_to_cm(h)
        
        el = (f'<draw:polyline draw:style-name="gr1" svg:width="{w_cm:.4f}cm" svg:height="{h_cm:.4f}cm" '
              f'svg:x="{x_cm:.4f}cm" svg:y="{y_cm:.4f}cm" svg:viewBox="0 0 1000 1000" draw:points="{pts_attr}"><text:p/></draw:polyline>')
        self.elements.append(el)

    def draw_polygon(self, points, **kwargs):
        # Calculate relative points and bounding box
        rel_pts = [(x - self.view_x, y - self.view_y) for (x, y) in points]
        minx, miny = min(p[0] for p in rel_pts), min(p[1] for p in rel_pts)
        maxx, maxy = max(p[0] for p in rel_pts), max(p[1] for p in rel_pts)
        w, h = max(1.0, maxx - minx), max(1.0, maxy - miny)
        
        # Internal integer scaling for ODF points
        pts_attr = " ".join([f"{int((p[0]-minx)*(1000/w))},{int((p[1]-miny)*(1000/h))}" for p in rel_pts])
        x_cm, y_cm = self._px_to_cm(minx), self._px_to_cm(miny)
        w_cm, h_cm = self._px_to_cm(w), self._px_to_cm(h)
        
        # Note: You must ensure 'gr1' is defined in your ODF style section
        style_name = kwargs.get("style_name", "gr1")
        
        # 2. Use <draw:polygon>
        el = (f'<draw:polygon draw:style-name="{style_name}" '
            f'svg:width="{w_cm:.4f}cm" svg:height="{h_cm:.4f}cm" '
            f'svg:x="{x_cm:.4f}cm" svg:y="{y_cm:.4f}cm" '
            f'svg:viewBox="0 0 1000 1000" '
            f'draw:points="{pts_attr}"><text:p/></draw:polygon>')
        
        self.elements.append(el)

    def draw_text(self, x, y, text, **kwargs):
        # Center text box roughly on coordinate
        x_cm, y_cm = self._px_to_cm(x - 10 - self.view_x), self._px_to_cm(y - 5 - self.view_y)
        el = (f'<draw:text-box svg:x="{x_cm:.4f}cm" svg:y="{y_cm:.4f}cm" svg:width="0.8cm" svg:height="0.4cm">'
              f'<text:p>{escape_xml(str(text))}</text:p></draw:text-box>')
        self.elements.append(el)

    def draw_zigzag(self, x, y, zig_dim, length, horizontal=False, **kwargs):
        num_steps = 5
        step_size = length / num_steps
        points = []
        for i in range(num_steps + 1):
            pos = i * step_size
            offset = zig_dim/2 if i % 2 == 1 else -zig_dim/2
            curr_x = x + (pos if horizontal else offset)
            curr_y = y + (offset if horizontal else pos)
            points.append((curr_x, curr_y))
        
        # Reuse the ODF polygon logic to handle scaling and XML wrapping
        self.draw_polyline(points, **kwargs)

    def draw_arrow(self, x1, y1, x2, y2, **kwargs):
        # 1. Draw the main line
        self.draw_line(x1, y1, x2, y2, **kwargs)
        
        # 2. Calculate triangle head
        angle = math.atan2(y2 - y1, x2 - x1)
        length = 8
        width = 4
        
        p1 = (x2, y2)
        p2 = (x2 - length * math.cos(angle) + width * math.sin(angle),
              y2 - length * math.sin(angle) - width * math.cos(angle))
        p3 = (x2 - length * math.cos(angle) - width * math.sin(angle),
              y2 - length * math.sin(angle) + width * math.cos(angle))
        
        # 3. Draw head
        self.draw_polygon([p1, p2, p3], style_name="filled_black", **kwargs)

    def draw_t_stop(self, x, y, direction='up', **kwargs):
        size = kwargs.get('size', 5)
        # Draw the horizontal bar of the 'T'
        if direction in ['up', 'down']:
            self.draw_line(x - size, y, x + size, y, **kwargs)
        else:  # left or right
            self.draw_line(x, y - size, x, y + size, **kwargs)

    def get_xml_fragment(self):
        return f'<draw:g draw:name="PneumaticGroup">\n{"".join(self.elements)}\n</draw:g>'
    
class PneumaticDesignerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ISO 1219 Pneumatic Symbol Designer")
        self.root.geometry("1100x700")
        self.num_ports = tk.IntVar(value=4)
        self.num_states = tk.IntVar(value=2)
        self.zoom_level = tk.DoubleVar(value=1.0)
        self.left_ops = {k: tk.BooleanVar() for k in ["Spring","Solenoid","Lever","Pilot","Detent"]}
        self.right_ops = {k: tk.BooleanVar() for k in ["Spring","Solenoid","Lever","Pilot","Detent"]}
        self.state_configs = []
        self.state_error_labels = []
        self._init_ui()

    def _init_ui(self):
        main_frame = ttk.Frame(self.root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        control_panel = ttk.LabelFrame(main_frame, text="Configuration", padding="10"); control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        ttk.Label(control_panel, text="Number of Ports:").grid(row=0, column=0, sticky="w", pady=5)
        port_cb = ttk.Combobox(control_panel, textvariable=self.num_ports, values=[2,3,4,5], state="readonly", width=5); port_cb.grid(row=0,column=1,sticky="e"); port_cb.bind("<<ComboboxSelected>>", self.rebuild_state_inputs)
        ttk.Label(control_panel, text="Number of States:").grid(row=1, column=0, sticky="w", pady=5)
        state_cb = ttk.Combobox(control_panel, textvariable=self.num_states, values=[2,3], state="readonly", width=5); state_cb.grid(row=1,column=1,sticky="e"); state_cb.bind("<<ComboboxSelected>>", self.rebuild_state_inputs)
        ttk.Label(control_panel, text="Zoom Level:").grid(row=2, column=0, sticky="w", pady=5)
        zoom_scale = tk.Scale(control_panel, variable=self.zoom_level, from_=0.5, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, length=100, command=lambda v: self.refresh_preview()); zoom_scale.grid(row=2,column=1,sticky="e")
        ttk.Separator(control_panel, orient='horizontal').grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)
        op_frame = ttk.Frame(control_panel); op_frame.grid(row=4, column=0, columnspan=2, sticky="ew")
        ttk.Label(op_frame, text="Left Operator").grid(row=0, column=0, sticky="w"); ttk.Label(op_frame, text="Right Operator").grid(row=0, column=1, sticky="w")
        r_idx = 1
        for name in self.left_ops:
            ttk.Checkbutton(op_frame, text=name, variable=self.left_ops[name], command=self.refresh_preview).grid(row=r_idx, column=0, sticky="w")
            ttk.Checkbutton(op_frame, text=name, variable=self.right_ops[name], command=self.refresh_preview).grid(row=r_idx, column=1, sticky="w")
            r_idx += 1
        ttk.Separator(control_panel, orient='horizontal').grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)
        self.state_input_frame = ttk.LabelFrame(control_panel, text="Flow Paths (e.g. 1-2, 3-T)"); self.state_input_frame.grid(row=6,column=0,columnspan=2,sticky="ew", pady=5)
        help_lbl = ttk.Label(control_panel, text="Format: '1-2' (connect),\n'1-T' (block).\nSeparate with commas.", font=("Arial",8), foreground="gray"); help_lbl.grid(row=7,column=0,columnspan=2)
        btn_frame = ttk.Frame(control_panel); btn_frame.grid(row=8,column=0,columnspan=2,pady=20)
        ttk.Button(btn_frame, text="Save LibreDraw (SVG)", command=self.save_svg).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save LibreDraw (.odg)", command=self.save_odg).pack(side=tk.LEFT, padx=5)
        self.canvas_frame = ttk.LabelFrame(main_frame, text="Preview", padding="10"); self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.canvas_frame, bg="white"); self.canvas.pack(fill=tk.BOTH, expand=True)
        self.rebuild_state_inputs()

    def rebuild_state_inputs(self, event=None):
        for w in self.state_input_frame.winfo_children(): w.destroy()
        self.state_configs.clear(); self.state_error_labels.clear()
        n_states = self.num_states.get()
        for i in range(n_states):
            lbl_text = f"Pos {i+1} (Left):" if i==0 else (f"Pos {i+1} (Right):" if i==n_states-1 else f"Pos {i+1} (Center):")
            ttk.Label(self.state_input_frame, text=lbl_text).grid(row=i,column=0,sticky="w",pady=2)
            var = tk.StringVar()
            if self.num_ports.get()==4:
                var.set("1-2, 4-3" if i==0 else "1-4, 2-3")
            elif self.num_ports.get()==5:
                var.set("1-2, 4-5" if i==0 else "1-4, 2-3")
            elif self.num_ports.get()==3:
                var.set("1-T, 2-3" if i==0 else "1-2, 3-T")
            entry = ttk.Entry(self.state_input_frame, textvariable=var, width=15); entry.grid(row=i,column=1,sticky="e",pady=2)
            entry.bind("<Return>", lambda e: self.refresh_preview()); entry.bind("<FocusOut>", lambda e: self.refresh_preview())
            err_lbl = tk.Label(self.state_input_frame, text="", fg="red", font=("Arial",8)); err_lbl.grid(row=i,column=2,sticky="w", padx=(6,0))
            self.state_configs.append(var); self.state_error_labels.append(err_lbl)
        self.refresh_preview()

    def is_valid_connection(self, token):
        token = token.strip()
        if not token: return True
        if '-' not in token: return False
        parts = token.split('-')
        if len(parts)!=2: return False
        left,right = parts[0].strip(), parts[1].strip()
        try:
            lv = int(left); 
            if lv <1 or lv>self.num_ports.get(): return False
        except ValueError:
            return False
        if right.upper()=='T': return True
        try:
            rv=int(right)
            if rv<1 or rv>self.num_ports.get(): return False
        except ValueError:
            return False
        return True

    def validate_all_state_inputs(self):
        invalid=[]
        for idx,var in enumerate(self.state_configs):
            raw=var.get(); tokens=[s.strip() for s in raw.split(',')]; bad=False
            for t in tokens:
                if t=="" : continue
                if not self.is_valid_connection(t): bad=True; break
            lbl=self.state_error_labels[idx]
            if bad: lbl.config(text="Invalid format"); invalid.append(idx)
            else: lbl.config(text="")
        return invalid

    def get_port_coords(self, port_num, box_x, box_y, box_w, box_h):
        ports = self.num_ports.get()
        mapping = {
            2: {1: (0.5, 1.0), 2: (0.5, 0.0)},
            3: {1: (0.5, 1.0), 2: (0.5, 0.0), 3: (0.8, 1.0)},
            4: {1: (0.3, 1.0), 3: (0.7, 1.0), 2: (0.3, 0.0), 4: (0.7, 0.0)},
            5: {1: (0.5, 1.0), 3: (0.8, 1.0), 5: (0.2, 1.0), 2: (0.8, 0.0), 4: (0.2, 0.0)}
        }
        pos_x, pos_y = mapping.get(ports, {}).get(port_num, (0.5, 0.5))
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
            r.draw_circle(px, ext_y, ANCHOR_RADIUS, width=0.5*scale)
            
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
        w = self.canvas.winfo_width(); h = self.canvas.winfo_height()
        if w < 10: w,h = 400,300
        scale = self.zoom_level.get()
        renderer = CanvasRenderer(self.canvas)
        self.draw_symbol_logic(renderer, w/2, h/2, scale=scale)

    def save_svg(self):
        invalid = self.validate_all_state_inputs()
        if invalid:
            if not messagebox.askyesno("Invalid inputs","Some inputs invalid. Continue export?"): return
        filename = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG files","*.svg"),("All files","*.*")])
        if not filename: return
        svg_r = SvgRenderer(); glue_points=[]
        self.draw_symbol_logic(svg_r, 300, 200, scale=1.0, collect_glue_points=glue_points)
        bx,by,bX,bY = svg_r.get_bounds(); padding = SVG_PADDING
        view_x = bx - padding; view_y = by - padding
        view_w = (bX - bx) + 2*padding; view_h = (bY - by) + 2*padding
        content = svg_r.get_svg(view_box=(view_x, view_y, view_w, view_h))
        try:
            with open(filename, "w", encoding="utf-8") as f: f.write(content)
            logger.info("Saved SVG: %s (view_box=(%.2f,%.2f,%.2f,%.2f))", filename, view_x, view_y, view_w, view_h)
            logger.debug("Glue points: %s", glue_points)
            messagebox.showinfo("Success","SVG saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    def save_odg(self):
        invalid = self.validate_all_state_inputs()
        if invalid:
            if not messagebox.askyesno("Invalid inputs","Some inputs invalid. Continue export?"): return
        filename = filedialog.asksaveasfilename(defaultextension=".odg", filetypes=[("LibreOffice Draw","*.odg"),("All files","*.*")])
        if not filename: return
        # compute bounds using SvgRenderer (same absolute coordinates)
        svg_r = SvgRenderer(); glue_points=[]
        self.draw_symbol_logic(svg_r, 300, 200, scale=1.0, collect_glue_points=glue_points)
        min_x,min_y,max_x,max_y = svg_r.get_bounds(); padding = SVG_PADDING
        view_x = min_x - padding; view_y = min_y - padding
        view_w = (max_x - min_x) + 2*padding; view_h = (max_y - min_y) + 2*padding
        try:
            self._write_odg_native(filename, glue_points, view_x, view_y, view_w, view_h)
            messagebox.showinfo("Success", f"Saved {os.path.basename(filename)}")
            logger.info("Saved ODG: %s", filename)
        except Exception as e:
            logger.exception("Failed to write ODG")
            messagebox.showerror("Error", f"Failed to save ODG: {e}")

    def _write_odg_native(self, odg_path, glue_points, view_x, view_y, view_w, view_h):
        px_to_cm = 2.54 / DPI_DEFAULT
        # Using the updated OdfRenderer with grouping logic
        odf = OdfRenderer(view_x, view_y, px_to_cm)
        self.draw_symbol_logic(odf, 300, 200, scale=1.0)
        group_xml = odf.get_xml_fragment()

        content_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" 
    xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" 
    xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" 
    xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0" office:version="1.2">
  <office:automatic-styles>
    <style:style style:name="gr1" style:family="graphic">
      <style:graphic-properties draw:stroke="solid" svg:stroke-width="0.05cm" svg:stroke-color="#000000" draw:fill="none"/>
    </style:style>
    <style:style style:name="filled_black" style:family="graphic">
        <style:graphic-properties 
            draw:fill="solid" 
            draw:fill-color="#000000" 
            draw:stroke="solid" 
            svg:stroke-color="#000000"/>
    </style:style>
  </office:automatic-styles>
  <office:body><office:drawing><draw:page draw:name="page1">{group_xml}</draw:page></office:drawing></office:body>
</office:document-content>'''

        with zipfile.ZipFile(odg_path, 'w') as zf:
            zf.writestr("mimetype", "application/vnd.oasis.opendocument.graphics")
            zf.writestr("content.xml", content_xml)
            zf.writestr("META-INF/manifest.xml", '<?xml version="1.0" encoding="UTF-8"?><manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.2"><manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.graphics"/><manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/></manifest:manifest>')

if __name__ == "__main__":
    log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.setLevel(log_level)
    root = tk.Tk()
    app = PneumaticDesignerApp(root)
    root.after(100, app.refresh_preview)
    root.mainloop()