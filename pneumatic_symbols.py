import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import math

# Define the standard size for the geometric glue point anchor
ANCHOR_RADIUS_BASE = 1.5 

class GraphicRenderer:
    """Abstract base for drawing to support both Canvas and SVG."""
    
    def draw_line(self, x1, y1, x2, y2, **kwargs):
        """Draw a line from (x1, y1) to (x2, y2).
        
        Args:
            x1: Starting x coordinate.
            y1: Starting y coordinate.
            x2: Ending x coordinate.
            y2: Ending y coordinate.
            **kwargs: Additional style parameters (width, color, etc.).
        """
        pass
    
    def draw_rect(self, x1, y1, x2, y2, **kwargs):
        """Draw a rectangle from (x1, y1) to (x2, y2).
        
        Args:
            x1: Top-left x coordinate.
            y1: Top-left y coordinate.
            x2: Bottom-right x coordinate.
            y2: Bottom-right y coordinate.
            **kwargs: Additional style parameters.
        """
        pass
    
    def draw_polygon(self, points, **kwargs):
        """Draw a polygon with the given points.
        
        Args:
            points: List of (x, y) tuples representing polygon vertices.
            **kwargs: Additional style parameters.
        """
        pass
    
    def draw_arrow(self, x1, y1, x2, y2, **kwargs):
        """Draw an arrow from (x1, y1) to (x2, y2).
        
        Args:
            x1: Starting x coordinate.
            y1: Starting y coordinate.
            x2: Ending x coordinate.
            y2: Ending y coordinate.
            **kwargs: Additional style parameters.
        """
        pass
    
    def draw_t_stop(self, x, y, direction='up', **kwargs):
        """Draw a T-shaped stop symbol at (x, y).
        
        Args:
            x: Center x coordinate.
            y: Center y coordinate.
            direction: Direction of the stem ('up' or 'down'). Defaults to 'up'.
            **kwargs: Additional style parameters.
        """
        pass
    
    def draw_zigzag(self, x, y, zig_width, height, horizontal=False, **kwargs):
        """Draw a zigzag/spring pattern.
        
        Args:
            x: Starting x coordinate.
            y: Starting y coordinate.
            zig_width: Amplitude of the zigzag.
            height: Total length of the zigzag.
            horizontal: If True, draw horizontally; else vertically. Defaults to False.
            **kwargs: Additional style parameters.
        """
        pass
    
    def draw_circle(self, x, y, r, **kwargs):
        """Draw a circle at (x, y) with radius r.
        
        Args:
            x: Center x coordinate.
            y: Center y coordinate.
            r: Radius of the circle.
            **kwargs: Additional style parameters (fill, width, etc.).
        """
        pass
    
    def draw_text(self, x, y, text, font_size=10, **kwargs):
        """Draw text at (x, y).
        
        Args:
            x: X coordinate of text position.
            y: Y coordinate of text position.
            text: Text string to draw.
            font_size: Font size in points. Defaults to 10.
            **kwargs: Additional style parameters.
        """
        pass


class CanvasRenderer(GraphicRenderer):
    """Draws directly to a Tkinter Canvas."""
    
    def __init__(self, canvas):
        """Initialize the Canvas renderer.
        
        Args:
            canvas: Tkinter Canvas object to draw on.
        """
        self.c = canvas

    def draw_line(self, x1, y1, x2, y2, **kwargs):
        """Draw a line from (x1, y1) to (x2, y2).
        
        Args:
            x1: Starting x coordinate.
            y1: Starting y coordinate.
            x2: Ending x coordinate.
            y2: Ending y coordinate.
            **kwargs: Additional style parameters (width, color, etc.).
        """
        width = kwargs.get('width', 2)
        self.c.create_line(x1, y1, x2, y2, width=width, fill="black")

    def draw_rect(self, x1, y1, x2, y2, **kwargs):
        """Draw a rectangle from (x1, y1) to (x2, y2).
        
        Args:
            x1: Top-left x coordinate.
            y1: Top-left y coordinate.
            x2: Bottom-right x coordinate.
            y2: Bottom-right y coordinate.
            **kwargs: Additional style parameters.
        """
        width = kwargs.get('width', 2)
        self.c.create_rectangle(x1, y1, x2, y2, outline="black", width=width)

    def draw_polygon(self, points, **kwargs):
        """Draw a polygon with the given points.
        
        Args:
            points: List of (x, y) tuples representing polygon vertices.
            **kwargs: Additional style parameters.
        """
        width = kwargs.get('width', 2)
        # Tkinter expects a flat list [x1, y1, x2, y2, ...]
        flat_points = [coord for pt in points for coord in pt]
        self.c.create_polygon(flat_points, outline="black", width=width, fill="")

    def draw_arrow(self, x1, y1, x2, y2, **kwargs):
        """Draw an arrow from (x1, y1) to (x2, y2).
        
        Args:
            x1: Starting x coordinate.
            y1: Starting y coordinate.
            x2: Ending x coordinate.
            y2: Ending y coordinate.
            **kwargs: Additional style parameters.
        """
        width = kwargs.get('width', 2)
        self.c.create_line(x1, y1, x2, y2, arrow=tk.LAST, width=width, fill="black")

    def draw_t_stop(self, x, y, direction='up', **kwargs):
        """Draw a T-shaped stop symbol at (x, y).
        
        Args:
            x: Center x coordinate.
            y: Center y coordinate.
            direction: Direction of the stem ('up' or 'down'). Defaults to 'up'.
            **kwargs: Additional style parameters (size, width, etc.).
        """
        size = kwargs.get('size', 5)
        width = kwargs.get('width', 2)
        if direction == 'up':
            self.c.create_line(x, y, x, y-size, width=width, fill="black") # stem
            self.c.create_line(x-size, y-size, x+size, y-size, width=width, fill="black") # bar
        else:
            self.c.create_line(x, y, x, y+size, width=width, fill="black")
            self.c.create_line(x-size, y+size, x+size, y+size, width=width, fill="black")

    def draw_zigzag(self, x, y, zig_width, height, horizontal=False, **kwargs):
        """Draw a zigzag/spring pattern.
        
        Args:
            x: Starting x coordinate.
            y: Starting y coordinate.
            zig_width: Amplitude of the zigzag.
            height: Total length of the zigzag.
            horizontal: If True, draw horizontally; else vertically. Defaults to False.
            **kwargs: Additional style parameters.
        """
        pts = []
        steps = 6
        
        if horizontal:
            # x is start_x, y is center_y, height is length (horizontal dimension), zig_width is amplitude (vertical dimension)
            step_w = height / steps
            pts.append((x, y))
            for i in range(1, steps):
                # Y offset is the amplitude
                y_offset = -zig_width/2 if i % 2 != 0 else zig_width/2
                pts.append((x + i * step_w, y + y_offset))
            pts.append((x + height, y))
        else:
            # Standard vertical spring (x is center X, y is start Y)
            step_h = height / steps
            pts.append((x, y))
            for i in range(1, steps):
                # X offset is the amplitude
                x_offset = -zig_width/2 if i % 2 != 0 else zig_width/2
                pts.append((x + x_offset, y + i * step_h))
            pts.append((x, y + height))
        
        stroke_width = kwargs.get('width', 2)
        # Flatten the list of tuples to a simple list of coords [x1, y1, x2, y2...]
        flat_pts = [coord for pt in pts for coord in pt]
        self.c.create_line(flat_pts, width=stroke_width, fill="black")

    def draw_circle(self, x, y, r, **kwargs):
        """Draw a circle at (x, y) with radius r.
        
        Args:
            x: Center x coordinate.
            y: Center y coordinate.
            r: Radius of the circle.
            **kwargs: Additional style parameters (fill, width, etc.).
        """
        width = kwargs.get('width', 2)
        # Fix for Tkinter: use empty string "" for no fill instead of "none"
        fill_color = kwargs.get('fill', "") 
        self.c.create_oval(x-r, y-r, x+r, y+r, outline="black", width=width, fill=fill_color)
    
    def draw_text(self, x, y, text, font_size=10, **kwargs):
        """Draw text at (x, y).
        
        Args:
            x: X coordinate of text position.
            y: Y coordinate of text position.
            text: Text string to draw.
            font_size: Font size in points. Defaults to 10.
            **kwargs: Additional style parameters.
        """
        self.c.create_text(x, y, text=text, fill="black", font=("Arial", int(font_size)))


class SvgRenderer(GraphicRenderer):
    """Generates an SVG string."""
    
    def __init__(self):
        """Initialize the SVG renderer."""
        self.elements = []

    def draw_line(self, x1, y1, x2, y2, **kwargs):
        """Draw a line from (x1, y1) to (x2, y2).
        
        Args:
            x1: Starting x coordinate.
            y1: Starting y coordinate.
            x2: Ending x coordinate.
            y2: Ending y coordinate.
            **kwargs: Additional style parameters (width, color, etc.).
        """
        w = kwargs.get('width', 2)
        self.elements.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="{w}" />')

    def draw_rect(self, x1, y1, x2, y2, **kwargs):
        """Draw a rectangle from (x1, y1) to (x2, y2).
        
        Args:
            x1: Top-left x coordinate.
            y1: Top-left y coordinate.
            x2: Bottom-right x coordinate.
            y2: Bottom-right y coordinate.
            **kwargs: Additional style parameters.
        """
        w_rect = abs(x2 - x1)
        h_rect = abs(y2 - y1)
        rx = min(x1, x2)
        ry = min(y1, y2)
        stroke = kwargs.get('width', 2)
        self.elements.append(f'<rect x="{rx}" y="{ry}" width="{w_rect}" height="{h_rect}" fill="none" stroke="black" stroke-width="{stroke}" />')

    def draw_polygon(self, points, **kwargs):
        """Draw a polygon with the given points.
        
        Args:
            points: List of (x, y) tuples representing polygon vertices.
            **kwargs: Additional style parameters.
        """
        pts_str = " ".join([f"{x},{y}" for x, y in points])
        stroke = kwargs.get('width', 2)
        self.elements.append(f'<polygon points="{pts_str}" fill="none" stroke="black" stroke-width="{stroke}" />')

    def draw_arrow(self, x1, y1, x2, y2, **kwargs):
        """Draw an arrow from (x1, y1) to (x2, y2).
        
        Args:
            x1: Starting x coordinate.
            y1: Starting y coordinate.
            x2: Ending x coordinate.
            y2: Ending y coordinate.
            **kwargs: Additional style parameters.
        """
        width = kwargs.get('width', 2)
        self.draw_line(x1, y1, x2, y2, width=width)
        # Calculate arrow head
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_len = 5 * width # Scale arrow head with line width
        angle1 = angle + math.pi / 6 + math.pi
        angle2 = angle - math.pi / 6 + math.pi
        
        ax1 = x2 + arrow_len * math.cos(angle1)
        ay1 = y2 + arrow_len * math.sin(angle1)
        ax2 = x2 + arrow_len * math.cos(angle2)
        ay2 = y2 + arrow_len * math.sin(angle2)
        
        self.elements.append(f'<polygon points="{x2},{y2} {ax1},{ay1} {ax2},{ay2}" fill="black" />')

    def draw_t_stop(self, x, y, direction='up', **kwargs):
        """Draw a T-shaped stop symbol at (x, y).
        
        Args:
            x: Center x coordinate.
            y: Center y coordinate.
            direction: Direction of the stem ('up' or 'down'). Defaults to 'up'.
            **kwargs: Additional style parameters (size, width, etc.).
        """
        size = kwargs.get('size', 5)
        width = kwargs.get('width', 2)
        if direction == 'up':
            self.draw_line(x, y, x, y-size, width=width) # stem
            self.draw_line(x-size, y-size, x+size, y-size, width=width) # bar
        else:
            self.draw_line(x, y, x, y+size, width=width)
            self.draw_line(x-size, y+size, x+size, y+size, width=width)

    def draw_zigzag(self, x, y, zig_width, height, horizontal=False, **kwargs):
        """Draw a zigzag/spring pattern.
        
        Args:
            x: Starting x coordinate.
            y: Starting y coordinate.
            zig_width: Amplitude of the zigzag.
            height: Total length of the zigzag.
            horizontal: If True, draw horizontally; else vertically. Defaults to False.
            **kwargs: Additional style parameters.
        """
        points = ""
        steps = 6
        
        if horizontal:
            # x is start_x, y is center_y, height is length (horizontal dimension), zig_width is amplitude (vertical dimension)
            step_w = height / steps
            points += f"{x},{y} "
            for i in range(1, steps):
                y_offset = -zig_width/2 if i % 2 != 0 else zig_width/2
                points += f"{x + i * step_w},{y + y_offset} "
            points += f"{x + height},{y}"
        else:
            # Standard vertical spring (x is center X, y is start Y)
            step_h = height / steps
            points = f"{x},{y} "
            for i in range(1, steps):
                x_offset = -zig_width/2 if i % 2 != 0 else zig_width/2
                points += f"{x + x_offset},{y + i * step_h} "
            points += f"{x},{y + height}"

        stroke = kwargs.get('width', 2)
        self.elements.append(f'<polyline points="{points}" fill="none" stroke="black" stroke-width="{stroke}" />')

    def draw_circle(self, x, y, r, **kwargs):
        """Draw a circle at (x, y) with radius r.
        
        Args:
            x: Center x coordinate.
            y: Center y coordinate.
            r: Radius of the circle.
            **kwargs: Additional style parameters (fill, width, etc.).
        """
        stroke = kwargs.get('width', 2)
        fill_color = kwargs.get('fill', "none")
        self.elements.append(f'<circle cx="{x}" cy="{y}" r="{r}" stroke="black" stroke-width="{stroke}" fill="{fill_color}" />')

    def draw_text(self, x, y, text, font_size=10, **kwargs):
        """Draw text at (x, y).
        
        Args:
            x: X coordinate of text position.
            y: Y coordinate of text position.
            text: Text string to draw.
            font_size: Font size in points. Defaults to 10.
            **kwargs: Additional style parameters.
        """
        self.elements.append(f'<text x="{x}" y="{y}" fill="black" font-family="Arial" font-size="{font_size}" text-anchor="middle">{text}</text>')

    def get_svg(self, width, height):
        """Generate the complete SVG document.
        
        Args:
            width: Width of the SVG canvas in pixels.
            height: Height of the SVG canvas in pixels.
            
        Returns:
            A string containing the complete SVG markup.
        """
        header = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" version="1.1">'
        footer = '</svg>'
        return header + "".join(self.elements) + footer


class PneumaticDesignerApp:
    """Main application for designing ISO 1219 pneumatic symbols."""
    
    def __init__(self, root):
        """Initialize the Pneumatic Designer application.
        
        Args:
            root: The root Tkinter window object.
        """
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
        
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface components."""
        # --- Main Layout ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left Panel: Controls
        control_panel = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 1. Basic Settings
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

        # 2. Operators
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

        # 3. State Flow Configuration
        self.state_input_frame = ttk.LabelFrame(control_panel, text="Flow Paths (e.g. 1-2, 3-T)")
        self.state_input_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=5)
        
        # Help Text
        help_lbl = ttk.Label(control_panel, text="Format: '1-2' (connect),\n'1-T' (block).\nSeparate with commas.", font=("Arial", 8), foreground="gray")
        help_lbl.grid(row=7, column=0, columnspan=2)

        # Buttons
        btn_frame = ttk.Frame(control_panel)
        btn_frame.grid(row=8, column=0, columnspan=2, pady=20)
        
        ttk.Button(btn_frame, text="Save LibreDraw (SVG)", command=self.save_svg).pack(side=tk.LEFT, padx=5)

        # Right Panel: Canvas
        self.canvas_frame = ttk.LabelFrame(main_frame, text="Preview", padding="10")
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Init dynamic inputs
        self.rebuild_state_inputs()

    def rebuild_state_inputs(self, event=None):
        """Rebuild dynamic state input fields based on current configuration.
        
        Args:
            event: Event object from tkinter callback. Defaults to None.
        """
        # Clear existing
        for widget in self.state_input_frame.winfo_children():
            widget.destroy()
        
        self.state_configs.clear()
        
        n_states = self.num_states.get()
        
        for i in range(n_states):
            lbl_text = f"Pos {i+1} (Left):" if i == 0 else (f"Pos {i+1} (Right):" if i == n_states-1 else f"Pos {i+1} (Center):")
            
            ttk.Label(self.state_input_frame, text=lbl_text).grid(row=i, column=0, sticky="w", pady=2)
            
            var = tk.StringVar()
            # Defaults based on common valves
            if self.num_ports.get() == 4:
                if i == 0: var.set("1-2, 4-3") # Parallel
                else: var.set("1-4, 2-3")      # Crossed
            elif self.num_ports.get() == 5:
                if i == 0: var.set("1-2, 4-5")
                else: var.set("1-4, 2-3")
            elif self.num_ports.get() == 3:
                if i == 0: var.set("1-T, 2-3")
                else: var.set("1-2, 3-T")
            
            entry = ttk.Entry(self.state_input_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, sticky="e", pady=2)
            # Update preview on enter or focus out
            entry.bind("<Return>", lambda e: self.refresh_preview())
            entry.bind("<FocusOut>", lambda e: self.refresh_preview())
            
            self.state_configs.append(var)
        
        self.refresh_preview()

    def get_port_coords(self, port_num, box_x, box_y, box_w, box_h):
        """Get the (x, y) coordinates for a specific port on a valve box.
        
        Args:
            port_num: Port number (1-5).
            box_x: X coordinate of the valve box top-left corner.
            box_y: Y coordinate of the valve box top-left corner.
            box_w: Width of the valve box.
            box_h: Height of the valve box.
            
        Returns:
            A tuple (x, y) representing the port coordinates.
        """
        ports = self.num_ports.get()
        pos_x = 0.5
        pos_y = 1.0 # Bottom by default
        
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

    def draw_symbol_logic(self, r, center_x, center_y, scale=1.0):
        """Core drawing logic for pneumatic symbols.
        
        Draws the complete pneumatic symbol including valve boxes, flow connections,
        port labels, and operators (spring, solenoid, lever, etc.).
        
        Args:
            r: GraphicRenderer instance (Canvas or SVG).
            center_x: X coordinate of the symbol center.
            center_y: Y coordinate of the symbol center.
            scale: Scaling factor for all dimensions. Defaults to 1.0.
        """
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
            
            # Draw Connections inside
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
                            # Blocked port
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
            
            # Label
            lbl_y = py + (15*scale) if py > center_y else py - (15*scale)
            r.draw_text(px, lbl_y, str(p), font_size=FONT_SIZE)
            
            # Line extension
            ext_y = py + (10*scale) if py > center_y else py - (10*scale)
            r.draw_line(px, py, px, ext_y, width=LINE_WIDTH)

            # Glue Point Anchor: Draw a small, filled circle at the end of the port line extension.
            # This is the functional "glue point" for vector editors.
            ANCHOR_RADIUS = ANCHOR_RADIUS_BASE * scale
            r.draw_circle(px, ext_y, ANCHOR_RADIUS, width=0.5*scale, fill="black")
            
        # 3. Draw Operators
        # Helper Sizes
        OP_LENGTH = 30 * scale # Standard length for Solenoid/Detent/Spring
        S_15 = 15 * scale
        S_10 = 10 * scale
        S_5  = 5 * scale
        
        # Operator Dimensions (1/3 of valve height, centered)
        OP_HEIGHT = BOX_SIZE / 3 # Amplitude for spring
        ly_center = center_y
        OP_Y_TOP = ly_center - (OP_HEIGHT / 2)
        OP_Y_BOT = ly_center + (OP_HEIGHT / 2)
        
        # Left Side
        lx = start_x
        l_offset = 0
        
        # Spring
        if self.left_ops["Spring"].get():
            spring_amp = OP_HEIGHT 
            spring_len = OP_LENGTH # Spring length now matches OP_LENGTH
            # x is end_x (valve side) - spring_len, y is center_y, zig_width is amplitude, height is length
            r.draw_zigzag(lx - spring_len - l_offset, ly_center, spring_amp, spring_len, horizontal=True, width=LINE_WIDTH)
            l_offset += OP_LENGTH
            
        # Solenoid
        if self.left_ops["Solenoid"].get():
            # Rectangle: (Lx, Ty, Rx, By)
            r.draw_rect(lx - OP_LENGTH - l_offset, OP_Y_TOP, lx - l_offset, OP_Y_BOT, width=LINE_WIDTH)
            # Diagonal line
            r.draw_line(lx - OP_LENGTH - l_offset, OP_Y_TOP, lx - l_offset, OP_Y_BOT, width=LINE_WIDTH) 
            l_offset += OP_LENGTH
            
        # Pilot
        if self.left_ops["Pilot"].get():
            r.draw_polygon([
                (lx - l_offset, ly_center), 
                (lx - l_offset - S_15, ly_center - S_10), 
                (lx - l_offset - S_15, ly_center + S_10)
            ], width=LINE_WIDTH)
            l_offset += S_15
            
        # Detent (Solenoid-sized box with notch)
        if self.left_ops["Detent"].get():
            dx1 = lx - l_offset - OP_LENGTH
            dx2 = lx - l_offset
            
            # Notch calculation
            center_x_notch = dx1 + OP_LENGTH / 2
            notch_start_x = center_x_notch - S_5
            notch_end_x = center_x_notch + S_5
            notch_y_peak = OP_Y_TOP + S_5 # Peak points down

            # 1. Draw three sides (Left, Right, Bottom)
            r.draw_line(dx1, OP_Y_TOP, dx1, OP_Y_BOT, width=LINE_WIDTH) # Left side
            r.draw_line(dx2, OP_Y_TOP, dx2, OP_Y_BOT, width=LINE_WIDTH) # Right side
            r.draw_line(dx1, OP_Y_BOT, dx2, OP_Y_BOT, width=LINE_WIDTH) # Bottom side
            
            # 2. Draw top segments (leaving a gap for the notch)
            r.draw_line(dx1, OP_Y_TOP, notch_start_x, OP_Y_TOP, width=LINE_WIDTH)
            r.draw_line(notch_end_x, OP_Y_TOP, dx2, OP_Y_TOP, width=LINE_WIDTH)
            
            # 3. Draw notch (two slopes, preventing the base line from being redrawn)
            r.draw_line(notch_start_x, OP_Y_TOP, center_x_notch, notch_y_peak, width=LINE_WIDTH)
            r.draw_line(center_x_notch, notch_y_peak, notch_end_x, OP_Y_TOP, width=LINE_WIDTH)
            
            l_offset += OP_LENGTH
            
        # Lever
        if self.left_ops["Lever"].get():
            # Trapezoid Lever with slanted face
            w_top = 25 * scale
            w_bot = 10 * scale
            lever_h = BOX_SIZE / 3
            lever_top = ly_center - (lever_h / 2)
            lever_bot = ly_center + (lever_h / 2)
            
            # Trapezoid Points (attached to wall at lx - l_offset)
            p_wall_top = (lx - l_offset, lever_top)
            p_wall_bot = (lx - l_offset, lever_bot)
            p_outer_bot = (lx - l_offset - w_bot, lever_bot)
            p_outer_top = (lx - l_offset - w_top, lever_top)
            
            r.draw_polygon([p_wall_top, p_wall_bot, p_outer_bot, p_outer_top], width=LINE_WIDTH)
            
            vx = p_outer_top[0] - p_outer_bot[0]
            vy = p_outer_top[1] - p_outer_bot[1]
            
            v_len = math.sqrt(vx*vx + vy*vy)
            nx, ny = vx/v_len, vy/v_len
            
            handle_len = 25 * scale
            hx, hy = p_outer_top[0] + nx * handle_len, p_outer_top[1] + ny * handle_len
            
            r.draw_line(p_outer_top[0], p_outer_top[1], hx, hy, width=LINE_WIDTH)
            r.draw_circle(hx, hy, S_5, width=LINE_WIDTH)
            
            l_offset += w_top

        # Right Side
        rx = start_x + total_w
        r_offset = 0
        
        # Spring
        if self.right_ops["Spring"].get():
            spring_amp = OP_HEIGHT
            spring_len = OP_LENGTH # Spring length now matches OP_LENGTH
            # x is start_x (valve side), y is center_y, zig_width is amplitude, height is length
            r.draw_zigzag(rx + r_offset, ly_center, spring_amp, spring_len, horizontal=True, width=LINE_WIDTH)
            r_offset += OP_LENGTH
            
        # Solenoid
        if self.right_ops["Solenoid"].get():
            # Rectangle: (Lx, Ty, Rx, By)
            r.draw_rect(rx + r_offset, OP_Y_TOP, rx + r_offset + OP_LENGTH, OP_Y_BOT, width=LINE_WIDTH)
            # Diagonal line
            r.draw_line(rx + r_offset, OP_Y_TOP, rx + r_offset + OP_LENGTH, OP_Y_BOT, width=LINE_WIDTH)
            r_offset += OP_LENGTH
            
        # Pilot
        if self.right_ops["Pilot"].get():
            r.draw_polygon([
                (rx + r_offset, ly_center), 
                (rx + r_offset + S_15, ly_center - S_10), 
                (rx + r_offset + S_15, ly_center + S_10)
            ], width=LINE_WIDTH)
            r_offset += S_15
            
        # Detent (Solenoid-sized box with notch)
        if self.right_ops["Detent"].get():
            dx1 = rx + r_offset
            dx2 = rx + r_offset + OP_LENGTH
            
            # Notch calculation
            center_x_notch = dx1 + OP_LENGTH / 2
            notch_start_x = center_x_notch - S_5
            notch_end_x = center_x_notch + S_5
            notch_y_peak = OP_Y_TOP + S_5 # Peak points down

            # 1. Draw three sides (Left, Right, Bottom)
            r.draw_line(dx1, OP_Y_TOP, dx1, OP_Y_BOT, width=LINE_WIDTH) # Left side
            r.draw_line(dx2, OP_Y_TOP, dx2, OP_Y_BOT, width=LINE_WIDTH) # Right side
            r.draw_line(dx1, OP_Y_BOT, dx2, OP_Y_BOT, width=LINE_WIDTH) # Bottom side
            
            # 2. Draw top segments (leaving a gap for the notch)
            r.draw_line(dx1, OP_Y_TOP, notch_start_x, OP_Y_TOP, width=LINE_WIDTH)
            r.draw_line(notch_end_x, OP_Y_TOP, dx2, OP_Y_TOP, width=LINE_WIDTH)
            
            # 3. Draw notch (two slopes, preventing the base line from being redrawn)
            r.draw_line(notch_start_x, OP_Y_TOP, center_x_notch, notch_y_peak, width=LINE_WIDTH)
            r.draw_line(center_x_notch, notch_y_peak, notch_end_x, OP_Y_TOP, width=LINE_WIDTH)
            
            r_offset += OP_LENGTH
            
        # Lever
        if self.right_ops["Lever"].get():
             # Mirror Trapezoid
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
            nx, ny = vx/v_len, vy/v_len
            
            handle_len = 25 * scale
            hx, hy = p_outer_top[0] + nx * handle_len, p_outer_top[1] + ny * handle_len
            
            r.draw_line(p_outer_top[0], p_outer_top[1], hx, hy, width=LINE_WIDTH)
            r.draw_circle(hx, hy, S_5, width=LINE_WIDTH)
            
            r_offset += w_top

    def refresh_preview(self):
        """Refresh the preview canvas with the current symbol configuration."""
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10: w, h = 400, 300
        
        scale = self.zoom_level.get()
        renderer = CanvasRenderer(self.canvas)
        self.draw_symbol_logic(renderer, w/2, h/2, scale=scale)

    def save_svg(self):
        """Save the current symbol design as an SVG file for LibreOffice Draw."""
        filename = filedialog.asksaveasfilename(defaultextension=".svg", 
                                                filetypes=[("SVG files", "*.svg"), ("All files", "*.*")],
                                                title="Save as SVG (LibreOffice Draw Compatible)")
        if not filename:
            return

        svg_r = SvgRenderer()
        # Drawing at scale 1.0 for export to keep coordinates clean
        self.draw_symbol_logic(svg_r, 300, 200, scale=1.0) 
        
        content = svg_r.get_svg(600, 400)
        
        try:
            with open(filename, "w") as f:
                f.write(content)
            messagebox.showinfo("Success", "File saved successfully!\nYou can open this .svg file in LibreOffice Draw, and the small black dots at the port ends will serve as reliable snap/glue points for connecting lines.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PneumaticDesignerApp(root)
    # Delay initial draw until window size is calculated
    root.after(100, app.refresh_preview)
    root.mainloop()
