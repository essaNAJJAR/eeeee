import sys
import os
import time
import math
import numpy as np
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    HAS_TK = True
except ImportError:
    HAS_TK = False
    tk = None
    ttk = None
import json
import threading

sys.path.insert(0, os.path.dirname(__file__))

from config.config_manager import load_config, save_config
from data.data_loader import load_ukdale, load_refit
from preprocessing.preprocessor import preprocess_signal
from feature_extraction.vmd import feedback_vmd
from feature_extraction.teo import teager_energy_operator
from feature_extraction.extract_features import extract_features
from fuzzy_logic.fuzzy_system import evaluate_fuzzy_logic
from deep_learning.model import VAEDCCNNAtt, predict_with_model, train_vae_dccnn_model
from federated_learning.federated_sim import run_federated_learning
from event_detection.detect_events import detect_events, classify_events
from evaluation.evaluation_framework import EvaluationFramework
from advanced.signal_quality import SignalQualityAssessor
from advanced.attention import AttentionMechanism
from streaming.data_stream import DataStream
from online_learning.online_learner import OnlineLearner
from multi_household.simulator import MultiHouseholdSimulator
from transfer_learning.manager import TransferLearningManager
from export.exporter import NILMExporter
from tests.run_tests import run_all_tests

try:
    import torch
    HAS_TORCH = True
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "N/A"
except ImportError:
    HAS_TORCH = False
    GPU_AVAILABLE = False
    GPU_NAME = "N/A"

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ── Color Palette ──────────────────────────────────────────────
C = {
    'bg':       '#0f1117', 'bg2':      '#161922', 'bg3':      '#1c2030',
    'bg4':      '#232840', 'card':     '#1a1f2e', 'card2':    '#222842',
    'border':   '#2a3050', 'border2':  '#353c5c',
    'accent':   '#6366f1', 'accent2':  '#818cf8', 'accent3':  '#4f46e5',
    'pink':     '#ec4899', 'pink2':    '#f472b6',
    'cyan':     '#22d3ee', 'cyan2':    '#06b6d4',
    'green':    '#10b981', 'green2':   '#34d399',
    'yellow':   '#f59e0b', 'orange':   '#f97316', 'red':      '#ef4444',
    'fg':       '#e2e8f0', 'fg2':      '#94a3b8', 'fg3':      '#64748b',
    'white':    '#ffffff',
}


# ── Utility Widgets ────────────────────────────────────────────
class Tooltip:
    def __init__(self, widget, text, delay=400):
        self.widget, self.text, self.delay = widget, text, delay
        self.tip_window = None
        self.after_id = None
        widget.bind('<Enter>', self._schedule)
        widget.bind('<Leave>', self._cancel)
        widget.bind('<ButtonPress>', self._cancel)

    def _schedule(self, e=None):
        self.after_id = self.widget.after(self.delay, self._show)

    def _cancel(self, e=None):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        self._hide()

    def _show(self):
        if self.tip_window:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.configure(bg=C['border2'])
        f = tk.Frame(tw, bg=C['bg4'], padx=1, pady=1)
        f.pack()
        tk.Label(f, text=self.text, justify=tk.LEFT, bg=C['bg4'], fg=C['fg'],
                 font=('Segoe UI', 9), padx=10, pady=6, wraplength=320).pack()
        self.tip_window = tw

    def _hide(self):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class GaugeCanvas(tk.Canvas):
    def __init__(self, parent, size=120, label="", color=C['accent'], **kw):
        super().__init__(parent, width=size, height=size + 24,
                         bg=C['card'], highlightthickness=0, **kw)
        self.size = size
        self.label = label
        self.color = color
        self.value = 0
        self._draw(0)

    def _draw(self, value):
        self.delete('all')
        s = self.size
        cx, cy, r = s // 2, s // 2, s // 2 - 12
        self.create_arc(cx - r, cy - r, cx + r, cy + r, start=225, extent=-270,
                        style='arc', outline=C['border'], width=8)
        extent = -270 * min(1.0, max(0.0, value))
        self.create_arc(cx - r, cy - r, cx + r, cy + r, start=225, extent=extent,
                        style='arc', outline=self.color, width=8)
        self.create_text(cx, cy - 4, text=f"{value * 100:.0f}%",
                         fill=C['fg'], font=('Segoe UI', 14, 'bold'))
        self.create_text(cx, cy + 18, text=self.label,
                         fill=C['fg3'], font=('Segoe UI', 8))

    def set_value(self, value):
        def _task():
            self.value = value
            self._draw(value)
        self.after(0, _task)


class MetricCard(tk.Frame):
    def __init__(self, parent, title="", value="0", color=C['accent'], **kw):
        super().__init__(parent, bg=C['card'], **kw)
        self.configure(highlightbackground=C['border'], highlightthickness=1)
        
        # Left color strip
        self.accent_strip = tk.Frame(self, bg=color, width=4)
        self.accent_strip.pack(side='left', fill='y')
        
        # Content area
        self.container = tk.Frame(self, bg=C['card'], padx=12, pady=10)
        self.container.pack(side='left', fill='both', expand=True)
        
        self.title_lbl = tk.Label(self.container, text=title.upper(), bg=C['card'], fg=C['fg2'],
                                 font=('Segoe UI', 8, 'bold'))
        self.title_lbl.pack(anchor='w')
        
        self.val_label = tk.Label(self.container, text=value, bg=C['card'], fg=C['white'],
                                   font=('Consolas', 16, 'bold'))
        self.val_label.pack(anchor='w', pady=(4, 0))
        self.color = color

    def set_value(self, value, color=None):
        self.val_label.configure(text=str(value))
        if color:
            self.val_label.configure(fg=color)
            self.accent_strip.configure(bg=color)


class ScrollableFrame(tk.Frame):
    def __init__(self, parent, bg=C['bg'], **kw):
        super().__init__(parent, bg=bg, **kw)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg=bg, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=bg)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        self.scrollable_frame.bind('<Enter>', self._bind_mousewheel)
        self.scrollable_frame.bind('<Leave>', self._unbind_mousewheel)
        
    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)
        
    def _bind_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
    def _unbind_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")
        
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


def create_section_card(parent, title):
    card = tk.Frame(parent, bg=C['card'], highlightbackground=C['border'], highlightthickness=1)
    
    header = tk.Frame(card, bg=C['bg3'], height=32)
    header.pack(fill='x', side='top')
    header.pack_propagate(False)
    
    tk.Label(header, text=f"  {title}", bg=C['bg3'], fg=C['accent2'], font=('Segoe UI', 9, 'bold')).pack(side='left')
    
    content = tk.Frame(card, bg=C['card'], padx=12, pady=12)
    content.pack(fill='both', expand=True)
    
    return card, content


def create_hover_button(parent, text, command, bg, fg, hover_bg, hover_fg, font=('Segoe UI', 9, 'bold'), height=1, width=None, relief='flat', padx=10, pady=5):
    btn = tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                    activebackground=hover_bg, activeforeground=hover_fg,
                    font=font, relief=relief, bd=0, padx=padx, pady=pady, cursor='hand2')
    btn.bind("<Enter>", lambda e: btn.config(bg=hover_bg, fg=hover_fg) if btn['state'] != 'disabled' else None)
    btn.bind("<Leave>", lambda e: btn.config(bg=bg, fg=fg) if btn['state'] != 'disabled' else None)
    return btn


class LogPanel(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C['bg'], **kw)
        toolbar = tk.Frame(self, bg=C['bg'])
        toolbar.pack(fill=tk.X, padx=4, pady=(4, 8))
        
        tk.Label(toolbar, text="📝 Log Console", bg=C['bg'], fg=C['fg'],
                 font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT)
        tk.Label(toolbar, text="  |  ", bg=C['bg'], fg=C['fg3'],
                 font=('Segoe UI', 9)).pack(side=tk.LEFT)
        
        self.filter_var = tk.StringVar(value="ALL")
        flt = ttk.Combobox(toolbar, textvariable=self.filter_var,
                           values=["ALL", "INFO", "SUCCESS", "WARNING", "ERROR"],
                           width=10, state='readonly')
        flt.pack(side=tk.LEFT, padx=4)
        flt.bind('<<ComboboxSelected>>', lambda e: self._apply_filter())
        
        self.search_var = tk.StringVar()
        se = tk.Entry(toolbar, textvariable=self.search_var, width=20,
                      bg=C['bg3'], fg=C['fg'], insertbackground=C['fg'],
                      font=('Consolas', 9), relief='flat', bd=4)
        se.pack(side=tk.LEFT, padx=4)
        se.bind('<Return>', lambda e: self._apply_filter())
        
        for txt, cmd in [("Clear", self._clear), ("Copy", self._copy),
                         ("Export", self._export_log)]:
            b = create_hover_button(toolbar, txt, cmd, C['bg3'], C['fg2'], C['accent'], C['white'], font=('Segoe UI', 8, 'bold'), padx=8, pady=3)
            b.pack(side=tk.RIGHT, padx=2)

        body = tk.Frame(self, bg=C['bg'])
        body.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self.text = tk.Text(body, wrap=tk.WORD, font=('Consolas', 9),
                            bg='#0a0d14', fg='#94a3b8', insertbackground=C['fg'],
                            selectbackground=C['accent3'], relief='flat', bd=0,
                            padx=8, pady=8, spacing1=1)
        scroll = ttk.Scrollbar(body, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.pack(fill=tk.BOTH, expand=True)

        for tag, fg in [('info', '#94a3b8'), ('success', C['green']),
                         ('warning', C['yellow']), ('error', C['red']),
                         ('header', C['accent2']), ('dim', C['fg3']),
                         ('accent', C['cyan'])]:
            self.text.tag_configure(tag, foreground=fg)
        self.text.tag_configure('header', font=('Consolas', 9, 'bold'))
        self.all_lines = []

    def log(self, msg, tag='info'):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.all_lines.append((line, tag))
        self.text.insert(tk.END, line + "\n", tag)
        self.text.see(tk.END)

    def _apply_filter(self):
        f = self.filter_var.get().lower()
        s = self.search_var.get().lower()
        self.text.delete('1.0', tk.END)
        for line, tag in self.all_lines:
            if f != 'all' and tag != f:
                continue
            if s and s not in line.lower():
                continue
            self.text.insert(tk.END, line + "\n", tag)
        self.text.see(tk.END)

    def _clear(self):
        self.all_lines.clear()
        self.text.delete('1.0', tk.END)

    def _copy(self):
        self.clipboard_clear()
        self.clipboard_append(self.text.get('1.0', tk.END))

    def _export_log(self):
        path = filedialog.asksaveasfilename(defaultextension=".log",
                                             filetypes=[("Log files", "*.log")])
        if path:
            with open(path, 'w') as f:
                for line, _ in self.all_lines:
                    f.write(line + "\n")


class DataTable(tk.Frame):
    def __init__(self, parent, columns=None, **kw):
        super().__init__(parent, bg=C['bg'], **kw)
        self.columns = columns or []
        self.tree = None
        self._build()

    def _build(self):
        if not self.columns:
            return
        style = ttk.Style()
        style.configure('Dark.Treeview', background=C['bg2'], foreground=C['fg'],
                        fieldbackground=C['bg2'], font=('Consolas', 9),
                        borderwidth=0, rowheight=24)
        style.configure('Dark.Treeview.Heading', background=C['bg4'],
                        foreground=C['fg2'], font=('Segoe UI', 9, 'bold'),
                        relief='flat')
        style.map('Dark.Treeview', background=[('selected', C['accent3'])],
                   foreground=[('selected', C['white'])])
        self.tree = ttk.Treeview(self, columns=self.columns, show='headings',
                                  style='Dark.Treeview', selectmode='browse')
        for col in self.columns:
            self.tree.heading(col, text=col, anchor=tk.W)
            self.tree.column(col, width=100, anchor=tk.W, minwidth=60)
        scroll_y = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        scroll_x = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        self.tree.grid(row=0, column=0, sticky='nsew')
        scroll_y.grid(row=0, column=1, sticky='ns')
        scroll_x.grid(row=1, column=0, sticky='ew')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def set_data(self, rows):
        if not self.tree:
            return
        self.tree.delete(*self.tree.get_children())
        for row in rows:
            self.tree.insert('', tk.END, values=row)

    def add_row(self, row):
        if self.tree:
            self.tree.insert('', tk.END, values=row)


class PipelineStepIndicator(tk.Frame):
    STEPS = ["Load", "Preprocess", "Features", "Classify", "Evaluate"]
    COLORS = {'pending': C['fg3'], 'running': C['yellow'], 'done': C['green'],
              'error': C['red']}

    def __init__(self, parent, bg=C['card'], **kw):
        super().__init__(parent, bg=bg, **kw)
        self.labels = []
        self.statuses = []
        for i, name in enumerate(self.STEPS):
            dot = tk.Canvas(self, width=14, height=14, bg=bg,
                            highlightthickness=0)
            dot.grid(row=0, column=i * 2, padx=(6, 2), pady=6)
            oval = dot.create_oval(2, 2, 12, 12, fill=C['fg3'], outline='')
            self.labels.append((dot, oval))
            lbl = tk.Label(self, text=name, bg=bg, fg=C['fg3'],
                           font=('Segoe UI', 8, 'bold'))
            lbl.grid(row=0, column=i * 2 + 1, padx=(0, 6))
            self.statuses.append(lbl)
            if i < len(self.STEPS) - 1:
                tk.Label(self, text="→", bg=bg, fg=C['border2'],
                         font=('Segoe UI', 8, 'bold')).grid(row=0, column=i * 2 + 2)

    def set_step(self, idx, status='done'):
        def _task():
            if idx < len(self.labels):
                dot, oval = self.labels[idx]
                color = self.COLORS.get(status, C['fg3'])
                dot.itemconfigure(oval, fill=color)
                self.statuses[idx].configure(fg=color)
        self.after(0, _task)

    def reset(self):
        def _task():
            for i in range(len(self.labels)):
                self.set_step(i, 'pending')
        self.after(0, _task)


# ── Main Application ───────────────────────────────────────────
class NILMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PyNILM v3.0  |  Non-Intrusive Load Monitoring System")
        self.root.geometry("1600x950")
        self.root.minsize(1200, 800)
        self.root.configure(bg=C['bg'])

        self.config = load_config()
        self.results = {}
        self.model = None
        self.data = None
        self.pipeline_start_time = None
        self.event_log = []

        self._create_styles()
        self._create_menu()
        self._create_main_layout()
        self._bind_shortcuts()
        self._update_clock()
        self._log("PyNILM System initialized.", 'info')
        self._log("Backend: " + ("PyTorch " + ("(GPU)" if GPU_AVAILABLE else "(CPU)") if HAS_TORCH else "NumPy fallback"), 'accent')

    # ── Styles ────────────────────────────────────────────────
    def _create_styles(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('.', background=C['bg'], foreground=C['fg'], font=('Segoe UI', 9))
        s.configure('TFrame', background=C['bg'])
        s.configure('TLabel', background=C['bg'], foreground=C['fg'], font=('Segoe UI', 9))
        s.configure('TButton', background=C['bg4'], foreground=C['fg'],
                    font=('Segoe UI', 9), borderwidth=0, padding=(10, 5))
        s.map('TButton', background=[('active', C['accent']), ('pressed', C['accent3'])],
              foreground=[('active', C['white'])])
        s.configure('Accent.TButton', background=C['accent'], foreground=C['white'],
                    font=('Segoe UI', 9, 'bold'), padding=(12, 7))
        s.map('Accent.TButton', background=[('active', C['accent2']), ('pressed', C['accent3'])])
        s.configure('Green.TButton', background=C['green'], foreground=C['white'],
                    font=('Segoe UI', 9, 'bold'))
        s.configure('Pink.TButton', background=C['pink'], foreground=C['white'],
                    font=('Segoe UI', 9, 'bold'))
        s.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'),
                    foreground=C['accent2'], background=C['bg'])
        s.configure('Subtitle.TLabel', font=('Segoe UI', 9),
                    foreground=C['fg3'], background=C['bg'])
        s.configure('Header.TLabel', font=('Segoe UI', 10, 'bold'),
                    foreground=C['fg'], background=C['bg'])
        s.configure('Dim.TLabel', foreground=C['fg3'], background=C['bg'])
        s.configure('Green.TLabel', foreground=C['green'], font=('Segoe UI', 9, 'bold'),
                    background=C['bg'])
        s.configure('Red.TLabel', foreground=C['red'], font=('Segoe UI', 9, 'bold'),
                    background=C['bg'])
        s.configure('Yellow.TLabel', foreground=C['yellow'], font=('Segoe UI', 9, 'bold'),
                    background=C['bg'])
        s.configure('Card.TLabelframe', background=C['card'], foreground=C['fg'],
                    font=('Segoe UI', 10, 'bold'), borderwidth=1, relief='solid')
        s.configure('Card.TLabelframe.Label', background=C['card'],
                    foreground=C['accent2'], font=('Segoe UI', 10, 'bold'))
        s.configure('Card.TFrame', background=C['card'])
        s.configure('Card.TButton', background=C['bg4'], foreground=C['fg'],
                    font=('Segoe UI', 9), padding=(8, 4))
        s.map('Card.TButton', background=[('active', C['accent'])],
              foreground=[('active', C['white'])])
        s.configure('TNotebook', background=C['bg'], borderwidth=0)
        s.configure('TNotebook.Tab', background=C['bg3'], foreground=C['fg3'],
                    font=('Segoe UI', 9), padding=(16, 7))
        s.map('TNotebook.Tab', background=[('selected', C['accent3'])],
              foreground=[('selected', C['white'])])
        s.configure('Horizontal.TProgressbar', background=C['accent'],
                    troughcolor=C['bg3'], borderwidth=0, thickness=6)
        s.configure('Treeview', background=C['bg2'], foreground=C['fg'],
                    fieldbackground=C['bg2'], font=('Consolas', 9),
                    borderwidth=0, rowheight=24)
        s.configure('Treeview.Heading', background=C['bg4'], foreground=C['fg2'],
                    font=('Segoe UI', 9, 'bold'))
        s.map('Treeview', background=[('selected', C['accent3'])],
              foreground=[('selected', C['white'])])
        s.configure('TScale', background=C['bg'], troughcolor=C['bg3'])
        s.configure('TScrollbar', background=C['bg4'], troughcolor=C['bg'],
                    borderwidth=0, arrowcolor=C['fg3'])
        s.configure('Horizontal.TSeparator', background=C['border'])
        
        # Custom listbox styling for Combobox dropdowns
        self.root.option_add('*TCombobox*Listbox.background', C['bg3'])
        self.root.option_add('*TCombobox*Listbox.foreground', C['fg'])
        self.root.option_add('*TCombobox*Listbox.selectBackground', C['accent'])
        self.root.option_add('*TCombobox*Listbox.selectForeground', C['white'])
        self.root.option_add('*TCombobox*Listbox.font', ('Segoe UI', 9))
        
        # Custom style overrides for Entry, Spinbox, Combobox
        s.configure('TEntry', fieldbackground=C['bg3'], foreground=C['fg'], borderwidth=0)
        s.configure('TSpinbox', fieldbackground=C['bg3'], foreground=C['fg'], borderwidth=0)
        s.configure('TCombobox', fieldbackground=C['bg3'], foreground=C['fg'], borderwidth=0)
        s.map('TCombobox', fieldbackground=[('readonly', C['bg3'])], foreground=[('readonly', C['fg'])])

    # ── Menu ──────────────────────────────────────────────────
    def _create_menu(self):
        mb = tk.Menu(self.root, bg=C['bg2'], fg=C['fg'],
                     activebackground=C['accent'], activeforeground=C['white'],
                     borderwidth=0)
        self.root.config(menu=mb)

        for label, items in [
            ("File", [
                ("Load Config...", self._load_config, "Ctrl+O"),
                ("Save Config...", self._save_config, "Ctrl+S"),
                None,
                ("Export Results...", self._export_results, "Ctrl+E"),
                ("Export Log...", self._export_log, None),
                None,
                ("Exit", self.root.quit, "Ctrl+Q"),
            ]),
            ("Data", [
                ("Load UK-DALE", lambda: self._load_dataset('UK-DALE'), "Ctrl+1"),
                ("Load REFIT", lambda: self._load_dataset('REFIT'), "Ctrl+2"),
                ("Generate Synthetic", self._generate_synthetic, "Ctrl+3"),
                None,
                ("Browse Data...", self._browse_data, None),
            ]),
            ("Analysis", [
                ("Run Full Pipeline", self._run_pipeline, "Ctrl+R"),
                ("Train Model", self._train_model, "Ctrl+T"),
                ("Run Federated Learning", self._run_fl, "Ctrl+F"),
                None,
                ("Preprocessing", self._run_preprocessing, None),
                ("Feature Extraction", self._run_features, None),
                ("Classification", self._run_classification, None),
                ("Fuzzy Logic", self._run_fuzzy, None),
                ("Event Detection", self._run_event_detection, None),
                ("Evaluation", self._run_evaluation, None),
            ]),
            ("Tools", [
                ("Run Tests", self._run_tests, "Ctrl+Shift+T"),
                ("Signal Quality", self._run_signal_quality, None),
                ("Multi-Household", self._run_multi_household, None),
                ("Online Learning", self._run_online_learning, None),
                ("Transfer Learning", self._run_transfer, None),
            ]),
            ("View", [
                ("Dashboard", lambda: self._show_tab('Dashboard'), None),
                ("Signals", lambda: self._show_tab('Signals'), None),
                ("Features", lambda: self._show_tab('Features'), None),
                ("Data Table", lambda: self._show_tab('Data Table'), None),
                ("Log", lambda: self._show_tab('Log'), None),
                None,
                ("Refresh Dashboard", self._refresh_dashboard, None),
            ]),
            ("Help", [
                ("About", self._show_about, None),
                ("Keyboard Shortcuts", self._show_shortcuts, None),
                ("Architecture", self._show_architecture, None),
            ]),
        ]:
            m = tk.Menu(mb, tearoff=0, bg=C['bg2'], fg=C['fg'],
                        activebackground=C['accent'], borderwidth=0)
            mb.add_cascade(label=label, menu=m)
            for item in items:
                if item is None:
                    m.add_separator()
                else:
                    txt, cmd, accel = item
                    kw = {'accelerator': accel} if accel else {}
                    m.add_command(label=txt, command=cmd, **kw)

    # ── Main Layout ───────────────────────────────────────────
    def _create_main_layout(self):
        # 1. Enable Immersive Dark Title Bar on Windows
        try:
            import ctypes
            self.root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            # DWMWA_USE_IMMERSIVE_DARK_MODE = 20 or 19
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(ctypes.c_int(1)), 4)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 19, ctypes.byref(ctypes.c_int(1)), 4)
        except Exception:
            pass

        # 2. Bottom Status Bar (Height 52px)
        self._create_status_bar()

        # 3. Sidebar Navigation Frame (Width 240px)
        self.sidebar = tk.Frame(self.root, bg=C['bg2'], width=240)
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        # 4. Main content stacked frames container
        self.main_content = tk.Frame(self.root, bg=C['bg'])
        self.main_content.pack(side='right', fill='both', expand=True)

        # 5. Build and register all pages
        self.current_page = None
        self.pages = {}
        page_builders = {
            'Dashboard': self._build_dashboard_page,
            'Data & Signals': self._build_data_signals_page,
            'Features': self._build_features_page,
            'Model & Training': self._build_training_evaluation_page,
            'Federated Learning': self._build_federated_learning_page,
            'Events & Fuzzy': self._build_events_fuzzy_page,
            'Tools & Exports': self._build_tools_page,
            'System Logs': self._build_logs_page
        }

        # First initialize all page container frames
        for name in page_builders.keys():
            self.pages[name] = tk.Frame(self.main_content, bg=C['bg'])

        # Setup helper mapping for viz_tabs to retain code compatibility
        self.viz_tabs = {}

        # Now execute each builder
        for name, builder in page_builders.items():
            builder()

        # Show initial page
        self._show_page('Dashboard')

    def _build_sidebar(self):
        # Sidebar Logo / Title
        logo_frame = tk.Frame(self.sidebar, bg=C['bg2'], pady=15)
        logo_frame.pack(fill='x')
        
        tk.Label(logo_frame, text="⚡ PyNILM", bg=C['bg2'], fg=C['accent2'],
                 font=('Segoe UI', 16, 'bold')).pack(anchor='w', padx=18)
        tk.Label(logo_frame, text="Non-Intrusive Load Monitoring", bg=C['bg2'], fg=C['fg3'],
                 font=('Segoe UI', 8, 'italic')).pack(anchor='w', padx=18)
        
        tk.Frame(self.sidebar, bg=C['border'], height=1).pack(fill='x', padx=10, pady=(0, 10))
        
        # Navigation Buttons container
        self.nav_buttons = {}
        tabs = [
            ("Dashboard", "📊"),
            ("Data & Signals", "📈"),
            ("Features", "🔬"),
            ("Model & Training", "🧠"),
            ("Federated Learning", "🌐"),
            ("Events & Fuzzy", "⚡"),
            ("Tools & Exports", "🛠️"),
            ("System Logs", "📝")
        ]
        
        for name, icon in tabs:
            btn_frame = tk.Frame(self.sidebar, bg=C['bg2'], height=40)
            btn_frame.pack(fill='x', padx=10, pady=2)
            btn_frame.pack_propagate(False)
            
            # Left active accent indicator strip (hidden by default)
            indicator = tk.Frame(btn_frame, bg=C['accent'], width=4)
            indicator.pack(side='left', fill='y')
            
            # The button itself
            btn = tk.Button(btn_frame, text=f"  {icon}   {name}", 
                            bg=C['bg2'], fg=C['fg2'],
                            activebackground=C['bg3'], activeforeground=C['white'],
                            font=('Segoe UI', 9, 'bold'), relief='flat', bd=0, 
                            anchor='w', cursor='hand2')
            btn.pack(side='left', fill='both', expand=True)
            
            # Mouse hover bind
            def _make_hover(b, ind, n=name):
                b.bind("<Enter>", lambda e: b.config(bg=C['bg3'], fg=C['white']) if self.current_page != n else None)
                b.bind("<Leave>", lambda e: b.config(bg=C['bg2'], fg=C['fg2']) if self.current_page != n else None)
                b.config(command=lambda: self._show_page(n))
            _make_hover(btn, indicator)
            
            self.nav_buttons[name] = (btn, indicator, btn_frame)

        # Space separator
        tk.Frame(self.sidebar, bg=C['bg2']).pack(fill='both', expand=True)

        # Quick Status Panel at the bottom of the sidebar
        status_card = tk.Frame(self.sidebar, bg=C['card'], highlightbackground=C['border'], highlightthickness=1, padx=8, pady=8)
        status_card.pack(fill='x', padx=10, pady=10)
        
        tk.Label(status_card, text="SYSTEM STATUS", bg=C['card'], fg=C['accent2'], font=('Segoe UI', 8, 'bold')).pack(anchor='w', pady=(0, 4))
        
        self.status_labels = {}
        for key, val in [('Config', 'Default'), ('Data', 'Not loaded'),
                          ('Model', 'Not trained'), ('Pipeline', 'Idle')]:
            r = tk.Frame(status_card, bg=C['card'])
            r.pack(fill=tk.X, pady=1)
            tk.Label(r, text=key + ":", bg=C['card'], fg=C['fg3'],
                     font=('Segoe UI', 8, 'bold'), width=8, anchor='w').pack(side='left')
            lbl = tk.Label(r, text=val, bg=C['card'], fg=C['red'] if 'Not' in val else C['fg2'], font=('Segoe UI', 8))
            lbl.pack(side='left')
            self.status_labels[key] = lbl
            
        # Quick run pipeline button
        run_btn = tk.Button(self.sidebar, text="⚡  RUN PIPELINE", bg=C['accent'], fg=C['white'],
                            activebackground=C['accent2'], activeforeground=C['white'],
                            font=('Segoe UI', 9, 'bold'), relief='flat', bd=0, pady=8, cursor='hand2')
        run_btn.pack(fill='x', padx=10, pady=(0, 15))
        run_btn.config(command=self._run_pipeline)
        Tooltip(run_btn, "Run the entire end-to-end pipeline (Ctrl+R)")

    def _show_page(self, name):
        self.current_page = name
        # Hide all pages
        for p_name, page in self.pages.items():
            page.pack_forget()
        # Show selected page
        self.pages[name].pack(fill='both', expand=True)
        
        # Update sidebar active styles
        for b_name, (btn, indicator, btn_frame) in self.nav_buttons.items():
            if b_name == name:
                btn.config(bg=C['bg3'], fg=C['accent2'])
                btn_frame.config(bg=C['bg3'])
                indicator.pack(side='left', fill='y')
            else:
                btn.config(bg=C['bg2'], fg=C['fg2'])
                btn_frame.config(bg=C['bg2'])
                indicator.pack_forget()

    def _create_status_bar(self):
        # Combined bottom bar
        self.bottom_bar = tk.Frame(self.root, bg=C['bg3'], height=52)
        self.bottom_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.bottom_bar.pack_propagate(False)
        
        # Left region: Status Indicators
        left_f = tk.Frame(self.bottom_bar, bg=C['bg3'])
        left_f.pack(side='left', fill='y', padx=10)
        
        self.status_data = tk.Label(left_f, text="📊 Data: No data", bg=C['bg3'], fg=C['fg2'], font=('Segoe UI', 8, 'bold'))
        self.status_data.pack(side='left', padx=10)
        
        self.status_model = tk.Label(left_f, text="🧠 Model: No model", bg=C['bg3'], fg=C['fg2'], font=('Segoe UI', 8, 'bold'))
        self.status_model.pack(side='left', padx=10)

        self.status_events = tk.Label(left_f, text="", bg=C['bg3'], fg=C['fg3'], font=('Segoe UI', 8))
        self.status_events.pack(side='left', padx=10)
        
        # Middle region: Pipeline indicators and progress
        mid_f = tk.Frame(self.bottom_bar, bg=C['bg3'])
        mid_f.pack(side='left', fill='both', expand=True, padx=20)
        
        self.pipeline_indicator = PipelineStepIndicator(mid_f, bg=C['bg3'])
        self.pipeline_indicator.pack(side='left', padx=10)
        
        prog_f = tk.Frame(mid_f, bg=C['bg3'])
        prog_f.pack(side='right', fill='x', expand=True, padx=10)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(prog_f, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', pady=(6, 2))
        
        self.pipeline_status = tk.Label(prog_f, text="Ready", bg=C['bg3'], fg=C['fg3'], font=('Segoe UI', 8))
        self.pipeline_status.pack(anchor='e')

        # Right region: Hardware and Clock
        right_f = tk.Frame(self.bottom_bar, bg=C['bg3'])
        right_f.pack(side='right', fill='y', padx=10)
        
        torch_s = C['green'] if HAS_TORCH else C['red']
        tk.Label(right_f, text="PyTorch", bg=C['bg3'], fg=torch_s, font=('Segoe UI', 8, 'bold')).pack(side='left', padx=5)
        
        gpu_s = C['green'] if GPU_AVAILABLE else C['fg3']
        tk.Label(right_f, text=GPU_NAME if GPU_AVAILABLE else "CPU", bg=C['bg3'], fg=gpu_s, font=('Segoe UI', 8)).pack(side='left', padx=5)
        
        self.status_clock = tk.Label(right_f, text="", bg=C['bg3'], fg=C['fg3'], font=('Consolas', 9))
        self.status_clock.pack(side='left', padx=10)

    def _update_clock(self):
        try:
            self.status_clock.configure(text=time.strftime("%Y-%m-%d  %H:%M:%S"))
        except Exception:
            pass
        self.root.after(1000, self._update_clock)

    # ── Dashboard Page ──────────────────────────────────────────
    def _build_dashboard_page(self):
        p = self.pages['Dashboard']
        
        # Top Cards Panel
        top = tk.Frame(p, bg=C['bg'], pady=10, padx=10)
        top.pack(fill='x')
        
        self.metric_cards = {}
        for i, (title, color) in enumerate([
            ('Accuracy', C['green']), ('F1 Score', C['accent2']),
            ('Events', C['cyan']), ('Appliances', C['pink']),
            ('Windows', C['yellow']), ('Features', C['orange']),
        ]):
            mc = MetricCard(top, title=title, value="--", color=color)
            mc.grid(row=0, column=i, padx=5, sticky='ew')
            self.metric_cards[title] = mc
        for i in range(6):
            top.grid_columnconfigure(i, weight=1)
            
        # Center Plot card
        plot_card, plot_content = create_section_card(p, "System Operations Overview Dashboard")
        plot_card.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        self.dash_fig = Figure(figsize=(12, 7), dpi=90, facecolor=C['bg'])
        self.dash_axes = self.dash_fig.subplots(2, 4)
        for ax in self.dash_axes.flat:
            ax.set_facecolor(C['bg2'])
            ax.tick_params(colors=C['fg3'], labelsize=6)
            for spine in ax.spines.values():
                spine.set_color(C['border'])
        self.dash_fig.tight_layout(pad=2.5)
        self.dash_canvas = FigureCanvasTkAgg(self.dash_fig, master=plot_content)
        self.dash_canvas.get_tk_widget().pack(fill='both', expand=True)

    # ── Data & Signals Page ─────────────────────────────────────
    def _build_data_signals_page(self):
        p = self.pages['Data & Signals']
        
        # Split Layout
        left_col = tk.Frame(p, bg=C['bg'], width=340)
        left_col.pack(side='left', fill='y', padx=10, pady=10)
        left_col.pack_propagate(False)
        
        right_col = tk.Frame(p, bg=C['bg'])
        right_col.pack(side='right', fill='both', expand=True, padx=(0, 10), pady=10)
        
        # Left card for parameters
        param_card, param_content = create_section_card(left_col, "Dataset Ingestion Settings")
        param_card.pack(fill='both', expand=True)
        
        # Add parameter fields to parameter content
        sf = ScrollableFrame(param_content, bg=C['card'])
        sf.pack(fill='both', expand=True)
        content_frame = sf.scrollable_frame
        
        for lbl_text, var_class, var_kw, row_kw in [
            ('Dataset', tk.StringVar, {'value': 'UK-DALE'}, {}),
            ('Home ID', tk.IntVar, {'value': 1}, {'from_': 1, 'to': 10}),
            ('Window', tk.IntVar, {'value': 256}, {'from_': 64, 'to': 2048, 'increment': 64}),
            ('Duration', tk.IntVar, {'value': 86400}, {'from_': 3600, 'to': 604800, 'increment': 3600}),
            ('Overlap', tk.IntVar, {'value': 128}, {'from_': 0, 'to': 512, 'increment': 32}),
        ]:
            r = tk.Frame(content_frame, bg=C['card'], pady=6)
            r.pack(fill=tk.X)
            tk.Label(r, text=lbl_text + ":", bg=C['card'], fg=C['fg2'],
                     font=('Segoe UI', 9, 'bold'), width=12, anchor='w').pack(side='left')
            var = var_class(**var_kw)
            setattr(self, lbl_text.lower().replace(' ', '_') + '_var', var)
            if lbl_text == 'Dataset':
                cb = ttk.Combobox(r, textvariable=var, values=["UK-DALE", "REFIT"],
                                   width=14, state='readonly')
                cb.pack(side='left', padx=4, fill='x', expand=True)
            else:
                ttk.Spinbox(r, textvariable=var, width=10, **row_kw).pack(side='left', padx=4)
                
        # Action Buttons
        tk.Frame(content_frame, bg=C['card'], height=15).pack()
        
        lb = create_hover_button(content_frame, "📥  LOAD DATASET", self._load_dataset_btn, C['accent'], C['white'], C['accent2'], C['white'], pady=8)
        lb.pack(fill='x', pady=4)
        Tooltip(lb, "Load specified dataset (Ctrl+1 or Ctrl+2)")
        
        sb = create_hover_button(content_frame, "⚙️  GENERATE SYNTHETIC", self._generate_synthetic, C['bg4'], C['fg'], C['accent3'], C['white'], pady=8)
        sb.pack(fill='x', pady=4)
        Tooltip(sb, "Generate synthetic training & testing data (Ctrl+3)")

        # Right Column grid
        right_col.rowconfigure(0, weight=3) # Signal card row
        right_col.rowconfigure(1, weight=2) # Table card row
        right_col.columnconfigure(0, weight=1)
        
        # Card 1: Signal plot
        signal_card, self.signal_plot_container = create_section_card(right_col, "Temporal Load Signals Analysis")
        signal_card.grid(row=0, column=0, sticky='nsew', pady=(0, 10))
        
        self.signal_empty_lbl = tk.Label(self.signal_plot_container, text="No signal visualization loaded. Load data to preview.", 
                                         bg=C['card'], fg=C['fg3'], font=('Segoe UI', 10, 'italic'))
        self.signal_empty_lbl.pack(expand=True)
        self.viz_tabs['Signals'] = self.signal_plot_container

        # Card 2: Data table
        table_card, table_content = create_section_card(right_col, "Data Window Statistics Preview")
        table_card.grid(row=1, column=0, sticky='nsew')
        
        self.data_table_preview = DataTable(table_content, columns=['Window', 'Label', 'Mean Power', 'Max Power', 'Std', 'Duration'])
        self.data_table_preview.pack(fill='both', expand=True)
        self.data_table = self.data_table_preview
        self.viz_tabs['Data Table'] = self.data_table

    # ── Features Page ───────────────────────────────────────────
    def _build_features_page(self):
        p = self.pages['Features']
        
        # Split Layout
        left_col = tk.Frame(p, bg=C['bg'], width=340)
        left_col.pack(side='left', fill='y', padx=10, pady=10)
        left_col.pack_propagate(False)
        
        right_col = tk.Frame(p, bg=C['bg'])
        right_col.pack(side='right', fill='both', expand=True, padx=(0, 10), pady=10)
        
        # Feature Settings Card
        feat_card, feat_content = create_section_card(left_col, "Feature Extraction Settings")
        feat_card.pack(fill='both', expand=True)
        
        # Parameters
        sf = ScrollableFrame(feat_content, bg=C['card'])
        sf.pack(fill='both', expand=True)
        c_frame = sf.scrollable_frame
        
        tk.Label(c_frame, text="Variational Mode Decomposition", bg=C['card'], fg=C['accent2'], font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(0, 10))
        
        # K Range Start/End variables
        self.vmd_k_start_var = tk.IntVar(value=3)
        self.vmd_k_end_var = tk.IntVar(value=7)
        
        r1 = tk.Frame(c_frame, bg=C['card'], pady=4)
        r1.pack(fill='x')
        tk.Label(r1, text="K Range Start:", bg=C['card'], fg=C['fg2'], font=('Segoe UI', 9, 'bold'), width=15, anchor='w').pack(side='left')
        ttk.Spinbox(r1, textvariable=self.vmd_k_start_var, from_=2, to=10, width=8).pack(side='left')
        
        r2 = tk.Frame(c_frame, bg=C['card'], pady=4)
        r2.pack(fill='x')
        tk.Label(r2, text="K Range End:", bg=C['card'], fg=C['fg2'], font=('Segoe UI', 9, 'bold'), width=15, anchor='w').pack(side='left')
        ttk.Spinbox(r2, textvariable=self.vmd_k_end_var, from_=2, to=15, width=8).pack(side='left')

        tk.Frame(c_frame, bg=C['card'], height=15).pack()
        
        # Buttons
        prep_btn = create_hover_button(c_frame, "🧹  RUN PREPROCESSING", self._run_preprocessing, C['bg4'], C['fg'], C['accent3'], C['white'], pady=8)
        prep_btn.pack(fill='x', pady=4)
        
        feat_btn = create_hover_button(c_frame, "🔬  EXTRACT FEATURES", self._run_features, C['accent'], C['white'], C['accent2'], C['white'], pady=8)
        feat_btn.pack(fill='x', pady=4)
        Tooltip(feat_btn, "Run VMD mode separation and extract TEO energy profiles")
        
        # Status box
        status_box = tk.Frame(c_frame, bg=C['bg3'], highlightbackground=C['border'], highlightthickness=1, padx=10, pady=10)
        status_box.pack(fill='x', pady=15)
        self.features_status_lbl = tk.Label(status_box, text="Status: Ready\nNo features extracted yet.", 
                                            bg=C['bg3'], fg=C['fg2'], font=('Segoe UI', 9), justify='left')
        self.features_status_lbl.pack(anchor='w')

        # Right Column content
        modes_card, self.features_plot_container = create_section_card(right_col, "Variational Mode Separation & Signal Analysis Features")
        modes_card.pack(fill='both', expand=True)
        
        empty_lbl1 = tk.Label(self.features_plot_container, text="No VMD separation plot loaded. Extract features to preview.", 
                              bg=C['card'], fg=C['fg3'], font=('Segoe UI', 10, 'italic'))
        empty_lbl1.pack(expand=True)
        
        self.viz_tabs['Features'] = self.features_plot_container

    # ── Training & Evaluation Page ──────────────────────────────
    def _build_training_evaluation_page(self):
        p = self.pages['Model & Training']
        
        # Split Layout
        left_col = tk.Frame(p, bg=C['bg'], width=360)
        left_col.pack(side='left', fill='y', padx=10, pady=10)
        left_col.pack_propagate(False)
        
        right_col = tk.Frame(p, bg=C['bg'])
        right_col.pack(side='right', fill='both', expand=True, padx=(0, 10), pady=10)
        
        # Left Column Grid
        left_col.rowconfigure(0, weight=3) # Hyperparameters
        left_col.rowconfigure(1, weight=2) # Gauges
        left_col.columnconfigure(0, weight=1)

        # Hyperparameters Card
        hp_card, hp_content = create_section_card(left_col, "Model Hyperparameters")
        hp_card.grid(row=0, column=0, sticky='nsew', pady=(0, 10))
        
        sf = ScrollableFrame(hp_content, bg=C['card'])
        sf.pack(fill='both', expand=True)
        c_frame = sf.scrollable_frame
        
        params = [('Classes', 'num_classes', tk.IntVar, 5, 2, 20),
                  ('Epochs', 'epochs', tk.IntVar, 50, 10, 500),
                  ('Latent Dim', 'latent_dim', tk.IntVar, 32, 8, 256),
                  ('Batch Size', 'batch_size', tk.IntVar, 32, 8, 256),
                  ('Learning Rate', 'lr', tk.DoubleVar, 0.001, 0.0001, 0.1)]
        for lbl, attr, cls, val, lo, hi in params:
            r = tk.Frame(c_frame, bg=C['card'], pady=5)
            r.pack(fill=tk.X)
            tk.Label(r, text=lbl + ":", bg=C['card'], fg=C['fg2'],
                     font=('Segoe UI', 9, 'bold'), width=14, anchor='w').pack(side='left')
            var = cls(value=val)
            setattr(self, attr + '_var', var)
            if cls == tk.DoubleVar:
                ttk.Entry(r, textvariable=var, width=10).pack(side='left', padx=4)
            else:
                ttk.Spinbox(r, textvariable=var, from_=lo, to=hi, width=8).pack(side='left', padx=4)
                
        tk.Frame(c_frame, bg=C['card'], height=10).pack()
        
        tb = create_hover_button(c_frame, "🧠  TRAIN MODEL", self._train_model, C['accent'], C['white'], C['accent2'], C['white'], pady=8)
        tb.pack(fill='x', pady=4)
        Tooltip(tb, "Train the VAE-DCCNN-Attention model (Ctrl+T)")
        
        bf = tk.Frame(c_frame, bg=C['card'])
        bf.pack(fill='x', pady=2)
        
        lb = create_hover_button(bf, "📂  LOAD MODEL", self._load_model, C['bg4'], C['fg'], C['accent3'], C['white'], pady=6)
        lb.pack(side='left', fill='x', expand=True, padx=(0, 2))
        
        sb = create_hover_button(bf, "💾  SAVE MODEL", self._save_model, C['bg4'], C['fg'], C['accent3'], C['white'], pady=6)
        sb.pack(side='right', fill='x', expand=True, padx=(2, 0))

        # Gauges Card
        gauge_card, gauge_content = create_section_card(left_col, "Evaluation Gauges")
        gauge_card.grid(row=1, column=0, sticky='nsew')
        
        gf = tk.Frame(gauge_content, bg=C['card'])
        gf.pack(fill='both', expand=True)
        self.gauge_accuracy = GaugeCanvas(gf, size=85, label="Accuracy", color=C['green'])
        self.gauge_accuracy.pack(side='left', padx=3, fill='both', expand=True)
        self.gauge_f1 = GaugeCanvas(gf, size=85, label="F1 Score", color=C['accent2'])
        self.gauge_f1.pack(side='left', padx=3, fill='both', expand=True)
        self.gauge_quality = GaugeCanvas(gf, size=85, label="Quality", color=C['cyan'])
        self.gauge_quality.pack(side='left', padx=3, fill='both', expand=True)

        # Right Column Grid content
        right_col.rowconfigure(0, weight=2)
        right_col.rowconfigure(1, weight=3)
        right_col.columnconfigure(0, weight=1)
        right_col.columnconfigure(1, weight=1)
        
        # Training Loss/Acc plots Card
        train_card, self.training_plot_container = create_section_card(right_col, "Model Training History Performance")
        train_card.grid(row=0, column=0, columnspan=2, sticky='nsew', pady=(0, 10))
        self.viz_tabs['Training'] = self.training_plot_container
        
        empty_lbl2 = tk.Label(self.training_plot_container, text="Model not trained. Train to see loss/accuracy curves.", 
                              bg=C['card'], fg=C['fg3'], font=('Segoe UI', 10, 'italic'))
        empty_lbl2.pack(expand=True)

        # Confusion Matrix Card (bottom left of right_col)
        cm_card, self.cm_plot_container = create_section_card(right_col, "Confusion Matrix")
        cm_card.grid(row=1, column=0, sticky='nsew', padx=(0, 5))
        self.viz_tabs['Confusion Matrix'] = self.cm_plot_container
        
        self.cm_fig = Figure(figsize=(5, 5), dpi=90, facecolor=C['bg'])
        self.cm_ax = self.cm_fig.add_subplot(111)
        self.cm_ax.set_facecolor(C['bg2'])
        self.cm_fig.tight_layout(pad=2.5)
        self.cm_canvas = FigureCanvasTkAgg(self.cm_fig, master=self.cm_plot_container)
        self.cm_canvas.get_tk_widget().pack(fill='both', expand=True)

        # Evaluation Report Card (bottom right of right_col)
        report_card, report_content = create_section_card(right_col, "Classification Metrics Report")
        report_card.grid(row=1, column=1, sticky='nsew', padx=(5, 0))
        self.viz_tabs['Evaluation'] = report_content
        
        self.eval_text = tk.Text(report_content, wrap='word', font=('Consolas', 9),
                                  bg='#0a0d14', fg='#94a3b8', relief='flat',
                                  padx=8, pady=8, spacing1=2)
        scroll = ttk.Scrollbar(report_content, command=self.eval_text.yview)
        self.eval_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.eval_text.pack(fill='both', expand=True)

    # ── Federated Learning Page ─────────────────────────────────
    def _build_federated_learning_page(self):
        p = self.pages['Federated Learning']
        
        # Split Layout
        left_col = tk.Frame(p, bg=C['bg'], width=340)
        left_col.pack(side='left', fill='y', padx=10, pady=10)
        left_col.pack_propagate(False)
        
        right_col = tk.Frame(p, bg=C['bg'])
        right_col.pack(side='right', fill='both', expand=True, padx=(0, 10), pady=10)
        
        # FL Parameters Card
        fl_card, fl_content = create_section_card(left_col, "Federated Learning Settings")
        fl_card.pack(fill='both', expand=True)
        
        sf = ScrollableFrame(fl_content, bg=C['card'])
        sf.pack(fill='both', expand=True)
        c_frame = sf.scrollable_frame
        
        for lbl, attr, val, lo, hi in [('Clients', 'fl_clients', 20, 2, 100),
                                         ('Rounds', 'fl_rounds', 30, 5, 200),
                                         ('Byzantine %', 'fl_byz', 10, 0, 50)]:
            r = tk.Frame(c_frame, bg=C['card'], pady=6)
            r.pack(fill=tk.X)
            tk.Label(r, text=lbl + ":", bg=C['card'], fg=C['fg2'],
                     font=('Segoe UI', 9, 'bold'), width=14, anchor='w').pack(side='left')
            var = tk.IntVar(value=val)
            setattr(self, attr + '_var', var)
            ttk.Spinbox(r, textvariable=var, from_=lo, to=hi, width=8).pack(side='left', padx=4)
            
        tk.Frame(c_frame, bg=C['card'], height=15).pack()
        
        fb = create_hover_button(c_frame, "🌐  RUN FEDERATED SIMULATION", self._run_fl, C['pink'], C['white'], C['pink2'], C['white'], pady=8)
        fb.pack(fill='x', pady=4)
        Tooltip(fb, "Run Byzantine-robust GAT/Multi-Krum Federated Learning simulation (Ctrl+F)")

        # Right Column content
        # Plot Card
        fl_plot_card, self.fl_plot_container = create_section_card(right_col, "Federated Learning Performance metrics")
        fl_plot_card.pack(fill='both', expand=True)
        
        empty_lbl3 = tk.Label(self.fl_plot_container, text="FL Simulation not run. Click run to see simulation statistics.", 
                              bg=C['card'], fg=C['fg3'], font=('Segoe UI', 10, 'italic'))
        empty_lbl3.pack(expand=True)
        self.viz_tabs['FL Results'] = self.fl_plot_container

    # ── Events & Fuzzy Page ─────────────────────────────────────
    def _build_events_fuzzy_page(self):
        p = self.pages['Events & Fuzzy']
        
        # Split Layout
        left_col = tk.Frame(p, bg=C['bg'], width=340)
        left_col.pack(side='left', fill='y', padx=10, pady=10)
        left_col.pack_propagate(False)
        
        right_col = tk.Frame(p, bg=C['bg'])
        right_col.pack(side='right', fill='both', expand=True, padx=(0, 10), pady=10)
        
        # Event Settings Card
        settings_card, settings_content = create_section_card(left_col, "Event Detection & Fuzzy Logic")
        settings_card.pack(fill='both', expand=True)
        
        sf = ScrollableFrame(settings_content, bg=C['card'])
        sf.pack(fill='both', expand=True)
        c_frame = sf.scrollable_frame
        
        tk.Label(c_frame, text="Event Thresholds", bg=C['card'], fg=C['accent2'], font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(0, 10))
        
        self.event_sens_var = tk.DoubleVar(value=0.5)
        r1 = tk.Frame(c_frame, bg=C['card'], pady=6)
        r1.pack(fill='x')
        tk.Label(r1, text="Sensitivity:", bg=C['card'], fg=C['fg2'], font=('Segoe UI', 9, 'bold'), width=12, anchor='w').pack(side='left')
        ttk.Scale(r1, variable=self.event_sens_var, from_=0.1, to=1.0).pack(side='left', padx=4, fill='x', expand=True)
        
        tk.Frame(c_frame, bg=C['card'], height=15).pack()
        
        btn_fuzzy = create_hover_button(c_frame, "🔮  EVALUATE FUZZY LOGIC", self._run_fuzzy, C['bg4'], C['fg'], C['accent3'], C['white'], pady=8)
        btn_fuzzy.pack(fill='x', pady=4)
        
        btn_detect = create_hover_button(c_frame, "⚡  RUN EVENT DETECTION", self._run_event_detection, C['accent'], C['white'], C['accent2'], C['white'], pady=8)
        btn_detect.pack(fill='x', pady=4)
        Tooltip(btn_detect, "Run the event detection algorithms on active signal stream")
        
        # Right Column content
        right_col.rowconfigure(0, weight=1)
        right_col.rowconfigure(1, weight=1)
        right_col.columnconfigure(0, weight=1)
        
        # Events Table Card
        ev_card, ev_content = create_section_card(right_col, "Detected Load Events Log")
        ev_card.grid(row=0, column=0, sticky='nsew', pady=(0, 10))
        
        self.events_table = DataTable(ev_content, columns=['#', 'Type', 'Time (s)', 'Power (W)', 'Appliance', 'Confidence'])
        self.events_table.pack(fill='both', expand=True)
        self.viz_tabs['Events'] = self.events_table
        
        # Data Table Card
        dt_card, dt_content = create_section_card(right_col, "Signal Window Features List")
        dt_card.grid(row=1, column=0, sticky='nsew')
        
        self.data_table_preview = DataTable(dt_content, columns=['Window', 'Label', 'Mean Power', 'Max Power', 'Std', 'Duration'])
        self.data_table_preview.pack(fill='both', expand=True)
        self.data_table = self.data_table_preview
        self.viz_tabs['Data Table'] = self.data_table

    # ── Tools Page ──────────────────────────────────────────────
    def _build_tools_page(self):
        p = self.pages['Tools & Exports']
        
        sf = ScrollableFrame(p, bg=C['bg'])
        sf.pack(fill='both', expand=True, padx=10, pady=10)
        c_frame = sf.scrollable_frame
        
        tk.Label(c_frame, text="⚙️  Advanced Diagnostic & Simulation Tools", bg=C['bg'], fg=C['accent2'], font=('Segoe UI', 14, 'bold')).pack(anchor='w', pady=(0, 15))
        
        grid_frame = tk.Frame(c_frame, bg=C['bg'])
        grid_frame.pack(fill='both', expand=True)
        
        tools = [
            ("Signal Quality Assessor", "Assess current grid load signal quality, signal-to-noise ratio (SNR), distortion, and metrics.", "🔍  ASSESS QUALITY", self._run_signal_quality, C['accent']),
            ("Multi-Household Simulator", "Simulate a demand-response power grid with 10 virtual households and aggregate demand profiles.", "🏢  RUN SIMULATOR", self._run_multi_household, C['green']),
            ("Online Learning & Concept Drift", "Inject simulated concept drift and adapt classification model coefficients in real time.", "📈  RUN ONLINE DRIFT", self._run_online_learning, C['pink']),
            ("Transfer Learning Manager", "Map pre-trained model domains from source buildings to target spaces for fine-tuning.", "✈️  INIT TRANSFER", self._run_transfer, C['cyan2']),
            ("System Verification Tests", "Execute the entire automated system verification suite, tests, and math validation code.", "✔️  RUN ALL TESTS", self._run_tests, C['yellow']),
            ("Data Exports Manager", "Save full model predictions, fuzzy logic states, and parameters into JSON/Excel reports.", "📤  EXPORT REPORT", self._export_results, C['accent3']),
        ]
        
        for i, (title, desc, btn_text, cmd, color) in enumerate(tools):
            card = tk.Frame(grid_frame, bg=C['card'], highlightbackground=C['border'], highlightthickness=1, padx=15, pady=15)
            card.grid(row=i // 2, column=i % 2, padx=6, pady=6, sticky='nsew')
            
            tk.Label(card, text=title, bg=C['card'], fg=C['white'], font=('Segoe UI', 11, 'bold')).pack(anchor='w', pady=(0, 6))
            tk.Label(card, text=desc, bg=C['card'], fg=C['fg2'], font=('Segoe UI', 9), wraplength=380, justify='left').pack(anchor='w', fill='both', expand=True, pady=(0, 12))
            
            btn = create_hover_button(card, btn_text, cmd, color, C['white'], C['white'], color, font=('Segoe UI', 9, 'bold'), pady=8)
            btn.pack(fill='x', side='bottom')
            
        grid_frame.grid_columnconfigure(0, weight=1)
        grid_frame.grid_columnconfigure(1, weight=1)

    # ── System Logs Page ────────────────────────────────────────
    def _build_logs_page(self):
        p = self.pages['System Logs']
        
        log_card, log_content = create_section_card(p, "System Operations & Execution Console")
        log_card.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.log_panel = LogPanel(log_content)
        self.log_panel.pack(fill='both', expand=True)
        self.viz_tabs['Log'] = self.log_panel

    def _show_tab(self, name):
        mapping = {
            'Dashboard': 'Dashboard',
            'Signals': 'Data & Signals',
            'Features': 'Features',
            'FL Results': 'Federated Learning',
            'Training': 'Model & Training',
            'Evaluation': 'Model & Training',
            'Confusion Matrix': 'Model & Training',
            'Events': 'Events & Fuzzy',
            'Data Table': 'Events & Fuzzy',
            'Log': 'System Logs'
        }
        target = mapping.get(name, 'Dashboard')
        self._show_page(target)

    def _refresh_dashboard(self):
        self._update_dashboard()

    # ── Logging ───────────────────────────────────────────────
    def _log(self, msg, tag='info'):
        def _task():
            if hasattr(self, 'log_panel'):
                self.log_panel.log(msg, tag)
            ts = time.strftime("%H:%M:%S")
            self.event_log.append(f"[{ts}] {msg}")
        self.root.after(0, _task)

    def _update_progress(self, value, text=None):
        def _task():
            self.progress_var.set(value)
            if text:
                self.pipeline_status.configure(text=text)
            self.root.update_idletasks()
        self.root.after(0, _task)

    def _set_status(self, key, text, ok=True):
        def _task():
            if key in self.status_labels:
                self.status_labels[key].configure(text=text)
                self.status_labels[key].configure(fg=C['green'] if ok else C['red'])
        self.root.after(0, _task)    # ── Keyboard Shortcuts ────────────────────────────────────
    def _bind_shortcuts(self):
        bindings = {
            '<Control-o>': self._load_config, '<Control-s>': self._save_config,
            '<Control-e>': self._export_results, '<Control-q>': self.root.quit,
            '<Control-r>': self._run_pipeline, '<Control-t>': self._train_model,
            '<Control-f>': self._run_fl, '<Control-1>': lambda: self._load_dataset('UK-DALE'),
            '<Control-2>': lambda: self._load_dataset('REFIT'),
            '<Control-3>': self._generate_synthetic,
            '<Control-Shift-T>': self._run_tests,
        }
        for key, cmd in bindings.items():
            self.root.bind(key, lambda e, c=cmd: c())

    def _show_shortcuts(self):
        messagebox.showinfo("Keyboard Shortcuts", (
            "Ctrl+1       Load UK-DALE\n"
            "Ctrl+2       Load REFIT\n"
            "Ctrl+3       Generate Synthetic\n"
            "Ctrl+R       Run Pipeline\n"
            "Ctrl+T       Train Model\n"
            "Ctrl+F       Run Federated Learning\n"
            "Ctrl+O       Load Config\n"
            "Ctrl+S       Save Config\n"
            "Ctrl+E       Export Results\n"
            "Ctrl+Shift+T Run Tests\n"
            "Ctrl+Q       Quit"))

    def _show_about(self):
        messagebox.showinfo("About PyNILM", (
            "PyNILM v3.0\n"
            "Non-Intrusive Load Monitoring System\n\n"
            "Architecture: FVMD-VAE-DCCNN-Att + Hybrid FL\n\n"
            "Components:\n"
            "  Variational Mode Decomposition (FVMD)\n"
            "  Teager Energy Operator (TEO)\n"
            "  Fuzzy Logic Controller (81 rules)\n"
            "  VAE-DCCNN-Attention Deep Learning\n"
            "  Multi-Krum + GAT Federated Learning\n"
            "  Event Detection & Classification\n"
            "  Online Learning & Concept Drift\n"
            "  Transfer Learning\n"
            "  Multi-Household Simulation\n\n"
            f"Backend: {'PyTorch ' + ('(GPU)' if GPU_AVAILABLE else '(CPU)') if HAS_TORCH else 'NumPy'}"))

    def _show_architecture(self):
        messagebox.showinfo("Architecture", (
            "Pipeline:\n"
            "  1. Data Loading (UK-DALE/REFIT HDF5)\n"
            "  2. Preprocessing (Z-score, SG, MinMax)\n"
            "  3. Feature Extraction (FVMD + TEO)\n"
            "  4. Classification (VAE-DCCNN-Att)\n"
            "  5. Fuzzy Logic Refinement\n"
            "  6. Event Detection & Classification\n"
            "  7. Evaluation & Export\n\n"
            "Federated Learning:\n"
            "  Multi-Krum aggregation\n"
            "  GAT Byzantine detection\n"
            "  Differential privacy"))

    def _browse_data(self):
        path = filedialog.askdirectory(title="Browse Data Directory")
        if path:
            self._log(f"Browsing: {path}", 'info')

    # ── Data Operations ───────────────────────────────────────
    def _load_config(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            try:
                self.config = load_config(path)
                self._log(f"Config loaded: {path}", 'success')
                self._set_status('Config', 'Loaded')
            except Exception as e:
                self._log(f"Config error: {e}", 'error')

    def _save_config(self):
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                             filetypes=[("JSON", "*.json")])
        if path:
            save_config(self.config, path)
            self._log(f"Config saved: {path}", 'success')

    def _load_dataset_btn(self):
        self._load_dataset(self.dataset_var.get())

    def _load_dataset(self, name):
        self._log(f"Loading {name}...", 'header')
        self._update_progress(10, f"Loading {name}...")
        self.pipeline_indicator.reset()
        self.pipeline_indicator.set_step(0, 'running')

        def _worker():
            try:
                cfg = dict(self.config['data'])
                cfg['windowSize'] = self.window_var.get()
                cfg['duration'] = self.duration_var.get()
                cfg['overlap'] = self.overlap_var.get()
                loader = load_ukdale if name == 'UK-DALE' else load_refit
                self.data = loader(self.home_id_var.get(), cfg)
                self.root.after(0, lambda: self._on_data_loaded(name))
            except Exception as e:
                self.root.after(0, lambda: self._log(f"Load error: {e}", 'error'))
                self.root.after(0, lambda: self.pipeline_indicator.set_step(0, 'error'))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_data_loaded(self, name):
        n = self.data['trainData'].shape[0]
        self._set_status('Data', f'{name} ({n} windows)')
        self._update_progress(100, "Data loaded")
        self.pipeline_indicator.set_step(0, 'done')
        self.status_data.configure(text=f"{name} | {n} windows | {self.data['trainData'].shape[1]} pts")
        self._log(f"  Train: {n} | Test: {self.data['testData'].shape[0]} | Apps: {len(self.data['applianceNames'])}", 'success')
        self._populate_data_table()
        self._update_dashboard()

    def _generate_synthetic(self):
        self._log("Generating synthetic data...", 'header')
        try:
            from data.data_loader import generate_smart_meter_data
            agg = generate_smart_meter_data(duration=43200, sampling_rate=1/6)
            self.data = {
                'trainData': agg['aggregate'][:2000].reshape(-1, 256)[:20],
                'testData': agg['aggregate'][2000:4000].reshape(-1, 256)[:10],
                'trainLabels': np.random.randint(0, 5, 20),
                'testLabels': np.random.randint(0, 5, 10),
                'aggregate': agg['aggregate'],
                'appliancePower': {}, 'applianceLabels': ['Kettle', 'Fridge', 'Microwave', 'Washer', 'Dishwasher'],
                'applianceNames': ['Kettle', 'Fridge', 'Microwave', 'Washer', 'Dishwasher'],
            }
            self._set_status('Data', 'Synthetic')
            self._log("  Synthetic data generated", 'success')
            self._populate_data_table()
            self._update_dashboard()
        except Exception as e:
            self._log(f"Error: {e}", 'error')

    def _populate_data_table(self):
        if self.data is None:
            return
        rows = []
        for i in range(min(len(self.data['testData']), 50)):
            w = self.data['testData'][i]
            rows.append((i, self.data['testLabels'][i],
                         "%.1f" % np.mean(w), "%.1f" % np.max(w),
                         "%.2f" % np.std(w), "%.1fs" % (len(w) * self.config['data']['samplingRate'])))
        self.data_table.set_data(rows)

    # ── Pipeline ──────────────────────────────────────────────
    def _run_pipeline(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Load data first.")
            return
        self.pipeline_start_time = time.time()
        self.pipeline_indicator.reset()
        self._log("Running Full Pipeline", 'header')
        threading.Thread(target=self._pipeline_worker, daemon=True).start()

    def _pipeline_worker(self):
        try:
            cfg = self.config
            window = self.data['testData'][0]

            self._update_progress(3, "Preprocessing...")
            self.pipeline_indicator.set_step(0, 'done')
            self.pipeline_indicator.set_step(1, 'running')
            preprocessed = preprocess_signal(window, cfg['preprocessing'])
            self.results['preprocessed'] = preprocessed
            self._log(f"  Preprocessing: {preprocessed['nOutliers']} outliers removed")

            self._update_progress(15, "FVMD...")
            self.pipeline_indicator.set_step(1, 'done')
            self.pipeline_indicator.set_step(2, 'running')
            k_start = self.vmd_k_start_var.get()
            k_end = self.vmd_k_end_var.get()
            k_range = list(range(k_start, k_end + 1)) if k_start <= k_end else cfg['featureExtraction']['vmd']['KRange']
            modes, omega, score, best_K = feedback_vmd(preprocessed['normalized'], k_range)
            teo_energy = np.zeros_like(modes)
            for k in range(modes.shape[0]):
                teo_energy[k] = teager_energy_operator(modes[k])
            features, feature_names = extract_features(modes, teo_energy)
            self.results.update({'modes': modes, 'teo_energy': teo_energy, 'features': features})
            self._log(f"  FVMD: K={best_K}, score={score:.4f} | Features: {len(features)} dims")

            self._update_progress(40, "Classification...")
            self.pipeline_indicator.set_step(2, 'done')
            self.pipeline_indicator.set_step(3, 'running')
            if self.model is None:
                self.model = VAEDCCNNAtt(cfg['deepLearning'])
            result = self.model.predict(features.reshape(1, -1))
            self.results['predictions'] = result['predictions']
            self.results['probabilities'] = result['probabilities']
            self._log(f"  Prediction: class={result['predictions'][0]}, conf={result['probabilities'].max():.4f}")

            self._update_progress(55, "Fuzzy Logic...")
            dp = float(np.mean(np.abs(np.diff(preprocessed['normalized']))))
            sigma = float(np.std(preprocessed['normalized']))
            dur = len(preprocessed['normalized']) * cfg['data']['samplingRate']
            freq = 1.0 / (dur + 1e-12)
            fuzzy_state, _ = evaluate_fuzzy_logic(min(1.0, dp/100), min(1.0, sigma/50), min(1.0, dur/3600), min(1.0, freq*3600))
            self.results['fuzzy_state'] = fuzzy_state
            self._log(f"  Fuzzy: {fuzzy_state}")

            self._update_progress(65, "Events...")
            events = detect_events(preprocessed['normalized'], cfg['data']['samplingRate'])
            n_events = len(events['indices'])
            self.results['events'] = events
            self._log(f"  Events: {n_events} detected")

            if n_events > 0:
                classifications = classify_events(events, preprocessed['normalized'])
                self.results['event_classifications'] = classifications
                self.root.after(0, lambda: self._populate_events_table(events, classifications))

            self._update_progress(80, "Evaluation...")
            self.pipeline_indicator.set_step(3, 'done')
            self.pipeline_indicator.set_step(4, 'running')
            gt = self.data['testLabels'][:len(result['predictions'])]
            if len(gt) == len(result['predictions']):
                eval_fw = EvaluationFramework()
                metrics = eval_fw.evaluate(result['predictions'], gt)
                self.results['metrics'] = metrics
                self.root.after(0, lambda: self._show_eval_report(eval_fw))
                self.root.after(0, lambda: self._update_gauges(metrics))
                self._log(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['macroF1']:.4f}", 'success')

            self._update_progress(95, "Updating dashboard...")
            self.pipeline_indicator.set_step(4, 'done')
            self.root.after(0, self._update_dashboard)
            self.root.after(0, self._update_confusion_matrix)

            elapsed = time.time() - self.pipeline_start_time
            self._update_progress(100, f"Complete ({elapsed:.1f}s)")
            self._set_status('Pipeline', f'Done ({elapsed:.1f}s)')
            self._log(f"Pipeline complete in {elapsed:.1f}s", 'success')

        except Exception as e:
            self._log(f"Pipeline error: {e}", 'error')
            import traceback
            self._log(traceback.format_exc(), 'dim')
            self._update_progress(0, f"Error: {e}")

    def _populate_events_table(self, events, classifications):
        rows = []
        for i in range(len(events['indices'])):
            rows.append((i + 1, events['types'][i],
                         "%.2f" % events['timestamps'][i],
                         "%.1f" % events['powerLevels'][i],
                         classifications['appliance'][i] if i < len(classifications['appliance']) else "?",
                         "%.3f" % classifications['confidence'][i] if i < len(classifications['confidence']) else "0"))
        self.events_table.set_data(rows)

    def _show_eval_report(self, eval_fw):
        self.eval_text.delete('1.0', tk.END)
        self.eval_text.insert('1.0', eval_fw.report())

    def _update_gauges(self, metrics):
        self.gauge_accuracy.set_value(metrics.get('accuracy', 0))
        self.gauge_f1.set_value(metrics.get('macroF1', 0))

    # ── Individual Steps ──────────────────────────────────────
    def _run_preprocessing(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Load data first.")
            return
        self._log("Preprocessing...")
        preprocessed = preprocess_signal(self.data['testData'][0], self.config['preprocessing'])
        self.results['preprocessed'] = preprocessed
        self._log(f"  Done: {preprocessed['nOutliers']} outliers", 'success')
        self._plot_signals(self.data['testData'][0], preprocessed)

    def _run_features(self):
        if 'preprocessed' not in self.results:
            messagebox.showwarning("Warning", "Run preprocessing first.")
            return
        self._log("Feature extraction...")
        k_start = self.vmd_k_start_var.get()
        k_end = self.vmd_k_end_var.get()
        k_range = list(range(k_start, k_end + 1)) if k_start <= k_end else self.config['featureExtraction']['vmd']['KRange']
        modes, omega, score, best_K = feedback_vmd(self.results['preprocessed']['normalized'], k_range)
        teo = np.zeros_like(modes)
        for k in range(modes.shape[0]):
            teo[k] = teager_energy_operator(modes[k])
        features, names = extract_features(modes, teo)
        self.results.update({'modes': modes, 'teo_energy': teo, 'features': features})
        self._log(f"  Modes: {modes.shape[0]}, Features: {len(features)}", 'success')
        self._plot_features(modes, teo)

    def _run_classification(self):
        if 'features' not in self.results:
            messagebox.showwarning("Warning", "Run feature extraction first.")
            return
        self._log("Classification...")
        if self.model is None:
            self.model = VAEDCCNNAtt(self.config['deepLearning'])
        result = self.model.predict(self.results['features'].reshape(1, -1))
        self.results['predictions'] = result['predictions']
        self.results['probabilities'] = result['probabilities']
        self._log(f"  Prediction: {result['predictions']}", 'success')

    def _run_fuzzy(self):
        self._log("Fuzzy logic evaluation...")
        states = set()
        for dp in [0.2, 0.5, 0.8]:
            for sg in [0.2, 0.5, 0.8]:
                state, _ = evaluate_fuzzy_logic(dp, sg, 0.5, 0.5)
                states.add(state)
        self._log(f"  States: {sorted(states)}", 'success')

    def _run_event_detection(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Load data first.")
            return
        self._log("Event detection...")
        events = detect_events(self.data['aggregate'][:5000], self.config['data']['samplingRate'])
        n = len(events['indices'])
        self._log(f"  Found {n} events", 'success')
        if n > 0:
            cls = classify_events(events, self.data['aggregate'][:5000])
            self.results['events'] = events
            self.results['event_classifications'] = cls
            self.root.after(0, lambda: self._populate_events_table(events, cls))

    def _run_evaluation(self):
        if 'metrics' not in self.results:
            messagebox.showwarning("Warning", "Run pipeline first.")
            return
        eval_fw = EvaluationFramework()
        eval_fw.metrics = self.results['metrics']
        self._show_eval_report(eval_fw)
        self._show_tab('Evaluation')

    def _run_fl(self):
        self._log("Running Federated Learning", 'header')
        self._update_progress(10, "Starting FL...")
        cfg = {'numClients': self.fl_clients_var.get(), 'numRounds': self.fl_rounds_var.get(),
               'byzantineFraction': self.fl_byz_var.get() / 100.0,
               'localEpochs': self.config['federatedLearning']['localEpochs'],
               'learningRate': self.config['federatedLearning']['learningRate']}

        def _worker():
            try:
                result = run_federated_learning(cfg)
                self.results['fl_results'] = result
                self._update_progress(100, "FL complete")
                self._log(f"  Accuracy: {result['history']['accuracy'][-1]:.4f} | Byzantine: {result['nByzantineDetected']}", 'success')
                self.root.after(0, lambda: self._plot_fl(result))
            except Exception as e:
                self._log(f"FL error: {e}", 'error')

        threading.Thread(target=_worker, daemon=True).start()

    def _train_model(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Load data first.")
            return
        self.pipeline_start_time = time.time()
        self._log("Training VAE-DCCNN-Att", 'header')
        self._update_progress(10, "Training...")
        cfg = {'numClasses': self.num_classes_var.get(), 'epochs': self.epochs_var.get(),
               'batchSize': self.batch_size_var.get(), 'learningRate': self.lr_var.get(),
               'latentDim': self.latent_dim_var.get(), 'vaeBeta': 0.5,
               'encoderChannels': [64, 128, 256], 'decoderChannels': [256, 128, 64],
               'attention': {'heads': 8, 'dim': 64}, 'inputDim': self.data['trainData'].shape[1]}

        def _worker():
            try:
                model, history = train_vae_dccnn_model(self.data['trainData'], self.data['trainLabels'], cfg)
                self.model = model
                self.results['training_history'] = history
                elapsed = time.time() - self.pipeline_start_time
                self._update_progress(100, f"Trained ({elapsed:.1f}s)")
                self._set_status('Model', f'Trained ({cfg["epochs"]}ep)')
                self.root.after(0, lambda: self.status_model.configure(text=f"Trained | {elapsed:.1f}s"))
                self._log(f"  Loss: {history['loss'][-1]:.4f} | Acc: {history['accuracy'][-1]:.4f} ({elapsed:.1f}s)", 'success')
                self.root.after(0, lambda: self._plot_training(history))
            except Exception as e:
                self._log(f"Training error: {e}", 'error')
                import traceback
                self._log(traceback.format_exc(), 'dim')

        threading.Thread(target=_worker, daemon=True).start()

    def _load_model(self):
        path = filedialog.askopenfilename(filetypes=[("Model", "*.pt *.json"), ("All", "*.*")])
        if path:
            try:
                if path.endswith('.pt'):
                    from deep_learning.model import _try_import_torch, _build_pytorch_classes
                    tm = _try_import_torch()
                    if tm:
                        _, Wrapper = _build_pytorch_classes(tm[0], tm[1])
                        self.model = Wrapper.load(path)
                    else:
                        self.model = VAEDCCNNAtt.load(path)
                else:
                    self.model = VAEDCCNNAtt.load(path)
                self._set_status('Model', 'Loaded')
                self._log(f"Model loaded: {path}", 'success')
            except Exception as e:
                self._log(f"Load error: {e}", 'error')

    def _save_model(self):
        if self.model is None:
            messagebox.showwarning("Warning", "No model to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pt",
                                             filetypes=[("PyTorch", "*.pt"), ("JSON", "*.json")])
        if path:
            self.model.save(path)
            self._log(f"Model saved: {path}", 'success')

    def _run_tests(self):
        self._log("Running Test Suite", 'header')
        self._update_progress(50, "Testing...")

        def _worker():
            try:
                runner = run_all_tests()
                self._update_progress(100, f"Tests: {runner.passed}/{runner.passed + runner.failed}")
                self._log(f"  {runner.passed}/{runner.passed + runner.failed} passed", 'success' if runner.failed == 0 else 'warning')
            except Exception as e:
                self._log(f"Test error: {e}", 'error')

        threading.Thread(target=_worker, daemon=True).start()

    def _run_signal_quality(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Load data first.")
            return
        assessor = SignalQualityAssessor()
        result = assessor.assess(self.data['aggregate'][:5000])
        self.gauge_quality.set_value(result['qualityScore'])
        self._log(f"  Quality: {result['qualityScore']:.3f} | SNR: {result['snr']:.1f}dB", 'success')

    def _run_multi_household(self):
        self._log("Multi-Household Simulation", 'header')

        def _worker():
            try:
                sim = MultiHouseholdSimulator(num_households=10, duration=8640)
                sim.generate_data()
                results = sim.analyze()
                self._log(f"  Grid: {results['grid_level']['total_mean']:.0f}W | Peak: {results['demand_response']['peak_hour']}:00", 'success')
            except Exception as e:
                self._log(f"Error: {e}", 'error')

        threading.Thread(target=_worker, daemon=True).start()

    def _run_online_learning(self):
        self._log("Online Learning Demo", 'header')
        learner = OnlineLearner(learning_rate=0.001, buffer_size=500)
        learner.initialize_model(256, 5)
        for _ in range(20):
            X = np.random.randn(10, 256).astype(np.float64)
            y = np.random.randint(0, 5, 10)
            learner.update(X, y)
        perf = learner.get_performance()
        if perf:
            self._log(f"  Accuracy: {perf[-1]['accuracy']:.4f} | Drifts: {learner.drifts_detected}", 'success')

    def _run_transfer(self):
        self._log("Transfer Learning Demo", 'accent')
        try:
            tm = TransferLearningManager()
            self._log("  Transfer learning manager initialized", 'success')
        except Exception as e:
            self._log(f"  Error: {e}", 'error')

    def _export_results(self):
        if not self.results:
            messagebox.showwarning("Warning", "No results to export.")
            return
        exporter = NILMExporter()
        exporter.export_all(self.results, self.config)
        self._log(f"Exported to: {exporter.output_path}", 'success')

    def _export_log(self):
        path = filedialog.asksaveasfilename(defaultextension=".log",
                                             filetypes=[("Log", "*.log")])
        if path:
            with open(path, 'w') as f:
                for line in self.event_log:
                    f.write(line + "\n")
            self._log(f"Log exported: {path}", 'success')

    # ── Visualization ─────────────────────────────────────────
    def _update_dashboard(self):
        if not HAS_MATPLOTLIB or self.data is None:
            return
        try:
            for ax in self.dash_axes.flat:
                ax.clear()
                ax.set_facecolor(C['bg2'])
                ax.tick_params(colors=C['fg3'], labelsize=6)
                for spine in ax.spines.values():
                    spine.set_color(C['border'])

            agg = self.data['aggregate'][:2000]
            self.dash_axes[0, 0].plot(agg, color=C['accent'], linewidth=0.4)
            self.dash_axes[0, 0].set_title('Aggregate Power', color=C['fg'], fontsize=8)

            self.dash_axes[0, 1].hist(agg, bins=60, color=C['accent3'], alpha=0.8, edgecolor='none')
            self.dash_axes[0, 1].set_title('Power Distribution', color=C['fg'], fontsize=8)

            if 'modes' in self.results:
                modes = self.results['modes']
                for k in range(min(4, modes.shape[0])):
                    self.dash_axes[0, 2].plot(modes[k, :300] + k * 3,
                                               color=[C['accent'], C['pink'], C['cyan'], C['green']][k],
                                               linewidth=0.4)
                self.dash_axes[0, 2].set_title('VMD Modes', color=C['fg'], fontsize=8)

            if 'probabilities' in self.results:
                probs = self.results['probabilities'][0]
                self.dash_axes[0, 3].bar(range(len(probs)), probs, color=C['accent2'], alpha=0.8)
                self.dash_axes[0, 3].set_title('Prediction Prob.', color=C['fg'], fontsize=8)

            self.dash_axes[1, 0].plot(agg[:500], color=C['green'], linewidth=0.5)
            self.dash_axes[1, 0].set_title('Signal Detail', color=C['fg'], fontsize=8)

            if 'preprocessed' in self.results:
                self.dash_axes[1, 1].plot(self.results['preprocessed']['normalized'][:500],
                                           color=C['cyan'], linewidth=0.4)
                self.dash_axes[1, 1].set_title('Preprocessed', color=C['fg'], fontsize=8)

            if 'metrics' in self.results and 'perClass' in self.results['metrics']:
                cls = list(self.results['metrics']['perClass'].keys())[:6]
                f1s = [self.results['metrics']['perClass'][c]['f1'] for c in cls]
                self.dash_axes[1, 2].barh(range(len(cls)), f1s, color=C['green'], height=0.5)
                self.dash_axes[1, 2].set_yticks(range(len(cls)))
                self.dash_axes[1, 2].set_yticklabels(cls, fontsize=6, color=C['fg3'])
                self.dash_axes[1, 2].set_title('F1 by Class', color=C['fg'], fontsize=8)

            if 'events' in self.results:
                ev = self.results['events']
                n = len(ev['indices'])
                self.dash_axes[1, 3].text(0.5, 0.5, f"Events\n{n}\n\n" +
                                           "\n".join(f"{t:.1f}s" for t in ev['timestamps'][:6]),
                                           ha='center', va='center', fontsize=9, color=C['fg'],
                                           transform=self.dash_axes[1, 3].transAxes, family='Consolas')
                self.dash_axes[1, 3].set_title('Events', color=C['fg'], fontsize=8)
                self.dash_axes[1, 3].axis('off')

            self.dash_fig.tight_layout(pad=2.0)
            self.dash_canvas.draw()

            if self.metric_cards:
                self.metric_cards['Windows'].set_value(str(self.data['trainData'].shape[0]))
                self.metric_cards['Appliances'].set_value(str(len(self.data['applianceNames'])))
                if 'metrics' in self.results:
                    self.metric_cards['Accuracy'].set_value("%.1f%%" % (self.results['metrics']['accuracy'] * 100))
                    self.metric_cards['F1 Score'].set_value("%.3f" % self.results['metrics']['macroF1'])
                if 'features' in self.results:
                    self.metric_cards['Features'].set_value(str(len(self.results['features'])))
                if 'events' in self.results:
                    self.metric_cards['Events'].set_value(str(len(self.results['events']['indices'])))
        except Exception as e:
            self._log(f"Dashboard error: {e}", 'error')

    def _plot_signals(self, original, preprocessed):
        if not HAS_MATPLOTLIB:
            return
        p = self.viz_tabs['Signals']
        for w in p.winfo_children():
            w.destroy()
        fig = Figure(figsize=(12, 6), dpi=90, facecolor=C['bg'])
        axes = fig.subplots(2, 2)
        data_pairs = [('Original', original, C['accent']), ('Cleaned', preprocessed['cleaned'], C['green']),
                      ('Smoothed', preprocessed['smoothed'], C['yellow']), ('Normalized', preprocessed['normalized'], C['cyan'])]
        for ax, (title, d, color) in zip(axes.flat, data_pairs):
            ax.set_facecolor(C['bg2'])
            ax.plot(d[:500], color=color, linewidth=0.5)
            ax.set_title(title, color=C['fg'], fontsize=9)
            ax.tick_params(colors=C['fg3'], labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(C['border'])
        fig.tight_layout(pad=2.0)
        canvas = FigureCanvasTkAgg(fig, master=p)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        canvas.draw()
        self._show_tab('Signals')

    def _plot_features(self, modes, teo):
        if not HAS_MATPLOTLIB:
            return
        p = self.viz_tabs['Features']
        for w in p.winfo_children():
            w.destroy()
        fig = Figure(figsize=(12, 6), dpi=90, facecolor=C['bg'])
        axes = fig.subplots(2, 2)
        colors = [C['accent'], C['pink'], C['cyan'], C['green']]
        for k in range(min(4, modes.shape[0])):
            axes[0, 0].plot(modes[k, :500] + k * 3, color=colors[k], linewidth=0.4, label=f'Mode {k+1}')
        axes[0, 0].set_title('VMD Modes', color=C['fg'], fontsize=9)
        axes[0, 0].legend(fontsize=6, facecolor=C['bg2'], edgecolor=C['border'], labelcolor=C['fg'])
        for k in range(min(4, teo.shape[0])):
            axes[0, 1].plot(teo[k, :500] + k * 0.5, color=colors[k], linewidth=0.4)
        axes[0, 1].set_title('TEO Energy', color=C['fg'], fontsize=9)
        spectrum = np.abs(np.fft.rfft(modes[0, :]))
        axes[1, 0].plot(spectrum[:120], color=C['accent'])
        axes[1, 0].set_title('Spectrum (Mode 1)', color=C['fg'], fontsize=9)
        axes[1, 1].imshow(modes[:, :300], aspect='auto', cmap='magma')
        axes[1, 1].set_title('Spectrogram', color=C['fg'], fontsize=9)
        for ax in axes.flat:
            ax.set_facecolor(C['bg2'])
            ax.tick_params(colors=C['fg3'], labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(C['border'])
        fig.tight_layout(pad=2.0)
        canvas = FigureCanvasTkAgg(fig, master=p)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        canvas.draw()
        self._show_tab('Features')

    def _plot_fl(self, fl_results):
        if not HAS_MATPLOTLIB:
            return
        p = self.viz_tabs['FL Results']
        for w in p.winfo_children():
            w.destroy()
        h = fl_results['history']
        fig = Figure(figsize=(12, 4), dpi=90, facecolor=C['bg'])
        axes = fig.subplots(1, 3)
        for ax, (title, key, color) in zip(axes, [('Loss', 'loss', C['accent']),
                                                    ('Accuracy', 'accuracy', C['green']),
                                                    ('F1 Score', 'f1', C['cyan'])]):
            ax.set_facecolor(C['bg2'])
            ax.plot(h['round'], h[key], color=color, linewidth=1.5)
            ax.set_title(title, color=C['fg'], fontsize=9)
            ax.set_xlabel('Round', color=C['fg3'], fontsize=8)
            ax.tick_params(colors=C['fg3'], labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(C['border'])
        fig.suptitle(f"FL: {fl_results['nTrusted']} trusted, {fl_results['nByzantineDetected']} Byzantine",
                     color=C['fg'], fontsize=11)
        fig.tight_layout(pad=2.5)
        canvas = FigureCanvasTkAgg(fig, master=p)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        canvas.draw()
        self._show_tab('FL Results')

    def _plot_training(self, history):
        if not HAS_MATPLOTLIB:
            return
        p = self.viz_tabs['Training']
        for w in p.winfo_children():
            w.destroy()
        fig = Figure(figsize=(12, 5), dpi=90, facecolor=C['bg'])
        axes = fig.subplots(1, 3)
        axes[0].set_facecolor(C['bg2'])
        axes[0].plot(history['loss'], color=C['accent'], linewidth=1.5)
        axes[0].set_title('Loss', color=C['fg'], fontsize=10)
        axes[0].set_xlabel('Epoch', color=C['fg3'])
        axes[1].set_facecolor(C['bg2'])
        axes[1].plot(history['accuracy'], color=C['green'], linewidth=1.5)
        axes[1].set_title('Accuracy', color=C['fg'], fontsize=10)
        axes[1].set_xlabel('Epoch', color=C['fg3'])
        axes[2].set_facecolor(C['bg2'])
        if len(history['loss']) > 1:
            lr_curve = [history['loss'][i] / max(history['loss'][0], 1e-8) for i in range(len(history['loss']))]
            axes[2].plot(lr_curve, color=C['cyan'], linewidth=1.5)
        axes[2].set_title('Loss Curve', color=C['fg'], fontsize=10)
        axes[2].set_xlabel('Epoch', color=C['fg3'])
        for ax in axes:
            ax.tick_params(colors=C['fg3'], labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(C['border'])
        fig.tight_layout(pad=2.0)
        canvas = FigureCanvasTkAgg(fig, master=p)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        canvas.draw()
        self._show_tab('Training')

    def _update_confusion_matrix(self):
        if not HAS_MATPLOTLIB or 'metrics' not in self.results:
            return
        try:
            eval_fw = EvaluationFramework()
            if eval_fw.confusion_matrix is not None:
                self.cm_ax.clear()
                self.cm_ax.set_facecolor(C['bg2'])
                cm = eval_fw.confusion_matrix
                im = self.cm_ax.imshow(cm, cmap='magma', aspect='auto')
                n = cm.shape[0]
                self.cm_ax.set_xticks(range(n))
                self.cm_ax.set_yticks(range(n))
                self.cm_ax.set_xticklabels(range(n), color=C['fg3'], fontsize=8)
                self.cm_ax.set_yticklabels(range(n), color=C['fg3'], fontsize=8)
                self.cm_ax.set_xlabel('Predicted', color=C['fg3'])
                self.cm_ax.set_ylabel('True', color=C['fg3'])
                self.cm_ax.set_title('Confusion Matrix', color=C['fg'], fontsize=10)
                for spine in self.cm_ax.spines.values():
                    spine.set_color(C['border'])
                for i in range(n):
                    for j in range(n):
                        color = C['white'] if cm[i, j] > cm.max() / 2 else C['fg']
                        self.cm_ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=9)
                self.cm_fig.colorbar(im, ax=self.cm_ax, fraction=0.046, pad=0.04)
                self.cm_fig.tight_layout(pad=2.0)
                self.cm_canvas.draw()
        except Exception as e:
            self._log(f"CM error: {e}", 'error')


def main():
    if not HAS_TK:
        print("tkinter not available.")
        return
    root = tk.Tk()
    app = NILMApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
