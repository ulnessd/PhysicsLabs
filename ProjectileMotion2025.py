# projectile_motion_gui_v2.py

import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from PyQt6.QtCore import Qt
import pyqtgraph as pg


# --- Custom Video Display Widget ---
class VideoCanvas(QWidget):
    """A custom widget to display video frames. The drawing is now handled by the main window."""

    def __init__(self):
        super().__init__()
        self.pixmap = None
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #111; border: 1px solid #444;")
        self.click_callback = None

    def set_frame(self, frame_image):
        """Sets the current video frame (as a QImage) to be displayed."""
        self.pixmap = QPixmap.fromImage(frame_image)
        self.update()

    def paintEvent(self, event):
        """Draws the video frame."""
        if not self.pixmap:
            return
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.pixmap)

    def mousePressEvent(self, event):
        """Handles mouse clicks on the canvas."""
        if self.click_callback:
            self.click_callback(event.position())


# --- Main Application Window ---
class ProjectileMotionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Projectile Motion Video Analyzer")
        self.setGeometry(100, 100, 1400, 800)

        # State variables
        self.video_capture = None
        self.total_frames = 0
        self.fps = 30
        self.app_state = "idle"  # States: idle, calibrating, collecting
        self.calibration_points = []
        self.meters_per_pixel = None
        self.data_points = {}  # {frame: (t, x_m, y_m)}
        self.pixel_points = {}  # {frame: (x_p, y_p)}

        self.initUI()
        self.apply_stylesheet()

    def initUI(self):
        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Control Panel ---
        control_panel = QVBoxLayout()
        control_panel.setSpacing(15)

        title = QLabel("Projectile Motion Lab")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_load = QPushButton("1. Load Video")
        self.btn_calibrate = QPushButton("2. Calibrate Scale")
        self.btn_collect = QPushButton("3. Collect Data")
        self.btn_analyze = QPushButton("4. Analyze & Plot")

        self.btn_load.clicked.connect(self.load_video)
        self.btn_calibrate.clicked.connect(self.start_calibration)
        self.btn_collect.clicked.connect(self.start_data_collection)
        self.btn_analyze.clicked.connect(self.analyze_data)

        self.instruction_label = QLabel("Click 'Load Video' to begin.")
        self.instruction_label.setWordWrap(True)

        self.data_table = QTableWidget()
        self.data_table.setColumnCount(3)
        self.data_table.setHorizontalHeaderLabels(["Time (s)", "X (m)", "Y (m)"])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        util_layout = QHBoxLayout()
        self.btn_undo = QPushButton("Undo Last Point")
        self.btn_clear = QPushButton("Clear All Data")
        self.btn_undo.clicked.connect(self.undo_last_point)
        self.btn_clear.clicked.connect(self.clear_all_data)
        util_layout.addWidget(self.btn_undo)
        util_layout.addWidget(self.btn_clear)

        self.results_label = QLabel("Results will be shown here.")
        self.results_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.results_label.setWordWrap(True)
        self.results_label.setFont(QFont("Courier New", 12))

        control_panel.addWidget(title)
        control_panel.addWidget(self.btn_load)
        control_panel.addWidget(self.btn_calibrate)
        control_panel.addWidget(self.btn_collect)
        control_panel.addWidget(self.btn_analyze)
        control_panel.addWidget(self.instruction_label)
        control_panel.addWidget(self.data_table, 1)
        control_panel.addLayout(util_layout)
        control_panel.addWidget(self.results_label, 1)

        # --- Right Display Panel ---
        display_panel = QVBoxLayout()

        self.video_canvas = VideoCanvas()
        self.video_canvas.click_callback = self.handle_canvas_click

        self.video_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_slider.valueChanged.connect(self.slider_moved)

        plots_layout = QHBoxLayout()
        pg.setConfigOption('background', '#19232D')
        pg.setConfigOption('foreground', 'd')
        self.plot_x = pg.PlotWidget(title="X-Position vs. Time")
        self.plot_y = pg.PlotWidget(title="Y-Position vs. Time")
        self.plot_x.setLabel('left', 'Position (m)')
        self.plot_x.setLabel('bottom', 'Time (s)')
        self.plot_y.setLabel('left', 'Position (m)')
        self.plot_y.setLabel('bottom', 'Time (s)')
        plots_layout.addWidget(self.plot_x)
        plots_layout.addWidget(self.plot_y)

        display_panel.addWidget(self.video_canvas, 2)
        display_panel.addWidget(self.video_slider)
        display_panel.addLayout(plots_layout, 1)

        main_layout.addLayout(control_panel, 1)
        main_layout.addLayout(display_panel, 2)

        self.update_button_states()

    def apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #2c3e50; color: #ecf0f1; font-family: Arial; }
            QPushButton { background-color: #3498db; color: white; border: none; padding: 10px; border-radius: 5px; font-size: 14px; }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:disabled { background-color: #555; color: #999; }
            QLabel { font-size: 14px; }
            QTableWidget { background-color: #34495e; border: 1px solid #444; gridline-color: #444; }
            QHeaderView::section { background-color: #3498db; color: white; padding: 5px; border: none; }
            QSlider::groove:horizontal { border: 1px solid #bbb; background: white; height: 10px; border-radius: 4px; }
            QSlider::handle:horizontal { background: #3498db; border: 1px solid #3498db; width: 18px; margin: -2px 0; border-radius: 9px; }
        """)

    def keyPressEvent(self, event):
        if not self.video_slider.isEnabled(): return
        current_frame = self.video_slider.value()
        if event.key() == Qt.Key.Key_Right and current_frame < self.total_frames - 1:
            self.video_slider.setValue(current_frame + 1)
        elif event.key() == Qt.Key.Key_Left and current_frame > 0:
            self.video_slider.setValue(current_frame - 1)

    def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if file_name:
            self.video_capture = cv2.VideoCapture(file_name)
            self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.video_slider.setMaximum(self.total_frames - 1)
            self.slider_moved(0)
            self.app_state = "loaded"
            self.instruction_label.setText("Video loaded. Proceed to Step 2: Calibrate Scale.")
            self.update_button_states()

    def start_calibration(self):
        self.app_state = "calibrating"
        self.calibration_points = []
        self.instruction_label.setText("CALIBRATING: Click on one end of the 1-meter stick.")
        self.update_button_states()

    def start_data_collection(self):
        self.app_state = "collecting"
        self.instruction_label.setText("COLLECTING: Use slider/keys to find a frame, then click on the ball.")
        self.update_button_states()

    def transform_click_to_video_coords(self, click_pos):
        """Converts a QPointF from the widget's coordinate system to the video's pixel coordinates."""
        if not self.video_capture:
            return None

        video_width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        canvas_width = self.video_canvas.width()
        canvas_height = self.video_canvas.height()

        video_aspect = video_width / video_height
        canvas_aspect = canvas_width / canvas_height

        if canvas_aspect > video_aspect:
            scale = canvas_height / video_height
            scaled_width = video_width * scale
            x_offset = (canvas_width - scaled_width) / 2
            y_offset = 0
        else:
            scale = canvas_width / video_width
            scaled_height = video_height * scale
            x_offset = 0
            y_offset = (canvas_height - scaled_height) / 2

        if not (
                x_offset <= click_pos.x() < canvas_width - x_offset and y_offset <= click_pos.y() < canvas_height - y_offset):
            return None

        video_x = (click_pos.x() - x_offset) / scale
        video_y = (click_pos.y() - y_offset) / scale

        return (int(video_x), int(video_y))

    def handle_canvas_click(self, pos):
        video_coords = self.transform_click_to_video_coords(pos)
        if video_coords is None: return

        if self.app_state == "calibrating":
            self.calibration_points.append(video_coords)
            if len(self.calibration_points) == 1:
                self.instruction_label.setText("CALIBRATING: Click on the other end of the 1-meter stick.")
            elif len(self.calibration_points) == 2:
                p1, p2 = self.calibration_points
                pixel_dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                self.meters_per_pixel = 1.0 / pixel_dist
                self.app_state = "calibrated"
                self.instruction_label.setText(
                    f"Scale calibrated: {self.meters_per_pixel:.5f} m/pixel. Proceed to Step 3.")
                self.update_button_states()

        elif self.app_state == "collecting":
            current_frame = self.video_slider.value()
            time = current_frame / self.fps

            video_height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            x_m = video_coords[0] * self.meters_per_pixel
            y_m = (video_height - video_coords[1]) * self.meters_per_pixel

            self.data_points[current_frame] = (time, x_m, y_m)
            self.pixel_points[current_frame] = video_coords

            self.slider_moved(current_frame)
            self.update_data_table()
            self.update_plots(live=True)
            self.update_button_states()

    def analyze_data(self):
        if len(self.data_points) < 3:
            self.results_label.setText("Not enough data points to analyze.")
            return

        points = list(self.data_points.values())
        times = np.array([p[0] for p in points])
        x_pos = np.array([p[1] for p in points])
        y_pos = np.array([p[2] for p in points])

        x_coeffs = np.polyfit(times, x_pos, 1)
        v_x = x_coeffs[0]
        x_fit_func = np.poly1d(x_coeffs)

        y_coeffs = np.polyfit(times, y_pos, 2)
        g = -2 * y_coeffs[0]
        y_fit_func = np.poly1d(y_coeffs)

        self.update_plots(live=False, x_fit=x_fit_func, y_fit=y_fit_func)

        results_text = (
            f"<pre>"
            f"--- Analysis Results ---<br><br>"
            f"<b>Horizontal Motion (Linear Fit)</b><br>"
            f"x(t) = {x_coeffs[0]:.3f} t + {x_coeffs[1]:.3f}<br>"
            f"<b>Velocity (v_x): {v_x:.3f} m/s</b><br><br>"
            f"<b>Vertical Motion (Quadratic Fit)</b><br>"
            f"y(t) = {y_coeffs[0]:.3f} t² + {y_coeffs[1]:.3f} t + {y_coeffs[2]:.3f}<br>"
            f"<b>Gravity (g):    {g:.3f} m/s²</b>"
            f"</pre>"
        )
        self.results_label.setText(results_text)
        self.app_state = "done"
        self.update_button_states()

    def slider_moved(self, value):
        if self.video_capture:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, value)
            ret, frame = self.video_capture.read()
            if ret:
                # --- CHANGE: Draw ALL collected markers on the current frame ---
                for frame_num, pos in self.pixel_points.items():
                    cv2.line(frame, (pos[0] - 12, pos[1] - 12), (pos[0] + 12, pos[1] + 12), (0, 0, 255), 3)
                    cv2.line(frame, (pos[0] - 12, pos[1] + 12), (pos[0] + 12, pos[1] - 12), (0, 0, 255), 3)

                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
                self.video_canvas.set_frame(q_img)

    def update_data_table(self):
        self.data_table.setRowCount(len(self.data_points))
        sorted_frames = sorted(self.data_points.keys())
        for i, frame_num in enumerate(sorted_frames):
            t, x, y = self.data_points[frame_num]
            self.data_table.setItem(i, 0, QTableWidgetItem(f"{t:.3f}"))
            self.data_table.setItem(i, 1, QTableWidgetItem(f"{x:.3f}"))
            self.data_table.setItem(i, 2, QTableWidgetItem(f"{y:.3f}"))

    def update_plots(self, live=False, x_fit=None, y_fit=None):
        if not self.data_points:
            self.plot_x.clear()
            self.plot_y.clear()
            return

        points = list(self.data_points.values())
        times = np.array([p[0] for p in points])
        x_pos = np.array([p[1] for p in points])
        y_pos = np.array([p[2] for p in points])

        self.plot_x.clear()
        self.plot_y.clear()

        self.plot_x.plot(times, x_pos, pen=None, symbol='o', symbolBrush=(255, 0, 0), symbolSize=8)
        self.plot_y.plot(times, y_pos, pen=None, symbol='o', symbolBrush=(255, 0, 0), symbolSize=8)

        if not live and x_fit is not None and y_fit is not None:
            fit_times = np.linspace(min(times), max(times), 100)
            fit_pen = pg.mkPen('y', width=3)
            self.plot_x.plot(fit_times, x_fit(fit_times), pen=fit_pen)
            self.plot_y.plot(fit_times, y_fit(fit_times), pen=fit_pen)

    def undo_last_point(self):
        if self.data_points:
            last_frame = max(self.data_points.keys())
            self.data_points.pop(last_frame)
            self.pixel_points.pop(last_frame)

            self.slider_moved(self.video_slider.value())
            self.update_data_table()
            self.update_plots(live=True)
            self.update_button_states()

    def clear_all_data(self):
        self.data_points = {}
        self.pixel_points = {}
        self.slider_moved(self.video_slider.value())
        self.update_data_table()
        self.update_plots(live=True)
        self.results_label.setText("Results will be shown here.")
        self.update_button_states()

    def update_button_states(self):
        self.btn_calibrate.setEnabled(False)
        self.btn_collect.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.video_slider.setEnabled(False)
        self.btn_undo.setEnabled(False)
        self.btn_clear.setEnabled(False)

        if self.app_state in ["loaded", "calibrated", "collecting", "done"]:
            self.btn_calibrate.setEnabled(True)
        if self.app_state in ["calibrated", "collecting", "done"]:
            self.btn_collect.setEnabled(True)
        if self.app_state in ["collecting", "done"]:
            self.video_slider.setEnabled(True)
        if self.data_points:
            self.btn_undo.setEnabled(True)
            self.btn_clear.setEnabled(True)
        if len(self.data_points) > 2:
            self.btn_analyze.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = ProjectileMotionGUI()
    main_win.show()
    sys.exit(app.exec())
