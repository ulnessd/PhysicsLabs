import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class VideoAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Projectile Motion Tracker")

        # Frame for all controls and output
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Load video button
        self.load_button = tk.Button(control_frame, text="Load Video", command=self.load_video)
        self.load_button.pack(pady=5)

        # Frame control
        self.frame_slider = tk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=400, resolution=1,
                                     command=self.update_frame)
        self.frame_slider.pack(pady=5)

        # Collect data button
        self.collect_button = tk.Button(control_frame, text="Collect Data", command=self.collect_data)
        self.collect_button.pack(pady=5)

        # Calibration button
        self.calibrate_button = tk.Button(control_frame, text="Calibrate", command=self.calibrate)
        self.calibrate_button.pack(pady=5)

        # Undo button
        self.undo_button = tk.Button(control_frame, text="Undo", command=self.undo_last_point)
        self.undo_button.pack(pady=5)

        # Clear all button
        self.clear_button = tk.Button(control_frame, text="Clear All", command=self.clear_all)
        self.clear_button.pack(pady=5)

        # Plot data button
        self.plot_button = tk.Button(control_frame, text="Plot Data", command=self.plot_data)
        self.plot_button.pack(pady=5)

        # Data console (could use a Text widget)
        self.data_console = tk.Text(control_frame, height=10, width=40)
        self.data_console.pack(pady=5)

        # Canvas to display video frame, placed on the side
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack(side=tk.RIGHT, padx=10, pady=10)

        # Variables for calibration and data collection
        self.calibration_points = []
        self.scale_factor = None
        self.data_points = []

        # Bind mouse click events to the canvas
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Placeholder for video data
        self.video = None
        self.current_frame = None

    def load_video(self):
        # File dialog to load video
        video_path = filedialog.askopenfilename()
        if video_path:  # Check if a file was selected
            self.video = cv2.VideoCapture(video_path)

            # Set the frame slider range based on video length
            frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_slider.config(to=frame_count - 1)
            self.update_frame(0)

    def update_frame(self, frame_no):
        if self.video is not None:
            # Load a specific frame from the video
            self.video.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
            ret, self.current_frame = self.video.read()
            if ret:
                self.display_frame()

    def display_frame(self):
        if self.current_frame is not None:
            # Convert frame to RGB format
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame_rgb)

            # Get the current size of the canvas
            self.canvas.update_idletasks()  # Ensure canvas size is updated
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # Calculate aspect ratios
            frame_width, frame_height = frame_image.size
            frame_aspect = frame_width / frame_height
            canvas_aspect = canvas_width / canvas_height

            # Determine new size while maintaining aspect ratio
            if frame_aspect > canvas_aspect:
                # Fit to width
                new_width = canvas_width
                new_height = int(canvas_width / frame_aspect)
            else:
                # Fit to height
                new_height = canvas_height
                new_width = int(canvas_height * frame_aspect)

            # Resize the image with maintained aspect ratio
            frame_image = frame_image.resize((new_width, new_height), Image.LANCZOS)

            # Convert to a Tkinter-compatible image
            self.photo = ImageTk.PhotoImage(image=frame_image)

            # Clear any previous images on the canvas
            self.canvas.delete("all")

            # Calculate positions to center the image on the canvas
            x_pos = (canvas_width - new_width) // 2
            y_pos = (canvas_height - new_height) // 2

            # Display the image on the canvas
            self.canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.photo)

            # Redraw the points
            self.redraw_points()

    def on_canvas_click(self, event):
        # Record the click position
        x, y = event.x, event.y

        if self.scale_factor is None:
            # Calibration mode: collect calibration points
            if len(self.calibration_points) < 2:
                self.calibration_points.append((x, y))
                self.data_console.insert(tk.END, f"Calibration point: {len(self.calibration_points)} at ({x}, {y})\n")

                # If two points are collected, calculate the scale factor
                if len(self.calibration_points) == 2:
                    self.calculate_scale_factor()
        else:
            # Data collection mode: collect data points
            frame_no = self.frame_slider.get()
            time = frame_no / self.video.get(cv2.CAP_PROP_FPS)
            self.data_points.append((x, y, time))
            self.data_console.insert(tk.END, f"Data point: ({x}, {y}) at time {time:.2f} seconds\n")
            self.draw_point(x, y)

    def calculate_scale_factor(self):
        # Calculate distance between the two calibration points
        (x1, y1), (x2, y2) = self.calibration_points
        pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Assume the real-world distance between the two points is 1 meter
        # Replace this with the actual length of the meter stick if different
        real_distance = 1.0  # meters
        self.scale_factor = real_distance / pixel_distance

        self.data_console.insert(tk.END, f"Calibration complete. Scale factor: {self.scale_factor:.4f} meters/pixel\n")

    def calibrate(self):
        # Reset calibration points to allow re-calibration
        self.calibration_points = []
        self.scale_factor = None
        self.data_console.insert(tk.END, "Calibration mode. Click on the two ends of the meter stick.\n")

        # Only remove points, not the video frame
        self.redraw_frame()

    def collect_data(self):
        # Switch to data collection mode
        if self.scale_factor is None:
            self.data_console.insert(tk.END, "Please calibrate the system first.\n")
        else:
            self.data_console.insert(tk.END, "Data collection mode. Click on the tennis ball to record positions.\n")

    def draw_point(self, x, y):
        # Draw an "X" at the specified location on the canvas
        size = 5  # Size of the "X"
        self.canvas.create_line(x - size, y - size, x + size, y + size, fill="red", width=2)
        self.canvas.create_line(x - size, y + size, x + size, y - size, fill="red", width=2)

    def redraw_points(self):
        # Redraw all points on the canvas
        for x, y, _ in self.data_points:
            self.draw_point(x, y)

    def undo_last_point(self):
        # Remove the last point
        if self.data_points:
            self.data_points.pop()
            self.data_console.insert(tk.END, "Last point removed.\n")
            self.redraw_frame()  # Refresh the frame to update the canvas

    def clear_all(self):
        # Clear all points and canvas, but not the video frame
        self.data_points = []
        self.data_console.insert(tk.END, "All points cleared.\n")
        self.redraw_frame()

    def redraw_frame(self):
        # Redraw the video frame and any points
        self.display_frame()

    def plot_data(self):
        if not self.data_points:
            self.data_console.insert(tk.END, "No data to plot. Please collect data first.\n")
            return

        if self.scale_factor is None:
            self.data_console.insert(tk.END, "Please calibrate the system before plotting.\n")
            return

        # Extract the data and apply the scale factor
        times = np.array([point[2] for point in self.data_points])
        x_positions = np.array([point[0] * self.scale_factor for point in self.data_points])
        y_positions = np.array([point[1] * self.scale_factor for point in self.data_points])

        # Invert y positions to account for image coordinate system
        y_positions = -y_positions

        # Create a new Matplotlib figure
        fig = Figure(figsize=(8, 6))

        # X vs Time plot with linear fit
        ax1 = fig.add_subplot(211)
        ax1.plot(times, x_positions, 'o', color='blue', label='Data')
        x_fit = np.polyfit(times, x_positions, 1)  # Linear fit
        ax1.plot(times, np.polyval(x_fit, times), '-', color='orange', label=f'Fit: y={x_fit[0]:.2f}x + {x_fit[1]:.2f}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('X Position (m)')
        ax1.set_title('X Position vs Time')
        ax1.legend()

        # Y vs Time plot with quadratic fit
        ax2 = fig.add_subplot(212)
        ax2.plot(times, y_positions, 'o', color='red', label='Data')
        y_fit = np.polyfit(times, y_positions, 2)  # Quadratic fit
        ax2.plot(times, np.polyval(y_fit, times), '-', color='green',
                 label=f'Fit: y={y_fit[0]:.2f}x^2 + {y_fit[1]:.2f}x + {y_fit[2]:.2f}')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title('Y Position vs Time')
        ax2.legend()

        # Adjust layout to prevent overlap
        fig.subplots_adjust(hspace=0.5)  # Increase hspace to add more space between plots

        # Display the plots in a new window
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Data Plots")

        # Embed the figure in a Tkinter canvas
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack()


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnalyzerApp(root)
    root.mainloop()
