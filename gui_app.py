import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import os
from model import DigitRecognitionModel

class DigitRecognitionGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Recognition - Neural Network Tester")
        self.root.geometry("800x600")
        self.root.resizable(False, False)

        self.model = None
        self.canvas_size = 280
        self.pixel_size = 10
        self.drawing = False

        self.image = Image.new("L", (28, 28), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.setup_ui()
        self.load_default_model()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        title_label = ttk.Label(main_frame, text="MNIST Digit Recognition",
                               font=("Arial", 20, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        left_frame = ttk.LabelFrame(main_frame, text="Draw a digit (0-9)", padding="10")
        left_frame.grid(row=1, column=0, padx=(0, 20), sticky=(tk.W, tk.E, tk.N, tk.S))

        self.canvas = tk.Canvas(left_frame, width=self.canvas_size, height=self.canvas_size,
                               bg='black', cursor='cross')
        self.canvas.grid(row=0, column=0, pady=(0, 10))

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_pixel)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        canvas_controls = ttk.Frame(left_frame)
        canvas_controls.grid(row=1, column=0, sticky=(tk.W, tk.E))

        ttk.Button(canvas_controls, text="Clear Canvas",
                  command=self.clear_canvas).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(canvas_controls, text="Predict Digit",
                  command=self.predict_digit).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(canvas_controls, text="Save Image",
                  command=self.save_image).grid(row=0, column=2)

        right_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        model_frame = ttk.LabelFrame(right_frame, text="Model Status", padding="5")
        model_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.model_status_label = ttk.Label(model_frame, text="No model loaded",
                                           foreground="red")
        self.model_status_label.grid(row=0, column=0, sticky=(tk.W, tk.E))

        model_controls = ttk.Frame(model_frame)
        model_controls.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

        ttk.Button(model_controls, text="Load Model",
                  command=self.load_model).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(model_controls, text="Load Default",
                  command=self.load_default_model).grid(row=0, column=1)

        prediction_frame = ttk.LabelFrame(right_frame, text="Prediction", padding="5")
        prediction_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.prediction_label = ttk.Label(prediction_frame, text="Draw a digit to predict",
                                         font=("Arial", 16, "bold"))
        self.prediction_label.grid(row=0, column=0, sticky=(tk.W, tk.E))

        self.confidence_label = ttk.Label(prediction_frame, text="Confidence: N/A")
        self.confidence_label.grid(row=1, column=0, sticky=(tk.W, tk.E))

        scores_frame = ttk.LabelFrame(right_frame, text="Confidence Scores", padding="5")
        scores_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.score_labels = {}
        for i in range(10):
            frame = ttk.Frame(scores_frame)
            frame.grid(row=i//5, column=i%5, padx=2, pady=2, sticky=(tk.W, tk.E))

            digit_label = ttk.Label(frame, text=f"{i}:", width=2)
            digit_label.grid(row=0, column=0)

            score_label = ttk.Label(frame, text="0.0%", width=6)
            score_label.grid(row=0, column=1)

            self.score_labels[i] = score_label

        instructions_frame = ttk.LabelFrame(right_frame, text="Instructions", padding="5")
        instructions_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))

        instructions_text = (
            "1. Load a trained model\n"
            "2. Draw a digit (0-9) on the black canvas\n"
            "3. Click 'Predict Digit' to see results\n"
            "4. Use 'Clear Canvas' to start over\n\n"
            "Tips:\n"
            "• Draw digits centered and large\n"
            "• Use white color (click and drag)\n"
            "• Model works best with clear digits"
        )

        instructions_label = ttk.Label(instructions_frame, text=instructions_text,
                                     justify=tk.LEFT, wraplength=250)
        instructions_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def start_drawing(self, event):
        self.drawing = True
        self.draw_pixel(event)

    def draw_pixel(self, event):
        if not self.drawing:
            return

        x, y = event.x, event.y

        brush_size = 8
        self.canvas.create_oval(x-brush_size, y-brush_size,
                               x+brush_size, y+brush_size,
                               fill='white', outline='white')

        pil_x = int(x / self.pixel_size)
        pil_y = int(y / self.pixel_size)

        brush_radius = 1
        for dx in range(-brush_radius, brush_radius + 1):
            for dy in range(-brush_radius, brush_radius + 1):
                px, py = pil_x + dx, pil_y + dy
                if 0 <= px < 28 and 0 <= py < 28:
                    self.image.putpixel((px, py), 255)

    def stop_drawing(self, event):
        self.drawing = False

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.prediction_label.config(text="Draw a digit to predict")
        self.confidence_label.config(text="Confidence: N/A")

        for i in range(10):
            self.score_labels[i].config(text="0.0%")

    def save_image(self):
        if self.image:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if filename:
                scaled_image = self.image.resize((280, 280), Image.NEAREST)
                scaled_image.save(filename)
                messagebox.showinfo("Success", f"Image saved as {filename}")

    def load_model(self):
        filename = filedialog.askopenfilename(
            title="Select trained model",
            filetypes=[("Keras model files", "*.keras"), ("H5 files", "*.h5"), ("All files", "*.*")]
        )

        if filename:
            try:
                self.model = DigitRecognitionModel()
                self.model.load_model(filename)
                self.model_status_label.config(text=f"Model loaded: {os.path.basename(filename)}",
                                             foreground="green")
                messagebox.showinfo("Success", "Model loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                self.model = None
                self.model_status_label.config(text="Failed to load model",
                                             foreground="red")

    def load_default_model(self):
        model_paths = []

        if os.path.exists("saved_models"):
            for file in os.listdir("saved_models"):
                if file.endswith(('.keras', '.h5')):
                    model_paths.append(os.path.join("saved_models", file))

        if model_paths:
            latest_model = max(model_paths, key=os.path.getmtime)
            try:
                self.model = DigitRecognitionModel()
                self.model.load_model(latest_model)
                self.model_status_label.config(text=f"Default model loaded: {os.path.basename(latest_model)}",
                                             foreground="green")
            except Exception as e:
                self.model = None
                self.model_status_label.config(text="No default model available",
                                             foreground="orange")
        else:
            self.model_status_label.config(text="No trained models found. Please train a model first.",
                                         foreground="orange")

    def predict_digit(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a trained model first!")
            return

        try:
            image_array = np.array(self.image)

            max_pixel_value = np.max(image_array)
            if max_pixel_value == 0:
                messagebox.showinfo("Info", "Please draw a digit first!")
                return

            print(f"Image max value: {max_pixel_value}, non-zero pixels: {np.count_nonzero(image_array)}")

            predicted_digit, confidence_scores = self.model.predict(image_array)

            max_confidence = confidence_scores[predicted_digit] * 100
            self.prediction_label.config(text=f"Predicted Digit: {predicted_digit}")
            self.confidence_label.config(text=f"Confidence: {max_confidence:.1f}%")

            for i in range(10):
                confidence_percent = confidence_scores[i] * 100
                color = "blue" if i == predicted_digit else "black"
                self.score_labels[i].config(text=f"{confidence_percent:.1f}%",
                                          foreground=color)

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

def main():
    root = tk.Tk()
    app = DigitRecognitionGUI(root)

    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")

    root.mainloop()

if __name__ == "__main__":
    main()