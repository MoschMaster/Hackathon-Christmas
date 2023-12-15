import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def read_submission_file(file_path):
    with open(file_path, 'r') as file:
        sled_routes = {}
        for index, line in enumerate(file):
            parts = line.split()
            sled_routes[f'Sled {index + 1}'] = [int(x) for x in parts[1:]] # Assuming delivery numbers are integers
        return sled_routes

def plot_route(sled_number):
    # Placeholder for route plotting logic
    # Here you would retrieve and plot the route of the selected sled
    print(f"Plotting route for {sled_number}")

# GUI Setup
root = tk.Tk()
root.title("Sled Route Viewer")

sled_routes = read_submission_file('santa_submission.txt')
sled_numbers = list(sled_routes.keys())

# Dropdown for selecting a sled
selected_sled = tk.StringVar()
sled_dropdown = ttk.Combobox(root, textvariable=selected_sled, values=sled_numbers)
sled_dropdown.grid(column=0, row=0)

# Button to plot route
plot_button = tk.Button(root, text="Plot Route", command=lambda: plot_route(selected_sled.get()))
plot_button.grid(column=1, row=0)

root.mainloop()