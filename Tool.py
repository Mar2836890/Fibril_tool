import tkinter as tk
from tkinter import filedialog, messagebox

import threading
import os
import sys
import pandas as pd
import platform
import shutil
import subprocess
import cv2
import numpy as np

from features_parallel import detect_fibril
from average_length import four_calc, avg_len_calc

#------------------------------Algorithms----------------------------------

# runs all the different algorithms based on the given selection
def algorithms(input_data, param_list):
    warning_time = False
    if input_data == "Fibril Analysis":
        result_file = "Fibril_result.csv"
        files = [f for f in os.listdir(images_test_folder) if f.endswith(".tif")]
        columns =  ["Filename","Total amount of found fibrils", "Avg Length Fibrils (um)", "Std Length Fibrils (um)", 
                    "Avg Width Fibrils (um)", "Std Width Fibrils (um)", "OrganizedOrientation (p-value of Chi-Square test of orientation distribution)", 
                    "PercAsg (Ratio of centriods to connected centroids of z-disks)",
                    "Orientation frequency |", "Orientation frequency /", "Orientation frequency -", "Orientation frequency \\" ]

    elif input_data == "Sacromere Analysis Tracing":
        result_file = 'Sacromere_results_Tracing.csv'
        files = [f for f in os.listdir(images_test_folder) if f.endswith(".tif")]
        columns = ["Filename", "Avg Lenght Sacromeres (um)", "ratio (ratio of z-disks and conected lines)", "z-lines amount", 
                    "distances amount (number of made connecting lines)", "check_distance (parameter value)", "deg_dif (parameter value)", 
                    "min_length_skeleton (parameter value)"]
        warning_time = True

    elif input_data == "Sacromere Analysis FFT":
        result_file = 'Sacromere_results_FFT.csv'
        files = [f for f in os.listdir(images_test_folder) if f.endswith(".png")]
        columns = ["Filename", "Avg Length Sacromere (um)", "Full Coverage (parameter value)", "Lines from Center (parameter value)"]
        
    try:
        results = []
        os.path.join(app_path, 'Results')
        for file in files:
            
            file1 = str(file)
            file_path = os.path.join(images_test_folder, file1)
            update_current_file(file, warning_time)  # Update UI with current file being processed
            
            if input_data == "Fibril Analysis":
                res = detect_fibril(images_test_folder, file, param_list)
                results.append(res[:12])
                im = res[12]
                
                path = os.path.join(results_folder, "Fibril_mask/" + file[:-4] + "_fibrils.png")
                cv2.imwrite(path, im)
                
            elif input_data == "Sacromere Analysis Tracing":
                avg_len = avg_len_calc(file_path, param_list, colour= [255, 255, 255], visualize= True, skeletonized = False)
                results.append([file1, *avg_len[:4], param_list[3][1], param_list[3][1], param_list[1][1]])
                
                path = os.path.join(results_folder, "Sacromere_mask/" + file[:-4] + "_Sacromere.png")
                cv2.imwrite(path, avg_len[4])
                
            elif input_data == "Sacromere Analysis FFT":
                avg_len = four_calc(file_path, param_list, visualize= False)
                results.append([file1, avg_len, param_list[3][1] , param_list[4][1]]) 
        
        
        # Create a DataFrame from the results
        df_results = pd.DataFrame(results, columns=columns)
        # Save the DataFrame to a CSV file
        path = os.path.join(results_folder, result_file)
        df_results.to_csv(path, index=False)   
        
        complete_processing("Processing Complete!", result_file)
    

    except:
        run_button.config(state=tk.NORMAL)  # Re-enable the button
        result_label.config(text=f" ")
        image_error(file)
            


#------------------------------Functions of the Tool----------------------------------

# Function to run the algorithm in a separate thread
def run_algorithm():
    input_check = True
    error_list = []

    input_data = clicked.get()
    result_label.config(text="Running...") 
    run_button.config(state=tk.DISABLED)  # Disable the button
    
    # Gather the parameter values from the entry fields, check input value
    params_list = []
    for param, entry, type_par in param_entries:
        param_value = entry.get()  
        try:
            # Convert the value to the correct type and check if it matches
            if type_par == int:
                param_value = int(param_value)
            elif type_par == float:
                param_value = float(param_value)
            elif type_par == bool:
                if param_value == "True":
                    param_value = True
                elif param_value == "False":
                    param_value = False
                else:
                    raise ValueError("Invalid boolean value. Expected 'True' or 'False'.")
                
            # If conversion is successful, append the value to the list
            params_list.append((param, param_value))
            
        except ValueError:
            # If conversion fails, mark as incorrect and add to error list
            error_list.append(param)
            input_check = False
            
    if input_check:
        # Run the algorithm in a separate thread to keep UI responsive
        thread = threading.Thread(target=algorithms, args=(input_data, params_list))
        thread.start()
    else:
        result_label.config(text=f" ")
        run_button.config(state=tk.NORMAL) 
        show_input_error(error_list)


# Function to handle completion of processing and enable file opening
def complete_processing(result, result_file_path):
    result_label.config(text=f" ")
    run_button.config(state=tk.NORMAL)  # Re-enable the button
    current_file_label.config(text="Processing: None")
    path = os.path.join(results_folder, result_file_path)

    # Open the CSV file in the default application
    try:
        if platform.system() == "Windows":
            os.startfile(path)
            
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", path], check=True)
        else:  # Linux
            subprocess.run(["xdg-open", path], check=True)
    except Exception as e:
        print(f"Error opening file: {e}")
        messagebox.showerror("Error", f"Could not open the file: {e}")
    

# Function to update the UI with the current file being processed
def update_current_file(file_name, warning):
    current_file_label.config(text=f"Processing: {file_name}")
    if warning:
        warning_time_label.config(text="Note: This method takes a long time to run!")
        # Schedule clearing the warning text after 5 seconds (5000 milliseconds)
        root.after(5000, lambda: warning_time_label.config(text=""))
        

# Function to update the parameter fields based on selected algorithm
def update_parameters():
    # Clear previous parameter widgets
    for widget in parameter_frame.winfo_children():
        widget.destroy()

    # Get the selected algorithm and its parameters
    selected_algorithm = clicked.get()
    params = alg_param[selected_algorithm]
    
    if selected_algorithm == "Sacromere Analysis FFT":
        intput_label= tk.Label(parameter_frame, text="Only takes 2D-FFT images (.png) as input \n")
    else:
        intput_label= tk.Label(parameter_frame, text="Only takes Binary mask images (.tiff) as input\n")
    intput_label.pack(pady=3)
    
    selected_method = clicked.get()
    description_label.config(text=method_descriptions[selected_method])
    
    # Create input fields for each parameter
    global param_entries  # List to store the parameter entry widgets
    param_entries = []  # Clear the list of entries
    
    for param, default_value, type, information in params:
        label = tk.Label(parameter_frame, text=param, font=("Arial", 15, "bold"))
        label.pack()
        
        label_inf = tk.Label(parameter_frame, text=information)
        label_inf.pack()

        # Create the entry widget for the parameter, using the default value
        entry = tk.Entry(parameter_frame)
        entry.insert(0, str(default_value))  # Set default value
        entry.pack()
        
        label_space = tk.Label(parameter_frame, text=" ")
        label_space.pack(pady=3)

        param_entries.append((param, entry, type))  # Store the (parameter_name, entry) tuple


# Function to handle image uploading
def upload_images():
    file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.png *.tif")])
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        destination = os.path.join(images_test_folder, filename)
        shutil.copy(file_path, destination)
    update_image_list()


# Function to update the list of images in the UI
def update_image_list():
    images_listbox.delete(0, tk.END)
    if not os.path.exists(images_test_folder):
        os.makedirs(images_test_folder)  # Ensure the directory exists (optional, based on needs)
    
    # Iterate over files in the current directory (or Images_to_test folder)
    for image_file in os.listdir(images_test_folder):
        images_listbox.insert(tk.END, image_file)


# Function to delete all images in the folder
def delete_all_images():
    for file in os.listdir(images_test_folder):
        os.remove(os.path.join(images_test_folder, file))
    update_image_list()


def show_input_error(error_list):
    popup = tk.Toplevel()
    popup.title("Error Input")
    popup.geometry("1000x200")
    
    message_label = tk.Label(popup, text=f"There was en error when checking the input of the parameters. The folowing paramters need to be changes: {error_list}.")
    message_label.pack(pady=20)

    # Create a button to close the pop-up
    close_button = tk.Button(popup, text="Close", command=popup.destroy)
    close_button.pack()
    
    
def image_error(file):
    popup = tk.Toplevel()
    popup.title("Error Input")
    popup.geometry("1000x200")
    
    message_label = tk.Label(popup, text="Something went wrong while processing the images, makes sure you have the right images oploaded. They should all be the same type (.tif or .png)")
    message_label.pack(pady=20)
    
    file_label = tk.Label(popup, text=f"The file that caused this issue was: {file}")
    file_label.pack(pady=20)

    # Create a button to close the pop-up
    close_button = tk.Button(popup, text="Close", command=popup.destroy)
    close_button.pack()
    

# Define the reset_parameters function to reset parameter entries to their default values
def reset_parameters():
    # Iterate over each parameter entry and reset to the default value
    for param, entry, type_par in param_entries:
        # Get the default value from the alg_param dictionary for the selected algorithm
        selected_algorithm = clicked.get()
        default_value = next((default for name, default, type, info in alg_param[selected_algorithm] if name == param), None)
        
        # Reset the entry to the default value
        entry.delete(0, tk.END)  # Clear the current value
        entry.insert(0, str(default_value))  # Insert the default value


#------------------------------Tkinter Main Window-----------------------------------


if __name__ == '__main__':    
    
    # Determine if the script is running as a bundled executable (PyInstaller)
    if getattr(sys, 'frozen', False):  # Running as a PyInstaller executable
        # Get the folder where the executable is located (not the _internal folder)
        app_path = os.path.dirname(sys.executable)
    else:
        # Running as a script (development mode)
        app_path = os.path.dirname(os.path.abspath(__file__))  # The current script directory

    # Define the relative paths for Results and Images_to_test folders
    results_folder = os.path.join(app_path, 'Results')
    images_test_folder = os.path.join(app_path, 'Images_to_test')

    # Ensure that the directories exist, otherwise create them
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    if not os.path.exists(images_test_folder):
        os.makedirs(images_test_folder)
    
    # list of changeble parameters per algorithm
    alg_param = {
        "Fibril Analysis": 
            [("Ratio (px/um - float)", 4.8177, float, "Ratio of the image in (px/um)"), 
            ("Radius (px - interger)", 14, int, "To determine whether a specific region is crowded, a threshold of 4 objects within a radius is set"), 
            ("Max_distance (px - interger)", 14, int, "To get the connected z-disks there is a max distance between z-disks that we use."), 
            ("Limit border (px - interger)", 16, int, "Limit for z-disks near the border border."), 
            ("Limit endpoints (px - interger)", 20, int, "Limit for distance between endpoints of z-disk.")],
            
        "Sacromere Analysis Tracing":
            [("Ratio (px/um - float)", 4.8177, float, "Ratio of the image in (px/um)"),
            ("min_length_skeleton (um - float)", 0.5, float, "Minimum lenght a z-disk skeleton must have." ),
            ("list_width (integer)", 100000, int, "Keep withing a list bound when checking points."),
            ("deg_dif (px - integer)" ,20, int, "Maximun distance between lines, to make sure they are not too far apart."),
            ("check_distance (px - integer)", 13, int, "Maximum distance og z-line couples.")],
            
        "Sacromere Analysis FFT":
            [("Ratio (px/um - float)", 4.8177, float, "Ratio of the image in (px/um)"),
            ("upper_s_len (um - float)", 2.0, float, "Maximum sacromere lenght"),
            ("lower_s_len (um - float)", 1.2, float, "Minimum sacromere lenght"),
            ("full_coverage (True or False)", False, bool, "Check all pixels in bandpass."),
            ("lines_from_center (degrees - integer)", 360, int,"How many degrees of the cirkel is checked."),
            ("interval_coverage (float)", 0.0004, float, "How many intervals in the cirkel is checked, value is multiplied by pi.")]
        }

    # Set up the Tkinter window
    root = tk.Tk()
    root.title("Algorithm Runner")
    root.geometry("1300x800")  # Adjust the size to accommodate the two-column layout

    # Configure the main frame
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Left Column Frame (for image management) with fixed dimensions
    left_frame = tk.Frame(main_frame, width=600, height=800)  # Fixed width and height
    left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")
    left_frame.grid_propagate(False)  # Prevent resizing based on content
    # Configure grid inside left_frame
    left_frame.columnconfigure(0, weight=1)  # Allow widgets in column 0 to stretch horizontally
    left_frame.rowconfigure(2, weight=1)    # Allow the images_listbox (row 2) to stretch vertically
        
    # Right Column Frame (for algorithm and parameters)
    right_frame = tk.Frame(main_frame, width=600, height=800)
    right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

    # Image management section in the left column
    upload_button = tk.Button(left_frame, text="Upload Images", command=upload_images)
    upload_button.grid(row=0, column=0, pady=5)

    delete_button = tk.Button(left_frame, text="Delete All Images", command=delete_all_images)
    delete_button.grid(row=1, column=0, pady=5)

    images_listbox = tk.Listbox(left_frame, width=40, height=10)
    images_listbox.grid(row=2, column=0)

    # Call update_image_list to initialize the listbox with current images
    update_image_list()
    
    # Add descriptions for each method
    method_descriptions = {
        "Fibril Analysis": """
Fibril Analysis: created by Linde Schoenmaker (2021) \n
Serves as a full analysis of the fibril structures in the image. It identifies fibrils in each image, calculating different metrics. First preprocessing must be done to create a Binary Mask of the original image. The Binary Mask retrieves Z-disks and establishes a coordinate system for further calculations. After preliminary image cleaning, the algorithm identifies neighboring Z-disks (forming sarcomeres) and connects them, repeating this process to trace entire fibrils.

Once tracing is complete, the algorithm calculates key metrics, including:
- Number of fibrils
- Average length and width of the fibrils (with standard deviations)
- Organized orientation (using a Chi-square test p-value for distribution)
- Full orientation distribution (| means vertical, - horizontal, / and \ diagonal)
            """,
        
        "Sacromere Analysis Tracing": """ 
Sarcomere Tracing Analysis: created by Tijmen van Wel (2023) \n
The algorithm identifies sarcomere structures by using the angles between z-lines. We begin by creating a Binary Mask of the original image to extract the z-disk structures. Each z-disk is then identified, with smaller disks being filtered out. Next, based on the orientation of the remaining z-disks, a perpendicular line is drawn between pairs of disks that share the same orientation. This process creates the sarcomere structures between the z-disks.
            
Once tracing is complete, the algorithm calculates the following metrics:
- Average sarcomere length
- Amount of traced z-disks
- Number of created lines between z-disks (amount of sarcomere structures that are traced)
- Ratio between number of z-lines and connected lines
- Some input parameter setting
            """,
            
        "Sacromere Analysis FFT": """
Sarcomere 2D-FFT method: created by Tijmen van Wel (2023). \n
We apply a 2D Fast Fourier Transform (2D-FFT) to the image to analyze its frequency content. In the original image, each pixel represents a specific brightness or intensity, reflecting spatial information. The 2D-FFT converts this spatial information into the frequency domain, where each pixel in the transformed image now corresponds to a frequency component. Bright spots indicate repeating patterns, like sarcomere spacing.
To reduce noise, we set frequency bounds based on typical sarcomere lengths, ensuring we focus on meaningful frequencies. Finally, we calculate the total brightness and the number of pixels at each distance, creating an average brightness distribution.

The algorithm returns the following information:
- Average sarcomere length
- Some input parameter setting
"""
    }
    # Add a label for the description in the left column
    description_label = tk.Label(left_frame,text=method_descriptions["Fibril Analysis"], height= 30, wraplength= 400, justify="left", anchor="n")
    description_label.grid(row=4, column=0, sticky="nsew")

    # Dropdown menu options
    options = ["Fibril Analysis", "Sacromere Analysis Tracing", "Sacromere Analysis FFT"]
    clicked = tk.StringVar()
    clicked.set("Fibril Analysis")

    # Create Dropdown menu in the right column
    drop = tk.OptionMenu(right_frame, clicked, *options)
    drop.grid(row=0, column=0, pady=10)

    # Frame to hold parameter fields in the right column
    parameter_frame = tk.Frame(right_frame)
    parameter_frame.grid(row=1, column=0, pady=10)

    # Position the buttons side-by-side using a frame in the right column
    button_frame = tk.Frame(right_frame)
    button_frame.grid(row=2, column=0, pady=5)

    run_button = tk.Button(button_frame, text="Run algorithm", command=run_algorithm)
    run_button.grid(row=0, column=0, padx=5)

    reset_button = tk.Button(button_frame, text="Reset Parameters", command=reset_parameters)
    reset_button.grid(row=0, column=1, padx=5)

    # Current File Label in the right column
    current_file_label = tk.Label(right_frame, text="No file is being processed.")
    current_file_label.grid(row=3, column=0, pady=5)

    # Result Label in the right column
    result_label = tk.Label(right_frame, text="")
    result_label.grid(row=4, column=0, pady=5)
    
    # Current File Label in the right column
    warning_time_label = tk.Label(right_frame, text="")
    warning_time_label.grid(row=5, column=0, pady=5)

    # List to store parameter entry widgets
    param_entries = []

    # Call update_parameters whenever the selected algorithm changes
    clicked.trace("w", lambda *args: update_parameters())

    # Initialize with default parameters for the first algorithm
    update_parameters()

    root.mainloop()




