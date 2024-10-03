import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Define input and output directories
input_dir = os.path.join('data', 'raw', 'qd-scan-automation-sample-1')
output_dir = os.path.join('outputs', 'qd-scan-automation-sample-1')

# Step 3: Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Step 4: Get all .txt files in the input directory
txt_files = glob.glob(os.path.join(input_dir, '*.dat'))

# Step 4: Process each .txt file
for txt_file in txt_files:
    # Get the base filename without the directory and extension
    base_filename = os.path.basename(txt_file)
    filename_without_ext = os.path.splitext(base_filename)[0]
    
    # Step 5: Read the data
    try:
        # Assuming data has two columns: wavelength (x) and intensity (y)
        data = np.loadtxt(txt_file)
    except Exception as e:
        print(f"Could not read {txt_file}: {e}")
        continue  # Skip to the next file if there's an error

    # Check if data has at least two columns
    if data.ndim != 2 or data.shape[1] < 2:
        print(f"Data in {txt_file} is not in the expected two-column format.")
        continue

    wavelength = data[:, 0]
    intensity = data[:, 1::2]
    
    # Step 5: Plot the data
    plt.figure()
    plt.plot(wavelength, intensity, label='Intensity')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title(f'{filename_without_ext}')
    plt.legend()
    plt.grid(True)

    # Save the plot in the output directory
    output_file = os.path.join(output_dir, f'{filename_without_ext}.png')
    plt.savefig(output_file)
    plt.show()
    plt.close()  # Close the figure to free up memory
    print(f"Processed {base_filename} and saved plot to {output_file}")
