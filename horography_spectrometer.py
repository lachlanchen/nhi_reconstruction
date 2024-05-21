import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from tensor_accumulation import TensorAccumulator  # Ensure this module has all necessary imports and classes
from tensor_shifter import ShiftCalculator, TensorShifter  # Ensure these classes are defined as shown earlier
import shutil
import datetime
from scipy.interpolate import griddata

class HorographySpectrometer:
    def __init__(self, file_path, intervals=1000, magic_code='otter'):
        self.file_path = file_path
        self.intervals = intervals
        self.magic_code = magic_code
        self.file_path, self.output_folder = self.setup_environment()
        self.shift_calculator = ShiftCalculator(width=8, steps=346, lines_per_mm=600, distance=84)
        self.tensor_shifter = TensorShifter(self.shift_calculator.compute_shift_vector())

    # def setup_output_folder(self):
    #     base_dir = os.path.dirname(self.file_path)
    #     output_folder = os.path.join(base_dir, f"data-{self.magic_code}")
    #     if not os.path.exists(output_folder):
    #         os.makedirs(output_folder)
    #     return output_folder

    def setup_environment(self):
        # Setup the directory with the given magic code
        base_dir = os.path.dirname(self.file_path)
        output_folder = f"data-{self.magic_code}"
        new_file_path = os.path.join(output_folder, os.path.basename(self.file_path))
        # os.makedirs(output_folder, exist_ok=True)
        if not os.path.exists(new_file_path):
            os.makedirs(output_folder, exist_ok=True)
            # Copy the tensor file to the new directory
            shutil.copy(self.file_path, new_file_path)
        return new_file_path, output_folder

    def load_and_accumulate(self):
        accumulator = TensorAccumulator(self.file_path)
        accumulated_tensor = accumulator.accumulate_continuous(self.intervals)
        return accumulated_tensor

    def shift(self, tensor):
        shifted_tensor = self.tensor_shifter.apply_shift(tensor)
        # self.visualize(tensor[0], shifted_tensor[0], 'shift')  # Visualize the first frame before and after shift
        return shifted_tensor


    def smooth_surface(self, tensor):
        """ Applies smoothing using grid data interpolation for each frame in the tensor."""
        timestamps, height, width = tensor.shape
        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)

        smoothed_tensor = torch.zeros_like(tensor)

        for i in range(timestamps):
            # Convert tensor to numpy for interpolation
            frame = tensor[i].numpy()
            # Flatten the arrays for griddata input
            points = np.vstack((y.ravel(), x.ravel())).T
            values = frame.ravel()

            # Interpolate using cubic spline
            grid_z = griddata(points, values, (y, x), method='cubic')
            smoothed_tensor[i] = torch.from_numpy(grid_z)

        return smoothed_tensor

    def visualize(self, original, processed, description, vis_folder):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original.numpy(), cmap='hot')
        plt.title('Original')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(processed.numpy(), cmap='hot')
        plt.title('Processed')
        plt.colorbar()
        plt.suptitle(f'Visualization of Changes: {description}')
        plt.savefig(os.path.join(vis_folder, f'{description}.png'))
        plt.close()

    def visualize_along_axis(self, tensor_old, tensor_new, axis, n_steps, vis_folder):
        steps = np.linspace(0, tensor_old.shape[axis] - 1, n_steps, dtype=int)
        axis_folder = os.path.join(vis_folder, f'axis_{axis}')
        os.makedirs(axis_folder, exist_ok=True)
        for step in steps:
            slice_description = f'step_{step}'
            self.visualize(tensor_old[:, :, step] if axis == 2 else tensor_old[:, step] if axis == 1 else tensor_old[step],
                           tensor_new[:, :, step] if axis == 2 else tensor_new[:, step] if axis == 1 else tensor_new[step],
                           slice_description, axis_folder)

    def multi_level_visualization(self, tensor_old, tensor_new, n_steps=10, description=""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        vis_folder = os.path.join(self.output_folder, f"visualization_{description}_{timestamp}")
        os.makedirs(vis_folder, exist_ok=True)
        for axis in range(3):  # Assuming tensor is 3D
            self.visualize_along_axis(tensor_old, tensor_new, axis, n_steps, vis_folder)



    def process(self):
        print("Starting processing of tensor data...")
        tensor_acc = self.load_and_accumulate()
        tensor_per = tensor_acc.permute(1, 2, 0)  # Ensure correct dimension order for processing
        tensor_shift = self.shift(tensor_per)
        # Additional steps like smoothing or further processing would go here, with visualization at each step
        self.multi_level_visualization(tensor_per, tensor_shift, description="shift")
        print("Processing complete. Output saved in:", self.output_folder)

        tensor_smooth = self.smooth_surface(tensor_shift)  # Smooth each frame
        self.multi_level_visualization(tensor_shift, tensor_smooth, description="smooth")
        print("Processing complete. Output saved in:", self.output_folder)


def main():
    parser = argparse.ArgumentParser(description='Process tensor data for spectrometer analysis.')
    parser.add_argument('file_path', type=str, help='Path to the tensor file.')
    args = parser.parse_args()
    spectrometer = HorographySpectrometer(args.file_path)
    spectrometer.process()

if __name__ == "__main__":
    main()
