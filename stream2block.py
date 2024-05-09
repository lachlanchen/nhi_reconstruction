import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import argparse
import subprocess

# Set up the device to use CPU as default
device = 'cpu'

class EventDataTo3DBlock:
    def __init__(self, csv_path, output_folder, height=None, width=None, model=None, use_cache=False):
        self.csv_path = csv_path
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        self.set_dynamic_dimensions()

        if model == 'D':
            self.height, self.width = 260, 346  # Davis
        elif model == 'X':
            self.height, self.width = 480, 640  # Dvxplorer
        elif height is not None and width is not None:
            self.height, self.width = height, width
        else:
            # self.set_dynamic_dimensions()
            pass

        self.cache_path = os.path.join(self.output_folder, f'frames_{self.height}_{self.width}.pt')
        if os.path.exists(self.cache_path) and use_cache:
            print("Loading frames from cache.")
            self.frames = torch.load(self.cache_path).to(device)
        else:
            self.load_and_process_data()
            torch.save(self.frames.cpu(), self.cache_path)
            print(f"Saved frames to {self.cache_path}.")

    def set_dynamic_dimensions(self):
        df = pd.read_csv(self.csv_path)
        self.height = df['y'].max() + 1
        self.width = df['x'].max() + 1

        print("height(y): ", self.height)
        print("width(x): ", self.width)

    def load_and_process_data(self):
        df = pd.read_csv(self.csv_path)
        unique_timestamps, indices = torch.unique(torch.tensor(df['event_timestamp'].astype('category').cat.codes), return_inverse=True)
        indices = indices.to(device)
        
        self.frames = torch.zeros((len(unique_timestamps), self.height, self.width), dtype=torch.uint8, device=device)
        x = torch.tensor(df['x'].values, dtype=torch.int, device=device)
        y = torch.tensor(df['y'].values, dtype=torch.int, device=device)
        polarity = torch.tensor(df['polarity'].map({True: 1, False: -1}).values, dtype=torch.uint8, device=device)
        self.frames.index_put_((indices, y, x), polarity, accumulate=True)

    def save_unique_timestamps_and_positions(self):
        df = pd.read_csv(self.csv_path)
        unique_timestamps_positions = df[['event_timestamp', 'y']].drop_duplicates()
        output_path = os.path.join(self.output_folder, 'unique_timestamps_and_y_positions.csv')
        unique_timestamps_positions.to_csv(output_path, index=False)
        print(f"Saved unique timestamps and Y positions to '{output_path}'.")

    def save_all_frames_as_images(self, folder='frames'):
        if not os.path.exists(folder):
            os.makedirs(folder)

        for i in range(len(self.frames)):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis('off')
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

            frame = self.frames[i].cpu().numpy()
            ax.imshow(frame, cmap='gray', extent=[0, 640, 480, 0])
            y_pos, x_pos = (frame == 1).nonzero()
            y_neg, x_neg = (frame == -1).nonzero()
            ax.scatter(x_pos, y_pos, color='green', s=1)
            ax.scatter(x_neg, y_neg, color='red', s=1)
            plt.savefig(f"{folder}/frame_{i:04d}.png", dpi=100)
            plt.close()

        print(f"Saved all frames as images in the {folder}/ directory.")

    def frames_to_video(self, folder='frames', output_video='frames_video.mp4', fps=30):
        folder_path = os.path.join(self.output_folder, folder)
        video_path = os.path.join(self.output_folder, output_video)
        command = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(folder_path, 'frame_%04d.png'),
            '-vf', f'scale={self.width}:{self.height}',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            video_path
        ]
        subprocess.run(command, check=True)
        print(f"Compiled frames into video: {video_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video data from event data.')
    parser.add_argument('csv_path', help='Path to the CSV file containing event data.')
    parser.add_argument('-output', dest='output_folder', default=None, help='Output folder for results. Default is the same as the CSV path.')
    parser.add_argument('-size', nargs=2, type=int, help='Specify video width and height as W H.')
    parser.add_argument('-model', choices=['D', 'X'], help='Specify camera model: D for Davis, X for Dvxplorer.')

    args = parser.parse_args()

    # Determine output folder based on whether a specific output directory was provided
    output_folder = args.output_folder if args.output_folder is not None else os.path.dirname(args.csv_path)

    data_processor = EventDataTo3DBlock(args.csv_path, output_folder, *args.size if args.size else (None, None), args.model)
    # data_processor.save_unique_timestamps_and_positions()
    # data_processor.save_all_frames_as_images()
    # data_processor.frames_to_video()
