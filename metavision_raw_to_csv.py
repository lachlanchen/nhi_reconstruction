import sys
# Ensure the correct path for system libraries
print(sys.path)
# sys.path.append("/usr/lib/python3/dist-packages/")

import os
import argparse
from datetime import datetime, timedelta
from metavision_core.event_io import EventsIterator
from tqdm import tqdm

def parse_time_argument(arg):
    """Parse time argument with units."""
    units = {"us": 1, "ms": 1000, "s": 1000000}
    if isinstance(arg, int):
        return arg
    if arg.isdigit():
        return int(arg) * units["s"]
    else:
        for unit in units:
            if arg.endswith(unit):
                try:
                    return int(float(arg[:-len(unit)]) * units[unit])
                except ValueError:
                    break
    raise argparse.ArgumentTypeError(f"Invalid time format: {arg}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert Metavision RAW or DAT to CSV format.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-file', dest='input_path', required=True,
                        help="Path to input RAW or DAT file.")
    parser.add_argument('-o', '--output-dir', default=None,
                        help="Directory to save the output CSV file. Defaults to the input file's directory.")
    parser.add_argument('-s', '--start-ts', type=parse_time_argument, default="0s",
                        help="Start time to begin processing data. Can be specified in us, ms, or s (default: s).")
    parser.add_argument('-d', '--max-duration', type=parse_time_argument, default="60s",
                        help="Maximum duration for processing data. Can be specified in us, ms, or s (default: s).")
    parser.add_argument('--delta-t', type=parse_time_argument, default="1s",
                        help="Duration of served event slice. Can be specified in us, ms, or s (default: s).")
    parser.add_argument('--start-datetime', type=str, default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        help="Start datetime for timestamps in HH:MM:SS.microsecond format. Defaults to current datetime.")
    parser.add_argument('--force', action='store_true', help="Force override of existing files.")
    return parser.parse_args()

def convert_raw_to_csv(
    input_path, output_dir=None, 
    start_ts=0, max_duration=60_000_000, delta_t=1_000_000, 
    start_datetime=None, force=False, description=None):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_dir is None:
        output_dir = os.path.dirname(input_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if start_datetime is None:
        start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    description = f"_{description}" if description else ""
    # Construct the output filename with all the info in args
    output_filename = (
        f"{os.path.basename(input_path)[:-4]}"
        f"{description}"
        f"_start_ts_{start_ts}"
        f"_max_duration_{max_duration}"
        f"_delta_t_{delta_t}"
        ".csv"
    )
    output_file = os.path.join(output_dir, output_filename)

    if os.path.exists(output_file) and not force:
        print(f"File {output_file} already exists. Skipping conversion.")
        return output_file

    mv_iterator = EventsIterator(
        input_path=input_path, 
        delta_t=delta_t,
        start_ts=start_ts, 
        max_duration=max_duration
    )

    # Parse the default start datetime for converting timestamps
    start_datetime = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")

    with open(output_file, 'w') as csv_file:
        # Write header
        csv_file.write("event_timestamp,x,y,polarity\n")
        total_iterations = max_duration // delta_t
        for evs in tqdm(mv_iterator, total=total_iterations):
            for ev in evs:
                # Calculate event time and format it
                event_time = start_datetime + timedelta(microseconds=int(ev['t']))
                formatted_time = event_time.strftime('%H:%M:%S.%f')  # Trim to microseconds precision
                csv_file.write(f"{formatted_time},{ev['x']},{ev['y']},{ev['p']}\n")

    print(f"File saved to {output_file}.")

    # Now read only the first two lines of the CSV
    with open(output_file, 'r') as csv_file:
        csv_file.readline()  # Skip header
        first_line = csv_file.readline()  # Read the first data line
        first_timestamp_str = first_line.split(',')[0]
        first_timestamp = datetime.strptime(first_timestamp_str, '%H:%M:%S.%f')

    # Calculate the time difference between the start_datetime and the first timestamp in the CSV
    time_difference = first_timestamp - start_datetime

    # Convert time difference to microseconds properly
    time_diff_in_us = time_difference.total_seconds() * 1_000_000

    # Print the results
    print(f"Original start time: {start_datetime.strftime('%H:%M:%S.%f')}")
    print(f"First event time from CSV: {first_timestamp.strftime('%H:%M:%S.%f')}")
    print(f"Time difference: {time_diff_in_us:.0f} microseconds")

    return output_file, int(time_diff_in_us)
       

def main():
    args = parse_args()
    convert_raw_to_csv(
        input_path=args.input_path,
        output_dir=args.output_dir,
        start_ts=args.start_ts,
        max_duration=args.max_duration,
        delta_t=args.delta_t,
        start_datetime=args.start_datetime,
        force=args.force
    )

if __name__ == "__main__":
    main()
