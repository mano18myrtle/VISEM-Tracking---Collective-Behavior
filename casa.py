import os
import numpy as np
import pandas as pd

def calculate_casa_metrics(track_data, fps):
    """
    Calculate CASA metrics for sperm tracking data.
    Handles column name variations and missing data.
    """
    casa_metrics = []

    # Standardize column names (case-insensitive)
    col_map = {
        'frame': 'Frame',
        'time': 'Frame',
        'id': 'ID',
        'x': 'X',
        'y': 'Y'
    }
    track_data = track_data.rename(columns=lambda x: col_map.get(x.lower(), x))

    # Verify required columns exist
    required_columns = ['Frame', 'ID', 'X', 'Y']
    missing_cols = [col for col in required_columns if col not in track_data.columns]
    
    if missing_cols:
        print(f"Missing columns detected: {missing_cols}")
        print("Available columns:", track_data.columns.tolist())
        raise ValueError("Required columns not found in tracking data")

    # Group data by track ID
    grouped_tracks = track_data.groupby('ID')

    for track_id, group in grouped_tracks:
        try:
            group = group.sort_values(by='Frame')
            
            # Convert frame numbers to time in seconds
            group['Time'] = group['Frame'] / fps
            x_coords = group['X'].values
            y_coords = group['Y'].values
            
            if len(group) < 2:
                continue  # Skip tracks with insufficient data

            # Time calculations
            total_time = group['Time'].iloc[-1] - group['Time'].iloc[0]
            if total_time <= 0:
                continue

            # VCL: Curvilinear Velocity
            dx = np.diff(x_coords)
            dy = np.diff(y_coords)
            distances = np.sqrt(dx**2 + dy**2)
            vcl = distances.sum() / total_time

            # VSL: Straight-Line Velocity
            straight_line_distance = np.sqrt((x_coords[-1] - x_coords[0])**2 + 
                                            (y_coords[-1] - y_coords[0])**2)
            vsl = straight_line_distance / total_time

            # VAP: Average Path Velocity (smoothed using moving average)
            window_size = min(3, len(x_coords))
            if window_size > 1:
                smooth_x = np.convolve(x_coords, np.ones(window_size)/window_size, mode='valid')
                smooth_y = np.convolve(y_coords, np.ones(window_size)/window_size, mode='valid')
                smooth_distances = np.sqrt(np.diff(smooth_x)**2 + np.diff(smooth_y)**2)
                vap = smooth_distances.sum() / total_time
            else:
                vap = 0.0

            # ALH: Amplitude of Lateral Head Displacement
            try:
                lateral_displacements = np.abs(y_coords - np.interp(x_coords, smooth_x, smooth_y))
                alh = lateral_displacements.max() / 2 if len(lateral_displacements) > 0 else 0.0
            except:
                alh = 0.0

            # Derived metrics
            lin = (vsl / vcl) * 100 if vcl > 0 else 0.0
            wob = (vap / vcl) * 100 if vcl > 0 else 0.0
            str_ = (vsl / vap) * 100 if vap > 0 else 0.0
            
            # BCF: Beat Cross Frequency
            try:
                zero_crossings = len(np.where(np.diff(np.sign(lateral_displacements)))[0])
                bcf = zero_crossings / total_time if total_time > 0 else 0.0
            except:
                bcf = 0.0

            casa_metrics.append({
                'ID': track_id,
                'VCL (μm/s)': vcl,
                'VSL (μm/s)': vsl,
                'VAP (μm/s)': vap,
                'ALH (μm)': alh,
                'LIN (%)': lin,
                'WOB (%)': wob,
                'STR (%)': str_,
                'BCF (Hz)': bcf,
                'Duration (s)': total_time
            })

        except Exception as e:
            print(f"Error processing track {track_id}: {str(e)}")
            continue

    return pd.DataFrame(casa_metrics)

def process_batch(batch_id, input_dir, output_dir, fps=30):
    """Process a single batch of tracking data with enhanced error handling"""
    try:
        # Input validation
        excel_path = os.path.join(input_dir, str(batch_id), f"{batch_id}.xlsx")
        if not os.path.exists(excel_path):
            print(f"Batch {batch_id}: Excel file not found")
            return

        # Read all sheets and combine with IDs
        xls = pd.ExcelFile(excel_path)
        all_data = []
        
        for sheet_name in xls.sheet_names:
            try:
                # Extract ID from sheet name (format: "ID_X_Class")
                parts = sheet_name.split('_')
                if len(parts) < 3 or not parts[1].isdigit():
                    continue
                
                track_id = int(parts[1])
                df = xls.parse(sheet_name)
                
                # Standardize column names
                df = df.rename(columns={
                    'Time': 'Frame',
                    'time': 'Frame',
                    'X Position': 'X',
                    'Y Position': 'Y'
                })
                
                # Add required columns if missing
                if 'Frame' not in df.columns:
                    if 'Time' in df.columns:
                        df['Frame'] = df['Time'] * fps
                    else:
                        df['Frame'] = df.index
                
                df['ID'] = track_id
                all_data.append(df[['Frame', 'ID', 'X', 'Y']])
            except Exception as e:
                print(f"Error processing sheet {sheet_name}: {str(e)}")
                continue

        if not all_data:
            print(f"Batch {batch_id}: No valid data found")
            return

        combined_df = pd.concat(all_data, ignore_index=True)

        # Calculate metrics
        casa_results = calculate_casa_metrics(combined_df, fps)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{batch_id}_casa_metrics.csv")
        casa_results.to_csv(output_path, index=False)
        print(f"Batch {batch_id} processed successfully. Results saved to {output_path}")

    except Exception as e:
        print(f"Fatal error processing batch {batch_id}: {str(e)}")

# Configuration
batch_ids = [11, 12, 13, 14, 15, 19, 21, 22, 23, 24, 29, 30, 35, 36, 38, 47, 52, 54, 60, 82]
input_base_dir = "/home/user/Mano/ml_env/Velocity"
output_base_dir = "/home/user/Mano/ml_env/CASA_Metrics"

# Process all batches
for bid in batch_ids:
    print(f"\n{' PROCESSING BATCH ' + str(bid) + ' ':=^50}")
    process_batch(
        batch_id=bid,
        input_dir=input_base_dir,
        output_dir=output_base_dir,
        fps=30  # Adjust based on actual video frame rate
    )

print("\nBatch processing completed!")

