import scipy.io as sio
import numpy as np
import pandas as pd
import json
from pathlib import Path
global PLOTQ # REF_IDX, CHANNELS
#REF_IDX = 70 # is using 1x 32 CH and 1x 64 CH grids # !!!OTBio Muovi/Muovi+/Syncstation specific!!!
#CHANNELS = [0, 64]# if Grid 1 is 32 CH [36,100] # if 1st grid is 32 CH and 2nd is 64 CH # !!!OTBio Muovi/Muovi+/Syncstation specific!!!
PLOTQ = False # !!set!!

def load_hdsemg_select_json(mat_path):
    """
    Load hdsemg-select JSON file for channel selection information.

    This function looks for a JSON file with the same name as the .mat file
    (but with .json extension) that contains channel selection info from hdsemg-select.

    Args:
        mat_path (str or Path): Path to the .mat file.

    Returns:
        dict or None: Dictionary containing:
            - 'channel_range': [start_idx, end_idx] for the first grid's EMG channels
            - 'good_channel_indices': List of channel indices (relative to channel_range) that are selected
            - 'bad_channel_indices': List of channel indices (relative to channel_range) marked as bad
            - 'ref_signals': List of dicts with reference signal info
            - 'grids': List of grid configurations
        Returns None if JSON file doesn't exist.

    Example:
        >>> config = load_hdsemg_select_json("data.mat")
        >>> print(config['channel_range'])  # [4, 68]
        >>> print(config['good_channel_indices'])  # [2, 3, 4, ...]
    """
    json_path = Path(mat_path).with_suffix('.json')

    if not json_path.exists():
        print(f"No hdsemg-select JSON found at {json_path}, using manual configuration")
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        print(f"Loaded hdsemg-select configuration from {json_path}")

        # Extract grid information (assuming first grid for now)
        if 'grids' not in data or len(data['grids']) == 0:
            print("Warning: No grids found in JSON file")
            return None

        first_grid = data['grids'][0]
        grid_channels = first_grid['channels']

        # Get the channel range (first to last channel index in the grid)
        channel_indices = [ch['channel_index'] for ch in grid_channels]
        channel_range = [min(channel_indices), max(channel_indices) + 1]

        # Get good channels (selected=True and no "Bad Channel" label)
        good_channel_indices = []
        bad_channel_indices = []

        for ch in grid_channels:
            relative_idx = ch['channel_index'] - channel_range[0]

            # Check if channel is selected and not labeled as bad
            is_bad = 'Bad Channel' in ch.get('labels', [])
            is_selected = ch.get('selected', False)

            if is_selected and not is_bad:
                good_channel_indices.append(relative_idx)
            else:
                bad_channel_indices.append(relative_idx)

        # Extract reference signal information
        ref_signals = first_grid.get('reference_signals', [])

        result = {
            'channel_range': channel_range,
            'good_channel_indices': good_channel_indices,
            'bad_channel_indices': bad_channel_indices,
            'ref_signals': ref_signals,
            'grids': data['grids'],
            'filename': data.get('filename', ''),
            'ied': first_grid.get('inter_electrode_distance_mm', None)
        }

        print(f"  Grid: {first_grid.get('rows', '?')}x{first_grid.get('columns', '?')}, "
              f"IED: {result['ied']}mm")
        print(f"  Channel range: {channel_range[0]}-{channel_range[1]-1}")
        print(f"  Good channels: {len(good_channel_indices)}/{len(grid_channels)}")
        print(f"  Bad channels: {len(bad_channel_indices)}")

        return result

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {json_path}: {e}")
        return None
    except Exception as e:
        print(f"Error loading hdsemg-select JSON: {e}")
        return None

def extract_raw_emg_metadata(mat_path, config, mat_source='otb+'):
    """
    Extracts and structures raw EMG metadata and signals from a .mat file.

    This function reads a `.mat` neurophysiological dataset and extracts:
    - Sampling frequency
    - Inter-electrode distance (IED)
    - Number of EMG channels
    - Reference signal
    - Raw EMG data
    
    Extracts from config:
    - channel_range (list of size 1,2): EMG channel indices from ... to
    - ref_path_measured_idx (int): index of the measured performed path of the force/torque reference
    
    Args:
        mat_path (str or Path): Path to the .mat file.
        config (Config): Configuration object containing settings such as start_time and sampling_frequency.
        mat_source (str, optional): Specifies the `.mat` file format ('otb+' only currently). Defaults to 'otb+'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, float, float, pd.DataFrame]: 
        - `rawEMG_Channels` (pd.DataFrame): Extracted EMG signal.
        - `refSignal` (pd.DataFrame): Extracted reference force/torque signal.
        - `fsamp` (float): Sampling frequency.
        - `ied` (float): Inter-electrode distance in mm.
        - `extras` (pd.DataFrame): Additional metadata extracted from the file.
    
    Raises:
        FileNotFoundError: If the provided `.mat` file is not found.
        ValueError: If the file format is incorrect or does not contain expected fields.

    Example:
        >>> rawEMG, refSignal, fsamp, ied, extras = extract_raw_emg_metadata("data.mat", config)
        >>> print(fsamp, ied)
    """
    try:
        mat = sio.loadmat(mat_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"MAT file not found at {mat_path}")
    channel_range = config.channel_range
    ref_path_measured_idx = config.ref_path_measured_idx
    if mat_source == 'otb+':
        # OTBiolab+ Specific Structure
        idxFrom = int(np.round(config.start_time * config.sampling_frequency))
        idxTo = int(np.round(config.end_time * config.sampling_frequency))

        # Extract reference signal
        refSignal = pd.DataFrame(mat['Data'][idxFrom:idxTo, ref_path_measured_idx])

        # Extract number of channels from the description
        try:
            description0 = mat['Description'][channel_range[0]][0][0]
            nCh = int(description0.split(' - ')[2].split(' ')[0][6:8]) * int(description0.split(' - ')[2].split(' ')[0][8:10])
            print(f"Description used: {description0}")
            if nCh%2 == 1:
                nCh = nCh - 1
            print(f" ... exporting {nCh} channels")
        except (KeyError, IndexError, ValueError):
            raise ValueError("Failed to parse channel count from the .mat file description.")

        # Extract raw EMG signal
        rawEMG_Channels = pd.DataFrame(mat["Data"][idxFrom:idxTo, channel_range[0]:channel_range[1]])

        # Extract sampling frequency and inter-electrode distance
        fsamp = mat['SamplingFrequency'][0][0]
        ied = float(description0.split(' - ')[2].split(' ')[0][2:4])
        print(f" ... IED: {ied}")

        # Extract additional metadata
        # Include bad_channels information in extras for export
        bad_channels_list = list(config.bad_channels) if hasattr(config, 'bad_channels') and config.bad_channels else []
        bad_channels_info = f"bad_channels={bad_channels_list}"
        extras = pd.DataFrame([
            description0.replace('(1)', ''),
            mat['Description'][ref_path_measured_idx][0][0],
            bad_channels_info,
            str(config)
        ])
    
    else:
        raise ValueError(f"Unsupported mat_source: {mat_source}. Only 'otb+' is implemented currently.")

    return rawEMG_Channels, refSignal, fsamp, ied, extras

def loadEMG_updConfig(mat, config, channel_range=None, ref_path_target_idx=None, ref_path_measured_idx=None, bad_channels=None, hdsemg_config=None, mat_source='otb+', n_std=7, sFrom=1, sTo=3, PLOTQ=False):
    """
    Extract raw EMG data from source file and update decomposition configurations.

    This function reads neurophysiological data and adapts the configuration parameters based
    on the recording metadata (e.g., sampling rate, signal trimming thresholds).

    Can be used in two modes:
    1. Manual mode: Provide channel_range, ref_path_target_idx, ref_path_measured_idx, bad_channels
    2. JSON mode: Provide hdsemg_config (from load_hdsemg_select_json())

    Args:
        mat (dict): Dictionary containing .mat file data.
        config (Config): Existing configuration object to update.
        channel_range (list of size 1,2, optional): EMG channel indices from ... to. If None, will use hdsemg_config.
        ref_path_target_idx (int, optional): index to target path. If None, will use hdsemg_config.
        ref_path_measured_idx (int, optional): index to performed path. If None, will use hdsemg_config.
        bad_channels (list, optional): channel indices (relative to channel_range) to be removed. If None, will use hdsemg_config.
        hdsemg_config (dict, optional): Configuration dict from load_hdsemg_select_json(). Takes precedence over manual parameters.
        mat_source (str, optional): Specifies the source format ('otb+' or 'original'). Defaults to 'otb+'.
        n_std (int, optional): Number of standard deviations for thresholding. Defaults to 7.
        sFrom (int, optional): Baseline force calculation start (in sec). Defaults to 1.
        sTo (int, optional): Baseline force calculation end (in sec). Defaults to 3.
        PLOTQ (bool, optional): Whether to plot the signals. Defaults to False.

    Returns:
        Tuple[dict, Config]: Updated `.mat` data and modified configuration.

    Raises:
        ValueError: If `mat_source` is unsupported or required parameters are missing.
    """
    if mat_source == 'otb+':
        # If hdsemg_config is provided, use it instead of manual parameters
        if hdsemg_config is not None:
            print("Using hdsemg-select JSON configuration")
            channel_range = hdsemg_config['channel_range']
            bad_channels = hdsemg_config['bad_channel_indices']

            # Find reference signals - look for the ones marked as selected
            # Typically target is the first ref, measured is the second
            ref_signals = hdsemg_config.get('ref_signals', [])
            selected_refs = [ref for ref in ref_signals if ref.get('selected', False)]

            if len(selected_refs) >= 2:
                ref_path_target_idx = selected_refs[0]['ref_index']
                ref_path_measured_idx = selected_refs[1]['ref_index']
                print(f"  Reference signals from JSON: target={ref_path_target_idx}, measured={ref_path_measured_idx}")
            elif len(selected_refs) == 1:
                # Use the same signal for both if only one is available
                ref_path_target_idx = selected_refs[0]['ref_index']
                ref_path_measured_idx = selected_refs[0]['ref_index']
                print(f"  Using single reference signal: {ref_path_measured_idx}")
            else:
                raise ValueError("No reference signals found in hdsemg-select JSON. Please select at least one reference signal.")
        else:
            # Manual mode - validate that required parameters are provided
            if channel_range is None or ref_path_target_idx is None or ref_path_measured_idx is None:
                raise ValueError("When not using hdsemg_config, channel_range, ref_path_target_idx, and ref_path_measured_idx must be provided")
            if bad_channels is None:
                bad_channels = []

        # Create the full list of channels
        all_channels = list(range(channel_range[0], channel_range[1]))
        # Filter out channels at indices specified in bad_channels
        good_channels = [ch for idx, ch in enumerate(all_channels) if idx not in bad_channels]
        print(f"Good channels used:\n {good_channels}")
        fsamp = int(mat['SamplingFrequency'][0][0])
        ref_path_target = mat['Data'][:, ref_path_target_idx]
        ref_path_measured = mat['Data'][:, ref_path_measured_idx]
        if PLOTQ:
            import matplotlib.pyplot as plt
            plt.plot(ref_path_target)
            plt.plot(ref_path_measured)
            plt.show()
            for i, goodCh in enumerate(good_channels):
                plt.plot(i+mat['Data'][:, goodCh]/(max(mat['Data'][:, goodCh])*1.3))
            plt.show()
        # Compute thresholds
        baseline_start = ref_path_measured[int(fsamp) * sFrom:int(fsamp * sTo)]
        threshold_start = baseline_start.mean() + baseline_start.std() * n_std

        baseline_end = ref_path_measured[::-1][int(fsamp) * sFrom:int(fsamp * sTo)]
        threshold_end = baseline_end.mean() + baseline_end.std() * n_std

        force_threshold = (threshold_start + threshold_end) / 2

        # Adjust config
        if sum(ref_path_target)==0:
            config.start_time = 0
            config.end_time = ref_path_target.shape[0]
        else:
            config.end_time = (1 + ref_path_target.shape[0] - np.where(ref_path_target[::-1] > force_threshold)[0][0]) / fsamp
            config.start_time = np.where(ref_path_target > force_threshold)[0][0] / fsamp
        config.sampling_frequency = fsamp
        n_good_channels = len(good_channels)
        config.extension_factor = int(np.round(1000 / n_good_channels))
        config.channel_range = channel_range
        config.ref_path_target_idx = ref_path_target_idx
        config.ref_path_measured_idx = ref_path_measured_idx
        config.bad_channels = bad_channels
        # ToDo Add all other decomposition settings to config to be saved in openhdemg EXTRAS later on
        print(f"EF: {round(config.extension_factor,2)}")

        # Load neural data
        mat["emg"] = mat["Data"][:, good_channels].transpose()  # needs update based on bad channels

    elif mat_source == 'original':
        # Load neural data from original source
        config.start_time = sFrom
        config.end_time = sTo

    return mat, config