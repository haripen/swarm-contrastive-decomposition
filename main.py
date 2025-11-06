import os
import numpy as np
import torch
from pathlib import Path
import scipy.io as sio

from config.structures import set_random_seed, Config
from models.scd import SwarmContrastiveDecomposition
from processing.postprocess import save_results
from utils.exporting import export_to_openhdemg_json, export_to_muedit_mat
from utils.preprocessing import loadEMG_updConfig, extract_raw_emg_metadata, load_hdsemg_select_json

set_random_seed(seed=42)


def train(path):
    print(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    acceptance_silhouette = 0.88
    extension_factor = 20 # will be updated
    time_differentiate = False
    notch_params = [50, 1.0, True] # powerline frequency, bandwidth, filter harmonics
    low_pass_cutoff = 500
    high_pass_cutoff = 10
    start_time = 0 # will be updated
    end_time = -1 # will be updated
    max_iterations = 250
    sampling_frequency = 2000 # will be updated
    peel_off_window_size_ms = 50 # 20 # ms
    output_final_source_plot = False
    use_coeff_var_fitness = True
    remove_bad_fr = True
    clamp_percentile = 0.999  

    config = Config(
        device=device,
        acceptance_silhouette=acceptance_silhouette,
        extension_factor=extension_factor,
        time_differentiate=time_differentiate,
        notch_params=notch_params,
        low_pass_cutoff=low_pass_cutoff,
        high_pass_cutoff=high_pass_cutoff,
        sampling_frequency=sampling_frequency,
        start_time=start_time,
        end_time=end_time,
        max_iterations=max_iterations,
        peel_off_window_size_ms=peel_off_window_size_ms,
        output_final_source_plot=output_final_source_plot,
        use_coeff_var_fitness=use_coeff_var_fitness,
        remove_bad_fr=remove_bad_fr,
        clamp_percentile=clamp_percentile,
    )

    # Load data
    if path.suffix == ".mat":
        mat = sio.loadmat(path)

        # Try to load hdsemg-select JSON configuration
        hdsemg_config = load_hdsemg_select_json(path)

        if hdsemg_config is not None:
            # Use JSON configuration
            print("Using hdsemg-select channel configuration")
            mat, config = loadEMG_updConfig(mat, config, hdsemg_config=hdsemg_config)
        else:
            # Fall back to manual configuration
            print("Using manual channel configuration")
            channel_range = [0, 64]  # Update these for your specific setup
            ref_path_measured_idx = 70
            ref_path_target_idx = 71
            bad_channels = []
            mat, config = loadEMG_updConfig(mat, config, channel_range, ref_path_target_idx, ref_path_measured_idx, bad_channels)

        neural_data = (
            torch.from_numpy(mat["emg"]).t().to(device=device, dtype=torch.float32)
        )  # time, channels
    elif path.suffix == ".npy":
        npy_data = np.load(path)
        neural_data = torch.from_numpy(npy_data).to(device=device, dtype=torch.float32)
    else:
        raise ValueError(
            "Data format not supported. Please provide data in .mat or .npy format."
        )
    start_index = int(config.start_time * sampling_frequency)
    end_index = int(config.end_time * sampling_frequency)
    neural_data = neural_data[start_index : end_index, : ]
#    if config.end_time == -1:
#        neural_data = neural_data[config.start_time * sampling_frequency : , :]
#    else:
#        neural_data = neural_data[config.start_time * sampling_frequency : config.end_time * sampling_frequency, :]

    # Initiate the model and run
    model = SwarmContrastiveDecomposition()
    predicted_timestamps, dictionary = model.run(neural_data, config)

    return dictionary, predicted_timestamps, mat, config


if __name__ == "__main__":
    # # Uncomment the next three lines to run in interactive window
    # import sys
    # sys.argv=['']
    # del sys

    HOME = Path.cwd().joinpath("data", "input")
    file_names = [f.name for f in HOME.iterdir() if f.is_file() and f.suffix == '.mat']
    if len(file_names) == 0:
        print(f'No .mat files in {HOME}')
    for file_name in file_names:
        #file_name = "emg"
        #path = HOME.joinpath(file_name).with_suffix(".npy")
        print(file_name)
        path = HOME.joinpath(file_name).with_suffix(".mat") # update HP
        output_path = (
	        Path(str(HOME).replace("input", "output"))
	        .joinpath(file_name)
	        .with_suffix(".pkl")
	    )
	
        dictionary, _, mat, config = train(path)
	
        save_results(output_path, dictionary)
        print(f"Saved results to {output_path}")

	    # Prepare Raw Data Info for openHDEMG
        rawEMG_Channels, refSignal, fsamp, ied, extras = extract_raw_emg_metadata(path, config)
        # Save decomposition result to openhdemg compressed json format
        export_to_openhdemg_json(config, output_path, rawEMG_Channels, refSignal, ied, fsamp, os.path.join(path), extras)
        # Save decomposition result to muEdit compatible .mat format for manual cleaning
        export_to_muedit_mat(
            str(output_path).replace('.pkl','.json')
        )
        
    print('--- ALL DONE ---')