"""Functions for exporting decomposition outputs"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
import openhdemg.library as emg

def allocate_openhdemg_file_structure():
    """
    Allocates and returns a dictionary of the openhdemg file structure.
    
    This file structure is used for storing high-density surface electromyography (HD-sEMG) data
    decomposed into single motor unit firings. The dictionary contains various types of data
    including raw HDEMG signals, reference signals, accuracy metrics, and more.

    The dictionary contains the following keys:
    - 'SOURCE': str, any length. Context: Decomposition tool used.
    - 'FILENAME': str, any length. Context: Filename of the data file that was decomposed.
    - 'RAW_SIGNAL': pandas DataFrame, each with shape (nSamp, nCh). Context: Raw sEMG signals.
    - 'REF_SIGNAL': pandas DataFrame, each with shape (nSamp, 1). Context: Reference force/torque signal.
    - 'ACCURACY': pandas DataFrame, each with shape (nMU, 1). Context: Accuracy of motor unit decomposition, i.e., SIL.
    - 'IPTS': pandas DataFrame, each with shape (nSamp, nMU). Context: Motor unit spike trains.
    - 'MUPULSES': list of numpy arrays, each entry a numpy array of int32 with shape (nFirings,). Context: Motor unit pulse timings as indices.
    - 'FSAMP': float, each with one value. Context: sEMG sampling frequency [Hz].
    - 'IED': float, each with one value. Context: Inter-electrode distance [mm].
    - 'EMG_LENGTH': int, each with one value corresponding to nSamp. Context: Length of the EMG signal.
    - 'NUMBER_OF_MUS': int, each with one value corresponding to nMU. Context: Number of motor units.
    - 'BINARY_MUS_FIRING': pandas DataFrame, each with shape (nSamp, nMU). Context: Binary matrix of motor unit firings.
    - 'EXTRAS': pandas DataFrame, each with shape (nExtras, 1). Context: Any additional information.
    (nSamp ... number of samples, nMU ... number of motor units, nCH ... number of sEMG channels)
    
    Returns:
        dict: A dictionary with the specified structure and empty lists for each key.
        
    by Harald Penasso 2024-11 (with some help of ChatGPT 4o)
    """
    decompfile = {
        'SOURCE': [],  # str (any length)
        'FILENAME': [],  # str (any length)
        'RAW_SIGNAL': [],  # DataFrame (nSamp, nCh)
        'REF_SIGNAL': [],  # DataFrame (nSamp, 1)
        'ACCURACY': [],  # DataFrame (nMU, 1)
        'IPTS': [],  # DataFrame (nSamp, nMU)
        'MUPULSES': [],  # list nMU (each entry a numpy array of int32 (nFirings,))
        'FSAMP': [],  # float 1
        'IED': [],  # float 1
        'EMG_LENGTH': [],  # int 1 (one value corresponding to nSamp)
        'NUMBER_OF_MUS': [],  # int 1 (one value corresponding to nMU)
        'BINARY_MUS_FIRING': [],  # DataFrame (nSamp, nMU)
        'EXTRAS': []  # DataFrame (nExtras, 1)
    }
    
    return decompfile

def export_to_openhdemg_json(config, out_path, rawEMG_Channels, refSignal, ied, fsamp, fn, extras = '', device_from = 'OTB'):
    """
    Export the decomposition results to the OpenHD-EMG JSON format.
    
    This function converts the decomposition outputs from the Swarm-Contrastive Decomposition pipeline 
    into the standardized OpenHD-EMG format for further analysis.

    Args:
        config (Config): Configuration object containing settings like start_time, sampling_frequency, etc.
        out_path (str): Path to the `.pkl` file containing decomposition results.
        rawEMG_Channels (pd.DataFrame): EMG channel signals extracted from the dataset.
        refSignal (pd.DataFrame): The reference signal, typically force or torque.
        ied (float): Inter-electrode distance in mm.
        fsamp (float): Sampling frequency in Hz.
        fn (str): Filename of the processed dataset.
        extras (str, optional): Additional metadata. Defaults to an empty string.
        device_from (str, optional): Source identifier for the data. Defaults to 'OTB'.

    Returns:
        None: Saves the data in JSON format.

    Raises:
        FileNotFoundError: If `out_path` is missing.
        ValueError: If unexpected data formats are encountered.

    Example:
        >>> export_to_openhdemg_json(config, 'output.pkl', rawEMG_Channels, refSignal, 5, 2000, 'subject1')
        Saved results to output.json in OpenHD-EMG compressed format.
    """

    import openhdemg.library as emg
    
    # Load decomposition results
    try:
        with open(out_path, 'rb') as f:
            decomp = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Decomposition results file not found: {out_path}")
    
    # Allocate structure
    decompfile = allocate_openhdemg_file_structure()

    # Populate structured dictionary
    decompfile['SOURCE'] = device_from
    decompfile['FILENAME'] = fn
    decompfile['RAW_SIGNAL'] = rawEMG_Channels
    decompfile['REF_SIGNAL'] = refSignal
    decompfile['ACCURACY'] = pd.DataFrame([sil.cpu().numpy() for sil in decomp['silhouettes']])
    decompfile['IPTS'] = pd.DataFrame(np.array(decomp['source'])[:, :, 0].T)
    decompfile['MUPULSES'] = [np.array(mup.cpu(), dtype='int32') for mup in decomp['timestamps']]
    decompfile['FSAMP'] = fsamp
    decompfile['IED'] = ied
    decompfile['EMG_LENGTH'] = decompfile['IPTS'].shape[0]
    decompfile['NUMBER_OF_MUS'] = len(decomp['silhouettes'])

    # Create binary firing matrix
    spikeMat = np.zeros((decompfile['NUMBER_OF_MUS'], decompfile['EMG_LENGTH']))
    for i in range(decompfile['NUMBER_OF_MUS']):
        spikeMat[i, decompfile['MUPULSES'][i]] = 1

    decompfile['BINARY_MUS_FIRING'] = pd.DataFrame(spikeMat.T)
    decompfile['EXTRAS'] = extras

    # Save the file
    output_json_path = Path(str(out_path).replace('.pkl', '.json'))
    emg.save_json_emgfile(decompfile, output_json_path, compresslevel=4)
    print(f"Saved results to {output_json_path} in OpenHD-EMG compressed JSON format.")

def allocate_muedit_file_structure():
    """
    Minimal muedit structure: only 'signal' and 'parameters' with required fields.

    Returns
    -------
    dict
        {'signal': {required minimal keys for muedit}, 'parameters': {source file path and name info}}
    """
    signal = {
        # required
        'data': np.empty((0, 0), dtype=float),     # (nb_channels x samples)  -- EMG only
        'fsamp': float('nan'),
        'nChan': 0.0,
        'ngrid': 0.0,
        'gridname': np.empty((1, 0), dtype=object),    # 1 x ngrid cell row
        'muscle': np.empty((1, 0), dtype=object),      # 1 x ngrid cell row
        'Pulsetrain': np.empty((1, 0), dtype=object),  # 1 x ngrid cell row; each cell: (nbMU_i x time)
        'Dischargetimes': np.empty((0, 0), dtype=object),  # ngrid x maxMU
        'path': np.empty((0,), dtype=float),           # 1 x n_samples double: produced path
        'target': np.empty((0,), dtype=float),         # 1 x n_samples double: target path
        'coordinates': [],                             # 1 x ngrid cell; each cell: [n_grid_channels x 2] double (row, col) 1-based indices
        'IED': np.empty((0,), dtype=float),            # 1 x ngrid double: IED (mm)
        'EMGmask': [],                                 # 1 x ngrid cell; each cell: [n_grid_channels x 1] double (0=keep, 1=discard): select channels
        'emgtype': [],                                 # 1 x ngrid double; each entry 1 for surface EMG (per manual)
    }
    params = {
        'pathname': '',                # char
        'filename': '',                # char
        }
    
    return {'signal': signal, 'parameters': params}

def export_to_muedit_mat(json_load_filepath, ngrid = 1):
    json_from_openhdemg = emg.emg_from_json(json_load_filepath)
    mat_save_filepath = json_load_filepath.replace(".json","_muedit.mat")
    nMU = json_from_openhdemg["IPTS"].shape[1]
    nCH = json_from_openhdemg["RAW_SIGNAL"].shape[1]
    dict_for_muedit = allocate_muedit_file_structure()
    dict_for_muedit["signal"]["data"] = np.transpose(json_from_openhdemg["RAW_SIGNAL"].to_numpy())
    dict_for_muedit["signal"]["fsamp"] = json_from_openhdemg["FSAMP"]
    dict_for_muedit["signal"]["nChan"] = nCH
    dict_for_muedit["signal"]["ngrid"] = ngrid # so far we support only one grid at-a-time decomposition and cleaning, doublicate removal is done in a seperate step
    dict_for_muedit["signal"]["gridname"] = np.array([[str(json_from_openhdemg["EXTRAS"].loc[0.0]).split(" - ")[-1].split(' ')[0].replace('HD','GR')]], dtype=object)
    dict_for_muedit["signal"]["muscle"] = np.array([[str(json_from_openhdemg["EXTRAS"].loc[0.0]).split(" - ")[0][1:].strip()]], dtype=object)
    # Build 1 x ngrid cell array for MATLAB
    pulsetrain_cell = np.empty((1, ngrid), dtype=object)
    # Each cell must be nMU x n_samples (double)
    ipts = json_from_openhdemg["IPTS"].to_numpy(dtype=np.float64, copy=True).T
    pulsetrain_cell[0, 0] =  ipts/ipts.max() # normalize for MUEdit
    # Assign into your struct and save
    dict_for_muedit["signal"]["Pulsetrain"] = pulsetrain_cell
    
    # Build ngrid x nMU MATLAB cell
    discharges_cell = np.empty((ngrid, nMU), dtype=object)
    # Fill row g=0 (your single grid) with 1 x n_i doubles
    for mu in range(nMU):
        seq = json_from_openhdemg["MUPULSES"][mu]+1 # +1 for 1-padded Matlab
        # make sure it's a 1 x n_i row vector of doubles
        arr = np.asarray(seq, dtype=np.float64).reshape(1, -1)
        discharges_cell[0, mu] = arr
    dict_for_muedit["signal"]["Dischargetimes"] = discharges_cell
    
    dict_for_muedit['signal']['IED'] = json_from_openhdemg["IED"]
    dict_for_muedit['signal']['target'] = np.transpose(json_from_openhdemg["REF_SIGNAL"]) # dirty fix, we'd have to take this from somewhere else or parr it directly
    dict_for_muedit['signal']['path'] = np.transpose(json_from_openhdemg["REF_SIGNAL"])
    dict_for_muedit['signal']['emgtype'] = np.ones((1,ngrid))
    
    # Build 1 x ngrid MATLAB cell
    bad_channel_bool = np.empty((1,ngrid), dtype=object)

    # Extract bad_channels from EXTRAS if available
    bad_channels = []
    try:
        if 'EXTRAS' in json_from_openhdemg and len(json_from_openhdemg['EXTRAS']) > 2:
            # Third entry in EXTRAS contains bad_channels info
            bad_channels_str = str(json_from_openhdemg['EXTRAS'].loc[2.0])
            if 'bad_channels=' in bad_channels_str:
                # Parse the bad_channels list from the string
                import ast
                bad_channels_list_str = bad_channels_str.split('bad_channels=')[1].strip()
                bad_channels = ast.literal_eval(bad_channels_list_str)
                print(f"  Loaded bad_channels from EXTRAS: {bad_channels}")
    except Exception as e:
        print(f"  Could not parse bad_channels from EXTRAS: {e}")

    # Create EMGmask: 0 = keep channel, 1 = discard channel
    emg_mask = np.zeros((nCH, 1))
    for bad_ch_idx in bad_channels:
        if 0 <= bad_ch_idx < nCH:
            emg_mask[bad_ch_idx, 0] = 1

    bad_channel_bool[0, 0] = emg_mask
    dict_for_muedit['signal']['EMGmask'] = bad_channel_bool
    
    # parameters
    dict_for_muedit['parameters']['pathname']   = str(Path(json_load_filepath).parent)
    dict_for_muedit['parameters']['filename']   = str(Path(json_load_filepath).name)
    
    sio.savemat(mat_save_filepath, dict_for_muedit, do_compression=True, long_field_names=True)
    print(f'Saved for cleaning in MUEdit: {mat_save_filepath}')