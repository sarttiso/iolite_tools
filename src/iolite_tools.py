import numpy as np
import pandas as pd
import os
from pathlib import Path
import radage
import re

# Get the path of the module file
module_file_path = os.path.abspath(__file__)

# Get the directory containing the module file
module_directory = Path(os.path.dirname(module_file_path))

iolite2pygeo_df = pd.read_csv(module_directory / 'iolite2pygeodb_dict.csv')


class Iolite2DB:
    """
    class for piping iolite data into pygeodb
    """

    def __init__(self):
        return


def parsesample(iolite_spot):
    """
    Take an iolite spot string and return the sample part of the name

    Parameters
    ----------
    iolite_spot : str
        Spot name in iolite.

    Returns
    -------
    sample_name : str
        sample name
    """
    # count number of underscores
    iolite_spot_split = iolite_spot.split('_')
    n_under = len(iolite_spot_split)

    # likely a standard
    if n_under == 2:
        sample_name = iolite_spot_split[0]
    # likely an unknown
    elif n_under >= 3:
        sample_name = iolite_spot_split[0:-2]

        # join with spaces
        sample_name = ' '.join(sample_name)

    return sample_name


def yyyymm_validator(test_str):
    """
    validate a string in a yyyy-mm format

    Parameters
    ----------
    test_str : str
        string to validate.

    Returns
    -------
    bool
        valid or not

    """
    pattern = r'^\d{4}-\d{2}$'
    return bool(re.match(pattern, test_str))


def match_prefix(spot_name, prefix_dict):
    """
    helper function for mapping sample prefixes to sample names, can be used
    with DataFrame.apply.
    Note that prefix_dict would need to be passed as a keyword argument when
    calling DataFrame.apply.
    """
    # default match is unknown
    match = 'unknown'
    for key in prefix_dict:
        if key in spot_name:
            match = prefix_dict[key]
            break
    return match


def read_excel_files(files, prefix_dict=None):
    """
    given list of paths to excel files output by Iolite 4, read them and make
    a dataframe.
    """
    dfs = []
    for file in files:
        cur_df = pd.read_excel(file, sheet_name='Data')
        cur_df['file'] = os.path.splitext(os.path.basename(file))[0]
        dfs.append(cur_df)

    df = pd.concat(dfs)

    # rename first column to be 'spot'
    df.rename({'Unnamed: 0': 'spot'}, axis=1, inplace=True)

    # drop empty columns (other Unnamed columns)
    idx_drop = np.atleast_1d(np.argwhere(
        ['Unnamed:' in x for x in list(df)]).squeeze())
    cols_drop = [list(df)[x] for x in idx_drop]
    # print(idx_drop)
    df.drop(cols_drop, axis=1, inplace=True)

    if prefix_dict is not None:
        df['sample'] = df['spot'].apply(match_prefix, prefix_dict=prefix_dict)

    return df


def excel2measurements(excel_paths, run_dates, run_numbers, run_type):
    """
    prepare a DataFrame of geochemical measurements for each analysis; suitable
    for updating geochemdb.Measurements after transformation to rows.

    Parameters
    ----------
    excel_paths : arraylike
        Paths to excel files with geochemical data exported from iolite4

    run_date : arraylike
        list of yyyy-mm's of the laser run date

    run_number : arraylike
        integer of the the run number

    run_type : str
        for example, U-Pb, trace, Hf; type of the run for which data was
        reduced. suffixed to 

    Returns
    -------
    df : pd.DataFrame
        DataFrame of measurements  with measurement units and uncertainties in
        columns

    """
    dfs = []
    for ii, excel_path in enumerate(excel_paths):
        # load dataframe
        df = pd.read_excel(excel_path, sheet_name='Data')

        # rename first column
        df.rename({'Unnamed: 0': 'analysis'}, axis=1, inplace=True)

        # rename spots yyyy-mm_run#_spot
        assert yyyymm_validator(
            run_dates[ii]), 'Run date must have yyyy-mm format.'
        df['analysis'] = f'{run_dates[ii]}_{run_numbers[ii]}_{run_type}_' + \
            df['analysis']

        # make spot index
        df.set_index('analysis', inplace=True)

        # keep only measurement columns
        cols_to_drop = [col for col in list(df)
                        if col not in iolite2pygeo_df['iolite'].tolist()]

        df.drop(columns=cols_to_drop, inplace=True)

        # measurement columns
        cols = list(df)

        # rename columns to be multiindex
        idx = np.array([x in cols for x in iolite2pygeo_df['iolite'].values])
        cols_new = pd.MultiIndex.from_arrays(
            [iolite2pygeo_df.iloc[idx]['quantity'],
             iolite2pygeo_df.iloc[idx]['unit']])

        # set new columns
        df = df.set_axis(cols_new, axis=1)

        dfs.append(df)

    # concatenate all dataframes
    df = pd.concat(dfs)

    return df


def get_ages(dfs):
    """
    produce radage.UPb age objects for each row in data files exported from
    Iolite 4
    """
    # cols
    cols = [
        'Final Pb206/U238_mean', 'Final Pb206/U238_2SE(prop)',
        'Final Pb207/U235_mean', 'Final Pb207/U235_2SE(prop)',
        'Final Pb207/Pb206_mean', 'Final Pb207/Pb206_2SE(prop)',
        'rho 206Pb/238U v 207Pb/235U', 'rho 207Pb/206Pb v 238U/206Pb'
    ]

    ages = []
    for df in dfs:
        for ii in range(df.shape[0]):
            ages.append(
                radage.UPb(df.iloc[ii][cols[0]],
                           df.iloc[ii][cols[1]] / 2,
                           df.iloc[ii][cols[2]],
                           df.iloc[ii][cols[3]] / 2,
                           df.iloc[ii][cols[4]],
                           df.iloc[ii][cols[5]] / 2,
                           df.iloc[ii][cols[6]],
                           df.iloc[ii][cols[7]],
                           name=df.iloc[ii, 0]))
    return ages
