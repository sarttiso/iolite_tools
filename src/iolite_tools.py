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

# dictionary linking units to type of measurement
unit2type_df = iolite2pygeo_df[['type', 'unit']].drop_duplicates()
unit2type_dict = dict(zip(unit2type_df['unit'], unit2type_df['type']))


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
        list of yyyy-mm's of the laser run date. used to label analysis and
        aliquot names

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
    excel_paths = np.atleast_1d(excel_paths)
    run_dates = np.atleast_1d(run_dates)
    run_numbers = np.atleast_1d(run_numbers)

    dfs = []
    for ii, excel_path in enumerate(excel_paths):
        # load dataframe
        df = pd.read_excel(excel_path, sheet_name='Data')

        # rename first column
        df.rename({'Unnamed: 0': 'analysis'}, axis=1, inplace=True)

        # save spots
        spot_names = df['analysis'].values

        # analyses are yyyy-mm_run_runtype_spot
        assert yyyymm_validator(
            run_dates[ii]), 'Run date must have yyyy-mm format.'
        df['analysis'] = f'{run_dates[ii]}_{run_numbers[ii]}_{run_type}_' + \
            df['analysis']

        # use spot names with run date and run number as aliquot names
        aliquots = [f'{run_dates[ii]}_{run_numbers[ii]}_' + spot
                    for spot in spot_names]

        # parse samples from spot names
        samples = [parsesample(spot) for spot in spot_names]

        # make multiindex analyses the index
        df.set_index(pd.MultiIndex.from_arrays([df['analysis'].values,
                                                aliquots,
                                                samples],
                                               names=['analysis',
                                                      'aliquot',
                                                      'sample']),
                     inplace=True)

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


def aliquots2sql(df, material=''):
    """
    Generate a DataFrame of aliquots for use with pygeodb.measurements_add()

    Parameters
    ----------
    df : pd.DataFrame
        ideally from excel2measurements, should have multiindex with
        'analysis', 'aliquot', and 'sample'
    material : str, optional
        Should match options in the Materials table.
        zircon, apatite, wholerock
        The default is ''.

    Returns
    -------
    df_sql : TYPE
        DESCRIPTION.

    """
    # make multiindex into columns
    df_sql = df.index.to_frame(index=False)

    # drop duplicates
    df_sql.drop_duplicates('analysis', inplace=True)

    # drop analysis
    df_sql.drop(columns=['analysis'], inplace=True)

    # add columns
    df_sql['material'] = material

    return df_sql


def analyses2sql(df, date='', instrument='', technique=''):
    """
    this function yields a dataframe that can be used to add/update analyses to
    the Analyses table in geochemdb.

    Parameters
    ----------
    df : pd.DataFrame
        ideally from excel2measurements, should have multiindex with
        'analysis', 'aliquot', and 'sample'
    date : str, optional
        yyyy-mm-dd of the analyses. The default is ''.
    instrument : str, optional
        should match options in Instruments table.
        Nu Plasma, Nu Plasma 3D, Agilent 7700x.
        The default is ''.
    technique : str, optional
        Should match options in the Techniques table.
        ICPMS, LASS ICPMS, ID-TIMS
        The default is ''.

    Returns
    -------
    df_sql : pd.DataFrame
        DataFrame ready for incorporation into Analyses table.

    """
    # make multiindex into columns
    df_sql = df.index.to_frame(index=False)

    # drop duplicates
    df_sql.drop_duplicates('analysis', inplace=True)

    # add columns
    df_sql['date'] = date
    df_sql['instrument'] = instrument
    df_sql['technique'] = technique

    return df_sql


def measurements2sql(df, refmat=''):
    """
    take a DataFrame generated by excel2measurements and put it into a format
    ready for adding/updating a Measurements table in geochemdb

    Parameters
    ----------
    df : pd.DataFrame
        dataframe generated by excel2measurements
    refmat : str
        name of the reference material used to reduce the data
    date : str
        datestring for the experiment start day (yyyy-mm-dd preferred)

    Returns
    -------
    df_sql : pd.DataFrame
        dataframe with top level measurement index stacked as a column and
        with columns to match Measurements table in geochemdb

    """
    # drop aliquot and sample from index
    df = df.droplevel(['aliquot', 'sample'])

    # make top level column index an index
    df = df.stack(level=0).copy()
    cols_df = list(df)

    n_rows = len(df)

    # each row should have at most two non-nan columns
    assert np.all((~df.isna()).sum(axis=1).values <= 2), \
        'too many values for measurement'

    # make df_sql
    cols_sql = ['mean',
                'measurement_unit',
                'uncertainty',
                'uncertainty_unit',
                'reference_material']
    df_sql = pd.DataFrame(index=df.index,
                          columns=cols_sql)

    unitcol_dict = {'mean': 'measurement_unit',
                    'uncertainty': 'uncertainty_unit'}

    # match mean and uncertainty columns
    for ii in range(n_rows):
        cur_idx = np.atleast_1d(np.argwhere(~df.iloc[ii].isna()).squeeze())
        for col_idx in cur_idx:
            cur_df_col = cols_df[col_idx]
            cur_type = unit2type_dict[cur_df_col]
            # update mean or uncertainty with value
            df_sql.iloc[ii][cur_type] = df.iloc[ii, col_idx]
            # update measurement type
            df_sql.iloc[ii][unitcol_dict[cur_type]] = cur_df_col

    # set reference materials
    df_sql['reference_material'] = refmat

    # make index into columns
    df_sql.reset_index(inplace=True)

    return df_sql


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
