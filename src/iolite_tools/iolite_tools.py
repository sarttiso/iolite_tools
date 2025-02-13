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

iolite2geochemdb_df = pd.read_csv(module_directory / 'iolite2geochemdb_dict.csv')

# dictionary linking units to type of measurement
unit2type_df = iolite2geochemdb_df[['type', 'unit']].drop_duplicates()
unit2type_dict = dict(zip(unit2type_df['unit'], unit2type_df['type']))

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
    # might be using a dash
    if len(iolite_spot_split) == 1:
        iolite_spot_split = iolite_spot.split('-')
    # if still 1, something is wrong
    assert len(iolite_spot_split) > 1, \
        'Spots should be either _ or - delimited.'

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
        df.rename({df.columns[0]: 'analysis'}, axis=1, inplace=True)

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
                        if col not in iolite2geochemdb_df['iolite'].tolist()]

        df.drop(columns=cols_to_drop, inplace=True)

        # measurement columns
        cols = list(df)

        # rename columns to be multiindex
        idx = np.array([np.argwhere(col == iolite2geochemdb_df['iolite'].values).squeeze() \
                        for col in cols])
        cols_new = pd.MultiIndex.from_arrays(
            [iolite2geochemdb_df.iloc[idx]['quantity'],
             iolite2geochemdb_df.iloc[idx]['unit']])

        # set new columns
        df = df.set_axis(cols_new, axis=1)

        dfs.append(df)

    # concatenate all dataframes
    df = pd.concat(dfs)

    return df


def aliquots2sql(df, material=''):
    """
    Generate a DataFrame of aliquots for use with geochemdb.measurements_add()

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
        Nu Plasma, Nu Plasma 3D, Agilent 7700x, Agilent 8900
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
    df = df.stack(level=0, future_stack=True).copy()
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
    for ii in df.index:
        cur_cols = df.loc[ii].dropna().index
        for col in cur_cols:
            cur_type = unit2type_dict[col]
            # update mean or uncertainty with value
            df_sql.loc[ii, cur_type] = df.loc[ii, col]
            # update measurement type
            df_sql.loc[ii, unitcol_dict[cur_type]] = col

    # set reference materials
    df_sql['reference_material'] = refmat

    # make index into columns
    df_sql.reset_index(inplace=True)

    return df_sql


def get_ages(df):
    """
    produce radage.UPb age objects from U-Pb measurements. assumes columns as 
    generated by excel2measurements, requires isotopic ratios with means and
    uncertainties, as well as error correlations
    """
    # extract only necessary columns
    df = df[['Pb206/U238', 
             'Pb207/U235',
            'Pb207/Pb206', 
             'rho 207Pb/206Pb v 238U/206Pb', 
             'rho 206Pb/238U v 207Pb/235U']].copy()
    
    # separate
    df68 = df['Pb206/U238'].rename(columns=unit2type_dict)
    df75 = df['Pb207/U235'].rename(columns=unit2type_dict)
    df76 = df['Pb207/Pb206'].rename(columns=unit2type_dict)
    df_rho68_75 = df['rho 206Pb/238U v 207Pb/235U'].rename(columns=unit2type_dict)
    df_rho76_86 = df['rho 207Pb/206Pb v 238U/206Pb'].rename(columns=unit2type_dict)
    
    ages = []
    for ii in range(df.shape[0]):
        ages.append(
            radage.UPb(df68.iloc[ii]['mean'],
                       df68.iloc[ii]['uncertainty'] / 2,
                       df75.iloc[ii]['mean'],
                       df75.iloc[ii]['uncertainty'] / 2,
                       df76.iloc[ii]['mean'],
                       df76.iloc[ii]['uncertainty'] / 2,
                       df_rho68_75.iloc[ii]['mean'],
                       df_rho76_86.iloc[ii]['mean'],
                       name=df.index[ii][0]))
    return ages


def propagate_standard_uncertainty():
    """
    for a dataframe export from iolite, propagate uncertainty into all observations such that each standard population has MSWD <= 1

    UNFINISHED
    """
    stand_strs = ['AusZ', 'GJ1', 'Plesovice', '9435', '91500', 'Temora']

    # files = glob.glob('exports/*run[0-9].xlsx')

    dfs = []
    for file in files:
        dfs.append(pd.read_excel(file, sheet_name='Data', index_col=0))

    # for each run, scale standard standard errors to enforce MSWD<=1
    for ii in range(len(files)):
        cur_scale = 1
        for jj in range(len(stand_strs)):
            # find standard analyses
            idx = dfs[ii].index.str.match(stand_strs[jj])
            curdat = dfs[ii][idx]
            mu = np.mean(curdat['Final Pb206/U238 age_mean'])
            mswd = np.sum((curdat['Final Pb206/U238 age_mean']-mu)**2 /
                          (curdat['Final Pb206/U238 age_2SE(prop)']/2*cur_scale)**2)/(np.sum(idx)-1)
            while mswd > 1:
                cur_scale = cur_scale + 0.01
                mswd = np.sum((curdat['Final Pb206/U238 age_mean']-mu)**2 /
                              (curdat['Final Pb206/U238 age_2SE(prop)']/2*cur_scale)**2)/(np.sum(idx)-1)
        # rescale all uncertainties
        dfs[ii][list(dfs[ii].filter(like='2SE'))] = cur_scale * dfs[ii][list(
            dfs[ii].filter(like='2SE'))]
        dfs[ii][list(dfs[ii].filter(like='2SD'))] = cur_scale * dfs[ii][list(
            dfs[ii].filter(like='2SD'))]

    # concatenate
    dat = pd.concat(dfs, axis=0)

    n_dat = len(dat)