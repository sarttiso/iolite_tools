import pandas as pd
import os
import radage

def match_prefix(spot_name, prefix_dict):
    """
    helper function for mapping sample prefixes to sample names, can be used with
    DataFrame.apply.
    Note that prefix_dict would need to be passed as a keyword argument when calling DataFrame.apply.
    """
    # default match is unknown
    match = 'unknown' 
    for key in prefix_dict:
        if key in spot_name:
            match = prefix_dict[key]
            break
    return match


def read_excel_files(files):
    """
    given list of paths to excel files output by Iolite 4, read them and make a dataframe.
    """
    dfs = []
    for file in files:
        cur_df = pd.read_excel(file, sheet_name='Data')
        cur_df['file'] = os.path.splitext(os.path.basename(file))[0]
        dfs.append(cur_df)

    df = pd.concat(dfs)
    
    return df


def get_ages(dfs):
    """
    produce radage.UPb age objects for each row in data files exported from Iolite 4
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