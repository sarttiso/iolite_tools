import pandas as pd
import os

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
        cur_df['file'] = os.path.basename(file).split('_')[0]
        dfs.append(cur_df)

    df = pd.concat(dfs)
    
    return df