'''
forestplot.py

Quick tool for making forest plots.

ngr.200408
'''
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_df(filename):
    return pd.read_csv(filename, index_col=0)


def make_fplot():
    """Makes a forest plot from dataframe with odds ratios, CIs, and variables.

    Typically feed a dataframe with
    cols=['variable','odds_ratio','lower_ci','upper_ci'] and rows specifying
    values for variables as "observations." Capitalizes first letter of a
    variable and removes undersscores and "cc" prefixes.

    Arg:
        df (pd.DataFrame): dataframe n_variables x 3, if df.index houses variable
            names, else n_variables x 4, specifying odds_ratio, confidence per
            variable.
        save (str): filename with extension (matplotlib compatible) to save
            figure to.
        odds_ratio (str, default='odds_ratio'): column name housing odds_ratio
        lower_ci (str, default='lower_ci'): column name housing CI/variable
        upper_ci (str, default='upper_ci'): column name housing CI/variable
        vars (str, default=None): if None, assumes variable names are in index;
            otherwise, specify column name housing variable names (observations).

    Returns:
        None. Displays a plot for %matplotlib inline
    """
