"""Common utilities.

For some naming convention for pandas DataFrame,
We denote that:
* 'df' as an object of pandas DataFrame.
* 'col' as column name or column index.
* 'coldata' as column data.
* 'row' as row index or id.
* 'rowdata' as row data.

For example:
```
df = pandas.DataFrame({
    'Sex': ['female', 'male', 'female'],
    'Name': ['Alice', 'Bob', 'Mary']
})
col = 'Sex'
coldata = df[col]
'''
coldata is an object of pandas Series
containing data like ['female', 'male', 'female']
'''
row = 0
rowdata = df.iloc[row]
'''
rowdata is an object of
pandas Series containing data like {'Name': 'Alice', 'Sex': 'female'}
'''
```
"""

import os
import types
import inspect
from datetime import datetime
import numpy
import pandas
from .const import Constant

constants = Constant(
    src_col='src_col',
    new_col='new_col',
    map_data='map_data'
)

def is_class(arg):
    """Check if the argument is a Class.

    Args:
        arg: Any data type.

    Returns:
        A bool. True for arg is a Class
        otherwise False.
    """
    return inspect.isclass(arg)

def is_func(arg):
    """Check if the argument is a function/method or not.

    Args:
        arg: Any data type.

    Returns:
        A bool. True for arg is a function or method
        otherwise False.
    """
    return isinstance(
        arg,
        (
            types.FunctionType,
            types.LambdaType,
            types.MethodType,
            types.BuiltinFunctionType,
            types.BuiltinMethodType,
        ),
    )

def is_strlist(arg):
    """Check if the argument is a list of string.

    Args:
        arg: Any data type.

    Returns:
        A bool. True for arg is a list of string
        otherwise False.
    """
    if isinstance(arg, list) and arg:
        # WARNING:
        # all([]) and all(()) and all({}) all return True.
        return all([isinstance(s, str) for s in arg])
    return False

def get_isotime(prefix='', suffix='', unit='seconds', raw=False):
    """Get iso8601 time format for the current time.

    Args:
        prefix: A prefix string prepending to time string.
        suffix: A suffix string appending to time string.
        unit: A string for time unit.
        raw: A flag to decide whether we replace '-' & ':' with '_'.
            For example, if raw = True, the result is '2018-04-05T11:22:33'.
            if raw = False, the result is '2018_04_05T11_22_33'

    Returns:
        A str means the current time with iso8601 format.
    """
    s = datetime.now().isoformat(timespec=unit)
    if not raw:
        s = s.replace('-', '_').replace(':', '_')
    return '{}{}{}'.format(prefix, s, suffix)

def print_df_summary(df, title=None):
    """Print a custom summary for pandas DataFrame.

    Args:
        df: An object of pandas DataFrame.
        title: A string representing the summary title.

    Returns:
        None
    """
    if isinstance(title, str) and title.strip():
        print('\n{} =========='.format(title.strip()))

    print('\n# Info')
    print(df.info())

    print('\n# Describe')
    print(df.describe())

    no_null_cols = True
    num_of_rows = len(df)
    for col in df:
        num_of_nulls = len(df[col][df[col].isnull()])
        if num_of_nulls != 0:
            if no_null_cols:
                no_null_cols = False
                print('\n# Null Summary')
            percent = num_of_nulls / num_of_rows * 100
            print('* {:s}: {:d} nulls ({:.2f}%)'.format(col, num_of_nulls, percent))

    if no_null_cols:
        print('All data is not null.')

    print('\n# First 3 data')
    print(df.head(3))

def get_df_with_target_cols(df, cols):
    """Get an object of pandas DataFrame with columns you want.

    Abbreviations:
        cols: column names

    Args:
        df: An object of pandas DataFrame.
        cols: A list of strings. Every string in this list
            represents a column name in df.

    Returns:
        An object of pandas DataFrame containing columns
        with column name in cols.

    Raises:
        TypeError: An error occurred when cols is not a list of strings.
    """
    if is_strlist(cols):
        return df[cols]
    else:
        raise TypeError('cols should be a list of strings.')

def get_series(df, col):
    """Get an object of pandas Series corresponding to a column in df.

    Args:
        df: An object of pandas DataFrame.
        col: A string representing a column name in df.

    Returns:
        An object of pandas Series.

    Raises:
        TypeError: An error occurred when col is not a string.
    """
    if isinstance(col, str):
        return df[col]
    else:
        raise TypeError('col should be a string.')

def get_traintest_with_target_feats(df_train, df_test, feat_cols, label_col):
    """Get train & test data with target features.

    Args:
        df_train: A object of pandas DataFrame representing ML train data.
        df_test: A object of pandas DataFrame representing ML test data.
        feat_cols: A list of strings for feature columns you want.
        label_col: A string representing the label column name in df_train.

    Returns:
        A tuple: (df_train, y_train, df_test) where
            df_train: An object of pandas DataFrame representing
                the train data with target features.
            y_train: An object of pandas Series representing
                the label column in the train data.
            df_test: An object of pandas DataFrame representing
                the test data with target features.
    """
    return (
        get_df_with_target_cols(df_train, feat_cols),
        get_series(df_train, label_col),
        get_df_with_target_cols(df_test, feat_cols),
    )


def is_valid_simple_map(simple_map):
    """Check if simple_map format is right.

    Args:
        simple_map: A dict with 3 keys
            including 'src_col', 'new_col', 'map_data'.

    Examples:
        simple_map = {
            'src_col': 'Sex',
            'new_col': 'derived_Sex',
            'map_data': {
                'male': 0,
                'female': 1,
            },
        }

    Returns:
        A bool. True for simple_map format is valid otherwise False.
    """
    src_col = constants.src_col
    new_col = constants.new_col
    map_data = constants.map_data
    if not isinstance(simple_map, dict):
        return False
    if (src_col not in simple_map or
            new_col not in simple_map or
            map_data not in simple_map):
        return False
    if (not isinstance(simple_map[src_col], str) or
            not isinstance(simple_map[new_col], str) or
            not isinstance(simple_map[map_data], dict)):
        return False
    return True

def update_df(df_train, df_test, simple_map=None, custom_update=None):
    """Update pandas DataFrame

    Args:
        df_train: An object of pandas DataFrame representing ML train data.
        df_test: An object of pandas DataFrame representing ML test data.
        simple_map: A dict, please refer to is_valid_simple_map
        custom_update: A function which args are df_train, df_test and
            returns modified df_train, df_test

    Returns:
        A tuple: (df_train, df_test) where
            df_train: An object of pandas DataFrame representin the train data.
            df_test: An object of pandas DataFrame representing the test data.
    """
    df_train = df_train.copy()
    df_test = df_test.copy()
    if is_func(custom_update):
        df_train, df_test = custom_update(df_train, df_test)
    elif is_valid_simple_map(simple_map):
        src_col = simple_map[constants.src_col]
        new_col = simple_map[constants.new_col]
        map_data = simple_map[constants.map_data]
        df_train[new_col] = df_train[src_col].map(map_data)
        df_test[new_col] = df_test[src_col].map(map_data)
        """For example:
        df_train['derived_Sex'] = df_train['Sex'].map({'male': 0, 'female': 1})
        """
    else:
        raise TypeError(
            'simple_map and custom_update can not be invalid at the same time.'
        )
    return (df_train, df_test)

def get_df_prds(id_col, ids, label_col, prds, to_int=True):
    """Get a DataFrame of predicted data.

    Abbreviations:
        prds: predictions

    Args:
        id_col: A str representing the id column name.
        ids: An array-like object (ex: a python built-in list or
            a pandas Series) representing ids of the test data.
        label_col: A str representing the label column name.
        prds: An array-like object (ex: a python built-in list or
            a pandas Series) representing the predicted results of
            the test data.
        to_int: A bool

    Returns:
        An object of pandas DataFrame representing the predicted data.
    """
    if not isinstance(id_col, str):
        raise TypeError('\'id_col\' should be a str.')
    if not hasattr(ids, '__iter__'):
        raise TypeError('\'ids\' should be an iterable object.')
    if not isinstance(label_col, str):
        raise TypeError('\'label_col\' should be a str.')
    if not hasattr(prds, '__iter__'):
        raise TypeError('\'prds\' should be an iterable object.')
    if not isinstance(to_int, bool):
        raise TypeError('\'to_int\' should be a bool.')
    if to_int:
        if isinstance(prds, (pandas.Series, numpy.ndarray)):
            prds = prds.astype(numpy.int32)
        else:
            prds = [int(v) for v in prds]

    return pandas.DataFrame({
        id_col: ids,
        label_col: prds,
    })

def write_sbmcsv(
        df_prds,
        abs_dir_path,
        title='submission',
        enable_time=True,
    ):
    """Output a csv file for kaggle submission

    Abbreviations:
        sbmcsv: submission csv file

    Args:
        df_prds: An object of pandas DataFrame representing the predicted data.
        abs_dir_path: A str representing the absolute path of the directory
            in which the csv file is placed.
        title: A str representing the file title.
        enable_time: A bool. True for time on csv file name otherwise False.

    IOs:
        Output a csv file.

    Returns:
        A tuple: (file_name, abs_file_path) where
            file_name: The file name of the output csv file.
            abs_file_path: The absolute file path of the output csv file.
    """
    time = get_isotime(prefix='_') if enable_time else ''
    file_name = '{}{}.csv'.format(title, time)
    abs_file_path = os.path.join(abs_dir_path, file_name)
    df_prds.to_csv(abs_file_path, index=False)
    return (file_name, abs_file_path)
