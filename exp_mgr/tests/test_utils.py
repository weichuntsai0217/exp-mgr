from unittest import mock
import pytest
import numpy
import pandas
from exp_mgr import Constant
from exp_mgr import utils

def get_consts():
    def fn():
        pass

    class Cls(object):
        @staticmethod
        def static_method():
            pass

        def method():
            pass

    obj = Cls()

    targets = [
        None,
        True,
        False,
        1,
        1.1,
        'string',
        list(),
        tuple(),
        dict(),
        set(),
        obj,
        list,
        Cls,
        all,
        fn,
        lambda x: x,
        Cls.static_method,
        obj.method,
    ]

    df = pandas.DataFrame({
        'id': [1, 2, 3],
        'Name': ['Alice', 'Bob', 'Mary'],
        'Sex': ['female', 'male', 'female'],
        'Sales': [300, 100, 200],
        'Cost': [100, 30, 150],
        'Label': [10.1, 11.1, 12.1],
    })

    simple_map = {
        'src_col': 'Sex',
        'new_col': 'derived_Sex',
        'map_data': {
            'male': 0,
            'female': 1,
        },
    }

    def custom_update(df_train, df_test):
        df_train['Profit'] = df_train['Sales'] - df_train['Cost']
        df_test['Profit'] = df_test['Sales'] - df_test['Cost']
        return (df_train, df_test)

    series_for_simple_map = pandas.Series([1, 0, 1])
    series_for_custom_update = pandas.Series([200, 70, 50])

    return Constant(
        fn=fn,
        cls=Cls,
        obj=obj,
        targets=targets,
        df=df,
        simple_map=simple_map,
        custom_update=custom_update,
        series_for_simple_map=series_for_simple_map,
        series_for_custom_update=series_for_custom_update,
    )

consts = get_consts()

def assert_all_true(arg1, arg2):
    if isinstance(arg1, list) and isinstance(arg2, list):
        assert (arg1 == arg2) == True
    else:
        assert all(arg1 == arg2) == True

def compare_values(func, answers):
    for i, val in enumerate(consts.targets):
        print(i)
        assert func(val) == answers[i]

# Test start
def test_is_class():
    compare_values(
        utils.is_class,
        [True if i in [11, 12] else False for i in range(len(consts.targets))]
    )


def test_is_func():
    compare_values(
        utils.is_func,
        [True if i >= 13 else False for i in range(len(consts.targets))]
    )

def test_is_strlist():
    right_list = ['aa', '', 'bb', 'cc']
    wrong_list = ['aa', 1, True, 'cc']
    assert utils.is_strlist(right_list) == True
    assert utils.is_strlist(wrong_list) == False
    compare_values(
        utils.is_strlist,
        [False]*len(consts.targets)
    )

@pytest.mark.parametrize('args, mock_return, expected', [
    (
        {},
        '2018-01-23T22:33:44',
        '2018_01_23T22_33_44',
    ),
    (
        {'prefix': '^', 'suffix': '$'},
        '2018-01-23T22:33:44',
        '^2018_01_23T22_33_44$',
    ),
    (
        {'prefix': '^', 'suffix': '$', 'raw': True},
        '2018-01-23T22:33:44',
        '^2018-01-23T22:33:44$',
    ),
    (
        {'prefix': '^', 'suffix': '$', 'unit': 'milliseconds'},
        '2018-01-23T22:33:44.789',
        '^2018_01_23T22_33_44.789$',
    ),
])
@mock.patch('exp_mgr.utils.datetime')
def test_get_isotime(mock_datetime, args, mock_return, expected):
    def get_mocks(mock_datetime):
        mock_obj = mock.MagicMock()
        mock_obj.isoformat = mock.MagicMock(return_value=mock_return)
        mock_datetime.now = mock.MagicMock(return_value=mock_obj)
        return mock_datetime, mock_obj 
    mock_datetime, mock_obj = get_mocks(mock_datetime)
    res = utils.get_isotime(**args)
    assert mock_obj.isoformat.call_count == 1
    if 'unit' in args:
        mock_obj.isoformat.assert_called_with(timespec=args['unit'])
    else:
        mock_obj.isoformat.assert_called_with(timespec='seconds')
    assert res == expected

@pytest.mark.parametrize('args, should_return, error', [
    (['Name', 'Sex'], True, None),
    (['Sales', 'Cost'], True, None),
    (['Sales', 'a'], False, KeyError),
    (['Sales', 1], False, TypeError),
    ('Sales', False, TypeError),
])
def test_get_df_with_target_cols(args, should_return, error):
    df = consts.df
    if should_return:
        filtered_df = utils.get_df_with_target_cols(df, args)
        res = [key for key, val in filtered_df.items()]
        assert res == args
    else:
        with pytest.raises(error):
            filtered_df = utils.get_df_with_target_cols(df, args)

@pytest.mark.parametrize('args, should_return, error', [
    ('Name', True, None),
    ('Sex', True, None),
    ('a', False, KeyError),
    (1, False, TypeError),
    (['a'], False, TypeError),
])
def test_get_series(args, should_return, error):
    df = consts.df
    if should_return:
        res = utils.get_series(df, args)
        assert_all_true(res, df[args])
    else:
        with pytest.raises(error):
            filtered_df = utils.get_series(df, args)

@pytest.mark.parametrize('feat_cols, label_col', [
    (['Name', 'Sales'], 'Cost'),
    (['Name', 'Sales', 'Cost'], 'Sex'),
])
def test_get_traintest_with_target_feats(feat_cols, label_col):
    df_train = consts.df.copy()
    df_test = consts.df.copy()
    df_train, y_train, df_test = utils.get_traintest_with_target_feats(
       df_train, df_test, feat_cols, label_col 
    )
    new_df_train_keys = [key for key, val in df_train.items()]
    new_df_test_keys = [key for key, val in df_test.items()]
    assert new_df_train_keys == feat_cols
    assert_all_true(y_train, consts.df[label_col])
    assert new_df_test_keys == feat_cols

@pytest.mark.parametrize('args, expected', [
    (
        consts.simple_map,
        True,
    ),
    (
        {
            'src_col': 'Sex',
            'map_data': {
                'male': 0,
                'female': 1,
            },
        },
        False,
    ),
    (
        {
            'src_col': [],
            'new_col': 'derived_Sex',
            'map_data': {
                'male': 0,
                'female': 1,
            },
        },
        False,
    ),
    (
        {
            'src_col': 'Sex',
            'new_col': 'derived_Sex',
            'map_data': True,
        },
        False,
    ),
])
def test_is_valid_simple_map(args, expected):
    assert utils.is_valid_simple_map(args) == expected

@pytest.mark.parametrize(
    'args, expected, should_return',
    [
        (
            {'simple_map': consts.simple_map},
            {
                'new_col': 'derived_Sex',
                'res': consts.series_for_simple_map
            },
            True
        ),
        (
            {'custom_update': consts.custom_update},
            {
                'new_col': 'Profit',
                'res': consts.series_for_custom_update
            },
            True
        ),
        (
            {
                'simple_map': consts.simple_map,
                'custom_update': consts.custom_update
            },
            {
                'new_col': 'Profit',
                'res': consts.series_for_custom_update
            },
            True
        ),
        (
            {
                'simple_map': consts.simple_map,
                'custom_update': 'a'
            },
            {
                'new_col': 'derived_Sex',
                'res': consts.series_for_simple_map,
            },
            True
        ),
        (
            {},
            {},
            False
        ),
        (
            {'simple_map': 'a'},
            {},
            False
        ),
        (
            {'custom_update': 'a'},
            {},
            False
        )
    ]
)
def test_update_df(args, expected, should_return):
    df_train = consts.df.copy()
    df_test = consts.df.copy()
    if should_return:
        df_train, df_test = utils.update_df(df_train, df_test, **args)
        new_col = expected['new_col']
        assert (new_col in df_train) == True
        assert (new_col in df_test) == True
        assert_all_true(df_train[new_col], expected['res'])
        assert_all_true(df_test[new_col], expected['res'])
    else:
        with pytest.raises(TypeError):
            utils.update_df(df_train, df_test, **args)

@pytest.mark.parametrize('args, should_return, expected', [
    (
        {
            'id_col': 'id',
            'ids': (consts.df['id']).copy(),
            'label_col': 'Label',
            'prds': (consts.df['Label']).copy(),
        },
        True,
        pandas.DataFrame({
            'id': (consts.df['id']).copy(),
            'Label': (consts.df['Label']).copy().astype(numpy.int32),
        }),
    ),
    (
        {
            'id_col': 'id',
            'ids': (consts.df['id']).copy(),
            'label_col': 'Label',
            'prds': (consts.df['Label']).copy(),
            'to_int': False,
        },
        True,
        pandas.DataFrame({
            'id': (consts.df['id']).copy(),
            'Label': (consts.df['Label']).copy(),
        }),
    ),
    (
        {
            'id_col': 'id',
            'ids': (consts.df['id']).copy(),
            'label_col': 'Label',
            'prds': (consts.df['Label']).copy(),
            'to_int': 'a',
        },
        False,
        None,
    ),
])
def test_get_df_prds(args, should_return, expected):
    if should_return:
        res = utils.get_df_prds(**args)
        id_col = args['id_col']
        label_col = args['label_col']
        assert_all_true(res[id_col], expected[id_col])
        assert_all_true(res[label_col], expected[label_col])
    else:
        with pytest.raises(TypeError):
            utils.get_df_prds(**args)

@pytest.mark.parametrize(
    'args, file_name, abs_file_path, call_count_of_get_isotime',
    [
        (
            {},
            'submission_2018_01_23T22_33_44.csv',
            '/a/b/submission_2018_01_23T22_33_44.csv',
            1,
        ),
        (
            {'title': 'Sales+Cost'},
            'Sales+Cost_2018_01_23T22_33_44.csv',
            '/a/b/Sales+Cost_2018_01_23T22_33_44.csv',
            1,
        ),
        (
            {'enable_time': False},
            'submission.csv',
            '/a/b/submission.csv',
            0,
        ),
        (
            {'title': 'Sales+Cost', 'enable_time': False},
            'Sales+Cost.csv',
            '/a/b/Sales+Cost.csv',
            0,
        ),
    ],
)
@mock.patch('exp_mgr.utils.get_isotime')
def test_write_sbmcsv(
        mock_get_isotime,
        args,
        file_name,
        abs_file_path,
        call_count_of_get_isotime,
    ):
    time = '_2018_01_23T22_33_44'
    abs_dir_path = '/a/b'
    mock_get_isotime.return_value = time
    mock_df_prds = mock.MagicMock()
    mock_df_prds.to_csv = mock.MagicMock(return_value=None)
    utils.write_sbmcsv(mock_df_prds, abs_dir_path, **args)
    assert mock_get_isotime.call_count == call_count_of_get_isotime
    if call_count_of_get_isotime == 1:
        mock_get_isotime.assert_called_with(prefix='_')
    mock_df_prds.to_csv.assert_called_with(
        abs_file_path,
        index=False,
    )
