import copy
from . import utils

class ExpMgr(object): # ExpMgr = Experiment Manager

    writable_attrs = [
        'src_train',
        'src_test',
        'data_train',
        'data_test',
        'id_col',
        'label_col',
        'summary_cv_file_path',
        'sbmcsv_dir_path',
    ]

    def __init__(self):
        # list default value below:
        self.src_train = None # a csv file path or a pandas DataFrame or a function
        self.src_test = None # a csv file path or a pandas DataFrame or a function
        self.data_train = None
        self.data_test = None
        self.id_col = 'auto_index'
        self.label_col = ''
        self.summary_cv_file_path = './summary_cv.csv'
        self.sbmcsv_dir_path = './'
        self.preprocess_history = [] # can not let user use `set` to change
        """
        self.preprocess_history = [
          {
            func: some_function,
            preprocess_desc: 'some desc', # default is some_function.__name__
            start_time: '2018-05-30T14:31:21.555735',
            end_time: '2018-06-03T00:00:00.000000',
            process_time: '3 days, 9:28:38.444265',
          }
        ]
        """
        self.cv_report = {} # triggered cv_report updated by reset_data_run_cv or run_tmp_cv
        """
        self.cv_report = {
            'start_time': '2018-05-30T14:31:21.555735', # datetime.datetime.now().isoformat()
            'end_time:    '2018-06-03T00:00:00.000000', # 
            'process_time': '3 days, 9:28:38.444265', # str(new_datetime - old_datetime)
            'result': ['0.85', '0.77', '0.88'],
            'output_to_file': 'summary_cv.csv', # triggered by output_summary_cv
            'mean': 0.8333333333333334,
            'std': 0.046427960923947055,
            'preprocess_desc': `join the desc in preprocess_history`,
            'feats_desc': 'any feats desc.',
            'model_desc': 'any model desc.',
        }
        """
        self.prediction_report = {} # triggered prediction_report update by reset_data_run_predict or run_tmp_predict
        """
        self.prediction_report = {
            'start_time': '2018-05-30T14:31:21.555735', # datetime.datetime.now().isoformat()
            'end_time:    '2018-06-03T00:00:00.000000', # 
            'process_time': '3 days, 9:28:38.444265', # str(new_datetime - old_datetime)
            'result': pandas.DataFrame({
                self.id_col: ids,
                self.label_col: prds,
            }),
            'output_to_file': 'DecisionTree_Sales+Cost_2018_01_02T22_33_44.csv', # triggered by output_prediction
            # if output_file_name = None, means no file generated.
            'preprocess_desc': `join the desc in preprocess_history`,
            'feats_desc': 'any feats desc.',
            'model_desc': 'any model desc.',
        }
        """

    def get(self, *args):
        if len(args) == 0:
            return copy.deepcopy(self.__dict__)
        elif len(args) == 1:
            key = args[0]
            if key in self.__dict__:
                return copy.deepcopy(self.__dict__[key])
            else:
                raise KeyError(
                    'The attribute \'{}\' doesn\'t exist.'.format(key)
                )
        else:
            res = {}
            for key in args:
                if key in self.__dict__:
                    res[key] = copy.deepcopy(self.__dict__[key])
                else:
                    raise KeyError(
                        'The attribute \'{}\' doesn\'t exist.'.format(key)
                    )
            return res

    def set(self, **kargs):
        if not kargs:
            raise ValueError('Arguments can not be empty.')
        # should validate and write case by case, maybe writable_attrs is not usefule.
        return self

    def _append_preprocess_history(self, func, preprocess_desc=None):
        if utils.is_func(func):
            if isinstance(preprocess_desc, str) and preprocess_desc.strip()
                preprocess_desc = preprocess_desc.strip()
            else:
                preprocess_desc = func.__name__
            self.preprocess_history.append({
                'func': func,
                'preprocess_desc': preprocess_desc,
            })
        else:
            raise TypeError('\'func\' should be a function.')

    def _load_data(self):
        attr_map = {
           'src_train': 'data_train',
           'src_test': 'data_test',
        }
        for src_attr, data_attr in attr_map.items():
            src_val = getattr(self, src_attr)
            if isinstance(src_val, str):
                setattr(self, data_attr, pandas.read_csv(src_val))
            elif isinstance(src_val, pandas.DataFrame):
                setattr(self, data_attr, src_val.copy())
            elif utils.is_func(src_val):
                setattr(self, data_attr, src_val())

    def _clear_preprocess_history(self):
        self.preprocess_history = []

    def reset_data(self):
        self._load_data()
        self._clear_preprocess_history()
        return self

    def update_data(self, func, preprocess_desc=None):
        if not utils.is_func(func):
            raise TypeError('\'func\' should be a function.')

    def preprocess(self, func, preprocess_desc=None):
        if not utils.is_func(func):
            raise TypeError('\'func\' should be a function.')
        self.reset_data()
        data = self.get('data_train', 'data_test')
        data_train, data_test = func(**data)
        self.set(data_train=data_train, data_test=data_test)
        return self

    def get_traintest_data(self, feats):
        if not utils.is_strlist(feats):
            raise TypeError('\'feats\' should be a list of strings.')
        data = self.get('data_train', 'data_test')
        for col in feats:
            if col not in data['data_train']:
                raise ValueError(
                    'col \'{}\' is not in train DataFrame.'.format(col)
                )
            if col not in data['data_test']:
                raise ValueError(
                    'col \'{}\' is not in test DataFrame.'.format(col)
                )
        return (
            data['data_train'][feats],
            data['data_train'][self.label_col],
            data['data_test'][feats],
        )

    def set_cv_desc(self):
        # for people forget feed desc.

    def set_prediction_desc(self):
        # for people forget feed desc.


"""########################################"""
emr = ExpMgr("""some init args""")
a = emr.get() # return all attributes with every attribute deep coied in a dict
b = emr.get('id_col', 'data_train') # return deep copied 'id_col' and 'data_train' attributes in a dict
c = emr.get('data_train') # return deep copied 'data_train' attribute (I would not wrap it into a tuple)
emr.set(id_col='Passenger_Id', src_test='/a/b/test.csv') # set attributes, would return self to chain object

emr.print_attrs() # print all attributes

# EDA phase ================
df_train, df_test = emr.update_data(add_col_a, preprocess_desc='any preprocess desc.')
#     update data & return emr.get('data_train', 'data_test')
#     add_col_a would be appended into preprocess_history
#     `update_data` would not reset data
#     plot.........................

df_train, df_test = emr.update_data(add_col_b)
#     Because preprocess_desc is not provided, default is add_col_b.__name__
#     update data & return emr.get('data_train', 'data_test')
#     add_col_b would be appended into preprocess_history
#     `update_data` would not reset data
#     plot.........................

# CV phase ==================
#     Case 1. You feel you can run cv directly without any further preprocess.
#         Note the method `run_cv` never resets data.
#         def cv_fn(data_train, y_train, data_test, emr):
#             Your cv here.....
#             return {
#                 'result': [0.88, 0.9, 0.5],
#             }
emr.run_cv(
        cv_fn,
        feats, # if feats is not provided, feats_desc would be set to 'All'.
        feats_desc='any feat desc.', # if feats provided but feats_desc is not provided, use '+'.join(feats) as defualt.
        model_desc='any model desc.', # no default.
    )
    .output_summary_cv()

#     Case 2. You want to reset data & update data and then run cv
#         Note `preprocess` would reset data then clean preprocess_history then update data then run cv,
#         and the input `update_fn` of preprocess would also be appended into preprocess_history
emr.preprocess(add_col_a_and_b, desc='some preprocess desc')
    .run_cv(
        cv_fn,
        feats,
        feats_desc='any feat desc.',
        model_desc='any model desc.',
    )
    .output_summary_cv()

# Predict phase ===================
#     Case 1. You feel you can run predict directly without any further preprocess.
#         Note the method 'run_predict' never resets data.
emr.run_predict(
        ml_fn,
        feats,
        feats_desc='any feat desc.',
        model_desc='any model desc.',
    ).outout_prediction(
        file_title='DecisionTree' + '+'.join(feats),
        time_title=False,
    )

#     Case 2. You want to reset data then clean preprocess_history then update data then run cv
emr.preprocess(add_col_a_and_b, desc='some preprocess desc')
    .run_predict(
        ml_fn,
        feats,
        feats_desc='any feat desc.',
        model_desc='any model desc.',
    ).outout_prediction(
        file_title='DecisionTree' + '+'.join(feats),
    )
    
"""########################################"""    
import copy
import pandas as pd
from .utils import isFunc, isStr, printDfSummary, updateDf, \
  getFilteredData, getFilteredTrainTest, getDfPredictions, \
  getClassName, writeSubmissionCsv

class ExpMgr(object): # ExpMgr = Experiment Manager
    trainKey = 'train'
    testKey = 'test'
    summaryKey = 'summary'

    @staticmethod
    def isValidCsvPaths(csvPaths):
        trainKey = ExpMgr.trainKey
        testKey = ExpMgr.testKey
        if (type(csvPaths) == dict) and \
          (trainKey in csvPaths) and \
          (testKey in csvPaths) and \
          isStr(csvPaths[trainKey]) and \
          isStr(csvPaths[testKey]):
            return True
        return False

    @staticmethod
    def getDftSbmTitle(model, feats): # getDftSbmTitle = get default submission title
        m = getClassName(model)
        f = '+'.join(feats)
        return '{m}_{f}'.format(m=m, f=f)

    def __init__(self, csvPaths, submissionDir, idKey, labelKey, preprocess=None):
        '''
        Args:
            csvPaths: a dict contains train/test csv absolute file path, ex:
	      csvPaths = {
	        'train': '/aaa/train.csv',
		'test': '/aaa/test.csv',
		'summary': '/bbb/summary.csv'
	      }
            submissionDir: a string means the absolute directory path
              containing submission csv files.
            idKey: id column name
            labelKey: label columne name
            preprocess: a function with args dfTrain and dfTest to preprocess
              train/test data and then it would return dfTrain, dfTest
        Returns:
            a instance of ExpMgr
        '''
        if ExpMgr.isValidCsvPaths(csvPaths) and \
            isStr(submissionDir) and \
            isStr(idKey) and \
            isStr(labelKey):
            self.__setCsvPaths(csvPaths)
            self.__resetDf(idKey, labelKey, preprocess)
            self.__submissionDir = submissionDir
        else:
            raise ValueError('\'csvPaths\' is not valid.')

    def __setCsvPaths(self, csvPaths):
        csvPaths = copy.deepcopy(csvPaths)
        if ExpMgr.summaryKey not in csvPaths:
            csvPaths[ExpMgr.summaryKey] = None
        self.__csvPaths = csvPaths

    def __resetDf(self, idKey, labelKey, preprocess=None):
        trainCsvPath = self.__csvPaths[ExpMgr.trainKey]
        testCsvPath = self.__csvPaths[ExpMgr.testKey]
        self.__idKey = idKey
        self.__labelKey = labelKey
        self.__dfTrain = pd.read_csv(trainCsvPath)
        self.__dfTest = pd.read_csv(testCsvPath)
        self.__dfTrain.set_index([idKey], inplace=True)
        self.__dfTest.set_index([idKey], inplace=True)
        # self.__dfTest.index.values => woudl get index value array (array-like)
        if isFunc(preprocess):
            self.__dfTrain, self.__dfTest = preprocess(
                self.__dfTrain, self.__dfTest)

    def resetData(self, preprocess=None):
        self.__resetDf(self.__idKey, self.__labelKey, preprocess)

    def copyData(self, toNumpy=False):
        dfTrain, dfTest = self.__dfTrain.copy(), self.__dfTest.copy()
        if toNumpy:
            return dfTrain.as_matrix(), dfTest.as_matrix()
        return dfTrain, dfTest

    def updateData(self, simpleMap=None, customMapFunc=None):
        self.__dfTrain, self.__dfTest = updateDf(
            self.__dfTrain, self.__dfTest, simpleMap, customMapFunc
        )

    def cv(self, model, feats, folds=5, writeToSummary=True):
        # TODO: if writeToSummary
        pass

    def trainAndPredict(
            self,
            model,
            feats,
            writeToSubmission=True,
            fileTitle=None,
            enableTime=True
        ):
        xTrain, yTrain, xTest = getFilteredTrainTest(
            self.__dfTrain,
            self.__dfTest,
            feats,
            self.__labelKey
        )
        model.fit(xTrain, yTrain)
        predictions = model.predict(xTest)
        dfPredictions = getDfPredictions(
            self.__idKey,
            self.__dfTest.index.values,
            self.__labelKey,
            predictions
        )
        if writeToSubmission:
            title = fileTitle or ExpMgr.getDftSbmTitle(model, feats)
            writeSubmissionCsv(
                dfPredictions,
                self.__submissionDir,
                title,
                enableTime,
            )
        return dfPredictions

    def showAttrs(self):
        print('# csv paths are')
        print(self.__csvPaths)
        print('\n')
        print('# train dataframe is')
        print(self.__dfTrain)
        print('\n')
        print('# test dataframe is')
        print(self.__dfTest)
        print('\n')

    def showSummary(self, dataType=None):
        if dataType == ExpMgr.trainKey:
            printDfSummary(self.__dfTrain, 'Train Data')
        elif dataType == ExpMgr.testKey:
            printDfSummary(self.__dfTest, 'Test Data')
        else:
            printDfSummary(self.__dfTrain, 'Train Data')
            printDfSummary(self.__dfTest, 'Test Data')

if __name__ == '__main__':
    from . import config
    # def __init__(self, csvPaths, submissionDir, idKey, labelKey, preprocess = None):
    em = ExpMgr(
        config.csvPaths, config.submissionDir, config.idKey, config.labelKey
    )
    em.updateData(simpleMap={
        'srcColKey': 'Sex',
        'newColKey': 'drvSexVal',
        'mapData': {'male': 0, 'female': 1},
    })
