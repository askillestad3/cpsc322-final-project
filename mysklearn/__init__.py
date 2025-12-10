"""__init__.py
@author askillestad3, milesmercer04
@date December 2025

Note: this is the package initializer for mysklearn
"""

from .myclassifiers import *
from .myevaluation import *
from .mypytable import *
from .myrandomforestclassifier import *

__all__ = [
    # Classifiers
    'MyKNeighborsClassifier',
    'MyDecisionTreeClassifier',
    'MyRandomForestClassifier',
    'MyDummyClassifier',
    # Evaluation
    'train_test_split',
    'stratified_kfold_split',
    'bootstrap_sample',
    'confusion_matrix',
    'accuracy_score',
    'binary_precision_score',
    'binary_recall_score',
    'binary_f1_score',
    'mutual_information',
    'pearson_r',
    'classification_report',
    # Data handling
    'MyPyTable'
]
