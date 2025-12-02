"""__init__.py
@author askillestad3, milesmercer04
@date December 2025

Note: this is the package initializer for mysklearn
"""

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
    'compute_precision_recall_f1',
    # Data handling
    'MyPyTable'
]
