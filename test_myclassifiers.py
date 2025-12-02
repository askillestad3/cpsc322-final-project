
from mysklearn import MyRandomForestClassifier

"""Test Data"""

# Define interview dataset
HEADER_INTERVIEW = ["level", "lang", "tweets", "phd", "interviewed_well"]
X_TRAIN_INTERVIEW = [
    ["Senior", "Java", "no", "no"],     # False
    ["Senior", "Java", "no", "yes"],    # False
    ["Mid", "Python", "no", "no"],      # True
    ["Junior", "Python", "no", "no"],   # True
    ["Junior", "R", "yes", "no"],       # True
    ["Junior", "R", "yes", "yes"],      # False
    ["Mid", "R", "yes", "yes"],         # True
    ["Senior", "Python", "no", "no"],   # False
    ["Senior", "R", "yes", "no"],       # True
    ["Junior", "Python", "yes", "no"],  # True
    ["Senior", "Python", "yes", "yes"], # True
    ["Mid", "Python", "no", "yes"],     # True
    ["Mid", "Java", "yes", "no"],       # True
    ["Junior", "Python", "no", "yes"]   # False
]
Y_TRAIN_INTERVIEW = ["False", "False", "True", "True", "True", "False", "True", "False",
                     "True","True", "True", "True", "True", "False"]

INTERVIEW_ATTRIBUTE_VALS = [
    ["Junior", "Mid", "Senior"],
    ["Java", "Python", "R"],
    ["no", "yes"],
    ["no", "yes"]
]

INTERVIEW_Y_VALS = ["False", "True"]

"""Utility Functions"""

def validate_decision_tree(tree: list | None, attribute_vals: list[list], y_vals: list):
    # Verify tree is not None
    assert tree is not None

    # Select the root of the decision tree
    root = tree[0]
    assert isinstance(root, str)
    
    # Determine the type of node
    match root:
        case "Leaf":        # Leaf node
            assert tree[1] in y_vals
            assert isinstance(tree[2], int)
            assert isinstance(tree[3], int)
            assert tree[2] < tree[3]

        case "Attribute":   # Attribute split
            assert tree[1][:3] == "att"
            attribute_idx = int(tree[1][3:])
            assert attribute_idx >= 0 and attribute_idx < len(attribute_vals)
            for value_branch in tree[2:]:
                assert value_branch[0] == "Value"
                assert value_branch[1] in attribute_vals[attribute_idx]
                validate_decision_tree(value_branch[2], attribute_vals, y_vals)

        case _:             # Unexpected node type
            assert False


def test_rf_init_sets_parameters():
    # Providing only N
    rf = MyRandomForestClassifier(N=100)
    assert rf.N == 100
    assert rf.M is 100
    assert rf.F is None
    assert rf.X_train is None
    assert rf.y_train is None
    assert len(rf.trees) == 0

    # Providing N and M
    rf = MyRandomForestClassifier(N=120, M=25)
    assert rf.N == 120
    assert rf.M == 25
    assert rf.F is None
    assert rf.X_train is None
    assert rf.y_train is None
    assert len(rf.trees) == 0

    # Providing N and F
    rf = MyRandomForestClassifier(N=75, F=3)
    assert rf.N == 75
    assert rf.M == 75
    assert rf.F == 3
    assert rf.X_train is None
    assert rf.y_train is None
    assert len(rf.trees) == 0

    # Providing N, M, and F
    rf = MyRandomForestClassifier(N=150, M=50, F=5)
    assert rf.N == 150
    assert rf.M == 50
    assert rf.F == 5
    assert rf.X_train is None
    assert rf.y_train is None
    assert len(rf.trees) == 0

def test_rf_fit_sets_X_train_and_y_train():
    rf = MyRandomForestClassifier(N=100)
    rf.fit(X_TRAIN_INTERVIEW, Y_TRAIN_INTERVIEW)
    assert rf.X_train == X_TRAIN_INTERVIEW
    assert rf.y_train == Y_TRAIN_INTERVIEW

def test_rf_fit_generates_correct_number_of_valid_trees():
    # Proviing only N
    rf = MyRandomForestClassifier(N=10)
    rf.fit(X_TRAIN_INTERVIEW, Y_TRAIN_INTERVIEW)
    assert len(rf.trees) == 10
    for tree in rf.trees:
        validate_decision_tree(tree.tree, INTERVIEW_ATTRIBUTE_VALS, INTERVIEW_Y_VALS)

    # Providing N and M
    rf = MyRandomForestClassifier(N=20, M=7)
    rf.fit(X_TRAIN_INTERVIEW, Y_TRAIN_INTERVIEW)
    assert len(rf.trees) == 7
    for tree in rf.trees:
        validate_decision_tree(tree.tree, INTERVIEW_ATTRIBUTE_VALS, INTERVIEW_Y_VALS)
    
    # Providing N and F
    rf = MyRandomForestClassifier(N=15, F=3)
    rf.fit(X_TRAIN_INTERVIEW, Y_TRAIN_INTERVIEW)
    assert len(rf.trees) == 15
    for tree in rf.trees:
        validate_decision_tree(tree.tree, INTERVIEW_ATTRIBUTE_VALS, INTERVIEW_Y_VALS)

    # Providing N, M, and F
    rf = MyRandomForestClassifier(N=25, M=10, F=2)
    rf.fit(X_TRAIN_INTERVIEW, Y_TRAIN_INTERVIEW)
    assert len(rf.trees) == 10
    for tree in rf.trees:
        validate_decision_tree(tree.tree, INTERVIEW_ATTRIBUTE_VALS, INTERVIEW_Y_VALS)

def test_rf_fit_creates_subtrees_with_proper_bootstrap_samples():
    rf = MyRandomForestClassifier(N=10)
    rf.fit(X_TRAIN_INTERVIEW, Y_TRAIN_INTERVIEW)
    assert len(rf.trees) == 10
    for tree in rf.trees:
        assert tree.X_train is not None
        assert tree.y_train is not None
        for x, y in zip(tree.X_train, tree.y_train):
            assert x in X_TRAIN_INTERVIEW
            assert y in Y_TRAIN_INTERVIEW
            x_idx = X_TRAIN_INTERVIEW.index(x)
            assert y == Y_TRAIN_INTERVIEW[x_idx]
