##############################################
# Programmer: Ian Myers
# Class: CPSC 322-02, Fall 2024
# Programming Assignment #5
# 11/07/24
# Description: This program tests all functions of myclassifiers
##############################################

#pylint: skip-file
import numpy as np
from scipy import stats
from mysklearn.myclassifiers import MyNaiveBayesClassifier


from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier,\
    MyDecisionTreeClassifier

def high_low_discretizer(val):
        if val >= 100:
            return "high"
        else:
            return "low"
        
# note: order is actual/received student value, expected/solution
def test_simple_linear_regression_classifier_fit():
    np.random.seed(0)
    X_train = [[val] for val in list(range(0,100))]
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]
    lin_reg = MySimpleLinearRegressionClassifier(discretizer=high_low_discretizer)
    lin_reg.fit(X_train,y_train)

    # assert lin_reg.slope and lin_reg.intercept
    # are correct
    # 1. desk calculation
    assert np.isclose(lin_reg.regressor.slope, 1.9249174584304438)
    assert np.isclose(lin_reg.regressor.intercept, 5.211786196055158)

def test_simple_linear_regression_classifier_predict():
    np.random.seed(0)
    X_train = [[val] for val in list(range(0,100))]
    # Test case 1
    y_train_1 = [row[0] * 3 + np.random.normal(5, 50) for row in X_train]
    lin_clf_1 = MySimpleLinearRegressionClassifier(discretizer=high_low_discretizer)
    lin_clf_1.fit(X_train,y_train_1)

    X_test_1 = [[3], [400], [150]]
    y_pred_1 = lin_clf_1.predict(X_test_1)
    y_test_1 = ["low", "high", "high"]
    assert y_pred_1 == y_test_1

    # Test case 2
    y_train_2 = [row[0] * 1 + np.random.normal(0, 100) for row in X_train]
    lin_clf_2 = MySimpleLinearRegressionClassifier(discretizer=high_low_discretizer)
    lin_clf_2.fit(X_train,y_train_2)

    X_test_2 = [[25], [0], [600]]
    y_pred_2 = lin_clf_2.predict(X_test_2)
    y_test_2 = ["low", "low", "high"]
    assert y_pred_2 == y_test_2

def test_kneighbors_classifier_kneighbors():
    # from in-class #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]

    kneighbor_1 = MyKNeighborsClassifier(categorical=False)
    kneighbor_1.fit(X_train_class_example1, y_train_class_example1)
    actual_distances_class_example1, actual_neighbor_indices_class_example1 = kneighbor_1.kneighbors([[0.33,1]])
    expected_distances_class_example1 = [0.67, 1, 1.053]
    expected_neighbor_indices_class_example1 = [0, 2, 3]
    assert np.isclose(actual_distances_class_example1[0][0],expected_distances_class_example1[0], atol=.01)
    assert np.isclose(actual_distances_class_example1[0][1],expected_distances_class_example1[1], atol = .01)
    assert np.isclose(actual_distances_class_example1[0][2],expected_distances_class_example1[2], atol=.01)
    assert actual_neighbor_indices_class_example1[0] == expected_neighbor_indices_class_example1

 # from in-class #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]

    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]

    kneighbor_2 = MyKNeighborsClassifier(categorical=False)
    kneighbor_2.fit(X_train_class_example2, y_train_class_example2)
    actual_distances_class_example2, actual_neighbor_indices_class_example2 = kneighbor_2.kneighbors([[2,3]])
    expected_distances_class_example2 = [1.4142135623730951, 1.4142135623730951, 2.0]
    expected_neighbor_indices_class_example2 = [0, 4, 6]
    assert np.isclose(actual_distances_class_example2[0][0],expected_distances_class_example2[0])
    assert np.isclose(actual_distances_class_example2[0][1],expected_distances_class_example2[1])
    assert np.isclose(actual_distances_class_example2[0][2],expected_distances_class_example2[2])
    assert actual_neighbor_indices_class_example2[0] == expected_neighbor_indices_class_example2

    # from Bramer
    X_train_bramer_example = [
    [0.8, 6.3],
    [1.4, 8.1],
    [2.1, 7.4],
    [2.6, 14.3],
    [6.8, 12.6],
    [8.8, 9.8],
    [9.2, 11.6],
    [10.8, 9.6],
    [11.8, 9.9],
    [12.4, 6.5],
    [12.8, 1.1],
    [14.0, 19.9],
    [14.2, 18.5],
    [15.6, 17.4],
    [15.8, 12.2],
    [16.6, 6.7],
    [17.4, 4.5],
    [18.2, 6.9],
    [19.0, 3.4],
    [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]

    kneighbor_3 = MyKNeighborsClassifier(categorical=False)
    kneighbor_3.fit(X_train_bramer_example, y_train_bramer_example)
    actual_distances_class_example3, actual_neighbor_indices_class_example3 = kneighbor_3.kneighbors([[9.1,11.0]])
    expected_distances_class_example3 = [0.608, 1.237, 2.202]
    expected_neighbor_indices_class_example3 = [6, 5, 7]
    assert np.isclose(actual_distances_class_example3[0][0],expected_distances_class_example3[0], atol=.01)
    assert np.isclose(actual_distances_class_example3[0][1],expected_distances_class_example3[1], atol=.01)
    assert np.isclose(actual_distances_class_example3[0][2],expected_distances_class_example3[2], atol=.01)
    assert actual_neighbor_indices_class_example3[0] == expected_neighbor_indices_class_example3


def test_kneighbors_classifier_predict():
    # from in-class #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]

    kneighbor_1 = MyKNeighborsClassifier(categorical=False)
    kneighbor_1.fit(X_train_class_example1, y_train_class_example1)
    y_pred_1 = kneighbor_1.predict([[0.33,1]])
    y_expected_1 = ["good"]
    assert y_pred_1 == y_expected_1

 # from in-class #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]

    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]

    kneighbor_2 = MyKNeighborsClassifier(categorical=False)
    kneighbor_2.fit(X_train_class_example2, y_train_class_example2)
    y_pred_2 = kneighbor_2.predict([[2,3]])
    y_expected_2 = ["yes"]
    assert y_pred_2 == y_expected_2

    # from Bramer
    X_train_bramer_example = [
    [0.8, 6.3],
    [1.4, 8.1],
    [2.1, 7.4],
    [2.6, 14.3],
    [6.8, 12.6],
    [8.8, 9.8],
    [9.2, 11.6],
    [10.8, 9.6],
    [11.8, 9.9],
    [12.4, 6.5],
    [12.8, 1.1],
    [14.0, 19.9],
    [14.2, 18.5],
    [15.6, 17.4],
    [15.8, 12.2],
    [16.6, 6.7],
    [17.4, 4.5],
    [18.2, 6.9],
    [19.0, 3.4],
    [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]

    kneighbor_3 = MyKNeighborsClassifier(categorical=False)
    kneighbor_3.fit(X_train_bramer_example, y_train_bramer_example)
    y_pred_3 = kneighbor_3.predict([[9.1,11.0]])
    y_expected_3 = ["+"]
    assert y_pred_3 == y_expected_3

    # testing categorical only data
    X_train_cat = [["yes", "no", "no"], ["yes", "yes", "yes"], ["yes", "no", "yes"]]
    y_train_cat = ["good", "bad", "worse"]

    kneighbor_4 = MyKNeighborsClassifier(3, True)
    kneighbor_4.fit(X_train_cat, y_train_cat)
    y_pred_4 = kneighbor_4.predict([["yes","no", "no"]])
    y_expected_4 = ["good"]
    assert y_pred_4 == y_expected_4

def test_dummy_classifier_fit():
    # Fit test A
    X_train_1 = [[val] for val in list(range(0,100))]
    y_train_1 = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy_1 = MyDummyClassifier()
    dummy_1.fit(X_train_1, y_train_1)
    assert dummy_1.most_common_label == "yes"

    # Fit test B
    X_train_2 = [[val] for val in list(range(0,100))]
    y_train_2 = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    dummy_2 = MyDummyClassifier()
    dummy_2.fit(X_train_2, y_train_2)
    assert dummy_2.most_common_label == "no"

    # Fit test C
    X_train_3 = [[val] for val in list(range(0,100))]
    y_train_3 = list(np.random.choice(["absolutely","yes", "no", "maybe", "definitely not"], 100, replace=True, p=[0.1, 0.2, 0.2, 0.4, 0.1]))
    dummy_3 = MyDummyClassifier()
    dummy_3.fit(X_train_3, y_train_3)
    assert dummy_3.most_common_label == "maybe"

def test_dummy_classifier_predict():
    # Predict test A
    X_train_1 = [[val] for val in list(range(0,100))]
    y_train_1 = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy_1 = MyDummyClassifier()
    dummy_1.fit(X_train_1, y_train_1)
    y_pred_1 = dummy_1.predict([[0], [50], [100]])
    assert y_pred_1 == ["yes", "yes", "yes"]

    # Fit test B
    X_train_2 = [[val] for val in list(range(0,100))]
    y_train_2 = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    dummy_2 = MyDummyClassifier()
    dummy_2.fit(X_train_2, y_train_2)
    y_pred_2 = dummy_2.predict([[0], [50], [100]])
    assert y_pred_2 == ["no", "no", "no"]

    # Fit test C
    X_train_3 = [[val] for val in list(range(0,100))]
    y_train_3 = list(np.random.choice(["absolutely","yes", "no", "maybe", "definitely not"], 100, replace=True, p=[0.1, 0.2, 0.2, 0.4, 0.1]))
    dummy_3 = MyDummyClassifier()
    dummy_3.fit(X_train_3, y_train_3)
    y_pred_3 = dummy_3.predict([[0], [50], [100]])
    assert y_pred_3 == ["maybe", "maybe", "maybe"]


def test_naive_bayes_classifier_fit():
    # in-class Naive Bayes example (lab task #1)
    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    naive_1 = MyNaiveBayesClassifier()
    naive_1.fit(X_train_inclass_example, y_train_inclass_example)
    expected_priors_inclass = {"yes":5/8, "no": 3/8}
    expected_posteriors_inclass = {"yes":{1: 4/5, 2: 1/5, 5 : 2/5, 6: 3/5}, "no" : {1: 2/3, 2: 1/3, 5: 2/3, 6: 1/3}}
    # Asserting priors are correct
    assert np.isclose(naive_1.priors["yes"], expected_priors_inclass["yes"]) 
    assert np.isclose(naive_1.priors["no"], expected_priors_inclass["no"]) 

    # Asserting posteriors are correct
    assert np.isclose(naive_1.posteriors["yes"]["01"], expected_posteriors_inclass["yes"][1])
    assert np.isclose(naive_1.posteriors["yes"]["02"], expected_posteriors_inclass["yes"][2])
    assert np.isclose(naive_1.posteriors["yes"]["15"], expected_posteriors_inclass["yes"][5])
    assert np.isclose(naive_1.posteriors["yes"]["16"], expected_posteriors_inclass["yes"][6])
    assert np.isclose(naive_1.posteriors["no"]["01"], expected_posteriors_inclass["no"][1])
    assert np.isclose(naive_1.posteriors["no"]["02"], expected_posteriors_inclass["no"][2])
    assert np.isclose(naive_1.posteriors["no"]["15"], expected_posteriors_inclass["no"][5])
    assert np.isclose(naive_1.posteriors["no"]["16"], expected_posteriors_inclass["no"][6])

    # MA7 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    naive_2 = MyNaiveBayesClassifier()
    naive_2.fit(X_train_iphone, y_train_iphone)
    expected_priors_iphone = {"yes":10/15, "no": 5/15}
    expected_posteriors_iphone = {"yes":{"01": 2/10, "02": 8/10, "11": 3/10, "12": 4/10, "13": 3/10, "2fair": 7/10, "2excellent": 3/10}, 
                                   "no" : {"01": 3/5, "02": 2/5, "11": 1/5, "12": 2/5, "13": 2/5, "2fair": 2/5, "2excellent": 3/5}}
    # Asserting priors are correct
    assert np.isclose(naive_2.priors["yes"], expected_priors_iphone["yes"]) 
    assert np.isclose(naive_2.priors["no"], expected_priors_iphone["no"]) 

    # Asserting posteriors are correct
    assert np.isclose(naive_2.posteriors["yes"]["01"], expected_posteriors_iphone["yes"]["01"])
    assert np.isclose(naive_2.posteriors["yes"]["02"], expected_posteriors_iphone["yes"]["02"])
    assert np.isclose(naive_2.posteriors["yes"]["11"], expected_posteriors_iphone["yes"]["11"])
    assert np.isclose(naive_2.posteriors["yes"]["12"], expected_posteriors_iphone["yes"]["12"])
    assert np.isclose(naive_2.posteriors["yes"]["13"], expected_posteriors_iphone["yes"]["13"])
    assert np.isclose(naive_2.posteriors["yes"]["2fair"], expected_posteriors_iphone["yes"]["2fair"])
    assert np.isclose(naive_2.posteriors["yes"]["2excellent"], expected_posteriors_iphone["yes"]["2excellent"])
    assert np.isclose(naive_2.posteriors["no"]["01"], expected_posteriors_iphone["no"]["01"])
    assert np.isclose(naive_2.posteriors["no"]["02"], expected_posteriors_iphone["no"]["02"])
    assert np.isclose(naive_2.posteriors["no"]["11"], expected_posteriors_iphone["no"]["11"])
    assert np.isclose(naive_2.posteriors["no"]["12"], expected_posteriors_iphone["no"]["12"])
    assert np.isclose(naive_2.posteriors["no"]["13"], expected_posteriors_iphone["no"]["13"])
    assert np.isclose(naive_2.posteriors["no"]["2fair"], expected_posteriors_iphone["no"]["2fair"])
    assert np.isclose(naive_2.posteriors["no"]["2excellent"], expected_posteriors_iphone["no"]["2excellent"])

    # Bramer 3.2 train dataset
    header_train = ["day", "season", "wind", "rain", "class"]
    X_train_train = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]
    
    naive_3 = MyNaiveBayesClassifier()
    naive_3.fit(X_train_train, y_train_train)
    expected_priors_bramer = {"on time":14/20, "late": 2/20, "very late": 3/20, "cancelled": 1/20}
    expected_posteriors_bramer = {"on time":{"0weekday": 9/14, "0saturday": 2/14, "0sunday": 1/14, "0holiday": 2/14, "1spring": 4/14, "1summer": 6/14, "1autumn": 2/14, "1winter": 2/14, "2none": 5/14, "2high": 4/14, "2normal": 5/14, "3none": 5/14, "3slight": 8/14, "3heavy": 1/14}, 
                                   "late" : {"0weekday": 1/2, "0saturday": 1/2, "0sunday": 0/2, "0holiday": 0/2, "1spring": 0/2, "1summer": 0/2, "1autumn": 0/2, "1winter": 2/2, "2none": 0/2, "2high": 1/2, "2normal": 1/2, "3none": 1/2, "3slight": 0/2, "3heavy": 1/2},
                                   "very late" :{"0weekday": 3/3, "0saturday": 0/3, "0sunday": 0/3, "0holiday": 0/3, "1spring": 0/3, "1summer": 0/3, "1autumn": 1/3, "1winter": 2/3, "2none": 0/3, "2high": 1/3, "2normal": 2/3, "3none": 1/3, "3slight": 0/3, "3heavy": 2/3},
                                   "cancelled" : {"0weekday": 0/1, "0saturday": 1/1, "0sunday": 0/1, "0holiday": 0/1, "1spring": 1/1, "1summer": 0/1, "1autumn": 0/1, "1winter": 0/1, "2none": 0/1, "2high": 1/1, "2normal": 0/1, "3none": 0/1, "3slight": 0/1, "3heavy": 1/1}
                                   }
    # Asserting priors are correct
    assert np.isclose(naive_3.priors["on time"], expected_priors_bramer["on time"]) 
    assert np.isclose(naive_3.priors["late"], expected_priors_bramer["late"]) 
    assert np.isclose(naive_3.priors["very late"], expected_priors_bramer["very late"]) 
    assert np.isclose(naive_3.priors["cancelled"], expected_priors_bramer["cancelled"]) 

    # Asserting posteriors are correct
    assert np.isclose(naive_3.posteriors["on time"]["0weekday"], expected_posteriors_bramer["on time"]["0weekday"])
    assert np.isclose(naive_3.posteriors["on time"]["0saturday"], expected_posteriors_bramer["on time"]["0saturday"])
    assert np.isclose(naive_3.posteriors["on time"]["0sunday"], expected_posteriors_bramer["on time"]["0sunday"])
    assert np.isclose(naive_3.posteriors["on time"]["0holiday"], expected_posteriors_bramer["on time"]["0holiday"])
    assert np.isclose(naive_3.posteriors["on time"]["1spring"], expected_posteriors_bramer["on time"]["1spring"])
    assert np.isclose(naive_3.posteriors["on time"]["1summer"], expected_posteriors_bramer["on time"]["1summer"])
    assert np.isclose(naive_3.posteriors["on time"]["1autumn"], expected_posteriors_bramer["on time"]["1autumn"])
    assert np.isclose(naive_3.posteriors["on time"]["1winter"], expected_posteriors_bramer["on time"]["1winter"])
    assert np.isclose(naive_3.posteriors["on time"]["2none"], expected_posteriors_bramer["on time"]["2none"])
    assert np.isclose(naive_3.posteriors["on time"]["2high"], expected_posteriors_bramer["on time"]["2high"])
    assert np.isclose(naive_3.posteriors["on time"]["2normal"], expected_posteriors_bramer["on time"]["2normal"])
    assert np.isclose(naive_3.posteriors["on time"]["3none"], expected_posteriors_bramer["on time"]["3none"])
    assert np.isclose(naive_3.posteriors["on time"]["3slight"], expected_posteriors_bramer["on time"]["3slight"])
    assert np.isclose(naive_3.posteriors["on time"]["3heavy"], expected_posteriors_bramer["on time"]["3heavy"])

    assert np.isclose(naive_3.posteriors["late"]["0weekday"], expected_posteriors_bramer["late"]["0weekday"])
    assert np.isclose(naive_3.posteriors["late"]["0saturday"], expected_posteriors_bramer["late"]["0saturday"])
    assert np.isclose(naive_3.posteriors["late"]["1winter"], expected_posteriors_bramer["late"]["1winter"])
    assert np.isclose(naive_3.posteriors["late"]["2high"], expected_posteriors_bramer["late"]["2high"])
    assert np.isclose(naive_3.posteriors["late"]["2normal"], expected_posteriors_bramer["late"]["2normal"])
    assert np.isclose(naive_3.posteriors["late"]["3none"], expected_posteriors_bramer["late"]["3none"])
    assert np.isclose(naive_3.posteriors["late"]["3heavy"], expected_posteriors_bramer["late"]["3heavy"])

    assert np.isclose(naive_3.posteriors["very late"]["0weekday"], expected_posteriors_bramer["very late"]["0weekday"])
    assert np.isclose(naive_3.posteriors["very late"]["1autumn"], expected_posteriors_bramer["very late"]["1autumn"])
    assert np.isclose(naive_3.posteriors["very late"]["1winter"], expected_posteriors_bramer["very late"]["1winter"])
    assert np.isclose(naive_3.posteriors["very late"]["2high"], expected_posteriors_bramer["very late"]["2high"])
    assert np.isclose(naive_3.posteriors["very late"]["2normal"], expected_posteriors_bramer["very late"]["2normal"])
    assert np.isclose(naive_3.posteriors["very late"]["3none"], expected_posteriors_bramer["very late"]["3none"])
    assert np.isclose(naive_3.posteriors["very late"]["3heavy"], expected_posteriors_bramer["very late"]["3heavy"])

    assert np.isclose(naive_3.posteriors["cancelled"]["0saturday"], expected_posteriors_bramer["cancelled"]["0saturday"])
    assert np.isclose(naive_3.posteriors["cancelled"]["1spring"], expected_posteriors_bramer["cancelled"]["1spring"])
    assert np.isclose(naive_3.posteriors["cancelled"]["2high"], expected_posteriors_bramer["cancelled"]["2high"])
    assert np.isclose(naive_3.posteriors["cancelled"]["3heavy"], expected_posteriors_bramer["cancelled"]["3heavy"])
    

def test_naive_bayes_classifier_predict():
        # in-class Naive Bayes example (lab task #1)
    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test_inclass_example = [[1,5]]
    naive_1 = MyNaiveBayesClassifier()
    naive_1.fit(X_train_inclass_example, y_train_inclass_example)
    y_pred_inclass_example = naive_1.predict(X_test_inclass_example)

    y_true_inclass_example = ["yes"]

    # Asserting prediction is correct are correct
    assert y_pred_inclass_example == y_true_inclass_example

    # MA7 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    X_test_iphone = [[2,2, "fair"], [1,1,"excellent"]]
    naive_2 = MyNaiveBayesClassifier()
    naive_2.fit(X_train_iphone, y_train_iphone)
    y_pred_iphone = naive_2.predict(X_test_iphone)

    y_true_iphone = ["yes", "no"]

    # Asserting prediction is correct are correct
    assert y_pred_iphone == y_true_iphone


    # Bramer 3.2 train dataset
    header_train = ["day", "season", "wind", "rain", "class"]
    X_train_train = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]
    X_test_bramer = [["weekday", "winter", "high", "heavy"],["weekday", "summer", "high", "heavy"], ["sunday", "summer", "normal", "slight"]]
    naive_3 = MyNaiveBayesClassifier()
    naive_3.fit(X_train_train, y_train_train)
    y_pred_bramer = naive_3.predict(X_test_bramer)

    y_true_bramer = ["very late", "on time", "on time"]

    # Asserting prediction is correct are correct
    assert y_pred_bramer == y_true_bramer
def test_decision_tree_classifier_fit():
    # interview dataset
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    # note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
    # note: the attribute values are sorted alphabetically
    tree_interview = \
            ["Attribute", "att0",
                ["Value", "Junior", 
                    ["Attribute", "att3",
                        ["Value", "no", 
                            ["Leaf", "True", 3, 5]
                        ],
                        ["Value", "yes", 
                            ["Leaf", "False", 2, 5]
                        ]
                    ]
                ],
                ["Value", "Mid",
                    ["Leaf", "True", 4, 14]
                ],
                ["Value", "Senior",
                    ["Attribute", "att2",
                        ["Value", "no",
                            ["Leaf", "False", 3, 5]
                        ],
                        ["Value", "yes",
                            ["Leaf", "True", 2, 5]
                        ]
                    ]
                ]
            ]
    interview_decision_tree = MyDecisionTreeClassifier()
    interview_decision_tree.fit(X_train_interview, y_train_interview)

    assert tree_interview == interview_decision_tree.tree 

    # iphone dataset
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    # note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
    # note: the attribute values are sorted alphabetically
    tree_iphone = \
            ["Attribute", "att0",
                ["Value", 1, 
                    ["Attribute", "att1",
                        ["Value", 1, 
                            ["Leaf", "yes", 1, 5]
                        ],
                        ["Value", 2, 
                            ["Attribute", "att2", 
                                ["Value", "excellent",
                                    ["Leaf", "yes", 1, 2]
                                ],
                                ["Value", "fair",
                                    ["Leaf", "no", 1, 2]
                                ]
                            ]
                        ],
                        ["Value", 3, 
                            ["Leaf", "no", 2, 5]
                        ]
                    ]
                ],
                ["Value", 2,
                    ["Attribute", "att2",
                        ["Value", "excellent", 
                            ["Leaf", "no", 2, 4]
                        ],
                        ["Value", "fair", 
                            ["Leaf", "yes", 6, 10]
                        ]
                    ]
                ]
            ]
    dec_tree_iphone = MyDecisionTreeClassifier()
    dec_tree_iphone.fit(X_train_iphone, y_train_iphone)
    assert tree_iphone == dec_tree_iphone.tree 

def test_decision_tree_classifier_predict():
    # interview dataset
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    interview_decision_tree = MyDecisionTreeClassifier()
    interview_decision_tree.fit(X_train_interview, y_train_interview)
    X_test_interview = [["Junior", "Java", "yes", "no"],["Junior", "Java", "yes", "yes"]]
    y_pred_interview = interview_decision_tree.predict(X_test_interview)
    y_true_interview = ["True", "False"]
    assert y_true_interview == y_pred_interview

    # iphone dataset
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    
    dec_tree_iphone = MyDecisionTreeClassifier()
    dec_tree_iphone.fit(X_train_iphone, y_train_iphone)
    X_test_iphone = [[2, 2, "fair"],[1,1, "excellent"]]
    y_pred_iphone = dec_tree_iphone.predict(X_test_iphone)
    y_true_iphone = ["yes", "yes"]
    assert y_true_iphone == y_pred_iphone


