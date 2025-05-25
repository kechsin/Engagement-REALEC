from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from pprint import pprint
import pandas as pd
import numpy as np
import os
import csv


def get_data(directory="../annotated_texts/"):
    datalist = []
    for file in os.listdir(directory):
        with open("../annotated_texts/" + file) as f:
            r = csv.reader(f)
            for line in r:
                if len(line[0]) > 0 and line[0] != "Sentence":
                    line2 = line[:-1]
                    line2.append(file)
                    datalist.append(line2)
    return datalist


def is_subcategory(categories_parent, parent, offspring):
    x = categories_parent.get(offspring)
    while x is not None:
        if x == parent:
            return 1
        x = categories_parent.get(x)
    return 0


class MyModel:
    def __init__(self):
        self.categories = ("Deny", "Counter", "Concur", "Pronounce", "Endorse", "Entertain", "Acknowledge", "Distance")
        self.big_categories = ("Proclaim", "Disclaim", "Attribute", "Contract", "Expand", "Heterogloss")
        self.categories_parent = {"Deny": "Disclaim",
                                  "Counter": "Disclaim",
                                  "Concur": "Proclaim",
                                  "Pronounce": "Proclaim",
                                  "Endorse": "Proclaim",
                                  "Entertain": "Expand",
                                  "Acknowledge": "Attribute",
                                  "Distance": "Attribute",
                                  "Disclaim": "Contract",
                                  "Proclaim": "Contract",
                                  "Attribute": "Expand",
                                  "Expand": "Heterogloss",
                                  "Contract": "Heterogloss"
        }
        self.vectorizer = None
        self.model = {}
        self.model_name = None
        self.X = None
        self.ys = None
        self.X_train = None
        self.X_test = None
        self.ys_train = {}
        self.ys_test = {}
        self.ys_pred = {}
        self.data = None
        dicts = {"precision": 0, "recall": 0, "f1-score": 0, "support": 0}
        self.mean_metrics = {"0": dicts.copy(), "1": dicts.copy(), "accuracy": 0,
                        "macro avg": dicts.copy(), "weighted avg": dicts.copy()}

    def preprocess(self, data):
        texts = [i[0] for i in data]
        self.data = data
        self.X = self.vectorizer.fit_transform(texts)
        self.X_train, self.X_test = train_test_split(self.X, random_state=157)
        self.ys = {}
        for i in self.categories:
            self.ys[i] = np.array([int(i in doc[1]) for doc in data])
            self.ys_train[i], self.ys_test[i] = train_test_split(self.ys[i], random_state=157)
        for sentence in data:
            if sentence[1] not in self.categories and len(sentence[1]) != 0:
                temp = sentence[1][sentence[1].find('|')+1:]
                xx = temp.split(";")
                for i in xx:
                    if i not in self.categories:
                        print(sentence[1], sentence[-1])

    def print(self):
        print("AVERAGE ", self.model_name)
        pprint(self.mean_metrics)
        print("\n\n")


class BaseLineModel(MyModel):  # CountVectorizer, LogisticRegression
    def __init__(self):
        super().__init__()
        self.vectorizer = CountVectorizer()
        self.model_name = "Logistic Regression"

    def regression(self, unprocessed_data=None, print_results=True):
        if self.X is None:
            self.preprocess(unprocessed_data)
        for category in self.categories:
            class_weights = compute_class_weight('balanced', classes=np.unique(self.ys[category]), y=self.ys[category])
            self.model[category] = LogisticRegression(penalty='l2', class_weight=dict(enumerate(class_weights)))
            self.model[category] = LogisticRegression()
            self.model[category].fit(self.X_train, self.ys_train[category])
            self.ys_pred[category] = self.model[category].predict(self.X_test)
            if print_results:
                print(category)
                print(classification_report(self.ys_test[category], self.ys_pred[category], zero_division=np.nan))
            clf_report = classification_report(self.ys_test[category], self.ys_pred[category],
                                               zero_division=np.nan, output_dict=True)
            for i in clf_report:
                if i == "accuracy":
                    self.mean_metrics[i] += clf_report[i]
                    continue
                for j in clf_report[i]:
                    self.mean_metrics[i][j] += clf_report[i][j]
        for category2 in self.big_categories:
            y = np.array([is_subcategory(self.categories_parent, parent=category2, offspring=doc[1]) for doc in self.data])
            y_train, y_test = train_test_split(y, random_state=157)
            self.model[category2] = LogisticRegression(penalty='l2')
            self.model[category2].fit(self.X_train, y_train)
            y_pred = self.model[category2].predict(self.X_test)
            if print_results:
                print(category2)
                print(classification_report(y_test, y_pred, zero_division=np.nan))
            clf_report = classification_report(y_test, y_pred,
                                               zero_division=np.nan, output_dict=True)
            for i in clf_report:
                if i == "accuracy":
                    self.mean_metrics[i] += clf_report[i]
                    continue
                for j in clf_report[i]:
                    self.mean_metrics[i][j] += clf_report[i][j]
        for i in self.mean_metrics:
            if type(self.mean_metrics[i]) is not dict:
                self.mean_metrics[i] /= (len(self.categories) + len(self.big_categories))
                continue
            for j in self.mean_metrics[i]:
                self.mean_metrics[i][j] /= (len(self.categories) + len(self.big_categories))
        return self.mean_metrics


class ForestModel(MyModel):
    def __init__(self):
        super().__init__()
        self.vectorizer = TfidfVectorizer()
        self.model_name = "Random Forest"

    def learn(self, unprocessed_data=None, n=100, print_results=True):
        if self.X is None:
            self.preprocess(unprocessed_data)
        for category in self.categories:
            class_weights = compute_class_weight('balanced', classes=np.unique(self.ys[category]), y=self.ys[category])
            self.model[category] = RandomForestClassifier(n_estimators=n, class_weight=dict(enumerate(class_weights)))
            self.model[category].fit(self.X_train, self.ys_train[category])
            self.ys_pred[category] = self.model[category].predict(self.X_test)
            if print_results:
                print(category)
                print(classification_report(self.ys_test[category], self.ys_pred[category], zero_division=np.nan))
            clf_report = classification_report(self.ys_test[category], self.ys_pred[category],
                                               zero_division=np.nan, output_dict=True)
            for i in clf_report:
                if i == "accuracy":
                    self.mean_metrics[i] += clf_report[i]
                    continue
                for j in clf_report[i]:
                    self.mean_metrics[i][j] += clf_report[i][j]
        for category2 in self.big_categories:
            y = np.array(
                [is_subcategory(self.categories_parent, parent=category2, offspring=doc[1]) for doc in self.data])
            y_train, y_test = train_test_split(y, random_state=157)
            self.model[category2] = RandomForestClassifier(n_estimators=n)
            self.model[category2].fit(self.X_train, y_train)
            y_pred = self.model[category2].predict(self.X_test)
            if print_results:
                print(category2)
                print(classification_report(y_test, y_pred, zero_division=np.nan))
            clf_report = classification_report(y_test,y_pred,
                                               zero_division=np.nan, output_dict=True)
            for i in clf_report:
                if i == "accuracy":
                    self.mean_metrics[i] += clf_report[i]
                    continue
                for j in clf_report[i]:
                    self.mean_metrics[i][j] += clf_report[i][j]
        for i in self.mean_metrics:
            if type(self.mean_metrics[i]) is not dict:
                self.mean_metrics[i] /= (len(self.categories) + len(self.big_categories))
                continue
            for j in self.mean_metrics[i]:
                self.mean_metrics[i][j] /= (len(self.categories) + len(self.big_categories))
        return self.mean_metrics

    def find_best(self, unprocessed_data=None):
        options = [5, 10, 20, 50, 100]
        if unprocessed_data is not None:
            self.preprocess(unprocessed_data)
        means = {}
        best_mean = 0
        best_option = None
        for i in options:
            mean_macro_avg_f1 = 0
            self.learn(n=i, print_results=False)
            for category in self.categories:
                df = classification_report(self.ys_test[category], self.ys_pred[category],
                                           zero_division=np.nan, output_dict=True)
                mean_macro_avg_f1 += df['weighted avg']['f1-score']
            mean_macro_avg_f1 = mean_macro_avg_f1 / len(self.categories)
            means[i] = mean_macro_avg_f1
            if mean_macro_avg_f1 > best_mean:
                best_mean = mean_macro_avg_f1
                best_option = i
        return best_option, means


class SVCModel(MyModel):
    def __init__(self, vectorizer):
        super().__init__()
        self.vectorizer = vectorizer()
        self.model_name = "Linear SVC"

    def learn(self, unprocessed_data=None, print_results=True):
        if self.X is None:
            self.preprocess(unprocessed_data)
        for category in self.categories:
            class_weights = compute_class_weight('balanced', classes=np.unique(self.ys[category]), y=self.ys[category])
            self.model[category] = LinearSVC(class_weight=dict(enumerate(class_weights)))
            self.model[category].fit(self.X_train, self.ys_train[category])
            self.ys_pred[category] = self.model[category].predict(self.X_test)
            if print_results:
                print(category)
                print(classification_report(self.ys_test[category], self.ys_pred[category], zero_division=np.nan))
            clf_report = classification_report(self.ys_test[category], self.ys_pred[category],
                                                   zero_division=np.nan, output_dict=True)
            for i in clf_report:
                if i == "accuracy":
                    self.mean_metrics[i] += clf_report[i]
                    continue
                for j in clf_report[i]:
                    self.mean_metrics[i][j] += clf_report[i][j]
        for category2 in self.big_categories:
            y = np.array(
                [is_subcategory(self.categories_parent, parent=category2, offspring=doc[1]) for doc in self.data])
            y_train, y_test = train_test_split(y, random_state=157)
            self.model[category2] = LinearSVC()
            self.model[category2].fit(self.X_train, y_train)
            y_pred = self.model[category2].predict(self.X_test)
            if print_results:
                print(category2)
                print(classification_report(y_test, y_pred, zero_division=np.nan))
            clf_report = classification_report(y_test, y_pred,
                                               zero_division=np.nan, output_dict=True)
            for i in clf_report:
                if i == "accuracy":
                    self.mean_metrics[i] += clf_report[i]
                    continue
                for j in clf_report[i]:
                    self.mean_metrics[i][j] += clf_report[i][j]
        for i in self.mean_metrics:
            if type(self.mean_metrics[i]) is not dict:
                self.mean_metrics[i] /= (len(self.categories) + len(self.big_categories))
                continue
            for j in self.mean_metrics[i]:
                self.mean_metrics[i][j] /= (len(self.categories) + len(self.big_categories))
        return self.mean_metrics


if __name__ == "__main__":
    np.random.seed(157)
    data = get_data()
    model1 = BaseLineModel()
    model1.regression(data)
    model1.print()
    model2 = ForestModel()
    means = model2.find_best(data)
    print(means)
    model2.learn(data, n=means[0])
    model2.print()
    model3_1 = SVCModel(CountVectorizer)
    model3_1.learn(data)
    model3_1.print()
    model3_1 = SVCModel(TfidfVectorizer)
    model3_1.learn(data)
    model3_1.print()
