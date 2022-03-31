import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


class DataModel:
    def __init__(self):
        self.file_name = "spam_or_not.pkl"
        self.df = pd.read_csv("emails.txt")

    def train_model(self, X_train, y_train):
        return Pipeline([("vectorizer", CountVectorizer()), ("nb", MultinomialNB())]).fit(X_train, y_train)

    def gen_data(self):
        return train_test_split(self.df["Message"], self.df["Category"], test_size=0.25, random_state=0)

    def save_to_file(self, model):
        joblib.dump(model, self.file_name)

    def load_model_from_file(self):
        return joblib.load(self.file_name)

    def get_accuracy_results(self, not_predicted, predicted):
        accuracy = "Accuracy: {} %".format(round(accuracy_score(not_predicted, predicted) * 100, 1))
        report = pd.DataFrame(classification_report(not_predicted, predicted, output_dict=True))
        return accuracy, report

    def get_predict_incorrect_visual(self, model):
        pred = model.predict(self.df["Message"])
        total_emails, total_not_spam, total_spam = self.count_emails()
        incorrect = abs(total_not_spam - (pred == "ham").sum())
        plt.title("Emails Predicted Correctly/Incorrectly From The Dataset")
        correct_percent = round(((total_emails - incorrect) / total_emails) * 100, 1)
        incorrect_percent = round((incorrect / total_emails) * 100, 1)
        plt.pie([correct_percent, incorrect_percent],
                labels=["Correct", "Incorrect"], autopct="%1.1f%%", pctdistance=1.15,
                labeldistance=1.33, textprops={"fontsize": 13})
        plt.axis("equal")
        plt.savefig("piechart.png")

    def show_confusion_matrix(self, not_predicted, predicted):
        labels = ["spam", "not spam"]
        mat = confusion_matrix(not_predicted, predicted, labels=["spam", "ham"])
        seaborn.heatmap(mat, yticklabels=labels,
                        xticklabels=labels, annot=True, fmt=".0f", square=True,
                        linewidths=0.6, annot_kws={"fontsize": 15})
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")

    def make_bar_graph_emails(self):
        plt.title("The number of emails used to train the model")
        total_emails, total_not_spam, total_spam = self.count_emails()
        names = ["Total Emails", "Not Spam", "Spam"]
        count = [total_emails, total_not_spam, total_spam]
        plt.bar(names, count)
        plt.savefig("bargraph_email.png")

    def count_emails(self):
        not_spam = len(self.df[self.df["Category"] == "ham"])
        spam = len(self.df[self.df["Category"] != "ham"])
        return len(self.df), not_spam, spam