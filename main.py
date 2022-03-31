import model
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)


@app.route("/")
def home_page():
    return render_template("homepage.html")


@app.route("/", methods=["POST"])
def home_page_post():
    email = request.form.get("email_text")
    pred = mod.predict(np.array([email]))
    text = ""

    if "ham" in pred:
        text = "The email entered is not spam."
    else:
        text = "The email entered is spam."

    return render_template("homepage.html", text=text, accuracy=accuracy)


@app.route("/visualizations")
def visualizations_page():
    return render_template("visualizations.html", report=report)


if __name__ == '__main__':
    data = model.DataModel()
    X_train, X_test, y_train, y_test = data.gen_data()
    mod = data.load_model_from_file()
    accuracy, report = data.get_accuracy_results(y_test, mod.predict(X_test))

    app.run(debug=True)
