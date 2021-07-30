from flask import Flask, render_template, request
from sent_dating import date_document, min_year, max_year

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():
    year, sents = date_document(request.form["paper"])
    return render_template("result.html", year=year, sents=sents, min_year=min_year, max_year=max_year)


if __name__ == "__main__":
    app.run()
