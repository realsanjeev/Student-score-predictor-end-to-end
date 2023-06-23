from flask import render_template, Flask, request, url_for

from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def index():
    pipeline = PredictPipeline()
    features = pipeline.get_features()
    in_features = dict()
    if request.method == "POST":
        in_features["gender"] = request.form.get("gender")
        in_features["race_ethnicity"] = request.form.get("race_ethnicity")
        in_features["parental_level_of_education"] = request.form.get("parental_level_of_education")
        in_features["lunch"] = request.form.get("lunch")
        in_features["test_preparation_course"] = request.form.get("test_preparation_course")

        in_features["writing_score"] = int(request.form.get("writing_score"))
        in_features["reading_score"] = int(request.form.get("reading_score"))

        scaled_data = pipeline.preprocess(in_features)
        result = int(pipeline.predict_score(scaled_data=scaled_data)[0])
        print("%")
        return render_template("index.html", features=features, result=result)
    elif request.method == "GET":
        return render_template("index.html", features=features)

if __name__=="__main__":
    app.run(debug=True)