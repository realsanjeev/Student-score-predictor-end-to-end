from flask import render_template, Flask, request, url_for

from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def index():
    pipeline = PredictPipeline()
    features = pipeline.get_features()
    if request.method == "POST":
        gender = request.form.get("gender")
        race_ethnicity = request.form.get("race_ethnicity")
        parental_level_of_education = request.form.get("parental_level_of_education")
        lunch = request.form.get("lunch")
        test_preparation_course = request.form.get("test_preparation_course")

        writing_score = int(request.form.get("writing_score"))
        reading_score = int(request.form.get("reading_score"))

        scaled_data = pipeline.preprocess(gender, race_ethnicity, 
                                          parental_level_of_education, 
                                          lunch, test_preparation_course, 
                                          reading_score, writing_score)
        result = 10
        print(gender,
            race_ethnicity,
            parental_level_of_education,
            lunch,
            test_preparation_course,
            writing_score,
            reading_score)
        return render_template("index.html", features=features, result=result)
    elif request.method == "GET":
        return render_template("index.html", features=features)

if __name__=="__main__":
    app.run(debug=True)