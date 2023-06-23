

class PredictPipeline:
    def __init__(self) -> None:
        self.features = {
            "categorical": {
                "gender":['male', 'female'],
                "race_ethnicity":['group A', 'group B', 'group C', 'group D'],
                "parental_level_of_education":["bachelor's degree", "some college", 
                                               "master's degree", "associate's degree"],
                "lunch":["standard", "free/reduced"],
                "test_preparation_course":["completed", "none"],
            },
            "numerical": [
                   "writing_score",
                    "reading_score"
                    ]
        }

    def preprocess(self, gender: str,
                    race_ethnicity: str,
                    parental_level_of_education: str,
                    lunch: str,
                    test_preparation_course: str,
                    reading_score: int,
                    writing_score: int):
        pass

    def get_features(self):
        return self.features 