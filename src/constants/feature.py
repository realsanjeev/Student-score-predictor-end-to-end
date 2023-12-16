from dataclasses import dataclass

@dataclass
class FeaturesConfig:
    NUMERICAL_FEATURES = [
                    "writing_score",
                    "reading_score"
                    ]
    CATEGORICAL_FEATURES = [
                    "gender",
                    "race_ethnicity",
                    "parental_level_of_education",
                    "lunch",
                    "test_preparation_course",
                ]
    TARGET_FEATURE = "math_score"
