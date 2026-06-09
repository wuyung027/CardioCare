import unittest
import pandas as pd

import sys
import os

# src 폴더의 모듈을 import하기 위한 경로 설정
sys.path.append(os.path.abspath("src"))

from inference import (
    load_model,
    create_sample_input,
    predict,
    validate_input_data,
)


class TestCardioCarePipeline(unittest.TestCase):
    """
    CardioCare 머신러닝 파이프라인 단위 테스트
    """

    @classmethod
    def setUpClass(cls):
        """
        모든 테스트에서 공통으로 사용할 모델과 샘플 데이터를 준비한다.
        """
        cls.model = load_model()
        cls.sample_input = create_sample_input()

    def test_prediction_shape_matches_input_shape(self):
        """
        테스트 1.
        예측 결과의 행 개수가 입력 데이터의 행 개수와 일치하는지 확인한다.
        """
        result = predict(self.model, self.sample_input)

        self.assertEqual(len(result), len(self.sample_input))
        self.assertIn("prediction", result.columns)

    def test_prediction_probability_range(self):
        """
        테스트 2.
        예측 확률이 0 이상 1 이하인지 확인한다.
        """
        result = predict(self.model, self.sample_input)

        self.assertIn("disease_probability", result.columns)
        self.assertTrue(result["disease_probability"].between(0, 1).all())

    def test_predict_proba_sum_to_one(self):
        """
        테스트 3.
        각 샘플에 대한 클래스별 예측 확률의 합이 1인지 확인한다.
        """
        probabilities = self.model.predict_proba(self.sample_input)
        row_sums = probabilities.sum(axis=1)

        for value in row_sums:
            self.assertAlmostEqual(value, 1.0, places=5)

    def test_clinical_input_range_validation(self):
        """
        테스트 4.
        임상적으로 범위가 정해진 특성값이 정상 범위일 때 검증을 통과하는지 확인한다.
        """
        self.assertTrue(validate_input_data(self.sample_input))

    def test_invalid_chol_range_raises_error(self):
        """
        테스트 5.
        chol 값이 비정상 범위이면 ValueError가 발생하는지 확인한다.
        """
        invalid_input = self.sample_input.copy()
        invalid_input.loc[0, "chol"] = 999

        with self.assertRaises(ValueError):
            validate_input_data(invalid_input)

    def test_deterministic_prediction(self):
        """
        테스트 6.
        같은 입력 데이터에 대해 같은 예측 결과가 나오는지 확인한다.
        """
        result_1 = predict(self.model, self.sample_input)
        result_2 = predict(self.model, self.sample_input)

        self.assertTrue(
            result_1["prediction"].equals(result_2["prediction"])
        )

        self.assertTrue(
            result_1["disease_probability"].equals(result_2["disease_probability"])
        )


if __name__ == "__main__":
    unittest.main()