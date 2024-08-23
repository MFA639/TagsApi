import unittest
from app import app, model, mlb_classes

class ModelTests(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.app = app.test_client()

    def test_model_loading(self):
        self.assertIsNotNone(model)
        self.assertIsNotNone(mlb_classes)

    def test_index(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode(), "Model loaded successfully!")

    def test_predict(self):
        response = self.app.post('/predict', json={'text': 'How to train a neural network?'})
        self.assertEqual(response.status_code, 200)
        self.assertTrue('error' not in response.json)

if __name__ == "__main__":
    unittest.main()

