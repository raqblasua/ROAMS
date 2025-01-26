import unittest
from app.main import app
import json

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_generate_text_success(self):
        response = self.app.post('/generate', json={"prompt": "Once upon a time"})
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("generated_text", data)
        self.assertIsInstance(data["generated_text"], str)

    def test_generate_text_no_prompt(self):
        response = self.app.post('/generate', json={})
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", data)
        self.assertEqual(data["error"], "Prompt is required")

    def test_generate_text_invalid_max_length(self):
        data = {
            "prompt": "Test",
            "max_length": -10,  
            "temperature": 1.0,
            "top_p": 0.9
        }

        response = self.app.post('/generate', json=data)

        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()
