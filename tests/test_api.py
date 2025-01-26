import unittest
import json
from app.main import app, db, RequestLog  

default_prompt = "Once upon a time"
class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test_requests.db'
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        app.config['TESTING'] = True
        cls.client = app.test_client()
        with app.app_context():
            db.create_all()  

    @classmethod
    def tearDownClass(cls):
        with app.app_context():
            db.drop_all()

    
    def test_generate_text_valid_prompt(self):
        response = self.client.post('/generate', 
            data=json.dumps({
                'prompt': default_prompt, 
                'max_length': 50, 
                'temperature': 1.0, 
                'top_p': 0.9}),
            content_type='application/json')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('generated_text', data)
        self.assertIsInstance(data['generated_text'], str)

    def test_generate_text_empty_prompt(self):
        response = self.client.post('/generate', 
            data=json.dumps({
                'prompt': '', 
                'max_length': 50, 
                'temperature': 1.0, 
                'top_p': 0.9}),
            content_type='application/json')

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'Prompt is required')

    def test_generate_text_invalid_max_length(self):
        response = self.client.post('/generate', 
            data=json.dumps({
                'prompt': default_prompt, 
                'max_length': -1, 
                'temperature': 1.0, 
                'top_p': 0.9}),
            content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'max_length must be greater than 0')

    def test_generate_text_invalid_temperature(self):
        response = self.client.post('/generate', 
            data=json.dumps({
                'prompt': default_prompt, 
                'max_length': 50, 
                'temperature': 3.0, 
                'top_p': 0.9}),
            content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'temperature must be between 0 and 2')

    def test_generate_text_invalid_top_p(self):
        response = self.client.post('/generate', 
            data=json.dumps({
                'prompt': default_prompt, 
                'max_length': 50, 
                'temperature': 1.0, 
                'top_p': 1.5}),
            content_type='application/json')

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'top_p must be between 0 and 1')

    def test_get_history_empty(self):
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertEqual(data['message'], 'No history found')

    def test_get_history_no_empty(self):
        with app.app_context():
            new_request = RequestLog(prompt='Test prompt', 
                    generated_text='Generated response')
            db.session.add(new_request)
            db.session.commit()

        response = self.client.get('/history')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertGreater(len(data), 0)
        self.assertEqual(data[0]['prompt'], 'Test prompt')

if __name__ == '__main__':
    unittest.main()