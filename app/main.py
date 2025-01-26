from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flasgger import Swagger

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///requests.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 

TOKEN= "Y355AlAzbY08YraTOO52pE7I8QgJz0ZRoH1GgYgqUz6sQiukQdt8lEelCMOACD7l"

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

template = {
  "swagger": "2.0",
  "info": {
    "title": "Text Generation API",
    "description": "API for generating text using GPT-2",
    "contact": {
      "responsibleOrganization": "BLANCO SUAREZ RAQUEL",
      "responsibleDeveloper": "BLANCO SUAREZ RAQUEL",
      "email": "blancosuarezraquel@gmail.com",
      "url": "https://www.linkedin.com/in/raqblasua/",
    },
    "version": "0.0.1"
  },
  "host": "localhost:5000", 
  "basePath": "/",  
  "schemes": [
    "http",
    "https"
  ],
}

swagger = Swagger(app, template=template)
db = SQLAlchemy(app)

class RequestLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prompt = db.Column(db.String(500), nullable=False)
    generated_text = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'<RequestLog {self.id}>'
    

def check_token():
    token = request.headers.get("Authorization")
    if token != f'Bearer={TOKEN}': return jsonify({"error": "Unauthorized"}), 403
    return None


@app.route('/generate', methods=['POST'])
def generate_text():
    """
    Generate text based on a prompt.
    ---
    tags:
      - Text Generation
    parameters:
      - in: header
        name: Authorization
        type: string
        required: true
        description: Bearer token for authentication.
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            prompt:
              type: string
              example: "Once upon a time"
            max_length:
              type: integer
              example: 50
            temperature:
              type: number
              example: 1.0
            top_p:
              type: number
              example: 0.9
    responses:
      200:
        description: Generated text
        schema:
          type: object
          properties:
            generated_text:
              type: string
              example: "Once upon a time, there was a brave knight..."
      400:
        description: Bad Request
      403:
        description: Unauthorized
    """

    auth_response = check_token()
    if auth_response: return auth_response

    data = request.json
    prompt= data.get('prompt', '')
    max_length = data.get('max_length', 50)
    temperature = data.get('temperature', 1.0)
    top_p = data.get('top_p', 0.9)

    if not prompt: return jsonify({"error": "Prompt is required"}), 400

    if max_length <= 0:
        return jsonify({"error": "max_length must be greater than 0"}), 400
    if not (0 < temperature <= 2):
        return jsonify({"error": "temperature must be between 0 and 2"}), 400
    if not (0 < top_p <= 1):
        return jsonify({"error": "top_p must be between 0 and 1"}), 400

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length, temperature=temperature, top_p=top_p, do_sample=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    new_request = RequestLog(prompt=prompt, generated_text=generated_text)
    db.session.add(new_request)
    db.session.commit()

    return jsonify({"generated_text": generated_text})

@app.route('/history', methods=['GET'])
def get_history():
    """
        Get the history of generated texts.
        ---
        tags:
        - History
        responses:
            200:
                description: List of generated texts
                schema:
                type: array
                items:
                    type: object
                    properties:
                    id:
                        type: integer
                        example: 1
                    prompt:
                        type: string
                        example: "Once upon a time"
                    generated_text:
                        type: string
                        example: "Once upon a time, there was a brave knight..."
            403:
                description: Unauthorized
    """

    try:
        auth_response = check_token()
        if auth_response: 
            return auth_response
        
        requests = RequestLog.query.all()
        if not requests:
            return jsonify({"message": "No history found"}), 404
        
        history = [{"id": r.id, "prompt": r.prompt, "generated_text": r.generated_text} for r in requests]
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)