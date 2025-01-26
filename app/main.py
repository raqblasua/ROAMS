from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from transformers import GPT2LMHeadModel, GPT2Tokenizer


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///requests.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 

model_name = "gpt2"  # Puedes usar cualquier otro modelo de Hugging Face
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

db = SQLAlchemy(app)

class RequestLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prompt = db.Column(db.String(500), nullable=False)
    generated_text = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'<RequestLog {self.id}>'

@app.route('/generate', methods=['POST'])
def generate_text():
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
    try:
        requests = RequestLog.query.all()
        if not requests:
            return jsonify({"message": "No history found"}), 404
        history = [{"id": r.id, "prompt": r.prompt, "generated_text": r.generated_text} for r in requests]
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)