from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt= data.get('prompt', '')
    max_length = data.get('max_length', 50)
    temperature = data.get('temperature', 1.0)
    top_p = data.get('top_p', 0.9)

    if not prompt: return jsonify({"error": "Prompt is required"}), 400

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length, temperature=temperature, top_p=top_p, do_sample=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"generated_text": generated_text})


if __name__ == '__main__':
    app.run(debug=True)