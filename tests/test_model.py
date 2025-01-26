from app.models import tokenizer, model

def generate_text(prompt, max_length=50, temperature=1.0, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(**inputs, 
                    max_length=max_length, 
                    temperature=temperature, 
                    top_p=top_p, 
                    do_sample=True )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    prompt = "Once upon a time"
    generated_text = generate_text(prompt)
    print("Generated Text:", generated_text)
