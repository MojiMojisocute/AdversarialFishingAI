from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_email(prompt):
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    email_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return email_text
