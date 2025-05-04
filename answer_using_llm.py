from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load FLAN-T5 (generative LLM)
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def answer_question_from_captions(captions, question):
    # Combine captions into a context block
    context = " ".join(captions)
    
    # Format prompt for instruction-based LLMs
    prompt = f"Given the following descriptions:\n{context}\nAnswer the question: {question}. Donot hallucinate. Provide short answer."

    # Tokenize and generate answer
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Example
captions = [
    "A man in white shirt is sitting on a bench in the park.",
    "He is reading a comic newspaper.",
    "A dog is lying next to him.",
    "The weather appears to be sunny and pleasant."
]
question = "What is the man reading ?"

answer = answer_question_from_captions(captions, question)
print("Answer:", answer)