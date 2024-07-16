from transformers import pipeline

# Load a public model directly
try:
    generator = pipeline('text-generation', model='gpt-2')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
