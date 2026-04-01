import numpy as np
from embeddings import OllamaEmbeddings

# Models to test
MODELS = [
    "nomic-embed-text",
    "mxbai-embed-large",
    "phi3:mini"
]

# Dataset
test_data = [
    ("SIP investment is good", "Systematic investment plan is beneficial", 1),
    ("Mutual funds are safe", "Funds managed by experts", 1),
    ("SIP investment", "I like cricket", 0),
    ("Stock market", "Pizza is tasty", 0),
]

def evaluate_model(model_name):
    obj = OllamaEmbeddings(model=model_name)
    similarities = []
    correct = 0
    for text1, text2, label in test_data:
        embedding1 = obj.ollama_embedding(text1.lower().strip())
        embedding2 = obj.ollama_embedding(text2.lower().strip())
        similarity = obj.cosine_similarity(embedding1, embedding2)
        prediction = 1 if similarity > 0.6 else 0
        print(f"Correct prediction for '{text1}' and '{text2}' with similarity {similarity:.4f}")
        if prediction == label:
            correct += 1
        similarities.append((similarity, label, prediction))
    print(f"Model: {model_name}, Accuracy: {correct/len(test_data)*100:.2f}%")

    return similarities

if __name__ == "__main__":
    for model in MODELS:
        print(f"Evaluating model: {model}")
        results = evaluate_model(model)
        for similarity, label, prediction in results:
            print(f"Similarity: {similarity:.4f}, Label: {label}, Prediction: {prediction}")
        print("\n")