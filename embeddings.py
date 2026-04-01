import ollama
import numpy as np

class OllamaEmbeddings:
    def __init__(self, model: str = "mxbai-embed-large:latest", **kwargs):
        self.model = model
        self.kwargs = kwargs

    def ollama_embedding(self, text: str) -> list[float]:
        response = ollama.embeddings(model=self.model, prompt=text, **self.kwargs)
        return response["embedding"]
    
    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate the cosine similarity between two vectors."""
        vec1 = np.array(vec1) # Convert the input lists to numpy arrays for easier calculations ex. [1, 2, 3] would become array([1, 2, 3])
        vec2 = np.array(vec2) # Convert the input lists to numpy arrays for easier calculations ex. [4, 5, 6] would become array([4, 5, 6])
        dot_product = np.dot(vec1, vec2) # Calculate the dot product of the two vectors ex. [1, 2, 3] and [4, 5, 6] would give us 1*4 + 2*5 + 3*6 = 32
        norm_vec1 = np.linalg.norm(vec1) # Calculate the magnitude (norm) of the first vector ex. [1, 2, 3] would give us sqrt(1^2 + 2^2 + 3^2) = sqrt(14)
        norm_vec2 = np.linalg.norm(vec2) # Calculate the magnitude (norm) of the second vector ex. [4, 5, 6] would give us sqrt(4^2 + 5^2 + 6^2) = sqrt(77)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)
    
obj = OllamaEmbeddings()
text1 = "SIP (Systematic Investment Plan) is good for long term financial investment"
text2 = "Systematic investment plan helps long term growth"
text3 = "I like cricket"
embedding1 = obj.ollama_embedding(text1.lower().strip()) # Convert the text to lowercase before generating the embedding
embedding2 = obj.ollama_embedding(text2.lower().strip())
embedding3 = obj.ollama_embedding(text3.lower().strip())
print("Embedding for text1:", embedding1)
print("Embedding for text2:", embedding2)
print("Embedding for text3:", embedding3)

similarity = obj.cosine_similarity(embedding1 , embedding2)
print("Cosine Similarity between 'embedding1' and 'embedding2':", similarity)

similarity = obj.cosine_similarity(embedding1 , embedding3)
print("Cosine Similarity between 'embedding1' and 'embedding3':", similarity)

