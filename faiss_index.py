import faiss
import numpy as np

class LogoIndex:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add_logo(self, vector, metadata):
        # Normaliza o vetor para norma unitária
        vector = vector / np.linalg.norm(vector)
        self.index.add(np.expand_dims(vector, axis=0))
        self.metadata.append(metadata)

    def search(self, query_vector, top_k):
        # Normaliza o vetor de consulta
        query_vector = query_vector / np.linalg.norm(query_vector)
        D, I = self.index.search(np.expand_dims(query_vector, axis=0), top_k)

        results = []
        threshold = 0.8
        for i in range(len(I[0])):
            logo_index = I[0][i]
            distance = D[0][i]
            metadata = self.metadata[logo_index]
            confidence = 1 / (1 + distance)
            if distance < threshold:
                results.append({
                    "distance": float(distance),
                    "confidence": round(confidence * 100, 2),
                    "metadata": metadata
                })
        print("Resultados FAISS:")
        for r in results:
            print(f"Nome: {r['metadata'].get('name', '')} | Distância: {r['distance']:.4f} | Confidence: {r['confidence']:.2f}")
        return results