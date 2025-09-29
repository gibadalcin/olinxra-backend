import faiss
import numpy as np

class LogoIndex:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add_logo(self, vector, metadata):
        # A API do FAISS espera um vetor 2D, por isso usamos np.expand_dims
        self.index.add(np.expand_dims(vector, axis=0))
        self.metadata.append(metadata)

    def search(self, query_vector, top_k):
        # A API do FAISS espera um vetor 2D para busca
        D, I = self.index.search(np.expand_dims(query_vector, axis=0), top_k)
        
        results = []
        for i in range(len(I[0])):
            logo_index = I[0][i]
            distance = D[0][i]
            metadata = self.metadata[logo_index]
            
            results.append({
                "distance": float(distance),
                "metadata": metadata
            })
            
        return results