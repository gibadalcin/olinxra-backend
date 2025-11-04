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
        # Threshold mais restritivo para evitar falsos positivos
        # Distância L2: 0.0 = perfeito, 0.3 = muito bom, 0.5 = aceitável
        threshold = 0.35  # Reduzido de 0.8 para 0.35 para maior precisão
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
        
        # Log detalhado de TODOS os candidatos (não apenas os aceitos)
        print(f"\n{'='*60}")
        print(f"🔍 FAISS Search - Top {top_k} candidatos:")
        print(f"{'='*60}")
        for i in range(len(I[0])):
            logo_index = I[0][i]
            distance = D[0][i]
            metadata = self.metadata[logo_index]
            confidence = 1 / (1 + distance)
            aceito = "✅ ACEITO" if distance < threshold else "❌ REJEITADO"
            print(f"{aceito} | Nome: {metadata.get('nome', 'N/A'):20s} | Distância: {distance:.4f} | Confidence: {confidence*100:.2f}%")
        print(f"{'='*60}")
        print(f"Threshold atual: {threshold} | Matches aceitos: {len(results)}")
        print(f"{'='*60}\n")
        
        return results