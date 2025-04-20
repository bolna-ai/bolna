from typing import List, Optional, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastembed import TextEmbedding

from bolna.helpers.logger_config import configure_logger
from bolna.memory.cache.base_cache import BaseCache

logger = configure_logger(__name__)


class VectorCache(BaseCache):
    def __init__(self, index_provider=None, embedding_model="BAAI/bge-small-en-v1.5"):
        super().__init__()
        self.index_provider = index_provider
        self.embedding_model = TextEmbedding(model_name=embedding_model)
        self.documents: List[str] = []
        self.embeddings: List[List[float]] = []

    def set(self, documents: List[str]):
        """
        Store documents and their embeddings in the cache.
        """
        self.documents = documents
        self.embeddings = list(self.embedding_model.passage_embed(documents))
        logger.info(f"Cached {len(documents)} documents.")

    def _get_most_similar_document(self, query_embedding: np.ndarray) -> str:
        """
        Return the document most similar to the query embedding.
        """
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        best_match_idx = np.argmax(similarities)
        return self.documents[best_match_idx]

    def get(self, query: Union[str, List[str]]) -> Optional[str]:
        """
        Retrieve the most relevant document for the given query.
        """
        if self.index_provider is not None:
            logger.info("Custom index_provider support is not implemented yet.")
            return None

        query_embedding = list(self.embedding_model.query_embed(query))[0]
        return self._get_most_similar_document(query_embedding)
