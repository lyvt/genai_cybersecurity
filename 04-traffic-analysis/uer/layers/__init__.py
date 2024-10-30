from .embeddings import WordEmbedding
from .embeddings import WordPosEmbedding
from .embeddings import WordPosSegEmbedding
from .embeddings import WordSinusoidalposEmbedding


str2embedding = {"word": WordEmbedding, "word_pos": WordPosEmbedding, "word_pos_seg": WordPosSegEmbedding,
                 "word_sinusoidalpos": WordSinusoidalposEmbedding}

__all__ = ["WordEmbedding", "WordPosEmbedding", "WordPosSegEmbedding",
           "WordSinusoidalposEmbedding", "str2embedding"]
