import re
import os
import gzip

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

_translator = None


def _safe_translate(text: str) -> str:
    """
    Best-effort translation.
    Default is OFF to avoid flaky network dependency during long experiments.
    Set WEBTEST_ENABLE_TRANSLATE_API=1 to enable remote translation.
    """
    if os.environ.get("WEBTEST_ENABLE_TRANSLATE_API", "0") != "1":
        return text
    global _translator
    if _translator is None:
        try:
            from translate import Translator
            _translator = Translator(from_lang='chinese', to_lang='english')
        except Exception:
            return text
    try:
        out = _translator.translate(text)
        if isinstance(out, str) and out.strip():
            return out
        return text
    except Exception:
        return text


def load_embedding_model():
    from gensim.models import KeyedVectors
    import gensim.downloader as api

    class _EmptyWordVectors:
        index_to_key = []

        def __contains__(self, _word):
            return False

        def __bool__(self):
            return False

    def _detect_has_header(path: str):
        opener = gzip.open if path.lower().endswith(".gz") else open
        try:
            with opener(path, "rt", encoding="utf-8", errors="ignore") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    return len(parts) == 2 and all(p.isdigit() for p in parts)
        except Exception:
            return None
        return None

    def _load_local(path: str):
        has_header = _detect_has_header(path)
        if has_header is True:
            modes = [False, True]
        elif has_header is False:
            modes = [True, False]
        else:
            modes = [True, False]

        last_exc = None
        for no_header in modes:
            try:
                model = KeyedVectors.load_word2vec_format(
                    path, binary=False, no_header=no_header
                )
                print(f"Loaded embedding from local cache: {path} (no_header={no_header})")
                return model
            except Exception as exc:
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Failed to load embeddings from path: {path}")

    model_path = os.environ.get("WEBTEST_GLOVE_PATH", "").strip()
    if not model_path:
        model_path = os.path.join(
            os.path.expanduser("~"),
            "gensim-data",
            "glove-wiki-gigaword-200",
            "glove-wiki-gigaword-200.gz",
        )
    if model_path and os.path.isfile(model_path):
        try:
            wv_from_bin = _load_local(model_path)
        except Exception as exc:
            print(f"WARNING: failed to load local embedding '{model_path}': {exc}")
            wv_from_bin = None
    else:
        wv_from_bin = None

    if wv_from_bin is None and os.environ.get("WEBTEST_ALLOW_GENSIM_DOWNLOAD", "0") == "1":
        try:
            wv_from_bin = api.load("glove-wiki-gigaword-200")
        except Exception as exc:
            print(f"WARNING: failed to download gensim embedding; using empty embedding model: {exc}")

    if wv_from_bin is None:
        print("WARNING: using empty embedding model; set WEBTEST_GLOVE_PATH or WEBTEST_ALLOW_GENSIM_DOWNLOAD=1 for semantic action text features.")
        wv_from_bin = _EmptyWordVectors()

    print("Loaded vocab size %i" % len(list(wv_from_bin.index_to_key)))
    return wv_from_bin


# pprint.pprint(wv_from_bin.most_similar(positive=['buy']))

def calculate_similarity(text1, text2, word_vectors):
    tokens1 = text1.lower().split()
    tokens2 = text2.lower().split()

    vectors1 = [word_vectors[word] for word in tokens1 if word in word_vectors]
    vectors2 = [word_vectors[word] for word in tokens2 if word in word_vectors]

    if len(vectors1) == 0 or len(vectors2) == 0:
        return 0.0  # 如果任意一个向量列表为空，则相似度为0

    # 计算文本的平均词向量
    vector1 = np.mean(vectors1, axis=0)
    vector2 = np.mean(vectors2, axis=0)

    # print(f"vector1 shape: {vector1.shape}, vector2 shape: {vector2.shape}")

    vector1 = np.array(vector1).reshape(1, -1)
    vector2 = np.array(vector2).reshape(1, -1)
    # print(f"Reshaped vector1 shape: {vector1.shape}, Reshaped vector2 shape: {vector2.shape}")

    similarity = cosine_similarity(vector1, vector2)[0][0]

    return similarity


def generate(button_text, word_vectors, threshold=0.4):
    base_terms = ["accept", "confirm", "submit", "ok", "yes", "next", "okay"]
    button_text = str(button_text or "").lower()
    if not word_vectors:
        return 0.0

    similarities = []
    for term in base_terms:
        similarity = calculate_similarity(term, button_text, word_vectors)
        similarities.append(similarity)

    similarity_score = max(similarities)
    if similarity_score > threshold:
        return 1
    return 0


def embedding(text, count, children, word_vectors):
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    text = str(text or "").strip()

    # Prefer deterministic text preprocessing over random fallbacks.
    # Many UI labels contain spaces/punctuation (e.g., "Sign in ->"),
    # and should still contribute stable semantic features.
    if text and chinese_pattern.search(text):
        text = _safe_translate(text)

    normalized = re.sub(r"[^A-Za-z]+", " ", text).strip().lower()
    if normalized:
        text_similar = generate(normalized, word_vectors)
    else:
        text_similar = 0.0
    combined_vector = np.concatenate((np.array([text_similar]), np.array([count]), np.array(children)))
    return combined_vector



# if __name__ == '__main__':
#     wv_from_bin = load_embedding_model()
#     translator = Translator(from_lang='chinese', to_lang='english')
#     # result = translator.translate("")
#     result1 = embedding("确认", 1, [1, 2, 3, 4, 5, 6], wv_from_bin)
#     result1 = embedding("okay", 1, [1, 2, 3, 4, 5, 6], wv_from_bin)
#     result = embedding("dimeshift", 1, [1,2,3,4,5,6], wv_from_bin)
#     print(result)
