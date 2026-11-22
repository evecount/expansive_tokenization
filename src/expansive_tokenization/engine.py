"""
Core Engine for Expansive Tokenization:
1. Subword Fragmentation & Fertility Auditing
2. Live Tokenizer Expansion & Compositional Semantic Embedding Initialization
3. Contextual Embedding Shift Analysis (768-D Cosine Geometry across Contexts of Interest)
4. JSON Export for Custom Tokenization Sets
"""

import json
import copy
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .presets import TokenizationSet, ContrastPair


def audit_tokenization_fragmentation(
    tokenizer: Any,
    token_set: TokenizationSet,
) -> Dict[str, Any]:
    """
    Audits how a baseline tokenizer fragments the specialized vocabulary of a TokenizationSet.

    Computes:
    - Subword Fertility (average tokens per specialized concept)
    - Semantic Fracture Rate (% of domain concepts split into 2+ fragments)
    - Per-term breakdown of subword pieces and token IDs
    """
    term_reports = []
    total_subwords = 0
    fractured_count = 0

    for category, terms in token_set.ontological_categories.items():
        for term in terms:
            # Normalize underscores to spaces for natural phrase baseline comparison,
            # while also testing the raw token string
            subwords = tokenizer.tokenize(term)
            ids = tokenizer.convert_tokens_to_ids(subwords)
            n_pieces = len(subwords)
            is_fractured = n_pieces > 1
            total_subwords += n_pieces
            if is_fractured:
                fractured_count += 1

            term_reports.append({
                "category": category,
                "term": term,
                "baseline_subwords": subwords,
                "baseline_token_ids": ids,
                "num_pieces": n_pieces,
                "fractured": is_fractured,
            })

    n_terms = max(len(term_reports), 1)
    fertility = round(total_subwords / n_terms, 2)
    fracture_rate_pct = round(100.0 * fractured_count / n_terms, 1)

    # Also measure corpus compression before expansion
    corpus_baseline_lengths = [len(tokenizer.tokenize(sent)) for sent in token_set.sample_corpus]

    return {
        "pack_id": token_set.pack_id,
        "title": token_set.title,
        "num_domain_terms": len(term_reports),
        "baseline_subword_fertility": fertility,
        "semantic_fracture_rate_pct": fracture_rate_pct,
        "total_baseline_pieces": total_subwords,
        "expanded_atomic_pieces": len(term_reports),
        "compression_ratio": round(total_subwords / n_terms, 2),
        "corpus_baseline_token_counts": corpus_baseline_lengths,
        "term_reports": term_reports,
    }


def expand_tokenizer_with_context_set(
    tokenizer: Any,
    model: Optional[Any],
    token_set: TokenizationSet,
    initialize_by_semantic_average: bool = True,
) -> Tuple[Any, Optional[Any], Dict[str, Any]]:
    """
    Expands a Hugging Face tokenizer (and optionally resizes & initializes a model's embedding layer)
    with the ontological categories and domain tokens of a TokenizationSet.

    When `initialize_by_semantic_average=True` and a PyTorch model is provided, each new token's
    embedding vector is initialized to the mean of its semantic definition's constituent embeddings
    (or its subword constituent embeddings), placing it immediately in the right semantic neighborhood!
    """
    expanded_tokenizer = copy.deepcopy(tokenizer)
    original_vocab_size = len(expanded_tokenizer)

    # 1. Register ontological category markers as special tokens
    category_markers = list(token_set.ontological_categories.keys())
    num_special_added = expanded_tokenizer.add_special_tokens(
        {"additional_special_tokens": category_markers}
    )

    # 2. Register domain terms as atomic tokens
    domain_terms = token_set.all_tokens
    num_domain_added = expanded_tokenizer.add_tokens(domain_terms)

    new_vocab_size = len(expanded_tokenizer)
    initialized_tokens: List[Dict[str, Any]] = []

    if model is not None:
        import torch

        model.resize_token_embeddings(new_vocab_size)

        if initialize_by_semantic_average:
            with torch.no_grad():
                input_embeddings = model.get_input_embeddings().weight

                for term in domain_terms:
                    new_id = expanded_tokenizer.convert_tokens_to_ids(term)
                    # Use semantic definition if available, else constituent subwords from original tokenizer
                    definition_text = token_set.semantic_definitions.get(
                        term, term.replace("_", " ").replace("-", " ")
                    )
                    source_ids = tokenizer.encode(definition_text, add_special_tokens=False)
                    if source_ids:
                        mean_vec = input_embeddings[source_ids].mean(dim=0)
                        input_embeddings[new_id] = mean_vec
                        initialized_tokens.append({
                            "term": term,
                            "new_token_id": int(new_id),
                            "initialized_from": definition_text,
                            "num_source_vectors_averaged": len(source_ids),
                            "vector_norm": round(float(torch.norm(mean_vec).item()), 4),
                        })

    # Compare corpus token counts before vs after
    before_counts = [len(tokenizer.tokenize(s)) for s in token_set.sample_corpus]
    after_counts = [len(expanded_tokenizer.tokenize(s)) for s in token_set.sample_corpus]

    summary = {
        "pack_id": token_set.pack_id,
        "original_vocab_size": original_vocab_size,
        "new_vocab_size": new_vocab_size,
        "special_category_tokens_added": num_special_added,
        "domain_atomic_tokens_added": num_domain_added,
        "corpus_tokens_before": sum(before_counts),
        "corpus_tokens_after": sum(after_counts),
        "token_savings_pct": round(
            100.0 * (1.0 - (sum(after_counts) / max(sum(before_counts), 1))), 1
        ),
        "initialized_tokens": initialized_tokens,
    }
    return expanded_tokenizer, model, summary


def _extract_target_word_embedding(
    tokenizer: Any,
    model: Any,
    sentence: str,
    target_word: str,
) -> np.ndarray:
    """
    Runs a sentence through a Transformer model and extracts the contextualized embedding
    vector (e.g. 768-D for BERT) corresponding to `target_word` (averaging subwords if split).
    """
    import torch

    inputs = tokenizer(sentence, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Support both AutoModel (last_hidden_state) and AutoModelForCausalLM (hidden_states[-1])
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        hidden = outputs.last_hidden_state[0].detach().cpu().numpy()
    else:
        hidden = outputs.hidden_states[-1][0].detach().cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    target_lower = target_word.lower()

    matched_indices = []
    for idx, tok in enumerate(tokens):
        clean_tok = tok.lower().lstrip("#Ġ ")
        if clean_tok and (clean_tok == target_lower or target_lower.startswith(clean_tok)):
            matched_indices.append(idx)

    if not matched_indices:
        # Fallback to mean pooling across content tokens
        return hidden.mean(axis=0)

    return hidden[matched_indices].mean(axis=0)


def compute_contextual_embedding_shift(
    tokenizer: Any,
    model: Any,
    contrast_pairs: List[ContrastPair],
) -> List[Dict[str, Any]]:
    """
    Measures how the exact same word shifts its 768-D embedding vector depending on
    the Context of Interest / Interlocutor (Sentence A vs. Sentence B).
    """
    results = []
    for pair in contrast_pairs:
        vec_a = _extract_target_word_embedding(tokenizer, model, pair.sentence_a, pair.target_word)
        vec_b = _extract_target_word_embedding(tokenizer, model, pair.sentence_b, pair.target_word)

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        cosine_sim = float(np.dot(vec_a, vec_b) / max(norm_a * norm_b, 1e-8))
        semantic_shift_distance = round(1.0 - cosine_sim, 4)
        l2_distance = round(float(np.linalg.norm(vec_a - vec_b)), 4)

        results.append({
            "target_word": pair.target_word,
            "context_a_label": pair.context_a_label,
            "context_b_label": pair.context_b_label,
            "sentence_a": pair.sentence_a,
            "sentence_b": pair.sentence_b,
            "embedding_dim": int(vec_a.shape[0]),
            "cosine_similarity": round(cosine_sim, 4),
            "semantic_shift_distance": semantic_shift_distance,
            "euclidean_l2_distance": l2_distance,
            "vec_a_preview": [round(float(x), 4) for x in vec_a[:5]],
            "vec_b_preview": [round(float(x), 4) for x in vec_b[:5]],
            "semantic_note": pair.semantic_note,
        })
    return results


def export_tokenization_set_json(token_set: TokenizationSet, filepath: str) -> str:
    """Exports a custom TokenizationSet to a clean, portable JSON specification."""
    payload = token_set.to_dict()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return filepath
