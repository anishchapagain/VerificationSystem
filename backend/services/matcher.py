"""
Signature Matching Service
============================
Module  : services/matcher.py
Purpose : Compare a query embedding against stored reference embeddings
          using cosine similarity and apply a threshold to produce a verdict.

Author  : Signature Verifier Team
Version : 1.0.0
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from backend.core.exceptions import NoReferenceSignatureError, VectorStoreError
from backend.core.logger import get_logger

log = get_logger("matcher")


@dataclass
class MatchResult:
    """
    Value object returned by the SignatureMatcher.

    Attributes:
        score          : Best cosine similarity score found [0.0 – 1.0].
        verdict        : True = MATCH (score >= threshold), False = NO MATCH.
        best_sig_id    : DB primary key of the closest matching signature.
        threshold_used : The threshold applied to reach this verdict.
        all_scores     : Dict of {signature_id: score} for all comparisons.
        confidence     : Human-readable confidence label.
    """
    score: float
    verdict: bool
    best_sig_id: Optional[int]
    threshold_used: float
    all_scores: dict

    @property
    def confidence(self) -> str:
        """Derive a human-readable confidence label from the score."""
        if self.score >= 0.95:
            return "Very High"
        elif self.score >= 0.90:
            return "High"
        elif self.score >= 0.85:
            return "Medium"
        elif self.score >= 0.70:
            return "Low"
        else:
            return "Very Low"

    def __repr__(self) -> str:
        verdict_str = "MATCH ✅" if self.verdict else "NO MATCH ❌"
        return (
            f"<MatchResult verdict={verdict_str} "
            f"score={self.score:.4f} "
            f"confidence={self.confidence} "
            f"best_sig_id={self.best_sig_id}>"
        )


class SignatureMatcher:
    """
    Compares a query embedding against all stored reference embeddings
    for a user and returns a MatchResult with a similarity score and verdict.

    Strategy
    --------
    1. Cosine similarity is computed between the query vector and every
       stored reference vector.
    2. The maximum score across all references is taken as the match score.
    3. If max_score >= threshold → MATCH; otherwise → NO MATCH.

    This "1-vs-all" approach means adding more reference signatures per user
    improves recall (catches more genuine signatures) without hurting precision,
    since the threshold is still applied to the maximum score.

    Args:
        threshold : Cosine similarity threshold for MATCH decision [0.0, 1.0].
                    Default 0.85 balances false acceptance vs. false rejection.
    """

    def __init__(self, threshold: float = 0.85) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0,1], got {threshold}")
        self.threshold = threshold
        log.info(f"SignatureMatcher initialized | threshold={threshold}")

    def match(
        self,
        query_embedding: np.ndarray,
        reference_embeddings: List[Tuple[int, np.ndarray]],
        user_id: int,
    ) -> MatchResult:
        """
        Find the best matching reference signature for the query embedding.

        Args:
            query_embedding      : 1-D float32 array of shape (embedding_dim,).
            reference_embeddings : List of (signature_id, embedding) tuples.
            user_id              : User ID, used only for logging context.

        Returns:
            MatchResult: Full match details including score and verdict.

        Raises:
            NoReferenceSignatureError : If reference_embeddings is empty.
            VectorStoreError          : If the similarity computation fails.
        """
        if not reference_embeddings:
            log.warning(f"No reference embeddings | user_id={user_id}")
            raise NoReferenceSignatureError(user_id)

        log.info(
            f"Starting match | user_id={user_id} | "
            f"num_references={len(reference_embeddings)} | "
            f"threshold={self.threshold}"
        )

        try:
            all_scores = self._compute_all_scores(query_embedding, reference_embeddings)
            best_sig_id = max(all_scores, key=all_scores.get)
            best_score = all_scores[best_sig_id]
            verdict = best_score >= self.threshold

            result = MatchResult(
                score=best_score,
                verdict=verdict,
                best_sig_id=best_sig_id,
                threshold_used=self.threshold,
                all_scores=all_scores,
            )

            log.info(
                f"Match complete | user_id={user_id} | "
                f"score={best_score:.4f} | verdict={result.verdict} | "
                f"confidence={result.confidence}"
            )
            return result

        except NoReferenceSignatureError:
            raise
        except Exception as exc:
            log.error(f"Matching failed | user_id={user_id} | error={exc}")
            raise VectorStoreError("cosine_similarity", detail=str(exc)) from exc

    def _compute_all_scores(
        self,
        query: np.ndarray,
        references: List[Tuple[int, np.ndarray]],
    ) -> dict:
        """
        Compute cosine similarity between the query and all reference embeddings.

        Uses sklearn's cosine_similarity which handles L2 normalization
        internally, making results consistent with the model's L2-normalized
        output.

        Args:
            query      : Query embedding (embedding_dim,).
            references : List of (sig_id, embedding) pairs.

        Returns:
            Dict mapping signature_id → cosine_similarity_score.
        """
        sig_ids = [sig_id for sig_id, _ in references]
        ref_matrix = np.vstack([emb for _, emb in references])  # (N, D)
        query_2d = query.reshape(1, -1)                          # (1, D)

        # Returns (1, N) matrix
        scores_matrix = cosine_similarity(query_2d, ref_matrix)
        scores = scores_matrix[0]  # (N,)

        return {sig_id: float(score) for sig_id, score in zip(sig_ids, scores)}

    def ensemble_match(
        self,
        query_embeddings: List[np.ndarray],
        reference_embeddings: List[Tuple[int, np.ndarray]],
        user_id: int,
    ) -> MatchResult:
        """
        Majority-vote ensemble over multiple query embeddings (for video input).

        Each embedding (from a different video frame) casts one vote.
        The final MatchResult uses the embedding that produced the highest score.

        Args:
            query_embeddings     : List of embeddings from multiple video frames.
            reference_embeddings : Stored reference embeddings.
            user_id              : User being verified.

        Returns:
            MatchResult: Based on the best-scoring frame.
        """
        if not query_embeddings:
            raise VectorStoreError("ensemble_match", detail="No query embeddings provided")

        log.info(
            f"Ensemble match | user_id={user_id} | "
            f"frames={len(query_embeddings)}"
        )

        results = [
            self.match(emb, reference_embeddings, user_id)
            for emb in query_embeddings
        ]

        # Select the result with the highest score
        best = max(results, key=lambda r: r.score)

        votes_match = sum(1 for r in results if r.verdict)
        votes_nomatch = len(results) - votes_match
        majority_verdict = votes_match > votes_nomatch

        log.info(
            f"Ensemble result | votes_match={votes_match} | "
            f"votes_nomatch={votes_nomatch} | "
            f"majority={majority_verdict} | best_score={best.score:.4f}"
        )

        # Return best result but override verdict with majority vote
        return MatchResult(
            score=best.score,
            verdict=majority_verdict,
            best_sig_id=best.best_sig_id,
            threshold_used=self.threshold,
            all_scores=best.all_scores,
        )
