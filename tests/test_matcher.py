"""
Unit Tests — SignatureMatcher
================================
"""
import numpy as np
import pytest
from backend.services.matcher import SignatureMatcher, MatchResult
from backend.core.exceptions import NoReferenceSignatureError


def _make_unit_vec(dim=512):
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def matcher():
    return SignatureMatcher(threshold=0.85)


class TestSignatureMatcher:

    def test_identical_embeddings_score_is_one(self, matcher):
        emb = _make_unit_vec()
        result = matcher.match(emb, [(1, emb)], user_id=1)
        assert abs(result.score - 1.0) < 1e-4
        assert result.verdict is True

    def test_orthogonal_embeddings_score_near_zero(self, matcher):
        e1 = np.zeros(512, dtype=np.float32); e1[0] = 1.0
        e2 = np.zeros(512, dtype=np.float32); e2[1] = 1.0
        result = matcher.match(e1, [(1, e2)], user_id=1)
        assert result.score < 0.1
        assert result.verdict is False

    def test_best_match_returns_highest_scoring_sig(self, matcher):
        query = _make_unit_vec()
        ref_close = query + np.random.randn(512).astype(np.float32) * 0.01
        ref_close /= np.linalg.norm(ref_close)
        ref_far = _make_unit_vec()
        result = matcher.match(query, [(1, ref_far), (2, ref_close)], user_id=1)
        assert result.best_sig_id == 2

    def test_no_references_raises(self, matcher):
        with pytest.raises(NoReferenceSignatureError):
            matcher.match(_make_unit_vec(), [], user_id=42)

    def test_confidence_labels(self, matcher):
        emb = _make_unit_vec()
        result = matcher.match(emb, [(1, emb)], user_id=1)
        assert result.confidence == "Very High"

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            SignatureMatcher(threshold=1.5)

    def test_ensemble_match_majority_vote(self, matcher):
        genuine = _make_unit_vec()
        # 2 genuine embeddings + 1 different = majority MATCH
        e1 = genuine + np.random.randn(512).astype(np.float32) * 0.01
        e2 = genuine + np.random.randn(512).astype(np.float32) * 0.01
        e3 = _make_unit_vec()   # very different
        for e in [e1, e2, e3]:
            e /= np.linalg.norm(e)
        result = matcher.ensemble_match([e1, e2, e3], [(1, genuine)], user_id=1)
        assert result.verdict is True  # 2 out of 3 match
