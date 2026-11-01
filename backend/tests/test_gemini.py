
import pytest
from app.ai.gemini_provider import GeminiProvider
from app.schemas.rms import RAGSource


@pytest.mark.asyncio
async def test_gemini_fallback_on_insufficient_evidence():
    """
    No API key or client needed — this tests YOUR logic, not Gemini's.
    Below-threshold relevance must always fall back to the deterministic
    mock, regardless of whether a real Gemini client is configured.
    """
    provider = GeminiProvider()

    low_relevance_source = RAGSource(
        document_id="doc_irrelevant_01",
        title="Unrelated Policy",
        clause="Section 1.1",
        excerpt="This policy does not address the student's actual question.",
        relevance_score=0.40,  # below the 0.65 threshold
    )

    draft = await provider.generate_draft(
        ticket_subject="Need emergency leave approval",
        ticket_description="I need to leave hostel urgently for a family emergency.",
        sources=[low_relevance_source],
        department="Hostel Affairs",
    )

    assert draft.grounding_status == "INSUFFICIENT_EVIDENCE"
    assert draft.needs_human_review is True
    assert draft.confidence == 0.0


@pytest.mark.asyncio
async def test_gemini_draft_generation_real():
    """
    Hits the real Gemini API. Requires GEMINI_API_KEY to be set in the
    environment. Run manually / in CI only when the key is available —
    this is the end-to-end confirmation that generation actually works,
    not a test you want gating every commit.
    """
    provider = GeminiProvider()

    if provider.client is None:
        pytest.skip("GEMINI_API_KEY not set — skipping real API test")

    valid_source = RAGSource(
        document_id="doc_hostel_leave_01",
        title="Hostel Leave Policy",
        clause="Section 4.2",
        excerpt="Students must submit leave requests 48 hours in advance via the hostel portal.",
        relevance_score=0.82,
    )

    draft = await provider.generate_draft(
        ticket_subject="Need emergency leave approval",
        ticket_description="I need to leave hostel urgently for a family emergency.",
        sources=[valid_source],
        department="Hostel Affairs",
    )

    assert draft.grounding_status == "GROUNDED"
    assert draft.needs_human_review is False
    assert draft.confidence == pytest.approx(0.82)
    assert len(draft.draft_response) > 0
    # Loosely check the response actually used the source material
    assert "48 hours" in draft.draft_response or "portal" in draft.draft_response