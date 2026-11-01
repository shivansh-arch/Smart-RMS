import os
import asyncio
import logging
from typing import List

from google import genai
from google.genai import types

from app.ai.provider import AIProvider
from app.ai.mock_provider import MockAIProvider
from app.schemas.rms import AIAnalysis, DraftResponse, RAGSource

logger = logging.getLogger(__name__)


class GeminiProvider(AIProvider):
    """Google Gemini AI provider with mock fallback."""

    def __init__(
        self,
        api_key: str = "",
        model_name: str = "gemini-2.5-flash",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self.fallback = MockAIProvider()
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    async def _generate_content(self, prompt: str) -> str:
        """Call Gemini without blocking the event loop."""
        if self.client is None:
            raise RuntimeError("Gemini client is not configured.")

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )

        generated_text = response.text
        if not generated_text or not generated_text.strip():
            raise ValueError("Gemini returned an empty response.")

        return generated_text.strip()

    async def analyze_ticket(
        self,
        ticket_subject: str,
        ticket_description: str,
    ) -> AIAnalysis:
        # Gemini isn't used for classification — your NLP pipeline already
        # does intent/department/priority deterministically and your eval
        # harness is built around that. Don't duplicate it with an LLM call.
        return await self.fallback.analyze_ticket(ticket_subject, ticket_description)

    async def generate_draft(
        self,
        ticket_subject: str,
        ticket_description: str,
        sources: List[RAGSource],
        department: str,
    ) -> DraftResponse:
        """Generate response prose via Gemini; grounding/confidence stays deterministic."""

        if self.client is None:
            return await self.fallback.generate_draft(
                ticket_subject, ticket_description, sources, department
            )

        # Same threshold as the mock — this decision is NOT delegated to Gemini.
        valid_sources = [s for s in sources if s.relevance_score >= 0.65]

        if not valid_sources:
            # No sufficient evidence → don't even call the LLM. Same contract as mock.
            return await self.fallback.generate_draft(
                ticket_subject, ticket_description, sources, department
            )

        policy_context = "\n\n".join(
            f"[Source {i+1}] {source.title} ({source.clause})\n{source.excerpt}"
            for i, source in enumerate(valid_sources)
        )

        prompt = (
            "You are drafting an official response for a university Redressal Desk. "
            "Write ONLY the body of a formal email to the student.\n\n"
            "Rules (do not break these):\n"
            "- Use ONLY the policy information given below. Do not invent policy numbers, "
            "dates, deadlines, or clauses not present in the sources.\n"
            "- If the sources don't fully answer the student's specific question, say the "
            "matter has been logged and routed to staff for manual confirmation — do not guess.\n"
            "- Cite the source(s) you use by title and clause, inline in the text.\n"
            "- Tone: formal, concise, reassuring. No markdown, no greeting beyond 'Dear Student,', "
            "no extra commentary outside the email body.\n\n"
            f"Student's ticket subject: {ticket_subject}\n"
            f"Student's ticket description: {ticket_description}\n"
            f"Assigned department: {department}\n\n"
            f"Available policy sources:\n{policy_context}\n\n"
            "Write the email body now:"
        )

        try:
            body = await self._generate_content(prompt)
        except Exception as exc:
            # Gemini failure must not crash the request — fall back to the
            # deterministic mock so staff still get a usable draft.
            logger.warning("Gemini generation failed, falling back to mock: %s", exc)
            return await self.fallback.generate_draft(
                ticket_subject, ticket_description, sources, department
            )

        primary_source = valid_sources[0]
        return DraftResponse(
            ticket_id="",
            draft_response=body,
            sources=valid_sources,
            confidence=primary_source.relevance_score,
            relevance_score=primary_source.relevance_score,
            grounding_status="GROUNDED",
            needs_human_review=False,
            requires_staff_edit=False,
            policy_compliance_passed=True,
        )