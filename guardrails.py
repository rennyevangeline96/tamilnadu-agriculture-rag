import os
import re
from dataclasses import dataclass
from nemoguardrails import LLMRails, RailsConfig

MAX_QUESTION_LENGTH = 1000

SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{20,}|lsv2_[A-Za-z0-9_\-]{20,}")

# Initialize NeMo rails engine once from the local config directory
_config_path = os.path.join(os.path.dirname(__file__), "guardrails_config")
rails_engine = LLMRails(RailsConfig.from_path(_config_path))


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str = ""


def validate_question(question: str) -> GuardrailResult:
    """Pre-LLM structural checks on input length and emptiness."""
    cleaned = question.strip()
    if not cleaned:
        return GuardrailResult(False, "Question cannot be empty.")
    if len(cleaned) > MAX_QUESTION_LENGTH:
        return GuardrailResult(
            False,
            f"Question is too long ({len(cleaned)} characters). Maximum allowed is {MAX_QUESTION_LENGTH}.",
        )
    return GuardrailResult(True)


def check_nemo_input(question: str) -> GuardrailResult:
    """Run NeMo semantic input rail to filter off-topic questions or prompt injections."""
    try:
        res = rails_engine.generate(
            messages=[{"role": "user", "content": question}],
            options={"rails": ["input"], "log": {"activated_rails": True}},
        )
        if hasattr(res, "log") and res.log and hasattr(res.log, "activated_rails"):
            blocked = any(getattr(r, "stop", False) for r in res.log.activated_rails)
            if blocked:
                return GuardrailResult(
                    False, "I can only help with Tamil Nadu agriculture scheme questions."
                )
        elif hasattr(res, "response"):
            if "I can only help with Tamil Nadu" in str(res.response):
                return GuardrailResult(
                    False, "I can only help with Tamil Nadu agriculture scheme questions."
                )
        return GuardrailResult(True)
    except Exception:
        # Fallback to allowed if an unexpected engine/network timeout occurs
        return GuardrailResult(True)


def validate_answer(answer: str) -> GuardrailResult:
    """Free output checks: detect empty answers or leaked secret keys."""
    if not answer.strip():
        return GuardrailResult(False, "Empty answer.")
    if SECRET_PATTERN.search(answer):
        return GuardrailResult(False, "Answer contained something that looks like a secret key.")
    return GuardrailResult(True)