"""LLM formatting action handlers for generation UI text fields."""
import random
import re
from typing import Optional

import gradio as gr
from loguru import logger

from acestep.constants import VALID_LANGUAGES
from acestep.inference import create_sample, format_sample
from acestep.text_tasks.caption_vocal_presence import ensure_caption_has_global_vocal_presence
from acestep.text_tasks.external_lm_mode import is_external_lm_active
from acestep.text_tasks.external_lm_tasks import (
    ExternalAIClientError,
    format_sample_with_external_provider,
    generate_lyrics_from_caption_with_external_provider,
)
from acestep.ui.gradio.i18n import get_i18n, t

from .llm_action_params import build_user_metadata, convert_lm_params
from .lyric_generation_fallback import (
    build_duration_aware_fallback_lyrics,
    build_lyrics_generation_seed,
    build_vocal_caption_prompt,
    is_tag_only_lyrics,
)
from .validation import clamp_duration_to_gpu_limit


_RANDOM_NARRATIVE_GENRES = (
    "cinematic orchestral pop",
    "neo-soul",
    "melodic drum and bass",
    "ambient folk",
    "dark synthwave",
    "afro-house",
    "indie rock ballad",
    "future garage",
    "latin pop",
    "lo-fi hip-hop",
)
_LYRICS_SECTION_TAG_PATTERN = re.compile(
    r"(?im)^\s*\[(verse|chorus|bridge|pre-chorus|intro|outro|hook|refrain|drop|"
    r"interlude|breakdown|post-chorus)(?:[^\]]*)\]\s*$"
)
_LYRIC_PROMPT_LEAKAGE_MARKERS = (
    "must include clear sung words",
    "do not output [instrumental",
    "target tempo:",
    "target duration:",
    "lyric requirements",
    "variation directive",
    "hidden nonce",
    "internal variation nonce",
    "retry nonce",
    "write 2-4 lines",
    "(2-4 sung lines",
    "start the narrative",
    "memorable hook",
    "develop the narrative",
    "in auto",
)
_GENERIC_BRACKET_TAG_PATTERN = re.compile(r"^\s*\[[^\]]+\]\s*$")
_SPOKEN_WORD_SAMPLE_PATTERN = re.compile(r"^\[\s*spoken word sample\s*:\s*(.*?)\s*\]$", flags=re.IGNORECASE)
_INSTRUMENTAL_SECTION_SUFFIX_PATTERN = re.compile(r"\s*-\s*instrumental\b", flags=re.IGNORECASE)
_XML_WRAPPER_TAG_PATTERN = re.compile(r"</?lyrics>", flags=re.IGNORECASE)
_PLACEHOLDER_DIRECTIVE_LINE_PATTERN = re.compile(
    r"^\s*\((?:[^)]*sung lines|[^)]*narrative|[^)]*hook)[^)]*\)\s*$",
    flags=re.IGNORECASE,
)
_INSTRUMENTAL_CAPTION_MARKERS = (
    "instrumental",
    "solo piano",
    "solo guitar",
    "solo violin",
    "without vocals",
    "wordless",
    "no vocals",
    "piano piece",
    "orchestral",
)


def _resolve_lyrics_generation_language(vocal_language: str) -> str:
    """Resolve a usable lyric-generation language from UI state and current input."""
    normalized_language = (vocal_language or "").strip().lower()
    if normalized_language in VALID_LANGUAGES and normalized_language not in {"unknown", "instrumental"}:
        return normalized_language

    ui_language = getattr(get_i18n(), "current_language", "")
    normalized_ui_language = (ui_language or "").strip().lower()
    if normalized_ui_language in VALID_LANGUAGES and normalized_ui_language != "unknown":
        return normalized_ui_language

    return "en"

_NON_LYRIC_TOKENS = set(
    "bar bars bridge breakdown chorus drop humming hook hum inst instrumental interlude intro la "
    "lyrics lyric music na none no only outro post pre refrain section tag verse vocal vocalise "
    "vocalize vocals".split()
)


def _format_failure_response(update_count: int, status_message: str):
    """Build a standardized failure response with update placeholders."""
    return (*([gr.update()] * update_count), status_message)


def _is_timeout_error_status(status_message: Optional[str]) -> bool:
    """Return ``True`` when status text indicates a timeout-like network failure."""
    normalized = (status_message or "").lower()
    return "timed out" in normalized or "timeout" in normalized


def _clean_optional_wrapped_quotes(text: Optional[str]) -> Optional[str]:
    """Strip a single layer of leading/trailing quote characters when present."""
    if text is None:
        return None
    if len(text) >= 2 and (
        (text.startswith("'") and text.endswith("'"))
        or (text.startswith('"') and text.endswith('"'))
    ):
        return text[1:-1]
    return text


def _ensure_structured_lyrics_sections(lyrics: Optional[str]) -> str:
    """Ensure lyrics include section directives for plain-text lyric inputs."""
    if not lyrics:
        return ""

    stripped = lyrics.strip()
    if not stripped:
        return ""
    if stripped.lower() == "[instrumental]":
        return stripped
    if _LYRICS_SECTION_TAG_PATTERN.search(stripped):
        return stripped

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return stripped

    if len(lines) <= 4:
        return "[Verse 1]\n" + "\n".join(lines)

    split_idx = max(2, len(lines) // 2)
    first_half = "\n".join(lines[:split_idx]).strip()
    second_half = "\n".join(lines[split_idx:]).strip()
    if not second_half:
        return "[Verse 1]\n" + first_half
    return f"[Verse 1]\n{first_half}\n\n[Chorus]\n{second_half}"


def _extract_word_tokens(text: str) -> list[str]:
    """Split text into Unicode-aware word tokens using alphanumeric runs."""
    tokens: list[str] = []
    current: list[str] = []
    for char in text:
        if char.isalnum() or (char == "'" and current):
            current.append(char)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens



def _line_has_lyric_words(line: str) -> bool:
    """Return ``True`` when a line contains meaningful singable word content."""
    if _GENERIC_BRACKET_TAG_PATTERN.fullmatch(line):
        return False

    normalized = _GENERIC_BRACKET_TAG_PATTERN.sub(" ", line)
    normalized = _XML_WRAPPER_TAG_PATTERN.sub(" ", normalized)
    normalized = re.sub(r"`{3,}", " ", normalized)
    normalized = re.sub(r"[*_#>|~]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False

    tokens = [token.lower() for token in _extract_word_tokens(normalized)]
    filtered_tokens = [
        token for token in tokens if not token.isdigit() and token not in _NON_LYRIC_TOKENS
    ]
    return bool(filtered_tokens)



def _sanitize_generated_lyrics(lyrics: Optional[str]) -> str:
    """Remove leaked control lines while preserving real lyric content."""
    if not lyrics:
        return ""

    cleaned_lines: list[str] = []
    for raw_line in lyrics.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        lower = stripped.lower()
        if lower in {"# lyric", "# lyrics"}:
            continue
        if lower.startswith("deliver these lyrics in "):
            continue
        if _PLACEHOLDER_DIRECTIVE_LINE_PATTERN.fullmatch(stripped):
            continue
        if any(marker in lower for marker in _LYRIC_PROMPT_LEAKAGE_MARKERS):
            continue

        spoken_match = _SPOKEN_WORD_SAMPLE_PATTERN.fullmatch(stripped)
        if spoken_match:
            spoken_content = spoken_match.group(1).strip()
            if spoken_content and _line_has_lyric_words(spoken_content):
                cleaned_lines.append(spoken_content)
            continue

        normalized = _XML_WRAPPER_TAG_PATTERN.sub("", stripped).strip()
        if not normalized:
            continue
        if _GENERIC_BRACKET_TAG_PATTERN.fullmatch(normalized):
            if not _LYRICS_SECTION_TAG_PATTERN.fullmatch(normalized):
                continue
            normalized = _INSTRUMENTAL_SECTION_SUFFIX_PATTERN.sub("", normalized)
            normalized = normalized.replace("  ", " ").replace(" ]", "]")
        cleaned_lines.append(normalized)

    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()
    return "\n".join(cleaned_lines).strip()



def _normalize_generated_lyrics(lyrics: Optional[str]) -> str:
    """Clean leaked prompt debris and ensure a stable lyric section structure."""
    sanitized = _sanitize_generated_lyrics(_clean_optional_wrapped_quotes(lyrics))
    return _ensure_structured_lyrics_sections(sanitized)


def _build_random_narrative_caption_seed() -> tuple[str, str]:
    """Return ``(genre, prompt)`` seed used for random caption-from-scratch generation."""
    genre = random.choice(_RANDOM_NARRATIVE_GENRES)
    prompt = (
        f"Create a brand-new {genre} music concept. "
        "Write a narrative caption with a clear linear structure "
        "(intro -> build -> chorus/drop -> outro), core instrumentation, "
        "singer gender and delivery mood/timbre, and energy/mix trajectory."
    )
    return genre, prompt


def _has_lyric_prompt_leakage(lyrics: str) -> bool:
    """Return ``True`` when generated lyrics contain prompt/control leakage text."""
    normalized = (lyrics or "").lower()
    return any(marker in normalized for marker in _LYRIC_PROMPT_LEAKAGE_MARKERS)


def _looks_like_placeholder_scaffold(lyrics: str) -> bool:
    """Return ``True`` when output still resembles a template scaffold, not final lyrics."""
    lines = [line.strip() for line in (lyrics or "").splitlines() if line.strip()]
    if not lines:
        return True

    placeholder_lines = sum(
        1 for line in lines if _PLACEHOLDER_DIRECTIVE_LINE_PATTERN.fullmatch(line)
    )
    if placeholder_lines > 0:
        return True

    tag_lines = sum(1 for line in lines if _GENERIC_BRACKET_TAG_PATTERN.fullmatch(line))
    non_tag_lines = [
        line for line in lines if not _GENERIC_BRACKET_TAG_PATTERN.fullmatch(line)
    ]
    non_placeholder_lines = [
        line for line in non_tag_lines if not _PLACEHOLDER_DIRECTIVE_LINE_PATTERN.fullmatch(line)
    ]
    if tag_lines >= 3 and len(non_placeholder_lines) < 2:
        return True
    return False


def _iter_structured_lyric_sections(lyrics: str) -> list[tuple[str, list[str]]]:
    """Return section names with their non-empty lyric lines from structured lyrics text."""
    sections: list[tuple[str, list[str]]] = []
    current_name: Optional[str] = None
    current_lines: list[str] = []

    for raw_line in (lyrics or "").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        match = _LYRICS_SECTION_TAG_PATTERN.fullmatch(stripped)
        if match:
            if current_name is not None:
                sections.append((current_name, current_lines))
            current_name = match.group(1).lower()
            current_lines = []
            continue

        if current_name is not None and _line_has_lyric_words(stripped):
            current_lines.append(stripped)

    if current_name is not None:
        sections.append((current_name, current_lines))
    return sections



def _has_inconsistent_repeated_section_structure(lyrics: str) -> bool:
    """Return ``True`` when repeated sections do not keep a matching line count."""
    comparable_sections = {
        "verse",
        "chorus",
        "pre-chorus",
        "hook",
        "refrain",
        "post-chorus",
    }
    line_counts_by_section: dict[str, list[int]] = {}

    for section_name, lines in _iter_structured_lyric_sections(lyrics):
        if section_name not in comparable_sections or not lines:
            continue
        line_counts_by_section.setdefault(section_name, []).append(len(lines))

    for counts in line_counts_by_section.values():
        if len(counts) >= 2 and len(set(counts)) > 1:
            return True
    return False



def _is_invalid_generated_lyrics(lyrics: str) -> bool:
    """Return ``True`` when generated lyrics should be retried or replaced."""
    return (
        is_tag_only_lyrics(lyrics)
        or _has_lyric_prompt_leakage(lyrics)
        or _looks_like_placeholder_scaffold(lyrics)
        or _has_inconsistent_repeated_section_structure(lyrics)
    )


def _execute_format_sample(
    llm_handler,
    caption: str,
    lyrics: str,
    bpm,
    audio_duration,
    key_scale: str,
    time_signature: str,
    lm_temperature: float,
    lm_top_k: int,
    lm_top_p: float,
    constrained_decoding_debug: bool,
    vocal_language: str = "unknown",
):
    """Run shared format-sample workflow.

    Returns:
        Tuple of ``(result_or_none, audio_duration_value_or_none, status_message)``.
    """
    external_active = is_external_lm_active()
    if not llm_handler.llm_initialized and not external_active:
        status_message = t("messages.lm_not_initialized")
        gr.Warning(status_message)
        return None, None, status_message

    user_metadata = build_user_metadata(
        bpm,
        audio_duration,
        key_scale,
        time_signature,
        vocal_language=vocal_language,
    )
    if external_active:
        try:
            result = format_sample_with_external_provider(
                caption=caption,
                lyrics=lyrics,
                user_metadata=user_metadata,
            )
        except ExternalAIClientError as exc:
            status_message = str(exc)
            gr.Warning(status_message)
            return None, None, status_message
    else:
        top_k_value, top_p_value = convert_lm_params(lm_top_k, lm_top_p)
        result = format_sample(
            llm_handler=llm_handler,
            caption=caption,
            lyrics=lyrics,
            user_metadata=user_metadata,
            temperature=lm_temperature,
            top_k=top_k_value,
            top_p=top_p_value,
            use_constrained_decoding=True,
            constrained_decoding_debug=constrained_decoding_debug,
        )

    if not result.success:
        status_message = result.status_message or t("messages.format_failed")
        gr.Warning(status_message)
        return None, None, status_message

    gr.Info(t("messages.format_success"))
    clamped_duration = clamp_duration_to_gpu_limit(result.duration, llm_handler)
    duration_value = clamped_duration if clamped_duration and clamped_duration > 0 else -1
    return result, duration_value, result.status_message


def handle_format_sample(
    llm_handler,
    caption: str,
    lyrics: str,
    bpm,
    audio_duration,
    key_scale: str,
    time_signature: str,
    lm_temperature: float,
    lm_top_k: int,
    lm_top_p: float,
    constrained_decoding_debug: bool = False,
    vocal_language: str = "unknown",
):
    """Format caption and lyrics together via LLM."""
    result, duration_value, status_message = _execute_format_sample(
        llm_handler=llm_handler,
        caption=caption,
        lyrics=lyrics,
        bpm=bpm,
        audio_duration=audio_duration,
        key_scale=key_scale,
        time_signature=time_signature,
        vocal_language=vocal_language,
        lm_temperature=lm_temperature,
        lm_top_k=lm_top_k,
        lm_top_p=lm_top_p,
        constrained_decoding_debug=constrained_decoding_debug,
    )

    if result is None:
        return _format_failure_response(update_count=8, status_message=status_message)

    normalized_caption = ensure_caption_has_global_vocal_presence(
        result.caption,
        lyrics=result.lyrics,
        vocal_language=result.language,
    )
    return (
        normalized_caption,
        result.lyrics,
        result.bpm,
        duration_value,
        result.keyscale,
        result.language,
        result.timesignature,
        True,
        status_message,
    )


def handle_format_caption(
    llm_handler,
    caption: str,
    lyrics: str,
    bpm,
    audio_duration,
    key_scale: str,
    time_signature: str,
    lm_temperature: float,
    lm_top_k: int,
    lm_top_p: float,
    constrained_decoding_debug: bool = False,
    vocal_language: str = "unknown",
):
    """Format only caption via LLM while leaving lyrics unchanged in UI wiring.

    Any outer single/double quotes added by the LLM are stripped from the
    returned caption for cleaner textbox display.
    """
    result, duration_value, status_message = _execute_format_sample(
        llm_handler=llm_handler,
        caption=caption,
        lyrics=lyrics,
        bpm=bpm,
        audio_duration=audio_duration,
        key_scale=key_scale,
        time_signature=time_signature,
        vocal_language=vocal_language,
        lm_temperature=lm_temperature,
        lm_top_k=lm_top_k,
        lm_top_p=lm_top_p,
        constrained_decoding_debug=constrained_decoding_debug,
    )

    if result is None:
        return _format_failure_response(update_count=7, status_message=status_message)

    normalized_caption = ensure_caption_has_global_vocal_presence(
        _clean_optional_wrapped_quotes(result.caption) or "",
        lyrics=result.lyrics,
        vocal_language=result.language,
    )
    return (
        normalized_caption,
        result.bpm,
        duration_value,
        result.keyscale,
        result.language,
        result.timesignature,
        True,
        status_message,
    )


def handle_format_lyrics(
    llm_handler,
    caption: str,
    lyrics: str,
    bpm,
    audio_duration,
    key_scale: str,
    time_signature: str,
    lm_temperature: float,
    lm_top_k: int,
    lm_top_p: float,
    constrained_decoding_debug: bool = False,
    vocal_language: str = "unknown",
):
    """Format only lyrics via LLM while leaving caption unchanged in UI wiring.

    Any outer single/double quotes added by the LLM are stripped from the
    returned lyrics for cleaner textbox display.
    """
    result, duration_value, status_message = _execute_format_sample(
        llm_handler=llm_handler,
        caption=caption,
        lyrics=lyrics,
        bpm=bpm,
        audio_duration=audio_duration,
        key_scale=key_scale,
        time_signature=time_signature,
        vocal_language=vocal_language,
        lm_temperature=lm_temperature,
        lm_top_k=lm_top_k,
        lm_top_p=lm_top_p,
        constrained_decoding_debug=constrained_decoding_debug,
    )

    if result is None:
        return _format_failure_response(update_count=7, status_message=status_message)

    structured_lyrics = _ensure_structured_lyrics_sections(
        _clean_optional_wrapped_quotes(result.lyrics)
    )

    return (
        structured_lyrics,
        result.bpm,
        duration_value,
        result.keyscale,
        result.language,
        result.timesignature,
        True,
        status_message,
    )


def _caption_looks_instrumental(caption: str) -> bool:
    """Return ``True`` when the caption strongly suggests an instrumental-only arrangement."""
    normalized = (caption or "").strip().lower()
    return any(marker in normalized for marker in _INSTRUMENTAL_CAPTION_MARKERS)



def _build_retry_vocal_caption_prompt(
    *,
    caption: str,
    vocal_language: str,
    bpm,
    audio_duration,
) -> str:
    """Build a stronger vocal-conversion prompt for local lyric retry formatting."""
    prompt_parts = [
        build_vocal_caption_prompt(
            caption=caption,
            vocal_language=vocal_language,
            bpm=bpm,
            audio_duration=audio_duration,
        ),
        "Convert this concept into a vocal version while preserving its groove, instrumentation, and mood.",
        "Write clear sung lyrics for a lead vocalist. Do not preserve instrumental-only framing.",
        "Do not return [Instrumental], [Music starts], empty section tags, or stage-direction placeholders.",
        "Keep repeated song sections structurally matched: Verse 1 and Verse 2 should use the same number of lines, and repeated choruses should keep the same line count and hook shape.",
        "Keep each sung lyric line short enough to sing naturally, ideally 20 syllables or fewer.",
        "If a phrase would run past a comma or pause, break it into a new lyric line instead of one long sentence.",
    ]
    if _caption_looks_instrumental(caption):
        prompt_parts.append(
            "The source caption may say instrumental, solo, or wordless. Override that and write a vocal topline with real words."
        )
    return "\n".join(prompt_parts)



def _build_lyrics_generation_query(
    *,
    caption: str,
    bpm,
    audio_duration,
    key_scale: str,
    time_signature: str,
    vocal_language: str,
    retry: bool = False,
) -> str:
    """Build a fresh lyric-generation query for the 5Hz sample-creation flow."""
    base_prompt = build_vocal_caption_prompt(
        caption=caption,
        vocal_language=vocal_language,
        bpm=bpm,
        audio_duration=audio_duration,
    )
    query_parts = [
        base_prompt,
        "This is a lyric-writing task for a vocal song with lead vocals.",
        "Return complete sung lyrics with clear words and a Verse -> Chorus -> Verse structure.",
        "Do not return [Instrumental], [Music starts], stage directions, placeholders, or control instructions.",
        "Keep repeated song sections structurally matched: Verse 1 and Verse 2 should use the same number of lines, and repeated choruses should keep the same line count and hook shape.",
        "Keep each sung lyric line short enough to sing naturally, ideally 20 syllables or fewer.",
        "If a phrase would run past a comma or pause, break it into a new lyric line instead of one long sentence.",
    ]
    if _caption_looks_instrumental(caption):
        query_parts.append(
            "The source caption may describe an instrumental arrangement. Reinterpret it as a vocal version and write sung lyrics anyway."
        )
    for label, value in (
        ("Preferred key", key_scale),
        ("Time signature", time_signature),
        ("Preferred vocal language", vocal_language),
    ):
        if value not in (None, "", "unknown"):
            query_parts.append(f"{label}: {value}")
    if retry:
        query_parts.append("Use a different hook and imagery from the previous draft.")
    return "\n".join(query_parts)



def _execute_generate_lyrics(
    llm_handler,
    caption: str,
    bpm,
    audio_duration,
    key_scale: str,
    time_signature: str,
    vocal_language: str,
    lm_temperature: float,
    lm_top_k: int,
    lm_top_p: float,
    constrained_decoding_debug: bool,
    retry: bool = False,
):
    """Run the dedicated lyric-generation flow for external or local LM backends."""
    external_active = is_external_lm_active()
    if external_active:
        try:
            result = generate_lyrics_from_caption_with_external_provider(
                caption=caption,
                bpm=bpm,
                audio_duration=audio_duration,
                key_scale=key_scale,
                time_signature=time_signature,
                vocal_language=vocal_language,
                retry=retry,
            )
        except ExternalAIClientError as exc:
            status_message = str(exc)
            gr.Warning(status_message)
            return None, None, status_message
        duration_value = clamp_duration_to_gpu_limit(result.duration, llm_handler)
        duration_value = duration_value if duration_value and duration_value > 0 else -1
        return result, duration_value, result.status_message

    if not llm_handler.llm_initialized:
        status_message = t("messages.lm_not_initialized")
        gr.Warning(status_message)
        return None, None, status_message

    top_k_value, top_p_value = convert_lm_params(lm_top_k, lm_top_p)
    if not retry:
        result = create_sample(
            llm_handler=llm_handler,
            query=_build_lyrics_generation_query(
                caption=caption,
                bpm=bpm,
                audio_duration=audio_duration,
                key_scale=key_scale,
                time_signature=time_signature,
                vocal_language=vocal_language,
                retry=False,
            ),
            instrumental=False,
            vocal_language=vocal_language,
            temperature=lm_temperature,
            top_k=top_k_value,
            top_p=top_p_value,
            use_constrained_decoding=True,
            constrained_decoding_debug=constrained_decoding_debug,
        )
        if not result.success:
            status_message = result.status_message or "Failed to generate lyrics from caption"
            gr.Warning(status_message)
            return None, None, status_message

        duration_value = clamp_duration_to_gpu_limit(result.duration, llm_handler)
        duration_value = duration_value if duration_value and duration_value > 0 else -1
        return result, duration_value, result.status_message

    # Local retry path: switch to rewrite-style lyric completion instead of repeating
    # the same sample-generation request that already returned tag-only/instrumental output.
    result, duration_value, status_message = _execute_format_sample(
        llm_handler=llm_handler,
        caption=_build_retry_vocal_caption_prompt(
            caption=caption,
            vocal_language=vocal_language,
            bpm=bpm,
            audio_duration=audio_duration,
        ),
        lyrics=build_lyrics_generation_seed(
            vocal_language=vocal_language,
            bpm=bpm,
            audio_duration=audio_duration,
            retry=True,
        ),
        bpm=bpm,
        audio_duration=audio_duration,
        key_scale=key_scale,
        time_signature=time_signature,
        vocal_language=vocal_language,
        lm_temperature=lm_temperature,
        lm_top_k=lm_top_k,
        lm_top_p=lm_top_p,
        constrained_decoding_debug=constrained_decoding_debug,
    )
    return result, duration_value, status_message



def handle_generate_lyrics_from_caption(
    llm_handler,
    caption: str,
    bpm,
    audio_duration,
    key_scale: str,
    time_signature: str,
    vocal_language: str = "unknown",
    lm_temperature: float = 0.85,
    lm_top_k: int = 0,
    lm_top_p: float = 0.9,
    constrained_decoding_debug: bool = False,
):
    """Generate fresh lyrics from caption while honoring selected metadata/language."""
    resolved_vocal_language = _resolve_lyrics_generation_language(vocal_language)
    variation_nonce = random.SystemRandom().randint(1000, 999999)
    result, duration_value, status_message = _execute_generate_lyrics(
        llm_handler=llm_handler,
        caption=caption,
        bpm=bpm,
        audio_duration=audio_duration,
        key_scale=key_scale,
        time_signature=time_signature,
        vocal_language=resolved_vocal_language,
        lm_temperature=lm_temperature,
        lm_top_k=lm_top_k,
        lm_top_p=lm_top_p,
        constrained_decoding_debug=constrained_decoding_debug,
        retry=False,
    )
    if result is None:
        if _is_timeout_error_status(status_message):
            fallback_lyrics = build_duration_aware_fallback_lyrics(
                caption=caption,
                vocal_language=resolved_vocal_language,
                bpm=bpm,
                audio_duration=audio_duration,
                variation_nonce=variation_nonce,
            )
            timeout_status = (
                f"{status_message} | Lyrics generated with fallback scaffold (LM timeout)."
            )
            return (
                fallback_lyrics,
                bpm,
                audio_duration,
                key_scale,
                resolved_vocal_language,
                time_signature,
                True,
                timeout_status,
            )
        return _format_failure_response(update_count=7, status_message=status_message)

    generated_lyrics = _normalize_generated_lyrics(result.lyrics)

    fallback_nonce = variation_nonce
    if _is_invalid_generated_lyrics(generated_lyrics):
        retry_nonce = random.SystemRandom().randint(1000, 999999)
        fallback_nonce = retry_nonce
        retry_result, retry_duration, retry_status = _execute_generate_lyrics(
            llm_handler=llm_handler,
            caption=caption,
            bpm=bpm,
            audio_duration=audio_duration,
            key_scale=key_scale,
            time_signature=time_signature,
            vocal_language=resolved_vocal_language,
            lm_temperature=lm_temperature,
            lm_top_k=lm_top_k,
            lm_top_p=lm_top_p,
            constrained_decoding_debug=constrained_decoding_debug,
            retry=True,
        )
        if retry_result is not None:
            retry_lyrics = _normalize_generated_lyrics(retry_result.lyrics)
            if not _is_invalid_generated_lyrics(retry_lyrics):
                result = retry_result
                duration_value = retry_duration
                status_message = retry_status
                generated_lyrics = retry_lyrics

    fallback_used = False
    if _is_invalid_generated_lyrics(generated_lyrics):
        fallback_used = True
        logger.warning(
            "Falling back to scaffold lyrics for caption={!r}; rejected lyrics preview={!r}",
            caption[:120],
            generated_lyrics[:240],
        )
        generated_lyrics = build_duration_aware_fallback_lyrics(
            caption=caption,
            vocal_language=resolved_vocal_language,
            bpm=bpm,
            audio_duration=audio_duration,
            variation_nonce=fallback_nonce,
        )

    generated_status = (
        "Lyrics generated from caption."
        if not fallback_used
        else "Lyrics generated with fallback scaffold (LM returned unusable output)."
    )
    if status_message:
        generated_status = f"{status_message} | {generated_status}"
    return (
        generated_lyrics,
        result.bpm,
        duration_value,
        result.keyscale,
        result.language or resolved_vocal_language,
        result.timesignature,
        True,
        generated_status,
    )

def handle_generate_random_narrative_caption(
    llm_handler,
    bpm,
    audio_duration,
    key_scale: str,
    time_signature: str,
    lm_temperature: float,
    lm_top_k: int,
    lm_top_p: float,
    constrained_decoding_debug: bool = False,
    vocal_language: str = "unknown",
):
    """Generate a caption from scratch using a random genre narrative seed."""
    genre, seed_prompt = _build_random_narrative_caption_seed()
    result, duration_value, status_message = _execute_format_sample(
        llm_handler=llm_handler,
        caption=seed_prompt,
        lyrics="",
        bpm=bpm,
        audio_duration=audio_duration,
        key_scale=key_scale,
        time_signature=time_signature,
        vocal_language=vocal_language,
        lm_temperature=lm_temperature,
        lm_top_k=lm_top_k,
        lm_top_p=lm_top_p,
        constrained_decoding_debug=constrained_decoding_debug,
    )

    if result is None:
        return _format_failure_response(update_count=7, status_message=status_message)

    random_status = f"{status_message} | Random genre seed: {genre}"
    normalized_caption = ensure_caption_has_global_vocal_presence(
        _clean_optional_wrapped_quotes(result.caption) or "",
        lyrics=result.lyrics,
        vocal_language=result.language,
    )
    return (
        normalized_caption,
        result.bpm,
        duration_value,
        result.keyscale,
        result.language,
        result.timesignature,
        True,
        random_status,
    )

