"""
ACE-Step Inference API Module

This module provides a standardized inference interface for music generation,
designed for third-party integration. It offers both a simplified API and
backward-compatible Gradio UI support.
"""

import math
import os
import re
import tempfile
from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from loguru import logger
import torch


from acestep.audio_utils import AudioSaver, apply_fade, generate_uuid_from_params, normalize_audio, get_lora_weights_hash
from acestep.text_tasks.caption_vocal_presence import ensure_caption_has_global_vocal_presence

# HuggingFace Space environment detection
IS_HUGGINGFACE_SPACE = os.environ.get("SPACE_ID") is not None

_LYRIC_TAG_LINE_PATTERN = re.compile(r"^\s*\[[^\]]+\]\s*$")
_LYRIC_WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")
_LYRIC_VOWEL_GROUP_PATTERN = re.compile(r"[aeiouy]+", re.IGNORECASE)
_LYRIC_COMMA_LINEBREAK_PATTERN = re.compile(r"\s*,\s*")
_LYRIC_DENSITY_WORDS_PER_BAR = 6.0
_LYRIC_DENSITY_MIN_WORD_BUDGET = 16
_LYRIC_DENSITY_MAX_WORD_BUDGET = 480
_LYRIC_DENSITY_SYLLABLES_PER_BEAT = 1.2
_LYRIC_DENSITY_VOCAL_ACTIVITY_RATIO = 0.55
_LYRIC_DENSITY_SYLLABLE_BUDGET_MULTIPLIER = 2.0
_LYRIC_DENSITY_MIN_SYLLABLE_BUDGET = 24
_LYRIC_DENSITY_MAX_SYLLABLE_BUDGET = 960
_LYRIC_DENSITY_WARNING_RATIO = 1.6
_LYRIC_DENSITY_AUTO_TRIM_RATIO = 3.0
_LYRIC_DENSITY_TRIM_TARGET_RATIO = 1.4

def _get_spaces_gpu_decorator(duration=180):
    """
    Get the @spaces.GPU decorator if running in HuggingFace Space environment.
    Returns identity decorator if not in Space environment.
    """
    if IS_HUGGINGFACE_SPACE:
        try:
            import spaces
            return spaces.GPU(duration=duration)
        except ImportError:
            logger.warning("spaces package not found, GPU decorator disabled")
            return lambda func: func
    return lambda func: func


@dataclass
class GenerationParams:
    """Configuration for music generation parameters.
    
    Attributes:
        # Text Inputs
        caption: A short text prompt describing the desired music (main prompt). < 512 characters
        lyrics: Lyrics for the music. Use "[Instrumental]" for instrumental songs. < 4096 characters
        instrumental: If True, generate instrumental music regardless of lyrics.
        
        # Music Metadata
        bpm: BPM (beats per minute), e.g., 120. Set to None for automatic estimation. 30 ~ 300
        keyscale: Musical key (e.g., "C Major", "Am"). Leave empty for auto-detection. A-G, #/♭, major/minor
        timesignature: Time signature (2 for '2/4', 3 for '3/4', 4 for '4/4', 6 for '6/8'). Leave empty for auto-detection.
        vocal_language: Language code for vocals, e.g., "en", "zh", "ja", or "unknown". see acestep/constants.py:VALID_LANGUAGES
        duration: Target audio length in seconds. If <0 or None, model chooses automatically. 10 ~ 600
        
        # Audio Post-Processing
        enable_normalization: Whether to apply loudness normalization to the output audio.
        normalization_db: Target loudness in dB for normalization (e.g., -1.0 for -1 dBFS peak).
        latent_shift: Additive shift applied to DiT latents before VAE decode (default 0, no shift).
        latent_rescale: Multiplicative rescale applied to DiT latents before VAE decode (default 1.0, no rescale).
        
        # Generation Parameters
        inference_steps: Number of diffusion steps (e.g., 8 for turbo, 32–100 for base model).
        guidance_scale: CFG (classifier-free guidance) strength. Higher means following the prompt more strictly. Only support for non-turbo model.
        seed: Integer seed for reproducibility. -1 means use random seed each time.
        
        # Advanced DiT Parameters
        use_adg: Whether to use Adaptive Dual Guidance (only works for base model).
        cfg_interval_start: Start ratio (0.0–1.0) to apply CFG.
        cfg_interval_end: End ratio (0.0–1.0) to apply CFG.
        shift: Timestep shift factor (default 1.0). When != 1.0, applies t = shift * t / (1 + (shift - 1) * t) to timesteps.
        
        # Task-Specific Parameters
        task_type: Type of generation task. One of: "text2music", "cover", "repaint", "lego", "extract", "complete".
        reference_audio: Path to a reference audio file for style transfer or cover tasks.
        src_audio: Path to a source audio file for audio-to-audio tasks.
        audio_codes: Audio semantic codes as a string (advanced use, for code-control generation).
        repainting_start: For repaint/lego tasks: start time in seconds for region to repaint.
        repainting_end: For repaint/lego tasks: end time in seconds for region to repaint (-1 for until end).
        audio_cover_strength: Strength of reference audio/codes influence (range 0.0–1.0). set smaller (0.2) for style transfer tasks.
        instruction: Optional task instruction prompt. If empty, auto-generated by system.
        
        # 5Hz Language Model Parameters for CoT reasoning
        thinking: If True, enable 5Hz Language Model "Chain-of-Thought" reasoning for semantic/music metadata and codes.
        lm_temperature: Sampling temperature for the LLM (0.0–2.0). Higher = more creative/varied results.
        lm_cfg_scale: Classifier-free guidance scale for the LLM.
        lm_top_k: LLM top-k sampling (0 = disabled).
        lm_top_p: LLM top-p nucleus sampling (1.0 = disabled).
        lm_negative_prompt: Negative prompt to use for LLM (for control).
        use_cot_metas: Whether to let LLM generate music metadata via CoT reasoning.
        use_cot_caption: Whether to let LLM rewrite or format the input caption via CoT reasoning.
        use_cot_language: Whether to let LLM detect vocal language via CoT.
    """
    # Required Inputs
    task_type: str = "text2music"
    instruction: str = "Fill the audio semantic mask based on the given conditions:"

    # Audio Uploads
    reference_audio: Optional[str] = None
    src_audio: Optional[str] = None

    # LM Codes Hints
    audio_codes: str = ""

    # Text Inputs
    caption: str = ""
    global_caption: str = ""  # Global/song-level caption for SFT-stems lego tasks
    lyrics: str = ""
    instrumental: bool = False

    # Metadata
    vocal_language: str = "unknown"
    bpm: Optional[int] = None
    keyscale: str = ""
    timesignature: str = ""
    duration: float = -1.0

    # Audio Post-Processing
    enable_normalization: bool = True
    normalization_db: float = -1.0
    fade_in_duration: float = 0.0   # Fade in duration in seconds. 0 = no fade in.
    fade_out_duration: float = 0.0  # Fade out duration in seconds. 0 = no fade out.

    # Latent Post-Processing (before VAE decode)
    latent_shift: float = 0.0       # Additive shift on DiT latents. Default 0 = no shift.
    latent_rescale: float = 1.0     # Multiplicative rescale on DiT latents. Default 1.0 = no rescale.

    # Advanced Settings
    inference_steps: int = 8
    seed: int = -1
    guidance_scale: float = 7.0
    use_adg: bool = False
    cfg_interval_start: float = 0.0
    cfg_interval_end: float = 1.0
    shift: float = 1.0
    infer_method: str = "ode"  # "ode" or "sde" - diffusion inference method
    # Custom timesteps (parsed from string like "0.97,0.76,0.615,0.5,0.395,0.28,0.18,0.085,0")
    # If provided, overrides inference_steps and shift
    timesteps: Optional[List[float]] = None

    repainting_start: float = 0.0
    repainting_end: float = -1
    chunk_mask_mode: str = "auto"  # "explicit" = 0/1 mask from repaint range; "auto" = all 2.0 (model decides)
    audio_cover_strength: float = 1.0
    cover_noise_strength: float = 0.0  # 0=pure noise (no cover), 1=closest to src audio

    # 5Hz Language Model Parameters
    thinking: bool = True
    lm_temperature: float = 0.85
    lm_cfg_scale: float = 2.0
    lm_top_k: int = 0
    lm_top_p: float = 0.9
    lm_negative_prompt: str = "NO USER INPUT"
    use_cot_metas: bool = True
    use_cot_caption: bool = True
    use_cot_lyrics: bool = False  # TODO: not used yet
    use_cot_language: bool = True
    use_constrained_decoding: bool = True

    cot_bpm: Optional[int] = None
    cot_keyscale: str = ""
    cot_timesignature: str = ""
    cot_duration: Optional[float] = None
    cot_vocal_language: str = "unknown"
    cot_caption: str = ""
    cot_lyrics: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class GenerationConfig:
    """Configuration for music generation.
    
    Attributes:
        batch_size: Number of audio samples to generate
        allow_lm_batch: Whether to allow batch processing in LM
        use_random_seed: Whether to use random seed
        seeds: Seed(s) for batch generation. Can be:
            - None: Use random seeds (when use_random_seed=True) or params.seed (when use_random_seed=False)
            - List[int]: List of seeds, will be padded with random seeds if fewer than batch_size
            - int: Single seed value (will be converted to list and padded)
        lm_batch_chunk_size: Batch chunk size for LM processing
        constrained_decoding_debug: Whether to enable constrained decoding debug
        audio_format: Output audio format, one of "mp3", "wav", "flac", "wav32", "opus", "aac". Default: "flac"
    """
    batch_size: int = 2
    allow_lm_batch: bool = False
    use_random_seed: bool = True
    seeds: Optional[List[int]] = None
    lm_batch_chunk_size: int = 8
    constrained_decoding_debug: bool = False
    audio_format: str = "flac"  # Default to FLAC for fast saving

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class GenerationResult:
    """Result of music generation.
    
    Attributes:
        # Audio Outputs
        audios: List of audio dictionaries with paths, keys, params
        status_message: Status message from generation
        extra_outputs: Extra outputs from generation
        success: Whether generation completed successfully
        error: Error message if generation failed
    """

    # Audio Outputs
    audios: List[Dict[str, Any]] = field(default_factory=list)
    # Generation Information
    status_message: str = ""
    extra_outputs: Dict[str, Any] = field(default_factory=dict)
    # Success Status
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class UnderstandResult:
    """Result of music understanding from audio codes.
    
    Attributes:
        # Metadata Fields
        caption: Generated caption describing the music
        lyrics: Generated or extracted lyrics
        bpm: Beats per minute (None if not detected)
        duration: Duration in seconds (None if not detected)
        keyscale: Musical key (e.g., "C Major")
        language: Vocal language code (e.g., "en", "zh")
        timesignature: Time signature (e.g., "4/4")
        
        # Status
        status_message: Status message from understanding
        success: Whether understanding completed successfully
        error: Error message if understanding failed
    """
    # Metadata Fields
    caption: str = ""
    lyrics: str = ""
    bpm: Optional[int] = None
    duration: Optional[float] = None
    keyscale: str = ""
    language: str = ""
    timesignature: str = ""
    
    # Status
    status_message: str = ""
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return asdict(self)


def _update_metadata_from_lm(
    metadata: Dict[str, Any],
    bpm: Optional[int],
    key_scale: str,
    time_signature: str,
    audio_duration: Optional[float],
    vocal_language: str,
    caption: str,
    lyrics: str,
) -> Tuple[Optional[int], str, str, Optional[float], str, str, str]:
    """Update metadata fields from LM output if not provided by user."""

    if bpm is None and metadata.get('bpm'):
        bpm_value = metadata.get('bpm')
        if bpm_value not in ["N/A", ""]:
            try:
                bpm = int(bpm_value)
            except (ValueError, TypeError):
                pass

    if not key_scale and metadata.get('keyscale'):
        key_scale_value = metadata.get('keyscale', metadata.get('key_scale', ""))
        if key_scale_value != "N/A":
            key_scale = key_scale_value

    if not time_signature and metadata.get('timesignature'):
        time_signature_value = metadata.get('timesignature', metadata.get('time_signature', ""))
        if time_signature_value != "N/A":
            time_signature = time_signature_value

    if audio_duration is None or audio_duration <= 0:
        audio_duration_value = metadata.get('duration', -1)
        if audio_duration_value not in ["N/A", ""]:
            try:
                audio_duration = float(audio_duration_value)
            except (ValueError, TypeError):
                pass

    if not vocal_language and metadata.get('vocal_language'):
        vocal_language = metadata.get('vocal_language')
    if not caption and metadata.get('caption'):
        caption = metadata.get('caption')
    if not lyrics and metadata.get('lyrics'):
        lyrics = metadata.get('lyrics')
    return bpm, key_scale, time_signature, audio_duration, vocal_language, caption, lyrics


def _count_lyric_words(lyrics: str) -> int:
    """Count lyric words while ignoring bracketed section-tag lines."""
    if not lyrics:
        return 0
    total = 0
    for line in lyrics.splitlines():
        stripped = line.strip()
        if not stripped or _LYRIC_TAG_LINE_PATTERN.fullmatch(stripped):
            continue
        total += len(_LYRIC_WORD_PATTERN.findall(stripped))
    return total


def _estimate_word_syllables(word: str) -> int:
    """Estimate syllable count for a single word using a lightweight heuristic."""
    normalized = re.sub(r"[^a-zA-Z]", "", word).lower()
    if not normalized:
        return 0

    groups = _LYRIC_VOWEL_GROUP_PATTERN.findall(normalized)
    syllables = len(groups)

    if normalized.endswith("e") and not normalized.endswith(("le", "ye")) and syllables > 1:
        syllables -= 1

    return max(1, syllables)


def _count_lyric_syllables(lyrics: str) -> int:
    """Estimate lyric syllables while ignoring bracketed section-tag lines."""
    if not lyrics:
        return 0
    total = 0
    for line in lyrics.splitlines():
        stripped = line.strip()
        if not stripped or _LYRIC_TAG_LINE_PATTERN.fullmatch(stripped):
            continue
        for word in _LYRIC_WORD_PATTERN.findall(stripped):
            total += _estimate_word_syllables(word)
    return total


def _parse_time_signature_numerator(time_signature: str) -> int:
    """Parse time-signature numerator with a safe default of 4."""
    if not time_signature:
        return 4
    token = str(time_signature).strip()
    if not token:
        return 4
    if "/" in token:
        token = token.split("/", 1)[0].strip()
    try:
        numerator = int(token)
    except (TypeError, ValueError):
        return 4
    if numerator <= 0:
        return 4
    return numerator


def _estimate_lyric_word_budget(
    duration: Optional[float],
    bpm: Optional[int],
    time_signature: str,
) -> Optional[int]:
    """Estimate a soft lyric word budget from duration, tempo, and meter."""
    if duration is None:
        return None
    try:
        duration_value = float(duration)
    except (TypeError, ValueError):
        return None
    if duration_value <= 0:
        return None

    effective_bpm = 110.0
    if bpm is not None:
        try:
            bpm_value = float(bpm)
            if bpm_value > 0:
                effective_bpm = max(30.0, min(300.0, bpm_value))
        except (TypeError, ValueError):
            pass

    beats_per_bar = max(1, _parse_time_signature_numerator(time_signature))
    beats = duration_value * (effective_bpm / 60.0)
    bars = beats / beats_per_bar
    estimated_words = int(round(bars * _LYRIC_DENSITY_WORDS_PER_BAR))
    return max(_LYRIC_DENSITY_MIN_WORD_BUDGET, min(_LYRIC_DENSITY_MAX_WORD_BUDGET, estimated_words))


def _estimate_calculated_vocal_syllable_capacity(
    duration: Optional[float],
    bpm: Optional[int],
    time_signature: str,
) -> Optional[int]:
    """Estimate raw vocal syllable capacity before applying policy budget multipliers."""
    if duration is None:
        return None
    try:
        duration_value = float(duration)
    except (TypeError, ValueError):
        return None
    if duration_value <= 0:
        return None

    effective_bpm = 110.0
    if bpm is not None:
        try:
            bpm_value = float(bpm)
            if bpm_value > 0:
                effective_bpm = max(30.0, min(300.0, bpm_value))
        except (TypeError, ValueError):
            pass

    beats_per_bar = max(1, _parse_time_signature_numerator(time_signature))
    meter_factor = max(0.7, min(1.3, 4.0 / beats_per_bar))
    beats = duration_value * (effective_bpm / 60.0)
    vocal_beats = beats * _LYRIC_DENSITY_VOCAL_ACTIVITY_RATIO
    estimated_syllables = int(
        round(vocal_beats * _LYRIC_DENSITY_SYLLABLES_PER_BEAT * meter_factor)
    )
    return max(1, estimated_syllables)


def _estimate_lyric_syllable_budget(
    duration: Optional[float],
    bpm: Optional[int],
    time_signature: str,
) -> Optional[int]:
    """Estimate syllable budget from vocal capacity with a soft policy multiplier."""
    base_capacity = _estimate_calculated_vocal_syllable_capacity(
        duration=duration,
        bpm=bpm,
        time_signature=time_signature,
    )
    if base_capacity is None:
        return None
    budget = int(round(base_capacity * _LYRIC_DENSITY_SYLLABLE_BUDGET_MULTIPLIER))
    return max(_LYRIC_DENSITY_MIN_SYLLABLE_BUDGET, min(_LYRIC_DENSITY_MAX_SYLLABLE_BUDGET, budget))


def _soft_trim_lyrics_to_word_budget(lyrics: str, word_budget: int) -> Tuple[str, bool]:
    """Trim non-tag lyric text to an approximate word budget while preserving section tags."""
    if word_budget <= 0 or not lyrics:
        return lyrics, False

    remaining = word_budget
    out_lines: List[str] = []
    trimmed = False

    for line in lyrics.splitlines():
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            continue
        if _LYRIC_TAG_LINE_PATTERN.fullmatch(stripped):
            out_lines.append(line)
            continue

        words = _LYRIC_WORD_PATTERN.findall(stripped)
        if not words:
            out_lines.append(line)
            continue

        if remaining <= 0:
            trimmed = True
            break

        if len(words) <= remaining:
            out_lines.append(line)
            remaining -= len(words)
            continue

        trimmed = True
        partial_words = words[:remaining]
        if partial_words:
            out_lines.append(" ".join(partial_words) + " ...")
            remaining = 0
        break

    return "\n".join(out_lines).strip(), trimmed


def _normalize_lyrics_for_generation(lyrics: str) -> str:
    """Normalize user lyrics into DiT-friendly line breaks before generation."""
    if not lyrics:
        return lyrics

    normalized_lines: List[str] = []
    for line in lyrics.splitlines():
        stripped = line.strip()
        if not stripped:
            normalized_lines.append("")
            continue
        if _LYRIC_TAG_LINE_PATTERN.fullmatch(stripped):
            normalized_lines.append(stripped)
            continue

        parts = [part.strip() for part in _LYRIC_COMMA_LINEBREAK_PATTERN.split(stripped) if part.strip()]
        if len(parts) <= 1:
            normalized_lines.append(stripped)
            continue
        normalized_lines.extend(parts)

    return "\n".join(normalized_lines)


def _apply_soft_lyric_density_guard(
    lyrics: str,
    duration: Optional[float],
    bpm: Optional[int],
    time_signature: str,
) -> Tuple[str, Optional[str], bool]:
    """Warn (and optionally soft-trim) when lyric density is too high for short durations."""
    stripped = (lyrics or "").strip()
    if not stripped or stripped.lower() in {"[instrumental]", "[inst]"}:
        return lyrics, None, False

    word_budget = _estimate_lyric_word_budget(duration=duration, bpm=bpm, time_signature=time_signature)
    syllable_budget = _estimate_lyric_syllable_budget(
        duration=duration,
        bpm=bpm,
        time_signature=time_signature,
    )
    if word_budget is None and syllable_budget is None:
        return lyrics, None, False

    word_count = _count_lyric_words(lyrics)
    syllable_count = _count_lyric_syllables(lyrics)

    word_ratio = (word_count / word_budget) if word_budget else 0.0
    syllable_ratio = (syllable_count / syllable_budget) if syllable_budget else 0.0
    density_ratio = max(word_ratio, syllable_ratio)
    if density_ratio <= _LYRIC_DENSITY_WARNING_RATIO:
        return lyrics, None, False

    budget_bits: List[str] = []
    if word_budget:
        budget_bits.append(f"{word_count} words vs ~{word_budget}")
    if syllable_budget:
        budget_bits.append(f"{syllable_count} syllables vs ~{syllable_budget}")
    warning = (
        f"[Lyric Density] Input lyrics may be too dense for this arrangement "
        f"({'; '.join(budget_bits)} suggested)."
    )

    if density_ratio <= _LYRIC_DENSITY_AUTO_TRIM_RATIO:
        return lyrics, warning, False

    trim_word_target = int(
        round((word_budget or max(1, word_count)) * _LYRIC_DENSITY_TRIM_TARGET_RATIO)
    )
    if syllable_budget and word_count > 0 and syllable_count > 0:
        avg_syllables_per_word = syllable_count / max(1, word_count)
        syllable_trim_target = int(
            round(syllable_budget * _LYRIC_DENSITY_TRIM_TARGET_RATIO)
            / max(0.1, avg_syllables_per_word)
        )
        trim_word_target = min(trim_word_target, max(1, syllable_trim_target))

    trimmed_lyrics, did_trim = _soft_trim_lyrics_to_word_budget(lyrics, trim_word_target)
    if not did_trim:
        return lyrics, warning, False

    warning += (
        f" Applied soft trim to ~{trim_word_target} words. "
        "Adjust duration/BPM or shorten lyrics for stronger alignment."
    )
    return trimmed_lyrics, warning, True


@_get_spaces_gpu_decorator(duration=180)
def generate_music(
    dit_handler,
    llm_handler,
    params: GenerationParams,
    config: GenerationConfig,
    save_dir: Optional[str] = None,
    progress=None,
) -> GenerationResult:
    """Generate music using ACE-Step model with optional LM reasoning.
    
    Args:
        dit_handler: Initialized DiT model handler (AceStepHandler instance)
        llm_handler: Initialized LLM handler (LLMHandler instance)
        params: Generation parameters (GenerationParams instance)
        config: Generation configuration (GenerationConfig instance)
        
    Returns:
        GenerationResult with generated audio files and metadata
    """
    try:
        # Phase 1: LM-based metadata and code generation (if enabled)
        audio_code_string_to_use = params.audio_codes
        lm_generated_metadata = None
        lm_generated_audio_codes_list = []
        lm_total_time_costs = {
            "phase1_time": 0.0,
            "phase2_time": 0.0,
            "total_time": 0.0,
        }

        # Extract mutable copies of metadata (will be updated by LM if needed)
        bpm = params.bpm
        key_scale = params.keyscale
        time_signature = params.timesignature
        audio_duration = params.duration
        dit_input_caption = params.caption
        dit_input_vocal_language = params.vocal_language
        dit_input_lyrics = _normalize_lyrics_for_generation(params.lyrics)
        lyrics_for_generation, lyric_density_warning, lyrics_soft_trimmed = _apply_soft_lyric_density_guard(
            lyrics=dit_input_lyrics,
            duration=audio_duration,
            bpm=bpm,
            time_signature=time_signature,
        )
        if lyrics_soft_trimmed:
            dit_input_lyrics = lyrics_for_generation
            params.lyrics = lyrics_for_generation
        elif lyrics_for_generation != dit_input_lyrics:
            dit_input_lyrics = lyrics_for_generation

        # Determine if we need to generate audio codes
        # If user has provided audio_codes, we don't need to generate them
        # Otherwise, check if we need audio codes (lm_dit mode) or just metas (dit mode)
        user_provided_audio_codes = bool(params.audio_codes and str(params.audio_codes).strip())

        # Determine infer_type: use "llm_dit" if we need audio codes, "dit" if only metas needed
        # For now, we use "llm_dit" if batch mode or if user hasn't provided codes
        # Use "dit" if user has provided codes (only need metas) or if explicitly only need metas
        # Note: This logic can be refined based on specific requirements
        need_audio_codes = not user_provided_audio_codes

        # Determine if we should use chunk-based LM generation (always use chunks for consistency)
        # Determine actual batch size for chunk processing
        actual_batch_size = config.batch_size if config.batch_size is not None else 1

        # Prepare seeds for batch generation
        # Use config.seed if provided, otherwise fallback to params.seed
        # Convert config.seed (None, int, or List[int]) to format that prepare_seeds accepts
        seed_for_generation = ""
        # Original code (commented out because it crashes on int seeds):
        # if config.seeds is not None and len(config.seeds) > 0:
        #     if isinstance(config.seeds, list):
        #         # Convert List[int] to comma-separated string
        #         seed_for_generation = ",".join(str(s) for s in config.seeds)

        if config.seeds is not None:
            if isinstance(config.seeds, list) and len(config.seeds) > 0:
                # Convert List[int] to comma-separated string
                seed_for_generation = ",".join(str(s) for s in config.seeds)
            elif isinstance(config.seeds, int):
                # Fix: Explicitly handle single integer seeds by converting to string.
                # Previously, this would crash because 'len()' was called on an int.
                seed_for_generation = str(config.seeds)

        # Use dit_handler.prepare_seeds to handle seed list generation and padding
        # This will handle all the logic: padding with random seeds if needed, etc.
        actual_seed_list, _ = dit_handler.prepare_seeds(actual_batch_size, seed_for_generation, config.use_random_seed)

        # LM-based Chain-of-Thought reasoning
        # Skip LM for cover/repaint tasks - these tasks use reference/src audio directly
        # and don't need LM to generate audio codes
        skip_lm_tasks = {"cover", "repaint"}
        
        # Determine if we should use LLM
        # LLM is needed for:
        # 1. thinking=True: generate audio codes via LM
        # 2. use_cot_caption=True: enhance/generate caption via CoT
        # 3. use_cot_language=True: detect vocal language via CoT
        # 4. use_cot_metas=True: fill missing metadata via CoT
        need_lm_for_cot = params.use_cot_caption or params.use_cot_language or params.use_cot_metas
        use_lm = (params.thinking or need_lm_for_cot) and llm_handler is not None and llm_handler.llm_initialized and params.task_type not in skip_lm_tasks
        lm_status = []
        
        if params.task_type in skip_lm_tasks:
            logger.info(f"Skipping LM for task_type='{params.task_type}' - using DiT directly")
        
        logger.info(f"[generate_music] LLM usage decision: thinking={params.thinking}, "
                   f"use_cot_caption={params.use_cot_caption}, use_cot_language={params.use_cot_language}, "
                   f"use_cot_metas={params.use_cot_metas}, need_lm_for_cot={need_lm_for_cot}, "
                   f"llm_initialized={llm_handler.llm_initialized if llm_handler else False}, use_lm={use_lm}")
        if lyric_density_warning:
            logger.warning("[generate_music] {}", lyric_density_warning)
            lm_status.append(lyric_density_warning)
        
        if use_lm:
            # Convert sampling parameters - handle None values safely
            top_k_value = None if not params.lm_top_k or params.lm_top_k == 0 else int(params.lm_top_k)
            top_p_value = None if not params.lm_top_p or params.lm_top_p >= 1.0 else params.lm_top_p

            # Build user_metadata from user-provided values
            user_metadata = {}
            if bpm is not None:
                try:
                    bpm_value = float(bpm)
                    if bpm_value > 0:
                        user_metadata['bpm'] = int(bpm_value)
                except (ValueError, TypeError):
                    pass

            if key_scale and key_scale.strip():
                key_scale_clean = key_scale.strip()
                if key_scale_clean.lower() not in ["n/a", ""]:
                    user_metadata['keyscale'] = key_scale_clean

            if time_signature and time_signature.strip():
                time_sig_clean = time_signature.strip()
                if time_sig_clean.lower() not in ["n/a", ""]:
                    user_metadata['timesignature'] = time_sig_clean

            if audio_duration is not None:
                try:
                    duration_value = float(audio_duration)
                    if duration_value > 0:
                        user_metadata['duration'] = int(duration_value)
                except (ValueError, TypeError):
                    pass

            user_metadata_to_pass = user_metadata if user_metadata else None

            # Determine infer_type based on whether we need audio codes
            # - "llm_dit": generates both metas and audio codes (two-phase internally)
            # - "dit": generates only metas (single phase)
            infer_type = "llm_dit" if need_audio_codes and params.thinking else "dit"

            # Use chunk size from config, or default to batch_size if not set
            max_inference_batch_size = int(config.lm_batch_chunk_size) if config.lm_batch_chunk_size > 0 else actual_batch_size
            num_chunks = math.ceil(actual_batch_size / max_inference_batch_size)

            all_metadata_list = []
            all_audio_codes_list = []

            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * max_inference_batch_size
                chunk_end = min(chunk_start + max_inference_batch_size, actual_batch_size)
                chunk_size = chunk_end - chunk_start
                chunk_seeds = actual_seed_list[chunk_start:chunk_end] if chunk_start < len(actual_seed_list) else None

                logger.info(f"LM chunk {chunk_idx+1}/{num_chunks} (infer_type={infer_type}) "
                            f"(size: {chunk_size}, seeds: {chunk_seeds})")

                # Use the determined infer_type
                # - "llm_dit" will internally run two phases (metas + codes)
                # - "dit" will only run phase 1 (metas only)
                result = llm_handler.generate_with_stop_condition(
                    caption=params.caption or "",
                    lyrics=lyrics_for_generation or "",
                    infer_type=infer_type,
                    temperature=params.lm_temperature,
                    cfg_scale=params.lm_cfg_scale,
                    negative_prompt=params.lm_negative_prompt,
                    top_k=top_k_value,
                    top_p=top_p_value,
                    target_duration=audio_duration,  # Pass duration to limit audio codes generation
                    user_metadata=user_metadata_to_pass,
                    use_cot_caption=params.use_cot_caption,
                    use_cot_language=params.use_cot_language,
                    use_cot_metas=params.use_cot_metas,
                    use_constrained_decoding=params.use_constrained_decoding,
                    constrained_decoding_debug=config.constrained_decoding_debug,
                    batch_size=chunk_size,
                    seeds=chunk_seeds,
                    progress=progress,
                )

                # Check if LM generation failed
                if not result.get("success", False):
                    error_msg = result.get("error", "Unknown LM error")
                    lm_status.append(f"❌ LM Error: {error_msg}")
                    # Return early with error
                    return GenerationResult(
                        audios=[],
                        status_message=f"❌ LM generation failed: {error_msg}",
                        extra_outputs={},
                        success=False,
                        error=error_msg,
                    )

                # Extract metadata and audio_codes from result dict
                if chunk_size > 1:
                    metadata_list = result.get("metadata", [])
                    audio_codes_list = result.get("audio_codes", [])
                    all_metadata_list.extend(metadata_list)
                    all_audio_codes_list.extend(audio_codes_list)
                else:
                    metadata = result.get("metadata", {})
                    audio_codes = result.get("audio_codes", "")
                    all_metadata_list.append(metadata)
                    all_audio_codes_list.append(audio_codes)

                # Collect time costs from LM extra_outputs
                lm_extra = result.get("extra_outputs", {})
                lm_chunk_time_costs = lm_extra.get("time_costs", {})
                if lm_chunk_time_costs:
                    # Accumulate time costs from all chunks
                    for key in ["phase1_time", "phase2_time", "total_time"]:
                        if key in lm_chunk_time_costs:
                            lm_total_time_costs[key] += lm_chunk_time_costs[key]

                    time_str = ", ".join([f"{k}: {v:.2f}s" for k, v in lm_chunk_time_costs.items()])
                    lm_status.append(f"✅ LM chunk {chunk_idx+1}: {time_str}")

            lm_generated_metadata = all_metadata_list[0] if all_metadata_list else None
            lm_generated_audio_codes_list = all_audio_codes_list

            # Set audio_code_string_to_use based on infer_type
            if infer_type == "llm_dit":
                # If batch mode, use list; otherwise use single string
                if actual_batch_size > 1:
                    audio_code_string_to_use = all_audio_codes_list
                else:
                    audio_code_string_to_use = all_audio_codes_list[0] if all_audio_codes_list else ""
            else:
                # For "dit" mode, keep user-provided codes or empty
                audio_code_string_to_use = params.audio_codes

            # Update metadata from LM if not provided by user
            if lm_generated_metadata:
                bpm, key_scale, time_signature, audio_duration, vocal_language, caption, lyrics = _update_metadata_from_lm(
                    metadata=lm_generated_metadata,
                    bpm=bpm,
                    key_scale=key_scale,
                    time_signature=time_signature,
                    audio_duration=audio_duration,
                    vocal_language=dit_input_vocal_language,
                    caption=dit_input_caption,
                    lyrics=lyrics_for_generation)
                generated_caption = ensure_caption_has_global_vocal_presence(
                    lm_generated_metadata.get('caption', '') or '',
                    lyrics=lm_generated_metadata.get('lyrics', '') or lyrics,
                    vocal_language=lm_generated_metadata.get('vocal_language', '') or vocal_language,
                    instrumental=lm_generated_metadata.get('instrumental'),
                )
                if generated_caption:
                    caption = generated_caption
                if not params.bpm:
                    params.cot_bpm = bpm
                if not params.keyscale:
                    params.cot_keyscale = key_scale
                if not params.timesignature:
                    params.cot_timesignature = time_signature
                if not params.duration:
                    params.cot_duration = audio_duration
                if not params.vocal_language:
                    params.cot_vocal_language = vocal_language
                if not params.caption:
                    params.cot_caption = caption
                if not params.lyrics:
                    params.cot_lyrics = lyrics

            # set cot caption and language if needed
            if params.use_cot_caption and lm_generated_metadata:
                dit_input_caption = ensure_caption_has_global_vocal_presence(
                    lm_generated_metadata.get('caption', '') or dit_input_caption,
                    lyrics=lm_generated_metadata.get('lyrics', '') or lyrics_for_generation,
                    vocal_language=lm_generated_metadata.get('vocal_language', '') or dit_input_vocal_language,
                    instrumental=lm_generated_metadata.get('instrumental'),
                )
            if params.use_cot_language and lm_generated_metadata:
                dit_input_vocal_language = lm_generated_metadata.get("vocal_language", dit_input_vocal_language)

        # Repaint/cover: no LM run, so conditioning must come from params (caption + lyrics from GUI).
        if params.task_type in ("repaint", "cover"):
            dit_input_caption = params.caption or dit_input_caption
            dit_input_lyrics = params.lyrics if params.lyrics is not None else dit_input_lyrics
            logger.info(f"[generate_music] Repaint/Cover task: using params.caption='{params.caption}', params.lyrics='{params.lyrics}'")
            logger.info(f"[generate_music] Final inputs: dit_input_caption='{dit_input_caption}', dit_input_lyrics='{dit_input_lyrics}'")

        # Phase 2: DiT music generation
        # Use seed_for_generation (from config.seed or params.seed) instead of params.seed for actual generation
        result = dit_handler.generate_music(
            captions=dit_input_caption,
            global_caption=params.global_caption,
            lyrics=dit_input_lyrics,
            bpm=bpm,
            key_scale=key_scale,
            time_signature=time_signature,
            vocal_language=dit_input_vocal_language,
            inference_steps=params.inference_steps,
            guidance_scale=params.guidance_scale,
            use_random_seed=config.use_random_seed,
            seed=seed_for_generation,  # Use config.seed (or params.seed fallback) instead of params.seed directly
            reference_audio=params.reference_audio,
            audio_duration=audio_duration,
            batch_size=config.batch_size if config.batch_size is not None else 1,
            # text2music (Custom mode) never uses src_audio; force None to
            # prevent stale UI values from leaking into generation.
            src_audio=None if params.task_type == "text2music" else params.src_audio,
            audio_code_string=audio_code_string_to_use,
            repainting_start=params.repainting_start,
            repainting_end=params.repainting_end,
            instruction=params.instruction,
            audio_cover_strength=params.audio_cover_strength,
            cover_noise_strength=params.cover_noise_strength,
            task_type=params.task_type,
            use_adg=params.use_adg,
            cfg_interval_start=params.cfg_interval_start,
            cfg_interval_end=params.cfg_interval_end,
            shift=params.shift,
            infer_method=params.infer_method,
            timesteps=params.timesteps,
            latent_shift=params.latent_shift,
            latent_rescale=params.latent_rescale,
            chunk_mask_mode=getattr(params, "chunk_mask_mode", "auto"),
            progress=progress,
        )

        # Check if generation failed
        if not result.get("success", False):
            return GenerationResult(
                audios=[],
                status_message=result.get("status_message", ""),
                extra_outputs={},
                success=False,
                error=result.get("error"),
            )

        # Extract results from dit_handler.generate_music dict
        dit_audios = result.get("audios", [])
        status_message = result.get("status_message", "")
        dit_extra_outputs = result.get("extra_outputs", {})

        # Use the seed list already prepared above (from config.seed or params.seed fallback)
        # actual_seed_list was computed earlier using dit_handler.prepare_seeds
        seed_list = actual_seed_list

        # Get base params dictionary
        base_params_dict = params.to_dict()

        # Save audio files using AudioSaver (format from config)
        audio_format = config.audio_format if config.audio_format else "flac"
        audio_saver = AudioSaver(default_format=audio_format)

        # Use handler's temp_dir for saving files
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # Build audios list for GenerationResult with params and save files
        # Audio saving and UUID generation handled here, outside of handler
        audios = []
        for idx, dit_audio in enumerate(dit_audios):
            # Create a copy of params dict for this audio
            audio_params = base_params_dict.copy()

            # Update audio-specific values
            audio_params["seed"] = seed_list[idx] if idx < len(seed_list) else None

            # Add LM-generated audio codes (only if non-empty, to preserve
            # user-provided codes when LM was used only for CoT metas)
            if lm_generated_audio_codes_list and idx < len(lm_generated_audio_codes_list):
                lm_code = lm_generated_audio_codes_list[idx]
                if lm_code and str(lm_code).strip():
                    audio_params["audio_codes"] = lm_code

            # Add LoRA state to params for UUID generation (ensures different UUIDs when only LoRA state changes)
            audio_params["lora_loaded"] = dit_handler.lora_loaded
            audio_params["use_lora"] = dit_handler.use_lora
            audio_params["lora_scale"] = dit_handler.lora_scale
            audio_params["lora_weights_hash"] = get_lora_weights_hash(dit_handler)

            # Get audio tensor and metadata
            audio_tensor = dit_audio.get("tensor")
            sample_rate = dit_audio.get("sample_rate", 48000)

            # --- NORMALIZATION & LOGGING ---
            if params.enable_normalization and params.normalization_db <= 0.0:
                 try:
                     peak_before = torch.max(torch.abs(audio_tensor)).item()
                     logger.info(f"[Normalization] Audio {idx} BEFORE: Peak={peak_before:.4f}, Target={params.normalization_db}dB")
                     
                     audio_tensor = normalize_audio(audio_tensor, params.normalization_db)
                     
                     peak_after = torch.max(torch.abs(audio_tensor)).item()
                     logger.info(f"[Normalization] Audio {idx} AFTER: Peak={peak_after:.4f}")
                     
                     # Update the tensor in the dict so downstream uses the normalized version ??
                     # Actually we use audio_tensor variable below, so it's fine.
                 except Exception as e:
                     logger.error(f"Normalization failed: {e}")
            # -------------------------------

            # --- FADE IN / FADE OUT ---
            if params.fade_in_duration > 0.0 or params.fade_out_duration > 0.0:
                try:
                    fade_in_samples = round(params.fade_in_duration * sample_rate)
                    fade_out_samples = round(params.fade_out_duration * sample_rate)
                    audio_tensor = apply_fade(audio_tensor, fade_in_samples, fade_out_samples)
                    logger.info(
                        f"[Fade] Audio {idx}: fade_in={params.fade_in_duration:.2f}s "
                        f"({fade_in_samples} samples), fade_out={params.fade_out_duration:.2f}s "
                        f"({fade_out_samples} samples)"
                    )
                except Exception as e:
                    logger.error(f"Fade application failed: {e}")
            # --------------------------

            # Generate UUID for this audio (moved from handler)
            batch_seed = seed_list[idx] if idx < len(seed_list) else seed_list[0] if seed_list else -1

            audio_code_str = lm_generated_audio_codes_list[idx] if (
                lm_generated_audio_codes_list and idx < len(lm_generated_audio_codes_list)) else audio_code_string_to_use
            if isinstance(audio_code_str, list):
                audio_code_str = audio_code_str[idx] if idx < len(audio_code_str) else ""

            audio_key = generate_uuid_from_params(audio_params)

            # Save audio file (handled outside handler)
            audio_path = None
            if audio_tensor is not None and save_dir is not None:

                try:
                    # Handle wav32 special case for extension
                    file_ext = "wav" if audio_format == "wav32" else audio_format
                    audio_file = os.path.join(save_dir, f"{audio_key}.{file_ext}")
                    
                    audio_path = audio_saver.save_audio(audio_tensor,
                                                        audio_file,
                                                        sample_rate=sample_rate,
                                                        format=audio_format,
                                                        channels_first=True)
                except Exception as e:
                    logger.error(f"[generate_music] Failed to save audio file: {e}")
                    audio_path = ""  # Fallback to empty path

            audio_dict = {
                "path": audio_path or "",  # File path (saved here, not in handler)
                "tensor": audio_tensor,  # Audio tensor [channels, samples], CPU, float32
                "key": audio_key,
                "sample_rate": sample_rate,
                "params": audio_params,
            }

            audios.append(audio_dict)

        # Merge extra_outputs: include dit_extra_outputs (latents, masks) and add LM metadata
        extra_outputs = dit_extra_outputs.copy()
        extra_outputs["lm_metadata"] = lm_generated_metadata

        # Merge time_costs from both LM and DiT into a unified dictionary
        unified_time_costs = {}

        # Add LM time costs (if LM was used)
        if use_lm and lm_total_time_costs:
            for key, value in lm_total_time_costs.items():
                unified_time_costs[f"lm_{key}"] = value

        # Add DiT time costs (if available)
        dit_time_costs = dit_extra_outputs.get("time_costs", {})
        if dit_time_costs:
            for key, value in dit_time_costs.items():
                unified_time_costs[f"dit_{key}"] = value

        # Calculate total pipeline time
        if unified_time_costs:
            lm_total = unified_time_costs.get("lm_total_time", 0.0)
            dit_total = unified_time_costs.get("dit_total_time_cost", 0.0)
            unified_time_costs["pipeline_total_time"] = lm_total + dit_total

        # Update extra_outputs with unified time_costs
        extra_outputs["time_costs"] = unified_time_costs

        if lm_status:
            status_message = "\n".join(lm_status) + "\n" + status_message
        else:
            status_message = status_message
        # Create and return GenerationResult
        return GenerationResult(
            audios=audios,
            status_message=status_message,
            extra_outputs=extra_outputs,
            success=True,
            error=None,
        )

    except Exception as e:
        logger.exception("Music generation failed")
        return GenerationResult(
            audios=[],
            status_message=f"Error: {str(e)}",
            extra_outputs={},
            success=False,
            error=str(e),
        )


def understand_music(
    llm_handler,
    audio_codes: str,
    temperature: float = 0.85,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    use_constrained_decoding: bool = True,
    constrained_decoding_debug: bool = False,
) -> UnderstandResult:
    """Understand music from audio codes using the 5Hz Language Model.
    
    This function analyzes audio semantic codes and generates metadata about the music,
    including caption, lyrics, BPM, duration, key scale, language, and time signature.
    
    If audio_codes is empty or "NO USER INPUT", the LM will generate a sample example
    instead of analyzing existing codes.
    
    Note: cfg_scale and negative_prompt are not supported in understand mode.
    
    Args:
        llm_handler: Initialized LLM handler (LLMHandler instance)
        audio_codes: String of audio code tokens (e.g., "<|audio_code_123|><|audio_code_456|>...")
                     Use empty string or "NO USER INPUT" to generate a sample example.
        temperature: Sampling temperature for generation (0.0-2.0). Higher = more creative.
        top_k: Top-K sampling (None or 0 = disabled)
        top_p: Top-P (nucleus) sampling (None or 1.0 = disabled)
        repetition_penalty: Repetition penalty (1.0 = no penalty)
        use_constrained_decoding: Whether to use FSM-based constrained decoding for metadata
        constrained_decoding_debug: Whether to enable debug logging for constrained decoding
        
    Returns:
        UnderstandResult with parsed metadata fields and status
        
    Example:
        >>> result = understand_music(llm_handler, audio_codes="<|audio_code_123|>...")
        >>> if result.success:
        ...     print(f"Caption: {result.caption}")
        ...     print(f"BPM: {result.bpm}")
        ...     print(f"Lyrics: {result.lyrics}")
    """
    # Check if LLM is initialized
    if not llm_handler.llm_initialized:
        return UnderstandResult(
            status_message="5Hz LM not initialized. Please initialize it first.",
            success=False,
            error="LLM not initialized",
        )
    
    # If codes are empty, use "NO USER INPUT" to generate a sample example
    if not audio_codes or not audio_codes.strip():
        audio_codes = "NO USER INPUT"
    
    try:
        # Call LLM understanding
        metadata, status = llm_handler.understand_audio_from_codes(
            audio_codes=audio_codes,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_constrained_decoding=use_constrained_decoding,
            constrained_decoding_debug=constrained_decoding_debug,
        )
        
        # Check if LLM returned empty metadata (error case)
        if not metadata:
            return UnderstandResult(
                status_message=status or "Failed to understand audio codes",
                success=False,
                error=status or "Empty metadata returned",
            )
        
        # Extract and convert fields
        caption = metadata.get('caption', '')
        lyrics = metadata.get('lyrics', '')
        keyscale = metadata.get('keyscale', '')
        language = metadata.get('language', metadata.get('vocal_language', ''))
        timesignature = metadata.get('timesignature', '')
        
        # Convert BPM to int
        bpm = None
        bpm_value = metadata.get('bpm')
        if bpm_value is not None and bpm_value != 'N/A' and bpm_value != '':
            try:
                bpm = int(bpm_value)
            except (ValueError, TypeError):
                pass
        
        # Convert duration to float
        duration = None
        duration_value = metadata.get('duration')
        if duration_value is not None and duration_value != 'N/A' and duration_value != '':
            try:
                duration = float(duration_value)
            except (ValueError, TypeError):
                pass
        
        # Clean up N/A values
        if keyscale == 'N/A':
            keyscale = ''
        if language == 'N/A':
            language = ''
        if timesignature == 'N/A':
            timesignature = ''
        
        return UnderstandResult(
            caption=caption,
            lyrics=lyrics,
            bpm=bpm,
            duration=duration,
            keyscale=keyscale,
            language=language,
            timesignature=timesignature,
            status_message=status,
            success=True,
            error=None,
        )
        
    except Exception as e:
        logger.exception("Music understanding failed")
        return UnderstandResult(
            status_message=f"Error: {str(e)}",
            success=False,
            error=str(e),
        )


@dataclass
class CreateSampleResult:
    """Result of creating a music sample from a natural language query.
    
    This is used by the "Simple Mode" / "Inspiration Mode" feature where users
    provide a natural language description and the LLM generates a complete
    sample with caption, lyrics, and metadata.
    
    Attributes:
        # Metadata Fields
        caption: Generated detailed music description/caption
        lyrics: Generated lyrics (or "[Instrumental]" for instrumental music)
        bpm: Beats per minute (None if not generated)
        duration: Duration in seconds (None if not generated)
        keyscale: Musical key (e.g., "C Major")
        language: Vocal language code (e.g., "en", "zh")
        timesignature: Time signature (e.g., "4")
        instrumental: Whether this is an instrumental piece
        
        # Status
        status_message: Status message from sample creation
        success: Whether sample creation completed successfully
        error: Error message if sample creation failed
    """
    # Metadata Fields
    caption: str = ""
    lyrics: str = ""
    bpm: Optional[int] = None
    duration: Optional[float] = None
    keyscale: str = ""
    language: str = ""
    timesignature: str = ""
    instrumental: bool = False
    
    # Status
    status_message: str = ""
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return asdict(self)


def create_sample(
    llm_handler,
    query: str,
    instrumental: bool = False,
    vocal_language: Optional[str] = None,
    temperature: float = 0.85,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    use_constrained_decoding: bool = True,
    constrained_decoding_debug: bool = False,
) -> CreateSampleResult:
    """Create a music sample from a natural language query using the 5Hz Language Model.
    
    This is the "Simple Mode" / "Inspiration Mode" feature that takes a user's natural
    language description of music and generates a complete sample including:
    - Detailed caption/description
    - Lyrics (unless instrumental)
    - Metadata (BPM, duration, key, language, time signature)
    
    Note: cfg_scale and negative_prompt are not supported in create_sample mode.
    
    Args:
        llm_handler: Initialized LLM handler (LLMHandler instance)
        query: User's natural language music description (e.g., "a soft Bengali love song")
        instrumental: Whether to generate instrumental music (no vocals)
        vocal_language: Allowed vocal language for constrained decoding (e.g., "en", "zh").
                       If provided, the model will be constrained to generate lyrics in this language.
                       If None or "unknown", no language constraint is applied.
        temperature: Sampling temperature for generation (0.0-2.0). Higher = more creative.
        top_k: Top-K sampling (None or 0 = disabled)
        top_p: Top-P (nucleus) sampling (None or 1.0 = disabled)
        repetition_penalty: Repetition penalty (1.0 = no penalty)
        use_constrained_decoding: Whether to use FSM-based constrained decoding
        constrained_decoding_debug: Whether to enable debug logging
        
    Returns:
        CreateSampleResult with generated sample fields and status
        
    Example:
        >>> result = create_sample(llm_handler, "a soft Bengali love song for a quiet evening", vocal_language="bn")
        >>> if result.success:
        ...     print(f"Caption: {result.caption}")
        ...     print(f"Lyrics: {result.lyrics}")
        ...     print(f"BPM: {result.bpm}")
    """
    # Check if LLM is initialized
    if not llm_handler.llm_initialized:
        return CreateSampleResult(
            status_message="5Hz LM not initialized. Please initialize it first.",
            success=False,
            error="LLM not initialized",
        )
    
    try:
        # Call LLM to create sample
        metadata, status = llm_handler.create_sample_from_query(
            query=query,
            instrumental=instrumental,
            vocal_language=vocal_language,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_constrained_decoding=use_constrained_decoding,
            constrained_decoding_debug=constrained_decoding_debug,
        )
        
        # Check if LLM returned empty metadata (error case)
        if not metadata:
            return CreateSampleResult(
                status_message=status or "Failed to create sample",
                success=False,
                error=status or "Empty metadata returned",
            )
        
        # Extract and convert fields
        caption = metadata.get('caption', '')
        lyrics = metadata.get('lyrics', '')
        keyscale = metadata.get('keyscale', '')
        language = metadata.get('language', metadata.get('vocal_language', ''))
        timesignature = metadata.get('timesignature', '')
        is_instrumental = metadata.get('instrumental', instrumental)
        
        # Convert BPM to int
        bpm = None
        bpm_value = metadata.get('bpm')
        if bpm_value is not None and bpm_value != 'N/A' and bpm_value != '':
            try:
                bpm = int(bpm_value)
            except (ValueError, TypeError):
                pass
        
        # Convert duration to float
        duration = None
        duration_value = metadata.get('duration')
        if duration_value is not None and duration_value != 'N/A' and duration_value != '':
            try:
                duration = float(duration_value)
            except (ValueError, TypeError):
                pass
        
        # Clean up N/A values
        if keyscale == 'N/A':
            keyscale = ''
        if language == 'N/A':
            language = ''
        if timesignature == 'N/A':
            timesignature = ''
        
        return CreateSampleResult(
            caption=caption,
            lyrics=lyrics,
            bpm=bpm,
            duration=duration,
            keyscale=keyscale,
            language=language,
            timesignature=timesignature,
            instrumental=is_instrumental,
            status_message=status,
            success=True,
            error=None,
        )
        
    except Exception as e:
        logger.exception("Sample creation failed")
        return CreateSampleResult(
            status_message=f"Error: {str(e)}",
            success=False,
            error=str(e),
        )


@dataclass
class FormatSampleResult:
    """Result of formatting user-provided caption and lyrics.
    
    This is used by the "Format" feature where users provide caption and lyrics,
    and the LLM formats them into structured music metadata and an enhanced description.
    
    Attributes:
        # Metadata Fields
        caption: Enhanced/formatted music description/caption
        lyrics: Formatted lyrics (may be same as input or reformatted)
        bpm: Beats per minute (None if not detected)
        duration: Duration in seconds (None if not detected)
        keyscale: Musical key (e.g., "C Major")
        language: Vocal language code (e.g., "en", "zh")
        timesignature: Time signature (e.g., "4")
        
        # Status
        status_message: Status message from formatting
        success: Whether formatting completed successfully
        error: Error message if formatting failed
    """
    # Metadata Fields
    caption: str = ""
    lyrics: str = ""
    bpm: Optional[int] = None
    duration: Optional[float] = None
    keyscale: str = ""
    language: str = ""
    timesignature: str = ""
    
    # Status
    status_message: str = ""
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return asdict(self)


def format_sample(
    llm_handler,
    caption: str,
    lyrics: str,
    user_metadata: Optional[Dict[str, Any]] = None,
    temperature: float = 0.85,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    use_constrained_decoding: bool = True,
    constrained_decoding_debug: bool = False,
) -> FormatSampleResult:
    """Format user-provided caption and lyrics using the 5Hz Language Model.
    
    This function takes user input (caption and lyrics) and generates structured
    music metadata including an enhanced caption, BPM, duration, key, language,
    and time signature.
    
    If user_metadata is provided, those values will be used to constrain the
    decoding, ensuring the output matches user-specified values.
    
    Note: cfg_scale and negative_prompt are not supported in format mode.
    
    Args:
        llm_handler: Initialized LLM handler (LLMHandler instance)
        caption: User's caption/description (e.g., "Latin pop, reggaeton")
        lyrics: User's lyrics with structure tags
        user_metadata: Optional dict with user-provided metadata to constrain decoding.
                      Supported keys: bpm, duration, keyscale, timesignature, language
        temperature: Sampling temperature for generation (0.0-2.0). Higher = more creative.
        top_k: Top-K sampling (None or 0 = disabled)
        top_p: Top-P (nucleus) sampling (None or 1.0 = disabled)
        repetition_penalty: Repetition penalty (1.0 = no penalty)
        use_constrained_decoding: Whether to use FSM-based constrained decoding for metadata
        constrained_decoding_debug: Whether to enable debug logging for constrained decoding
        
    Returns:
        FormatSampleResult with formatted metadata fields and status
        
    Example:
        >>> result = format_sample(llm_handler, "Latin pop, reggaeton", "[Verse 1]\\nHola mundo...")
        >>> if result.success:
        ...     print(f"Caption: {result.caption}")
        ...     print(f"BPM: {result.bpm}")
        ...     print(f"Lyrics: {result.lyrics}")
    """
    # Check if LLM is initialized
    if not llm_handler.llm_initialized:
        return FormatSampleResult(
            status_message="5Hz LM not initialized. Please initialize it first.",
            success=False,
            error="LLM not initialized",
        )
    
    try:
        # Call LLM formatting
        metadata, status = llm_handler.format_sample_from_input(
            caption=caption,
            lyrics=lyrics,
            user_metadata=user_metadata,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_constrained_decoding=use_constrained_decoding,
            constrained_decoding_debug=constrained_decoding_debug,
        )
        
        # Check if LLM returned empty metadata (error case)
        if not metadata:
            return FormatSampleResult(
                status_message=status or "Failed to format input",
                success=False,
                error=status or "Empty metadata returned",
            )
        
        # Extract and convert fields
        result_caption = metadata.get('caption', '')
        result_lyrics = metadata.get('lyrics', lyrics)  # Fall back to input lyrics
        keyscale = metadata.get('keyscale', '')
        language = metadata.get('language', metadata.get('vocal_language', ''))
        timesignature = metadata.get('timesignature', '')
        
        # Convert BPM to int
        bpm = None
        bpm_value = metadata.get('bpm')
        if bpm_value is not None and bpm_value != 'N/A' and bpm_value != '':
            try:
                bpm = int(bpm_value)
            except (ValueError, TypeError):
                pass
        
        # Convert duration to float
        duration = None
        duration_value = metadata.get('duration')
        if duration_value is not None and duration_value != 'N/A' and duration_value != '':
            try:
                duration = float(duration_value)
            except (ValueError, TypeError):
                pass
        
        # Clean up N/A values
        if keyscale == 'N/A':
            keyscale = ''
        if language == 'N/A':
            language = ''
        if timesignature == 'N/A':
            timesignature = ''
        
        return FormatSampleResult(
            caption=result_caption,
            lyrics=result_lyrics,
            bpm=bpm,
            duration=duration,
            keyscale=keyscale,
            language=language,
            timesignature=timesignature,
            status_message=status,
            success=True,
            error=None,
        )
        
    except Exception as e:
        logger.exception("Format sample failed")
        return FormatSampleResult(
            status_message=f"Error: {str(e)}",
            success=False,
            error=str(e),
        )
