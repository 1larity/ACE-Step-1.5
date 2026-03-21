"""Prompt-bank helpers for bulk caption enhancement review."""

from __future__ import annotations

from pathlib import Path

DEFAULT_CER_PROMPTS = [
    "Minimal techno, instrumental",
    "Ambient choir soundscape, no drums, no vocals",
    "Hyperpop breakup anthem, female vocals, glitchy drop, bilingual English/Japanese",
    "Blackened doom folk lament, male vocals, very slow, haunted atmosphere",
    "Jazz waltz torch song, smoky female vocals, 3/4",
    "Math rock with spoken word, anxious energy, shifting accents",
    "Bollywood disco banger, duet vocals, huge chorus, dramatic strings",
    "Dream pop lullaby, whispered female vocals, tiny arrangement, very intimate mix",
    "Industrial rap track, distorted male vocals, metallic percussion, no melody",
    "Salsa dura with brass section, call-and-response vocals, live club energy",
    "Bluegrass murder ballad, male vocals, fiddle and banjo, cinematic buildup",
    "Shoegaze hymn, androgynous vocals, wall of guitars, blurred chorus",
    "Afro-house sunrise anthem, female vocal hook, percussive build, long DJ-friendly outro",
    "Post-rock crescendo, instrumental, starts sparse and ends enormous",
    "K-pop summer single, female group vocals, rap verse, euphoric final chorus",
    "Dark cabaret tango, theatrical female vocals, accordion, upright bass",
    "Chiptune pop song, cute lead vocal, retro game textures, punchy chorus",
    "Drone metal meditation, almost no drums, enormous sustained guitars, wordless vocal textures",
    "Reggaeton heartbreak duet, male and female vocals, glossy modern mix",
    "Progressive trance instrumental, evolving pads, arpeggiators, long breakdown, no vocals",
    "Country soul ballad with gospel choir, male lead, organ swell in the bridge",
    "Arabic pop with trap drums, female vocals, ornate strings, dramatic pre-chorus",
    "Noise rock confession, strained male vocals, explosive chorus, ugly-beautiful mix",
    "Bossa nova lounge piece, intimate female vocals, brushed drums, nylon-string guitar",
    "Symphonic power metal anthem, soaring male vocals, double-kick drums, heroic outro",
]


def load_cer_prompts(prompt_file: Path | None, prompt_limit: int) -> list[str]:
    """Load CER prompts from a file or the built-in default bank."""

    prompts = list(DEFAULT_CER_PROMPTS)
    if prompt_file:
        prompts = [
            line.strip()
            for line in prompt_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if prompt_limit > 0:
        return prompts[:prompt_limit]
    return prompts
