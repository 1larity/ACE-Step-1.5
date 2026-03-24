"""Localized lyric pools for fallback scaffold generation."""

from __future__ import annotations


def resolve_fallback_section_pools(
    vocal_language: str,
    theme: str,
) -> tuple[list[str], list[str], list[str]]:
    """Return verse and chorus line pools for the requested vocal language."""
    language = _normalize_vocal_language(vocal_language)
    if language == "ja":
        return _JA_VERSE_1, _JA_CHORUS, _JA_VERSE_2
    if language in {"zh", "yue"}:
        return _ZH_VERSE_1, _ZH_CHORUS, _ZH_VERSE_2
    if language == "ko":
        return _KO_VERSE_1, _KO_CHORUS, _KO_VERSE_2
    return (
        [
            f"I carry the pulse of {theme} like a spark in my chest",
            "Every small heartbeat keeps pulling me west",
            "Streetlight reflections are writing our names",
            "We turn the silence to wildfire and flames",
            "All of my doubt starts to loosen and fall",
            "Your voice keeps rising above it all",
        ],
        [
            "Hold me in the light, we can outrun the night",
            "Sing it till the skyline opens wide",
            "When the kick drum lands, our shadows come alive",
            "We keep the fire bright until the morning tide",
        ],
        [
            "Rain on the windows keeps time with the snare",
            "Hope in the low end is filling the air",
            "Breath on the downbeat, we lean into sound",
            "Turn every tremble to something profound",
            "One final chorus and then we let go",
            "Leaving a trail of electric glow",
        ],
    )


def _normalize_vocal_language(vocal_language: str) -> str:
    """Collapse locale variants to a compact language identifier."""
    normalized = (vocal_language or "").strip().lower()
    if not normalized or normalized == "unknown":
        return "en"
    return normalized.replace("_", "-").split("-", maxsplit=1)[0]


_JA_VERSE_1 = [
    "胸の奥で灯るリズムが夜を照らす",
    "揺れる街のネオンが未来をなぞる",
    "静かな吐息をメロディーに変えて",
    "ほどけた不安も光へ溶けていく",
]
_JA_CHORUS = [
    "今この声で夜を越えていこう",
    "高鳴る鼓動を空へ放とう",
    "つないだ願いが明日をひらく",
    "消えない火花を抱いて歌おう",
]
_JA_VERSE_2 = [
    "窓を打つ雨さえビートに変わる",
    "重なるステップが心をほどく",
    "震える想いも強さに変えて",
    "最後のサビまで夢を離さない",
]

_ZH_VERSE_1 = [
    "胸口的节拍点亮整片夜色",
    "霓虹在街角轻轻描摹承诺",
    "把沉默呼吸都唱成了温热",
    "摇晃的不安也慢慢化成火",
]
_ZH_CHORUS = [
    "用这一声穿过漫长黑夜",
    "让心跳把天空一点点照亮",
    "握紧的愿望正在前方盛放",
    "把不熄的光都唱进你眼眶",
]
_ZH_VERSE_2 = [
    "窗上的雨点跟着军鼓坠落",
    "低频里的希望填满了轮廓",
    "每一次颤抖都变得更执着",
    "最后一段副歌把梦紧紧握着",
]

_KO_VERSE_1 = [
    "가슴속 리듬이 밤을 밝히고",
    "도시의 네온이 내일을 그려",
    "조용한 숨결도 멜로디가 되고",
    "흩어진 불안은 빛으로 번져",
]
_KO_CHORUS = [
    "이 목소리로 긴 밤을 넘어",
    "커지는 심장을 하늘에 띄워",
    "붙잡은 소원이 문을 열어",
    "꺼지지 않는 불꽃 안고 노래해",
]
_KO_VERSE_2 = [
    "창가의 빗방울도 박자를 타고",
    "낮은 울림 속 희망이 차올라",
    "떨리던 마음도 힘으로 바뀌고",
    "마지막 후렴에 꿈을 더 안아",
]