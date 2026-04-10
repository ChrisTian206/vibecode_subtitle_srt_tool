from __future__ import annotations


def _format_timestamp(total_seconds: float) -> str:
    if total_seconds < 0:
        total_seconds = 0.0

    milliseconds = int(round((total_seconds % 1) * 1000))
    total_int = int(total_seconds)
    seconds = total_int % 60
    minutes = (total_int // 60) % 60
    hours = total_int // 3600
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        start = _format_timestamp(float(segment["start"]))
        end = _format_timestamp(float(segment["end"]))
        text = segment["text"].strip()

        lines.append(str(index))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines).strip() + "\n"
