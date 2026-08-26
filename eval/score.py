"""Офлайн-скоринг eval-набора против работающего сервиса транскрибации.

Прогоняет фразы из phrases.yaml через POST /v1/audio/transcriptions,
считает WER/CER относительно эталонов, печатает таблицу и сохраняет
артефакты прогона (results.json, summary.md).

Пример:
    python eval/score.py --url http://spark.local:8337 \
        --token $TRANSCRIBE_TOKEN --label baseline
    python eval/score.py ... --prompt-file glossary.txt --label prompt

Аудиозаписи ищутся в eval/audio/<id>.(wav|mp3|m4a|flac|ogg|pcm);
отсутствующие пропускаются с предупреждением.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import re
import sys

import httpx
import yaml

AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".oga", ".opus", ".pcm")
_MIME = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
    ".flac": "audio/flac", ".ogg": "audio/ogg", ".oga": "audio/ogg",
    ".opus": "audio/ogg", ".pcm": "application/octet-stream",
}


def _normalize(text: str) -> list[str]:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return text.split()


def _levenshtein(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = _normalize(reference), _normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    ref = list("".join(_normalize(reference)))
    hyp = list("".join(_normalize(hypothesis)))
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def find_audio(audio_dir: pathlib.Path, phrase_id: str) -> pathlib.Path | None:
    for ext in AUDIO_EXTS:
        candidate = audio_dir / f"{phrase_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def transcribe(client: httpx.Client, url: str, token: str,
               audio_path: pathlib.Path, language: str,
               prompt: str) -> str:
    headers = {"Authorization": f"Bearer {token}"}
    data = {"language": language}
    if prompt:
        data["prompt"] = prompt
    with audio_path.open("rb") as fh:
        files = {"file": (audio_path.name, fh, _MIME.get(audio_path.suffix, "audio/wav"))}
        r = client.post(f"{url}/v1/audio/transcriptions", headers=headers,
                        files=files, data=data, timeout=600)
    r.raise_for_status()
    return r.json()["text"]


def run(args: argparse.Namespace) -> int:
    root = pathlib.Path(__file__).resolve().parent
    phrases = yaml.safe_load((root / "phrases.yaml").read_text(encoding="utf-8"))["phrases"]
    audio_dir = pathlib.Path(args.audio_dir)
    token = args.token or os.environ.get("TRANSCRIBE_TOKEN", "")
    if not token:
        print("ошибка: укажите --token или переменную TRANSCRIBE_TOKEN", file=sys.stderr)
        return 2
    prompt = pathlib.Path(args.prompt_file).read_text(encoding="utf-8").strip() \
        if args.prompt_file else args.prompt

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "runs" / f"{stamp}_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, skipped = [], []
    with httpx.Client() as client:
        health = client.get(f"{args.url}/health").json()
        print(f"сервис: {health['model']} [{health['status']}] "
              f"device={health['backend']['device']} label={args.label}\n")
        for p in phrases:
            audio = find_audio(audio_dir, p["id"])
            if audio is None:
                skipped.append(p["id"])
                continue
            hyp = transcribe(client, args.url, token, audio, args.language, prompt)
            rows.append({
                "id": p["id"], "class": p["class"],
                "ref": p["text"], "hyp": hyp,
                "wer": round(wer(p["text"], hyp), 4),
                "cer": round(cer(p["text"], hyp), 4),
            })

    def avg(key: str, cls: str | None = None) -> float:
        vals = [r[key] for r in rows if cls is None or r["class"] == cls]
        return sum(vals) / len(vals) if vals else float("nan")

    header = f"{'id':8} {'class':7} {'WER':>7} {'CER':>7}  hyp"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(f"{r['id']:8} {r['class']:7} {r['wer']:7.2%} {r['cer']:7.2%}  {r['hyp']}")
    lines.append("-" * len(header))
    for cls in (None, "ru", "mixed", "ids"):
        name = cls or "TOTAL"
        lines.append(f"{name:16} WER={avg('wer', cls):.2%} CER={avg('cer', cls):.2%} "
                     f"(n={sum(1 for r in rows if cls is None or r['class'] == cls)})")
    table = "\n".join(lines)
    print(table)
    if skipped:
        print(f"\nпропущено (нет аудио): {', '.join(skipped)}")

    safe_args = {**vars(args), "token": "***" if args.token else ""}
    (out_dir / "results.json").write_text(
        json.dumps({"health": health, "args": safe_args,
                    "prompt": prompt, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (out_dir / "summary.md").write_text(
        f"# Eval run: {args.label}\n\n```\n{table}\n```\n", encoding="utf-8")
    print(f"\nартефакты: {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8337")
    ap.add_argument("--token", default="")
    ap.add_argument("--audio-dir", default=str(pathlib.Path(__file__).parent / "audio"))
    ap.add_argument("--language", default="ru")
    ap.add_argument("--prompt", default="", help="initial_prompt поверх конфига сервиса")
    ap.add_argument("--prompt-file", default="", help="файл с глоссарием для initial_prompt")
    ap.add_argument("--label", default="run", help="метка прогона для артефактов")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
