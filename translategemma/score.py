#!/usr/bin/env python3
"""Score translations the way scores.csv was scored: slip, chrF and COMET per sentence.

    pip install sacrebleu unbabel-comet
    python3 score.py --language am translations.jsonl        lines {"id", "source", "reference", "output"}

prints one CSV row per sentence: id,slip,chrf,comet

slip = 1 when the translation holds a letter from a writing system that is neither the target language's nor Latin
(Latin is allowed because names and acronyms are copied legitimately). chrF: sacrebleu's CHRF() with its defaults,
per sentence, 0 to 100. COMET: Unbabel/wmt22-comet-da on (source, translation, reference), times 100.
"""
import argparse, json, unicodedata

RANGES = {
    "ethiopic": [(0x1200, 0x137F), (0x1380, 0x139F), (0x2D80, 0x2DDF), (0xAB00, 0xAB2F)],
    "myanmar": [(0x1000, 0x109F), (0xAA60, 0xAA7F), (0xA9E0, 0xA9FF)],
    "khmer": [(0x1780, 0x17FF), (0x19E0, 0x19FF)],
    "devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "tamil": [(0x0B80, 0x0BFF)],
    "bengali": [(0x0980, 0x09FF)],
    "arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "cjk": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x3000, 0x303F), (0xFF00, 0xFFEF)],
    "sinhala": [(0x0D80, 0x0DFF)],
    "georgian": [(0x10A0, 0x10FF), (0x2D00, 0x2D2F), (0x1C90, 0x1CBF)],
    "lao": [(0x0E80, 0x0EFF)],
    "latin": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF)],
}
TARGET = {"am": "ethiopic", "ti": "ethiopic", "my": "myanmar", "km": "khmer", "hi": "devanagari", "zh": "cjk",
          "si": "sinhala", "ka": "georgian", "lo": "lao", "es": "latin", "sw": "latin"}


def script_of(ch):
    o = ord(ch)
    for name, rs in RANGES.items():
        if any(a <= o <= b for a, b in rs):
            return name
    return "other"


def slip(output, language):
    target = TARGET[language]
    return int(any(script_of(c) not in (target, "latin")
                   for c in output if unicodedata.category(c).startswith("L")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, choices=sorted(TARGET))
    ap.add_argument("translations")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.translations)]
    from sacrebleu.metrics import CHRF
    from comet import download_model, load_from_checkpoint
    chrf = CHRF()
    model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    comet = model.predict([{"src": r["source"].strip(), "mt": r["output"].strip(), "ref": r["reference"].strip()}
                           for r in rows], batch_size=32, gpus=1, progress_bar=False).scores
    print("id,slip,chrf,comet")
    for r, c in zip(rows, comet):
        print(f'{r["id"]},{slip(r["output"], a.language)},{chrf.sentence_score(r["output"], [r["reference"]]).score:.2f},'
              f'{100 * c:.2f}')


if __name__ == "__main__":
    main()
