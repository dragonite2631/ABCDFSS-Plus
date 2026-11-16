import csv
import glob
import json
import os
from typing import Dict, List, Tuple

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARTS_DIR = os.path.join(ROOT_DIR, "results", "parts")
OUT_JSON = os.path.join(ROOT_DIR, "results", "table4_results.json")
OUT_CSV = os.path.join(ROOT_DIR, "results", "table4_results.csv")

BENCHMARKS = ["deepglobe", "isic", "lung", "fss", "suim"]
CDFSS_BENCHMARKS = ["deepglobe", "isic", "lung", "fss"]
SHOTS = [1, 5]


def load_part_rows(parts_dir: str = PARTS_DIR) -> Dict[Tuple[str, int], dict]:
    merged: Dict[Tuple[str, int], dict] = {}
    for path in sorted(glob.glob(os.path.join(parts_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for row in data.get("results", []):
            key = (row["benchmark"], int(row["shot"]))
            merged[key] = row
    return merged


def format_table4_html_row(merged: Dict[Tuple[str, int], dict]) -> str:
    cells: List[str] = ["<td><b>ABCDFSS-Plus (every-episode, no-pp)</b></td>"]
    for b in CDFSS_BENCHMARKS:
        for s in SHOTS:
            r = merged.get((b, s))
            if r:
                cells.append(f'<td align="center">{r["mIoU"]:.1f}</td>')
                cells.append(f'<td align="center">{r["FB-IoU"]:.1f}</td>')
            else:
                cells.extend(['<td align="center">-</td>', '<td align="center">-</td>'])

    # CD-FSS Avg. (4 benchmarks)
    for s in SHOTS:
        vals = [merged[(b, s)] for b in CDFSS_BENCHMARKS if (b, s) in merged]
        if len(vals) == len(CDFSS_BENCHMARKS):
            avg_m = sum(v["mIoU"] for v in vals) / len(vals)
            avg_fb = sum(v["FB-IoU"] for v in vals) / len(vals)
            cells.append(f'<td align="center"><b>{avg_m:.1f}</b></td>')
            cells.append(f'<td align="center"><b>{avg_fb:.1f}</b></td>')
        else:
            cells.extend(['<td align="center">-</td>', '<td align="center">-</td>'])

    # SUIM
    for s in SHOTS:
        r = merged.get(("suim", s))
        if r:
            cells.append(f'<td align="center">{r["mIoU"]:.1f}</td>')
            cells.append(f'<td align="center">{r["FB-IoU"]:.1f}</td>')
        else:
            cells.extend(['<td align="center">-</td>', '<td align="center">-</td>'])

    return "    <tr>" + "".join(cells) + "</tr>"


def main():
    merged = load_part_rows()
    expected = [(b, s) for b in BENCHMARKS for s in SHOTS]
    missing = [k for k in expected if k not in merged]

    ordered_rows = [merged[k] for k in expected if k in merged]
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "ABCDFSS-Plus",
                "table": "Table 4 (every-episode, no-pp)",
                "missing_units": [f"{b}:{s}" for b, s in missing],
                "results": ordered_rows,
            },
            f,
            indent=2,
        )

    fieldnames = [
        "benchmark",
        "shot",
        "mIoU",
        "FB-IoU",
        "episode_count",
        "available_episodes",
        "elapsed_sec",
        "seed",
        "inference_mode",
        "num_epochs",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f"Merged {len(ordered_rows)}/10 units -> {OUT_CSV}")
    if missing:
        print("Missing units:", [f"{b}:{s}" for b, s in missing])
    print("\nTable 4 HTML row (paste into REPORT.md):")
    print(format_table4_html_row(merged))


if __name__ == "__main__":
    main()
