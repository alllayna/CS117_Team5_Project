# prepare_widerperson.py
"""This script turns the WiderPerson real-person dataset into the same label format our game screenshots use.

It saves each photo as an image file and writes its person boxes as YOLO labels, so the baseline and the games can be
scored by the same evaluation script. Pedestrians, riders, and partly hidden people count as people; crowd boxes are
left out because they cover groups instead of single people.
"""

import argparse
from pathlib import Path

import pyarrow.parquet as pq

PERSON_CATEGORIES = {0, 1, 2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    rows = pq.read_table(args.parquet).to_pylist()
    n_boxes = 0
    for row in rows:
        name = Path(row["image"]["path"]).name
        (out / "images" / name).write_bytes(row["image"]["bytes"])
        w, h = row["width"], row["height"]
        lines = []
        for (x, y, bw, bh), cat in zip(row["objects"]["bbox"], row["objects"]["category"]):
            if cat in PERSON_CATEGORIES and bw > 0 and bh > 0:
                lines.append(f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
        (out / "labels" / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_boxes += len(lines)
    print(f"Wrote {len(rows)} images and {n_boxes} person boxes to {out}")


if __name__ == "__main__":
    main()
