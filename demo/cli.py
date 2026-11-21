"""Command-line demo.

  python demo/cli.py --models models/ --title "Xerox TONER CARTRIDGE ..." --description "25K PAGE YIELD ..."
  python demo/cli.py --models models/            # interactive prompt
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.predictor import Extractor


def show(extractor, title, description):
    rows = extractor.extract(title, description)
    for system in dict.fromkeys(r["system"] for r in rows):
        print(f"\n== {system} ==")
        for attr, value, source in extractor.as_table(rows, system):
            print(f"  {attr:<28} {value:<40} [{source}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="models")
    ap.add_argument("--title"); ap.add_argument("--description", default="")
    ap.add_argument("--freq", help="json with training span counts (optional)")
    a = ap.parse_args()
    ex = Extractor(a.models, json.loads(Path(a.freq).read_text()) if a.freq else None)
    if a.title:
        show(ex, a.title, a.description); return
    while True:
        title = input("\nProduct title (empty to quit): ").strip()
        if not title:
            break
        show(ex, title, input("Description: ").strip())


if __name__ == "__main__":
    main()
