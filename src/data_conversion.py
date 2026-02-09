"""Process the data files and convert all the .data files into the .csv format,
 ensuring that all the data formats are consistent"""
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT = BASE_DIR / "Data"

def convert_data_file(data_path: Path):
    print(f"Processing: {data_path}")

    encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]

    for enc in encodings:
        try:
            df = pd.read_csv(
                data_path,
                header=None,
                sep=",",
                engine="python",
                na_values=["?", "NaN"],
                encoding=enc
            )
            print(f"  ✓ Read with encoding={enc}")
            break

        except Exception:
            try:
                df = pd.read_csv(
                    data_path,
                    header=None,
                    sep=r"\s+",
                    engine="python",
                    na_values=["?", "NaN"],
                    encoding=enc
                )
                print(f"  ✓ Read with encoding={enc} (whitespace)")
                break
            except Exception:
                df = None

    if df is None:
        print(f"Failed to read: {data_path}")
        return

    out_path = data_path.with_suffix(".csv")
    df.to_csv(out_path, index=False)
    print(f"  ✓ Saved -> {out_path} | shape={df.shape}")


def main():
    data_files = list(ROOT.rglob("*.data"))

    if not data_files:
        print("No .data files found.")
        return

    print(f"Found {len(data_files)} .data files\n")

    for data_file in data_files:
        csv_path = data_file.with_suffix(".csv")
        if csv_path.exists():
            print(f"Skip (csv exists): {csv_path}")
            continue
        convert_data_file(data_file)

    print("\n All .data files converted.")

if __name__ == "__main__":
    main()
