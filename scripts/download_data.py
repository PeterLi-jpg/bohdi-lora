"""Download HealthBench datasets to data/raw/."""

import urllib.request
from pathlib import Path

URLS = {
    "healthbench_hard": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl",
    "healthbench": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl",
    "healthbench_consensus": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl",
}

def main():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, url in URLS.items():
        dest = out_dir / f"{name}.jsonl"
        if dest.exists():
            print(f"  {name} already exists, skipping")
            continue
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  -> {dest}")

    print("done")


if __name__ == "__main__":
    main()
