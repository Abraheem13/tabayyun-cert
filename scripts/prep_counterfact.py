"""Build the compact Tier-D fact pool from CounterFact (Meng et al., ROME, NeurIPS 2022; MIT licence).

Source: Hugging Face dataset `azhx/counterfact` (the original CounterFact records).
Output: data/counterfact/counterfact_4k.json, the first 4,000 records with
  prompt (subject filled in), subject, relation_id, target_true, target_new, paraphrase.
Needs: pip install huggingface_hub pandas pyarrow.   Usage: python scripts/prep_counterfact.py
"""
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
p = hf_hub_download("azhx/counterfact", "data/train-00000-of-00001-05d11247db7abce8.parquet", repo_type="dataset")
df = pd.read_parquet(p)
out = []
for _, r in df.iterrows():
    rw = r["requested_rewrite"]
    if not rw["prompt"].rstrip().endswith("{}") and "{}" not in rw["prompt"]:
        continue
    out.append({"case_id": int(r["case_id"]), "prompt": rw["prompt"].format(rw["subject"]),
                "subject": rw["subject"], "relation_id": rw["relation_id"],
                "target_true": rw["target_true"]["str"], "target_new": rw["target_new"]["str"],
                "paraphrase": [str(x) for x in r["paraphrase_prompts"][:2]]})
    if len(out) == 4000:
        break
dst = ROOT / "data" / "counterfact"; dst.mkdir(parents=True, exist_ok=True)
(dst / "counterfact_4k.json").write_text(json.dumps(out, ensure_ascii=False))
print("saved", len(out), "records;", len({o["relation_id"] for o in out}), "relations")
