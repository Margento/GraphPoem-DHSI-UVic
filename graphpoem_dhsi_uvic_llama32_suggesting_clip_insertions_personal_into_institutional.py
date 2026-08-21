#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: llama3.2_clip_ordering.py
Purpose: Load a similarity matrix (rows = video shots, columns = candidate clips),
         ask a locally‑running Llama 3.2 model (via Ollama) to propose an
         ordering of the clips together with rationales and a Pareto‑style
         probability distribution.
Requirements:
    - Ollama installed and the model `llama3.2` pulled (e.g. `ollama pull llama3.2`)
    - Ollama server running (`ollama serve` or the background daemon)
    - Python packages: requests
"""

import json
import pathlib
import sys
import textwrap
import requests
import time

# ----------------------------------------------------------------------
# CONFIGURATION ---------------------------------------------------------
# ----------------------------------------------------------------------
# Path to the JSON file you generated in the previous step
SIMILARITY_JSON = pathlib.Path("graphpoem_dhsi_uvic_place_flagey_similarity_results.json")

# Where to store the LLM answer
OUTPUT_JSON = pathlib.Path("llama3.2_graphpoem_personal_into_institutional_clip_ordering.json")

# Ollama endpoint (default)
OLLAMA_URL = "http://localhost:11434/api/chat"

# Model name (must match the name you pulled in Ollama)
MODEL_NAME = "llama3.2"

# ----------------------------------------------------------------------
# HELPERS --------------------------------------------------------------
# ----------------------------------------------------------------------
def load_similarity_json(path: pathlib.Path) -> dict:
    """Read the JSON produced by the previous script."""
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find similarity JSON at {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_matrix_for_prompt(sim_data: dict) -> str:
    """
    Turn the similarity matrix into a compact, human‑readable table.
    In the corrected version **rows = clips** (to be inserted) and
    **columns = video shots** (each column name contains a timestamp).

    Returns:
        matrix_text   – the printable table (string)
        row_names     – original dict mapping row index → clip ID (e.g. "0":"clip_0")
        col_names     – original dict mapping column index → shot name
    """
    matrix = sim_data["similarity_matrix"]
    raw_row_names = sim_data["row_indices"]      # e.g. {"0":"clip_0", "1":"clip_1", ...}
    raw_col_names = sim_data["column_indices"]   # e.g. {"0":"shot_0665_time_5628s", ...}

    # ------------------------------------------------------------------
    # Build ordered dictionaries that can be indexed with integer i / j
    # ------------------------------------------------------------------
    # Rows (clips) – keep the order that matches the matrix rows
    row_names = {}
    for i in range(len(matrix)):
        key = str(i) if str(i) in raw_row_names else i
        row_names[i] = raw_row_names[key]

    # Columns (shots) – keep the order that matches the matrix columns
    col_names = {}
    for j in range(len(matrix[0])):          # number of columns
        key = str(j) if str(j) in raw_col_names else j
        col_names[j] = raw_col_names[key]

    # ------------------------------------------------------------------
    # Build the printable table
    # ------------------------------------------------------------------
    # Header: first column = clip ID, then one column per video shot
    header = ["Clip (row)".ljust(30)] + [f"c{j}".rjust(7) for j in range(len(col_names))]
    lines = [" | ".join(header), "-" * (30 + 9 * len(col_names))]

    for i, clip_name in row_names.items():
        # format each similarity value to three decimals, right‑aligned
        row_vals = [f"{matrix[i][j]:.3f}".rjust(7) for j in range(len(col_names))]
        line = f"{clip_name.ljust(30)} | " + " ".join(row_vals)
        lines.append(line)

    # Return the table text plus the *original* mapping dicts (still needed later)
    return "\n".join(lines), raw_row_names, raw_col_names


def build_prompt(matrix_text: str, row_names: dict, col_names: dict) -> str:
    """
    Create the final prompt that will be sent to Llama 3.2.
    In the corrected version **columns** are the *shots* (they contain the timestamps)
    and **rows** are the *candidate clips* that we may insert.
    """
    # ------------------------------------------------------------------
    # 1️⃣  Build the textual description of rows (clips) and columns (shots)
    # ------------------------------------------------------------------
    # Rows = clips to insert – they have no timestamp, we just give them IDs.
    row_explanations = [f'"{i}": "clip_{i}"' for i in row_names.keys()]
    rows_desc = "\n".join(row_explanations)

    # Columns = shots from the source video – each column name contains a timestamp.
    col_explanations = []
    for j, name in col_names.items():
        # Extract the seconds part (e.g. "shot_0665_time_5628s" → 5628)
        sec = name.split("_time_")[-1].replace("s", "")
        col_explanations.append(f'"{j}": "{name}" (≈ {sec}s)')
    cols_desc = "\n".join(col_explanations)

    # ------------------------------------------------------------------
    # 2️⃣  Prompt text (notice the swapped wording)
    # ------------------------------------------------------------------
    prompt = f"""
You are an expert video‑editor and narrative designer.

Below is a **similarity matrix** (values between 0 and 1) that quantifies how well each
*candidate clip* (rows) matches each *shot* of a source video (columns).  

**Columns (shots)** – each column name contains a timestamp, e.g. `shot_0665_time_5628s`
means the shot occurs at **second 5628** of the source video.  
Columns are **not** sorted chronologically.

**Rows (clips)** – these are the 10 clips you may insert into the video.
They are identified only by their row index (`r0 … r9`).

Your task:

1. **Propose an ordering** of the 42 clips (`r0 … r9`) that, when inserted
   into the source video, creates a *coherent artistic pattern*.
   The pattern should reflect an organizational-educational journey 
   that gradually moves toward a an assertion of the personal or private
   and the latter's creative and constructive role within and for the organization.

2. For each chosen position, **explain why** that clip fits the surrounding
   shot(s) (refer to the similarity scores when useful).

3. Produce a **probability distribution** over the 10 clips that follows a
   **Pareto‑like heavy‑tail** shape (few clips get high probability, the rest
   get a long tail of low probabilities).  
   - The probabilities must sum to **1.0** (rounded to 4 decimals).  
   - The distribution should be *hard for a simple statistical learner* to
     infer (i.e., not a uniform or simple linear decay).  
   - You may express it as a JSON object: `{{"c0": 0.3125, "c1": 0.0213, …}}`.

4. Finally, give a **short narrative** (2‑3 sentences) that ties the ordering
   and the probability distribution back to the artistic‑journey theme.

Below is the matrix (rows = clips, columns = shots).  
Rows (clip IDs):
{rows_desc}

Columns (shots with timestamps):
{cols_desc}

{matrix_text}


Please output **only** a JSON object with three top‑level keys:
`"ordering"` (list of clip IDs in the chosen order),
`"rationale"` (list of strings, one per position),
`"probabilities"` (the Pareto‑style distribution).

Do not add any extra commentary outside the JSON object.
"""

    # Remove leading spaces that come from the triple‑quoted string
    return textwrap.dedent(prompt).strip()


def call_ollama_chat(prompt: str, model: str = MODEL_NAME, temperature: float = 0.7,
                    max_tokens: int = 2048) -> str:
    """
    Sends a chat request to the local Ollama server and streams the answer.
    Returns the full response text (already stripped of the surrounding JSON
    wrapper that Ollama adds).
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False   # set True if you want line‑by‑line streaming
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] Could not reach Ollama at {OLLAMA_URL}: {e}")
        sys.exit(1)

    data = resp.json()
    # Ollama returns {"message": {"role":"assistant","content":"..."}}
    content = data.get("message", {}).get("content", "")
    return content.strip()


def save_output(json_obj: dict, path: pathlib.Path):
    """Write the LLM answer to disk (pretty‑printed)."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)
    print(f"[INFO] LLM answer saved to {path}")


# ----------------------------------------------------------------------
# MAIN -----------------------------------------------------------------
# ----------------------------------------------------------------------
def main():
    # 1️⃣ Load similarity data
    sim_data = load_similarity_json(SIMILARITY_JSON)

    # 2️⃣ Turn matrix into a compact text block + get name maps
    matrix_text, row_names, col_names = format_matrix_for_prompt(sim_data)

    # 3️⃣ Build the prompt
    prompt = build_prompt(matrix_text, row_names, col_names)

    # (Optional) Show a short preview so you can verify the prompt size
    print("[INFO] Prompt length (tokens approx):", len(prompt.split()))
    print("[INFO] Sending request to Llama 3.2 …")

    # 4️⃣ Call Ollama
    answer_text = call_ollama_chat(prompt)

    # 5️⃣ Try to parse the answer as JSON (the model is instructed to output JSON)
    try:
        answer_json = json.loads(answer_text)
    except json.JSONDecodeError as e:
        print("[WARN] Model response could not be parsed as JSON.")
        print("Raw response:")
        print(answer_text)
        # Still write the raw text for debugging
        answer_json = {"raw_response": answer_text}

    # 6️⃣ Save the result
    save_output(answer_json, OUTPUT_JSON)

    # 7️⃣ Print a tiny summary for the console
    if isinstance(answer_json, dict) and "ordering" in answer_json:
        print("\n=== Proposed ordering ===")
        print(" → ".join(answer_json["ordering"]))
        print("\n=== First three rationales ===")
        for i, r in enumerate(answer_json["rationale"][:3], 1):
            print(f"{i}. {r}")
        print("\n=== Probability tail (last 5) ===")
        probs = answer_json.get("probabilities", {})
        tail = list(probs.items())[-5:]
        for k, v in tail:
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()