# agent.py
import os
import csv
import json
import asyncio
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from browser_use import Agent, ChatGoogle

# -----------------------------
# Config 
# -----------------------------
INPUT_CSV = os.getenv("INPUT_CSV", "input.csv")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "output.csv")
MAX_STEPS = int(os.getenv("MAX_STEPS", "40"))

# load .env so GOOGLE_API_KEY is available
load_dotenv(override=True)


# -----------------------------
# Structured output schema (what we want back from the agent)
# -----------------------------
class CompanyInfo(BaseModel):
    club_name: Optional[str] = Field(None, description="Club name as found/verified")
    post_code: Optional[str] = Field(None, description="Postcode used for lookup")
    number_1: Optional[str] = Field(None, description="Primary public phone number in UK format")
    number_2: Optional[str] = Field(None, description="Secondary phone number if found")
    number_3: Optional[str] = Field(None, description="Tertiary phone number if found")
    source_url: Optional[str] = Field(None, description="Authoritative URL used to verify the result")
    confidence: Optional[float] = Field(None, description="0.0–1.0 confidence")
    notes: Optional[str] = Field(None, description="Short reasoning or disambiguation notes")


# -----------------------------
# Row container for writing out (unused but kept for reference)
# -----------------------------
@dataclass
class CompanyResult:
    club_name: str
    post_code: str
    number_1: Optional[str] = None
    number_2: Optional[str] = None
    number_3: Optional[str] = None
    source_url: Optional[str] = None
    confidence: Optional[float] = None
    notes: Optional[str] = None


# -----------------------------
# Build the per-row task
# -----------------------------
def build_task(club_name: str, postcode: str) -> str:
    # Use only the club_name and postcode variables passed in
    return f"""
You are a precise UK business lookup agent.

Goal:
- Using ONLY the club name and postcode provided, identify the business located there and find any public phone numbers for the business.

Input:
- club name: {club_name}
- postcode: {postcode}

Rules:
- Prefer authoritative sources: the official business website, Google Business Profile, Companies House, Yell/118, etc.
- Return phone numbers in UK national format (e.g., 020..., 0161..., 07...).
- If you find more than one phone number put them in number_1, number_2, number_3 (leave empty if not found).
- If multiple businesses share the same address/postcode, choose the best match and explain briefly in 'notes'.
- Do NOT hallucinate. If unsure, leave fields empty and explain why in 'notes'.
- Provide a verification 'source_url'. If multiple sources are found, pick one authoritative URL (official website or Companies House or Google Business Profile).
- If fields in the row are already populated, only provide values for missing fields.

Output:
Return ONLY a JSON object with these exact fields (use null for missing):
{{
  "club_name": string | null,
  "post_code": string | null,
  "number_1": string | null,
  "number_2": string | null,
  "number_3": string | null,
  "source_url": string | null,
  "confidence": number | null,
  "notes": string | null
}}
Be concise in 'notes' (one or two short sentences).
"""


# -----------------------------
# CSV helpers (more robust)
# -----------------------------
def _normalize_header_cell(cell: str) -> str:
    """
    Normalize a header cell to a predictable token:
      - strip whitespace
      - collapse internal whitespace to single space
      - lowercase
      - replace non-alphanumeric chars with underscore
      - collapse multiple underscores
    Examples:
      " post_code " -> "post_code"
      " Phone Number\t" -> "phone_number"
    """
    if cell is None:
        return ""
    s = str(cell).strip()
    s = re.sub(r"\s+", " ", s)            # collapse whitespace
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)    # replace non-alnum with underscore
    s = re.sub(r"_+", "_", s)            # collapse multiple underscores
    s = s.strip("_")
    return s


def sniff_dialect_and_header(path: str) -> Tuple[csv.Dialect, bool]:
    """
    Try to detect CSV dialect robustly. First use csv.Sniffer,
    then fallback to trying comma vs tab heuristics.
    Returns (dialect, has_header)
    """
    with open(path, "rb") as f:
        sample = f.read(8192)

    sniffer = csv.Sniffer()
    # attempt sniffer first
    try:
        decoded = sample.decode("utf-8", errors="ignore")
        dialect = sniffer.sniff(decoded)
        has_header = sniffer.has_header(decoded)
        return dialect, has_header
    except Exception:
        pass

    # fallback heuristic: count commas vs tabs on the first line
    try:
        text = sample.decode("utf-8", errors="ignore")
        first_line = text.splitlines()[0] if text.splitlines() else ""
        comma_count = first_line.count(",")
        tab_count = first_line.count("\t")
        class _D(csv.Dialect):
            delimiter = "," if comma_count >= tab_count else "\t"
            quotechar = '"'
            doublequote = True
            skipinitialspace = True
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL
        dialect = _D()
        # best-effort header detection: assume header if first line contains non-numeric tokens
        has_header = bool(re.search(r"[A-Za-z]", first_line))
        return dialect, has_header
    except Exception:
        # absolute fallback
        class _D(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            doublequote = True
            skipinitialspace = True
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL
        return _D(), True


def read_rows(path: str) -> List[List[str]]:
    dialect, _ = sniff_dialect_and_header(path)
    # Use csv.reader with the detected dialect. Strip BOM via utf-8-sig.
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, dialect)
        rows = [row for row in reader if row and any(c.strip() for c in row)]
    return rows


def get_col_indices(header: List[str]) -> Tuple[int, int]:
    """
    Return indices for club_name and post_code.
    Accepts a wide range of common variants by normalizing header cells.
    Raises ValueError if they cannot be found.
    """
    normalized = [_normalize_header_cell(h) for h in header]
    # helpful debug print to surface what we detected
    print("DEBUG: normalized header ->", normalized)

    def find_one(variants: List[str]) -> Optional[int]:
        for v in variants:
            if v in normalized:
                return normalized.index(v)
        return None

    club_variants = [
        "club_name", "clubname", "club", "name", "organisation", "organisation_name", "organisationname", "business"
    ]
    pc_variants = [
        "post_code", "postcode", "postal_code", "postalcode", "zip", "zip_code", "postal", "postalcode"
    ]

    club_idx = find_one(club_variants)
    pc_idx = find_one(pc_variants)

    if club_idx is None or pc_idx is None:
        # give a helpful error showing available candidates
        raise ValueError(
            "Header row must include club_name and post_code (or a common variant). "
            f"Found columns: {normalized}"
        )
    return club_idx, pc_idx


# -----------------------------
# Per-row agent execution
# -----------------------------
async def run_for_row(llm: ChatGoogle, club_name: str, postcode: str) -> CompanyInfo | None:
    agent = Agent(
        task=build_task(club_name, postcode),
        llm=llm,
        output_model_schema=CompanyInfo,
        max_failures=2,
        step_timeout=120,
        max_actions_per_step=8,
    )
    history = await agent.run(max_steps=MAX_STEPS)

    # Best case: structured_output parsed into our schema
    data = getattr(history, "structured_output", None)
    if data is not None:
        # pydantic model-like object
        if hasattr(data, "model_dump"):
            try:
                return CompanyInfo(**data.model_dump())
            except Exception:
                pass
        if isinstance(data, dict):
            try:
                return CompanyInfo(**data)
            except Exception:
                pass

    # Fallback: try JSON from final_result()
    final = None
    if hasattr(history, "final_result"):
        try:
            final = history.final_result()
        except Exception:
            final = None

    if isinstance(final, str) and final.strip():
        try:
            obj = json.loads(final)
            return CompanyInfo(**obj)
        except Exception:
            return CompanyInfo(notes=final.strip())

    return None


# -----------------------------
# Main (sequential processing)
# -----------------------------
async def main():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not found in environment. Put it in your .env")

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    # read all rows
    rows = read_rows(INPUT_CSV)
    if not rows:
        print("No data rows found.")
        return

    # header
    dialect, has_header = sniff_dialect_and_header(INPUT_CSV)
    print("DEBUG: detected delimiter:", repr(dialect.delimiter), "has_header:", has_header)

    if not has_header:
        raise ValueError("Your file appears to have no header row. Please add headers like: club_name,post_code,number_1,number_2,number_3,source_url")

    header = rows[0]
    data_rows = rows[1:]

    # show raw header for debugging
    print("DEBUG: raw header row ->", header)

    club_idx, pc_idx = get_col_indices(header)

    # Build output header (preserve original order; ensure these exist)
    output_header = header[:]

    def ensure_col(name: str) -> int:
        if name in output_header:
            return output_header.index(name)
        output_header.append(name)
        return len(output_header) - 1

    # Ensure columns that the user specified exist (match exact names requested)
    out_club_idx = ensure_col("club_name")
    out_pc_idx = ensure_col("post_code")
    out_num1_idx = ensure_col("number_1")
    out_num2_idx = ensure_col("number_2")
    out_num3_idx = ensure_col("number_3")
    out_src_idx = ensure_col("source_url")
    out_conf_idx = ensure_col("confidence")
    out_notes_idx = ensure_col("notes")

    llm = ChatGoogle(model="gemini-flash-latest")

    print(f"🔎 Processing {len(data_rows)} row(s) from {INPUT_CSV} ...")

    out_rows: List[List[str]] = []

    for i, row in enumerate(data_rows, start=1):
        # normalize row length to output_header length
        row = row + [""] * (len(output_header) - len(row))

        club_name = row[club_idx].strip() if len(row) > club_idx else ""
        postcode = row[pc_idx].strip() if len(row) > pc_idx else ""

        # if both empty, nothing to do
        if not club_name and not postcode:
            out_rows.append(row)
            continue

        # If target output fields already exist and are filled, skip lookup
        already_has_number = False
        try:
            already_has_number = bool(row[out_num1_idx].strip())
        except Exception:
            already_has_number = False

        try:
            # Only call the agent if we need to fill missing fields
            if already_has_number and row[out_src_idx].strip():
                print(f"[{i}/{len(data_rows)}] → Skipping (already has number/source): {club_name} {postcode}")
            else:
                info = await run_for_row(llm, club_name, postcode)
                if info:
                    # Only overwrite empty fields (preserve existing data)
                    row[out_club_idx] = info.club_name or row[out_club_idx] or club_name or ""
                    row[out_pc_idx] = info.post_code or row[out_pc_idx] or postcode or ""
                    row[out_num1_idx] = info.number_1 or row[out_num1_idx] or ""
                    row[out_num2_idx] = info.number_2 or row[out_num2_idx] or ""
                    row[out_num3_idx] = info.number_3 or row[out_num3_idx] or ""
                    row[out_src_idx] = info.source_url or row[out_src_idx] or ""
                    row[out_conf_idx] = "" if info.confidence is None else f"{info.confidence:.2f}"
                    row[out_notes_idx] = info.notes or row[out_notes_idx] or ""
                    print(f"[{i}/{len(data_rows)}] ✓ {club_name} {postcode} -> {row[out_num1_idx] or '—'}")
                else:
                    row[out_notes_idx] = row[out_notes_idx] or "No structured result returned."
                    print(f"[{i}/{len(data_rows)}] ? {club_name} {postcode} -> no result")
        except Exception as e:
            row[out_notes_idx] = f"Agent error: {e}"
            print(f"[{i}/{len(data_rows)}] ✗ {club_name} {postcode} -> Error: {e}")

        out_rows.append(row)

    # write output
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, dialect)
        writer.writerow(output_header)
        writer.writerows(out_rows)

    print(f"✅ Wrote {len(out_rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
