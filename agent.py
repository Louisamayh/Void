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
    company_name: Optional[str] = Field(None, description="Company/club name as found/verified")
    post_code: Optional[str] = Field(None, description="Postcode used for lookup")
    website: Optional[str] = Field(None, description="Official company website")
    email: Optional[str] = Field(None, description="Primary/general public email (e.g., info@, enquiries@)")
    numbers: Optional[List[str]] = Field(None, description="List of phone numbers discovered (most general first)")
    source_url: Optional[str] = Field(None, description="Authoritative URL used to verify the result")
    confidence: Optional[float] = Field(None, description="0.0–1.0 confidence")
    notes: Optional[str] = Field(None, description="Short reasoning or disambiguation notes")


# -----------------------------
# Row container (for reference)
# -----------------------------
@dataclass
class CompanyResult:
    company_name: str
    post_code: str
    website: Optional[str] = None
    email: Optional[str] = None
    numbers: Optional[List[str]] = None
    source_url: Optional[str] = None
    confidence: Optional[float] = None
    notes: Optional[str] = None


# -----------------------------
# Build the per-row task
# -----------------------------
def build_task(company_name: str, postcode: str, uri: str | None) -> str:
    uri_text = uri or ""
    return f"""
You are a precise UK business lookup agent.

Goal:
- Using ONLY the company name and postcode provided, identify the business located there and find:
  1) official website,
  2) a public contact email (prefer general inbox like info@ / enquiries@),
  3) one or more phone numbers.
- If you cannot confidently find the info using name+postcode alone, you MAY use the provided URI (same row) as a fallback hint/source.

Input:
- company name: {company_name}
- postcode: {postcode}
- fallback URI (optional): {uri_text}

Rules:
- Prefer authoritative sources: the official company website/contact page, Google Business Profile, Companies House filings, league/association directories.
- Return email in lowercase and ensure it matches 'local@domain.tld'.
- If multiple emails are visible, choose the most general inbox; mention alternatives briefly in 'notes'.
- Return phone numbers in an E.164-like or clearly dialable format when possible (strip spaces/extraneous text).
- If multiple phone numbers are visible, include several (most general first) in the 'numbers' list.
- Do NOT hallucinate. If unsure, leave fields empty and explain why in 'notes'.
- Provide a verification 'source_url'. If multiple sources are found, pick one authoritative URL (official website preferred).
- Only provide values for fields that are missing in the row when this script merges results.

Output:
Return ONLY a JSON object with these exact fields (use null for missing):
{{
  "company_name": string | null,
  "post_code": string | null,
  "website": string | null,
  "email": string | null,
  "numbers": string[] | null,
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
    if cell is None:
        return ""
    s = str(cell).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s


def sniff_dialect_and_header(path: str) -> Tuple[csv.Dialect, bool]:
    with open(path, "rb") as f:
        sample = f.read(8192)

    sniffer = csv.Sniffer()
    try:
        decoded = sample.decode("utf-8", errors="ignore")
        dialect = sniffer.sniff(decoded)
        has_header = sniffer.has_header(decoded)
        return dialect, has_header
    except Exception:
        pass

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
        has_header = bool(re.search(r"[A-Za-z]", first_line))
        return dialect, has_header
    except Exception:
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
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, dialect)
        rows = [row for row in reader if row and any(c.strip() for c in row)]
    return rows


def get_col_indices(header: List[str]) -> Tuple[int, int, Optional[int], dict]:
    """
    Returns indices for companyname, postcode, URI (optional),
    and a dict of optional indices for website/email/number.
    """
    normalized = [_normalize_header_cell(h) for h in header]
    print("DEBUG: normalized header ->", normalized)

    def find_one(variants: List[str]) -> Optional[int]:
        for v in variants:
            if v in normalized:
                return normalized.index(v)
        return None

    company_variants = [
        "companyname", "company_name", "organisation", "organisation_name", "club_name", "name", "business"
    ]
    pc_variants = [
        "postcode", "post_code", "postal_code", "postalcode", "zip", "zip_code", "postal"
    ]
    uri_variants = ["uri", "url", "link", "source"]

    website_variants = ["website", "site", "homepage"]
    email_variants = ["email", "e_mail", "mail"]
    number_variants = ["number", "phone", "telephone", "tel", "mobile", "phone_number"]

    company_idx = find_one(company_variants)
    pc_idx = find_one(pc_variants)
    uri_idx = find_one(uri_variants)

    if company_idx is None or pc_idx is None:
        raise ValueError(
            "Header row must include companyname and postcode (or a common variant). "
            f"Found columns: {normalized}"
        )

    optional = {
        "website": find_one(website_variants),
        "email": find_one(email_variants),
        "number": find_one(number_variants),
    }

    return company_idx, pc_idx, uri_idx, optional


# -----------------------------
# Per-row agent execution
# -----------------------------
async def run_for_row(llm: ChatGoogle, company_name: str, postcode: str, uri: Optional[str]) -> CompanyInfo | None:
    agent = Agent(
        task=build_task(company_name, postcode, uri),
        llm=llm,
        output_model_schema=CompanyInfo,
        max_failures=2,
        step_timeout=120,
        max_actions_per_step=8,
    )
    history = await agent.run(max_steps=MAX_STEPS)

    data = getattr(history, "structured_output", None)
    if data is not None:
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
# Utilities
# -----------------------------
_phone_clean_re = re.compile(r"[^\d+]+")

def _clean_phone(p: str) -> str:
    p = p.strip()
    # remove extraneous text and spaces, keep digits and leading +
    p = _phone_clean_re.sub("", p)
    # avoid empty/too-short fragments
    return p

def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if not x:
            continue
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# -----------------------------
# Main — process rows, then write with dynamic number columns
# -----------------------------
async def main():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not found in environment. Put it in your .env")

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    rows = read_rows(INPUT_CSV)
    if not rows:
        print("No data rows found.")
        return

    dialect, has_header = sniff_dialect_and_header(INPUT_CSV)
    print("DEBUG: detected delimiter:", repr(dialect.delimiter), "has_header:", has_header)

    if not has_header:
        raise ValueError("Your file appears to have no header row. Please add headers like: companyname,postcode,URI,website,email,number")

    header = rows[0]
    data_rows = rows[1:]

    print("DEBUG: raw header row ->", header)

    company_idx, pc_idx, uri_idx, opt = get_col_indices(header)

    # Build base output header (preserve original order; ensure these exist)
    output_header = header[:]

    def ensure_col(name: str) -> int:
        if name in output_header:
            return output_header.index(name)
        output_header.append(name)
        return len(output_header) - 1

    # Ensure columns we need exist (exact names requested)
    out_company_idx = ensure_col("companyname")
    out_pc_idx = ensure_col("postcode")
    out_uri_idx = ensure_col("URI")
    out_website_idx = ensure_col("website")
    out_email_idx = ensure_col("email")
    out_number_idx = ensure_col("number")

    # Also keep useful metadata
    out_src_idx = ensure_col("source_url")
    out_conf_idx = ensure_col("confidence")
    out_notes_idx = ensure_col("notes")

    # We'll collect processed rows in memory to compute the max phone count
    processed_rows: List[List[str]] = []
    phone_lists: List[List[str]] = []

    llm = ChatGoogle(model="gemini-flash-latest")

    print(f"🔎 Processing {len(data_rows)} row(s) from {INPUT_CSV} ...")

    for i, row in enumerate(data_rows, start=1):
        # Expand row to current output_header length
        row = row + [""] * (len(output_header) - len(row))

        company_name = row[company_idx].strip() if len(row) > company_idx else ""
        postcode = row[pc_idx].strip() if len(row) > pc_idx else ""
        uri = row[uri_idx].strip() if (uri_idx is not None and len(row) > uri_idx) else ""

        # Existing values (don't overwrite if already present)
        existing_website = row[out_website_idx].strip() if len(row) > out_website_idx else ""
        existing_email = row[out_email_idx].strip() if len(row) > out_email_idx else ""
        existing_number = row[out_number_idx].strip() if len(row) > out_number_idx else ""

        # Skip completely blank inputs
        if not company_name and not postcode and not uri:
            processed_rows.append(row)
            phone_lists.append([])
            print(f"[{i}/{len(data_rows)}] → Skipped blank row")
            continue

        # Attempt lookup if any of website/email/number is missing
        needs_lookup = not (existing_website and existing_email and existing_number)

        if needs_lookup:
            try:
                info = await run_for_row(llm, company_name, postcode, uri or None)
                if info:
                    # update company/postcode if missing
                    row[out_company_idx] = row[out_company_idx] or info.company_name or company_name or ""
                    row[out_pc_idx] = row[out_pc_idx] or info.post_code or postcode or ""
                    # prefer website if missing
                    if not existing_website and info.website:
                        row[out_website_idx] = info.website.strip()
                    # prefer email if missing
                    if not existing_email and info.email:
                        row[out_email_idx] = (info.email or "").strip().lower()
                    # capture numbers list (clean/dedupe)
                    nums = []
                    if info.numbers:
                        nums = [_clean_phone(n) for n in info.numbers if isinstance(n, str)]
                        nums = [n for n in nums if len(n) >= 7]  # simple sanity filter
                        nums = _dedupe_keep_order(nums)

                    # also if 'number' already exists and we fetched nums, keep existing at front if not duplicate
                    if existing_number:
                        clean_existing = _clean_phone(existing_number)
                        if len(clean_existing) >= 7 and clean_existing not in nums:
                            nums = [clean_existing] + nums

                    # We'll assign these later when writing (for dynamic number columns)
                    phone_lists.append(nums)

                    # source/confidence/notes
                    row[out_src_idx] = row[out_src_idx] or (info.source_url or "")
                    row[out_conf_idx] = row[out_conf_idx] or ("" if info.confidence is None else f"{info.confidence:.2f}")
                    row[out_notes_idx] = row[out_notes_idx] or (info.notes or "")

                    print(f"[{i}/{len(data_rows)}] ✓ {company_name} {postcode} -> site:{row[out_website_idx] or '—'} email:{row[out_email_idx] or '—'} phones:{len(nums) or '—'}")
                else:
                    phone_lists.append([_clean_phone(existing_number)] if existing_number else [])
                    if not row[out_notes_idx]:
                        row[out_notes_idx] = "No structured result returned."
                    print(f"[{i}/{len(data_rows)}] ? {company_name} {postcode} -> no result")
            except Exception as e:
                phone_lists.append([_clean_phone(existing_number)] if existing_number else [])
                row[out_notes_idx] = f"Agent error: {e}"
                print(f"[{i}/{len(data_rows)}] ✗ {company_name} {postcode} -> Error: {e}")
        else:
            # no lookup needed; keep existing number as the only phone
            phone_lists.append([_clean_phone(existing_number)] if existing_number else [])
            print(f"[{i}/{len(data_rows)}] → Skipping lookup (already has website/email/number)")

        processed_rows.append(row)

    # Determine how many phone columns we need
    max_phones = 0
    for nums in phone_lists:
        max_phones = max(max_phones, len(nums))

    # Ensure header has number columns: number, number_2, ..., number_N
    number_base_idx = output_header.index("number")
    if max_phones <= 1:
        # only 'number' needed
        pass
    else:
        # add number_2 ... number_max
        for k in range(2, max_phones + 1):
            col_name = f"number_{k}"
            if col_name not in output_header:
                output_header.append(col_name)

    # Write output with padded rows
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(
            f,
            delimiter=dialect.delimiter,
            quotechar='"',
            doublequote=True,
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writerow(output_header)

        for row, nums in zip(processed_rows, phone_lists):
            # Make sure row length matches header length
            row = row + [""] * (len(output_header) - len(row))

            # Fill phone columns
            # Put first number into 'number', subsequent into number_2, number_3...
            for k in range(max_phones):
                val = nums[k] if k < len(nums) else ""
                if k == 0:
                    row[number_base_idx] = row[number_base_idx] or val
                else:
                    col_name = f"number_{k+1}"
                    col_idx = output_header.index(col_name)
                    row[col_idx] = val

            writer.writerow(row)

    print(f"✅ Wrote {len(processed_rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
