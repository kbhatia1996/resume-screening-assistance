"""
Resume Parser – line-by-line explained version
Local-first, no paid services. Parses PDF/DOCX/TXT into a normalized JSON.

Folders (recommended):
  data/resumes/raw/     -> input resumes
  data/resumes/parsed/  -> JSON outputs

Dependencies (install later):
  pdfplumber==0.11.0    # for PDF text extraction
  python-docx==1.1.2    # for DOCX text extraction

Usage (after you set up Python):
  python resume_parser.py data/resumes/raw/jane_doe.pdf --out data/resumes/parsed
"""

# -------------------------------
# 1) Standard library imports
# -------------------------------
from __future__ import annotations  # enables forward references & newer typing behavior
import argparse  # for a simple CLI interface
import json  # to write the normalized JSON output
import re  # regular expressions for emails/phones/section headers
from dataclasses import dataclass, asdict  # lightweight structured containers
from pathlib import Path  # path-safe filesystem operations
from typing import Dict, List, Optional, Tuple  # type hints for clarity

# -------------------------------
# 2) Optional third-party imports
#    We import lazily inside functions so the file can load even if deps aren't installed yet.
# -------------------------------
# (No top-level third-party imports so the script doesn't crash if you haven't installed them.)

# -------------------------------
# 3) Small, built-in skills lexicon
#    This is only for demo; you can replace with a richer list or load from a JSON later.
# -------------------------------
DEFAULT_SKILLS = {
    "programming": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "sql", "bash", "matlab", "scala"
    ],
    "ml_ai": [
        "machine learning", "deep learning", "nlp", "computer vision", "pytorch",
        "tensorflow", "scikit-learn", "xgboost", "lightgbm", "llm", "rag", "langchain"
    ],
    "data_platforms": [
        "spark", "hadoop", "kafka", "airflow", "dbt", "snowflake", "bigquery"
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "git", "github actions"
    ],
    "web_backend": [
        "fastapi", "flask", "django", "spring", "node", "express", "graphql", "rest"
    ],
}

# -------------------------------
# 4) Data classes to keep structure explicit & typed
# -------------------------------
@dataclass
class ExperienceItem:
    role: Optional[str]
    company: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    description: Optional[str]

@dataclass
class EducationItem:
    degree: Optional[str]
    institution: Optional[str]
    year: Optional[str]

@dataclass
class CertificationItem:
    name: Optional[str]
    year: Optional[str]

@dataclass
class ParsedResume:
    name: Optional[str]
    contact: Dict[str, Optional[str]]
    summary: Optional[str]
    skills: List[str]
    experience: List[ExperienceItem]
    education: List[EducationItem]
    certifications: List[CertificationItem]
    raw_text: str

# -------------------------------
# 5) File type detection
# -------------------------------

def detect_file_type(path: Path) -> str:
    """Return 'pdf', 'docx', or 'txt' based on file suffix."""
    suffix = path.suffix.lower()  # normalize case like .PDF -> .pdf
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".docx":
        return "docx"
    if suffix == ".txt":
        return "txt"
    raise ValueError(f"Unsupported file type: {suffix}")

# -------------------------------
# 6) Text extraction per file type (lazy imports inside)
# -------------------------------

def extract_text_pdf(path: Path) -> str:
    """Extract text from a PDF using pdfplumber (handles multi-page PDFs)."""
    try:
        import pdfplumber  # local import so script doesn't hard-fail without deps
    except ImportError as e:
        raise ImportError(
            "pdfplumber is required for PDF parsing. Install with: pip install pdfplumber"
        ) from e

    text_parts: List[str] = []  # we collect page texts here
    with pdfplumber.open(str(path)) as pdf:  # open the PDF file
        for page in pdf.pages:  # iterate through each page
            page_text = page.extract_text() or ""  # extract text; guard against None
            text_parts.append(page_text)  # accumulate
    return "\n".join(text_parts)  # join pages with newlines


def extract_text_docx(path: Path) -> str:
    """Extract text from a DOCX using python-docx."""
    try:
        from docx import Document  # local import
    except ImportError as e:
        raise ImportError(
            "python-docx is required for DOCX parsing. Install with: pip install python-docx"
        ) from e

    doc = Document(str(path))  # load the DOCX file
    paragraphs = [p.text for p in doc.paragraphs]  # read each paragraph's text
    return "\n".join(paragraphs)  # join paragraphs with newlines


def extract_text_txt(path: Path) -> str:
    """Read plain text with UTF-8 fallback to latin-1 to avoid decode crashes."""
    try:
        return path.read_text(encoding="utf-8")  # first try utf-8
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")  # fallback if encoding is odd

# -------------------------------
# 7) Text cleanup utilities
# -------------------------------

def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace, fix stray hyphenations, and strip edges."""
    text = re.sub(r"\u00A0", " ", text)  # replace non-breaking spaces with normal spaces
    text = re.sub(r"-\s*\n\s*", "", text)  # join words split by hyphen at line end
    text = re.sub(r"\r\n?|\n", "\n", text)  # normalize CRLF/CR to LF
    text = re.sub(r"[\t ]+", " ", text)  # collapse tabs/multiple spaces inside lines
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse 3+ newlines to 2
    return text.strip()  # trim leading/trailing whitespace

# -------------------------------
# 8) Contact info extraction (regex-based, language-agnostic)
# -------------------------------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")  # simple email regex
PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}"  # permissive phone
)
LINKEDIN_RE = re.compile(r"https?://(www\.)?linkedin\.com/[^\s]+", re.IGNORECASE)  # linkedin URLs
GITHUB_RE = re.compile(r"https?://(www\.)?github\.com/[^\s]+", re.IGNORECASE)  # github URLs
NAME_LINE_RE = re.compile(r"^[A-Z][A-Za-z\-.' ]{1,40}$")  # naive: single name line at top (tweak as needed)


def extract_contact_and_name(text: str) -> Tuple[Optional[str], Dict[str, Optional[str]]]:
    """Pull name (best-effort) + contact fields (email/phone/linkedin/github)."""
    # Heuristic: assume the first 5 lines may contain the candidate's name
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]  # keep non-empty lines
    candidate_name: Optional[str] = None  # default until we find one
    for ln in lines[:5]:  # scan first few lines
        if EMAIL_RE.search(ln):  # if line has email, it's probably not the name line
            continue
        if PHONE_RE.search(ln):  # same for phones
            continue
        # A very rough rule: looks like a capitalized name with limited punctuation
        if 2 <= len(ln.split()) <= 5 and NAME_LINE_RE.match(ln):
            candidate_name = ln
            break

    email = (EMAIL_RE.search(text).group(0) if EMAIL_RE.search(text) else None)  # first email hit
    phone = (PHONE_RE.search(text).group(0) if PHONE_RE.search(text) else None)  # first phone hit
    linkedin = (LINKEDIN_RE.search(text).group(0) if LINKEDIN_RE.search(text) else None)
    github = (GITHUB_RE.search(text).group(0) if GITHUB_RE.search(text) else None)

    contact = {"email": email, "phone": phone, "linkedin": linkedin, "github": github}
    return candidate_name, contact

# -------------------------------
# 9) Section splitting based on common headings
# -------------------------------
SECTION_HEADERS = [
    "summary", "professional summary", "about", "profile",
    "experience", "work experience", "employment", "professional experience",
    "education", "academics",
    "skills", "technical skills",
    "projects",
    "certifications", "licenses",
]

HEADER_RE = re.compile(
    r"^(?P<h>" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*:?$",
    re.IGNORECASE
)


def split_into_sections(text: str) -> Dict[str, str]:
    """Return a dict of {section_name_lower: section_text}. Unlabeled text goes under 'preamble'."""
    sections: Dict[str, List[str]] = {}  # accumulate lines per section key
    current_key = "preamble"  # text before first recognized header
    sections[current_key] = []

    for raw_line in text.splitlines():  # iterate over lines
        line = raw_line.strip()  # normalize spaces at edges
        if not line:
            continue  # skip empty lines
        m = HEADER_RE.match(line.lower())  # does this line look like a known header?
        if m:
            current_key = m.group("h").lower()  # switch bucket
            if current_key not in sections:
                sections[current_key] = []  # start new list if first time
        else:
            sections[current_key].append(line)  # collect line into current bucket

    # join lists into single blocks of text
    return {k: "\n".join(v).strip() for k, v in sections.items() if v}

# -------------------------------
# 10) Skill extraction via keyword lookup
# -------------------------------

def extract_skills(text: str, lexicon: Dict[str, List[str]] = DEFAULT_SKILLS) -> List[str]:
    """Simple case-insensitive keyword scan across categories; returns unique normalized skills."""
    found: List[str] = []
    lowered = text.lower()
    for category, words in lexicon.items():  # iterate each category (not used, but kept to group later)
        for w in words:
            # match whole word or standalone token; avoid partials like 'awsome' matching 'aws'
            pattern = r"(?<![A-Za-z0-9_])" + re.escape(w) + r"(?![A-Za-z0-9_])"
            if re.search(pattern, lowered):
                found.append(w)
    # deduplicate while preserving order
    seen = set()
    unique = [x for x in found if not (x in seen or seen.add(x))]
    return unique

# -------------------------------
# 11) Experience/Education/Cert parsing – conservative baseline
#     We keep it simple: treat section text as blocks and try to capture common patterns.
# -------------------------------
DATE_RE = re.compile(
    r"(?P<start>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4}|\d{4})\s*(?:-|to|–|—|until)\s*(?P<end>(?:Present|Current|Now|\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4}))",
    re.IGNORECASE,
)


def parse_experience_block(text: str) -> List[ExperienceItem]:
    """Very light heuristic parsing of experience lines into structures."""
    items: List[ExperienceItem] = []
    # Split on blank lines to get chunks (each chunk ≈ one job)
    chunks = re.split(r"\n\s*\n", text.strip())
    for chunk in chunks:
        lines = [ln.strip("- •\t ") for ln in chunk.splitlines() if ln.strip()]
        if not lines:
            continue
        header = lines[0]
        # Try to separate "Role at Company" or "Company - Role"
        role, company = None, None
        if " at " in header.lower():
            parts = re.split(r"\bat\b", header, flags=re.IGNORECASE)
            role = parts[0].strip("- •| ") if parts else None
            company = parts[1].strip("- •| ") if len(parts) > 1 else None
        elif " - " in header:
            parts = header.split(" - ", 1)
            company = parts[0].strip()
            role = parts[1].strip() if len(parts) > 1 else None
        else:
            # If unknown, guess role = header, leave company None
            role = header

        # Look for a date range anywhere in chunk
        m = DATE_RE.search(chunk)
        start, end = (m.group("start"), m.group("end")) if m else (None, None)

        # Description = remaining lines joined
        description = " ".join(lines[1:]) if len(lines) > 1 else None

        items.append(ExperienceItem(role=role, company=company, start_date=start, end_date=end, description=description))
    return items


def parse_education_block(text: str) -> List[EducationItem]:
    """Parse education lines looking for degree, institution, year (very heuristically)."""
    items: List[EducationItem] = []
    for line in text.splitlines():
        line = line.strip("- •\t ")
        if not line:
            continue
        # year if present
        year_match = re.search(r"(20\d{2}|19\d{2})", line)
        year = year_match.group(0) if year_match else None
        # naive split by comma – common pattern: Degree, Institution, Year
        parts = [p.strip() for p in line.split(",")]
        degree = parts[0] if parts else None
        institution = parts[1] if len(parts) > 1 else None
        items.append(EducationItem(degree=degree, institution=institution, year=year))
    return items


def parse_certifications_block(text: str) -> List[CertificationItem]:
    """Parse certifications – each line is a cert; try to capture a year if present."""
    items: List[CertificationItem] = []
    for line in text.splitlines():
        line = line.strip("- •\t ")
        if not line:
            continue
        year_match = re.search(r"(20\d{2}|19\d{2})", line)
        year = year_match.group(0) if year_match else None
        items.append(CertificationItem(name=line, year=year))
    return items

# -------------------------------
# 12) High-level parse function that ties everything together
# -------------------------------

def parse_resume(path: Path) -> ParsedResume:
    """Detect type -> extract text -> clean -> segment -> extract fields -> return ParsedResume."""
    ftype = detect_file_type(path)  # get file type from extension

    # Text extraction
    if ftype == "pdf":
        raw = extract_text_pdf(path)
    elif ftype == "docx":
        raw = extract_text_docx(path)
    else:  # txt
        raw = extract_text_txt(path)

    # Normalize whitespace & line endings
    text = normalize_whitespace(raw)

    # Contact + name
    name, contact = extract_contact_and_name(text)

    # Section splitting
    sections = split_into_sections(text)

    # Summary – pick from 'summary'/'professional summary' if present, else preamble top
    summary_candidates = [
        sections.get("summary"),
        sections.get("professional summary"),
        sections.get("about"),
        sections.get("profile"),
    ]
    summary = next((s for s in summary_candidates if s), None)
    if summary is None:
        # If no explicit summary, take first 3-5 lines from preamble (if present)
        preamble = sections.get("preamble", "")
        lines = [ln for ln in preamble.splitlines() if ln.strip()]
        summary = " ".join(lines[:5]) if lines else None

    # Skills
    skills_text = sections.get("skills") or sections.get("technical skills") or text
    skills = extract_skills(skills_text)

    # Experience / Education / Certifications
    exp_text = (
        sections.get("experience")
        or sections.get("work experience")
        or sections.get("employment")
        or sections.get("professional experience")
        or ""
    )
    experience = parse_experience_block(exp_text) if exp_text else []

    edu_text = sections.get("education") or sections.get("academics") or ""
    education = parse_education_block(edu_text) if edu_text else []

    cert_text = sections.get("certifications") or sections.get("licenses") or ""
    certifications = parse_certifications_block(cert_text) if cert_text else []

    # Bundle into dataclass
    parsed = ParsedResume(
        name=name,
        contact=contact,
        summary=summary,
        skills=skills,
        experience=experience,
        education=education,
        certifications=certifications,
        raw_text=text,
    )
    return parsed

# -------------------------------
# 13) Save helper – write ParsedResume as JSON
# -------------------------------

def save_parsed_json(parsed: ParsedResume, out_dir: Path, source_path: Path) -> Path:
    """Write the parsed content to out_dir with the same stem as input, .json extension."""
    out_dir.mkdir(parents=True, exist_ok=True)  # ensure output directory exists
    out_path = out_dir / f"{source_path.stem}.json"  # derive output file name

    # Convert dataclasses inside lists to dicts recursively
    def convert(obj):
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items()}
        return obj

    payload = convert(parsed)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path

# -------------------------------
# 14) CLI entrypoint – so you can run it directly later
# -------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a resume (PDF/DOCX/TXT) into normalized JSON")
    parser.add_argument("input", type=str, help="Path to input resume file")
    parser.add_argument("--out", type=str, default="data/resumes/parsed", help="Output directory for JSON")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    parsed = parse_resume(in_path)
    out_path = save_parsed_json(parsed, Path(args.out), in_path)
    print(f"✅ Parsed -> {out_path}")


if __name__ == "__main__":  # standard Python entrypoint guard
    main()
