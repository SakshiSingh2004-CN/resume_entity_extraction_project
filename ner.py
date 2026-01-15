import os
import re
import time
import json
import logging
import phonenumbers
import docx
import mysql.connector
import snowflake.connector
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# third-party libs imported with proper alias
import pandas as pd
import spacy

# ---------------- CONFIG ----------------
FOLDER = r"C:\Users\Dell\OneDrive\Desktop\new_ner_model\ATS_DS_100_Unique_Names_Phones"
CSV_DIR = os.path.join(FOLDER, "csv_output")   # output folder for CSVs
os.makedirs(CSV_DIR, exist_ok=True)

# load spacy model (friendly error if not installed)
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    raise RuntimeError("SpaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm") from e

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{6,}\d")  # slightly more permissive

SKILL_KEYWORDS = [
    "python","java","c++","sql","mysql","mongodb","numpy","pandas","excel",
    "tableau","power bi","aws","azure","gcp","git","github","html","css",
    "javascript","react","node","flask","django","ml","machine learning",
    "deep learning","nlp","data analysis","docker","kubernetes"
]
SKILL_KEYWORDS = [k.lower() for k in SKILL_KEYWORDS]

SUPPORTED_EXTENSIONS = ["pdf", "docx", "txt", "json"]

# ---------------- TEXT EXTRACTION ----------------
def extract_text(path):
    ext = path.lower().split('.')[-1]

    if ext == "pdf":
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    elif ext == "docx":
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == "txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext == "json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Flatten JSON values into a single string
            def flatten(d):
                if isinstance(d, dict):
                    return " ".join(flatten(v) for v in d.values())
                elif isinstance(d, list):
                    return " ".join(flatten(i) for i in d)
                else:
                    return str(d)
            return flatten(data)
    else:
        return ""

# ---------------- NAME EXTRACTION ----------------
def get_name(text):
    if not text:
        return None

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    header = " ".join(lines[:5])

    # Regex Name (First Last or First Middle Last)
    m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", header)
    if m and len(m.group(1).split()) >= 2:
        return m.group(1)

    # SpaCy Name (search first 300 chars)
    for ent in nlp(text[:300]).ents:
        if ent.label_ == "PERSON" and 2 <= len(ent.text.split()) <= 3:
            return ent.text

    # First Line Fallback
    if lines and lines[0].replace(" ", "").isalpha() and len(lines[0].split()) <= 3:
        return lines[0].title()

    return None

# ---------------- PHONE EXTRACTION ----------------
def get_phone(text):
    if not text:
        return None

    for m in PHONE_RE.findall(text):
        try:
            p = phonenumbers.parse(m, "IN")
            if phonenumbers.is_valid_number(p):
                return phonenumbers.format_number(p, phonenumbers.PhoneNumberFormat.E164)
        except Exception:
            # try parsing without country (may still fail)
            try:
                p = phonenumbers.parse(m, None)
                if phonenumbers.is_valid_number(p):
                    return phonenumbers.format_number(p, phonenumbers.PhoneNumberFormat.E164)
            except Exception:
                continue
    return None

# ---------------- SECTION FINDER ----------------
def get_section(text, keys):
    lines = text.splitlines()
    capture = False
    result = []
    stop_keywords = ["education","experience","summary","contact","project","projects","skills","certification","certifications"]

    lower_keys = [k.lower() for k in keys]
    for ln in lines:
        low = ln.lower()
        # start capturing when any key appears as a whole word or phrase
        if any(k in low for k in lower_keys):
            capture = True
            continue

        # stop when another common section appears
        if capture and any(stop in low for stop in stop_keywords):
            break

        if capture:
            if ln.strip():
                result.append(ln.strip())

    return result

# ---------------- SKILL EXTRACTION ----------------
def extract_skills(text):
    if not text:
        return []
    section = get_section(text, ["skill", "skills"])
    if section:
        combined = " ".join(section).lower()
        return sorted({s for s in SKILL_KEYWORDS if s in combined})
    # fallback: scan whole text
    found = sorted({s for s in SKILL_KEYWORDS if s in text.lower()})
    return found

# ---------------- PROJECT EXTRACTION ----------------
def extract_projects(text):
    if not text:
        return []
    section = get_section(text, ["project", "projects"])
    if section:
        cleaned = [p.strip() for p in section if len(p.strip()) > 5]
        return cleaned
    return []

# ---------------- CSV GENERATION ----------------
def generate_csvs():
    rows_names, rows_emails, rows_phones, rows_skills, rows_projects = [], [], [], [], []

    for file in os.listdir(FOLDER):
        if not any(file.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            continue

        path = os.path.join(FOLDER, file)
        logging.info("Processing: %s", file)
        text = extract_text(path)

        # Basic fields
        rows_names.append({"file": file, "name": get_name(text)})
        rows_emails.append({"file": file, "email": (EMAIL_RE.findall(text) or [None])[0]})
        rows_phones.append({"file": file, "phone": get_phone(text)})

        # Skills
        for s in (extract_skills(text) or []):
            rows_skills.append({"file": file, "skill": s})

        # Projects
        for p in (extract_projects(text) or []):
            rows_projects.append({"file": file, "project": p})

    # Save CSVs
    pd.DataFrame(rows_names).to_csv(os.path.join(CSV_DIR, "name.csv"), index=False)
    pd.DataFrame(rows_emails).to_csv(os.path.join(CSV_DIR, "email.csv"), index=False)
    pd.DataFrame(rows_phones).to_csv(os.path.join(CSV_DIR, "phone.csv"), index=False)
    pd.DataFrame(rows_skills).to_csv(os.path.join(CSV_DIR, "skills.csv"), index=False)
    pd.DataFrame(rows_projects).to_csv(os.path.join(CSV_DIR, "projects.csv"), index=False)
    logging.info("All CSV files generated successfully in %s", CSV_DIR)

# ---------------- MYSQL SYNC ----------------
def ensure_table_and_file_column(cursor, db_name):
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS resume_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name TEXT,
        email TEXT,
        phone VARCHAR(100),
        skill TEXT,
        project TEXT,
    ) ENGINE=InnoDB;
    """)
    # ensure unique index on file (safe-guard)
    try:
        cursor.execute("ALTER TABLE resume_data ADD UNIQUE INDEX IF NOT EXISTS idx_resume_file_unique (file)")
    except Exception:
        # MySQL doesn't support IF NOT EXISTS for ALTER ADD INDEX on some versions -> ignore if fails
        pass

def sync_csv_to_mysql():
    try:
        logging.info("Syncing CSVs to MySQL...")

        names = pd.read_csv(os.path.join(CSV_DIR, "name.csv"))
        emails = pd.read_csv(os.path.join(CSV_DIR, "email.csv"))
        phones = pd.read_csv(os.path.join(CSV_DIR, "phone.csv"))
        skills = pd.read_csv(os.path.join(CSV_DIR, "skills.csv"))
        projects = pd.read_csv(os.path.join(CSV_DIR, "projects.csv"))

        df = names.merge(emails, on="file", how="left").merge(phones, on="file", how="left")

        db = mysql.connector.connect(
            host=os.environ.get("MYSQL_HOST", "localhost"),
            user=os.environ.get("MYSQL_USER", "root"),
            password=os.environ.get("MYSQL_PASSWORD", "shakshi"),
            database=os.environ.get("MYSQL_DB", "sak")
        )
        cursor = db.cursor()
        ensure_table_and_file_column(cursor, db.database)
        db.commit()

        upsert_query = """
        INSERT INTO resume_data (file, name, email, phone, skill, project)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            name = VALUES(name),
            email = VALUES(email),
            phone = VALUES(phone),
            skill = VALUES(skill),
            project = VALUES(project)
        """

        for file in df["file"].unique():
            row = df[df["file"] == file].iloc[0]
            name = row.get("name", None)
            email = row.get("email", None)
            phone = str(row.get("phone", "")) if pd.notna(row.get("phone", "")) else None

            resume_skills = skills[skills["file"] == file]["skill"].tolist()
            resume_skills = [s for s in resume_skills if isinstance(s, str) and s.strip()]
            skills_str = ", ".join(sorted(set(resume_skills))) if resume_skills else None

            resume_projects = projects[projects["file"] == file]["project"].tolist()
            resume_projects = [p for p in resume_projects if isinstance(p, str) and p.strip()]
            projects_str = ", ".join(sorted(set(resume_projects))) if resume_projects else None

            cursor.execute(upsert_query, (file, name, email, phone, skills_str, projects_str))

        db.commit()

        # Delete removed rows
        cursor.execute("SELECT file FROM resume_data")
        db_files = set([row[0] for row in cursor.fetchall() if row[0] is not None])
        csv_files = set(df["file"].unique())
        files_to_delete = db_files - csv_files
        for f in files_to_delete:
            cursor.execute("DELETE FROM resume_data WHERE file = %s", (f,))
        db.commit()

        cursor.close()
        db.close()

        logging.info("✔ Live sync complete!")

    except Exception as e:
        logging.exception("Error during sync: %s", e)

# ---------------- WATCHDOG ----------------
class CSVChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            time.sleep(0.2)
            sync_csv_to_mysql()
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            time.sleep(0.2)
            sync_csv_to_mysql()

# ---------------- S3 → SNOWFLAKE ----------------
def upload_to_s3_and_snowflake():
    # Use environment variables for credentials (do NOT hardcode)
    # AWS_ACCESS_KEY = os.environ.get("AKIATKZJPIZFLJJXRWF3")
    # AWS_SECRET_KEY = os.environ.get("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")
    # BUCKET_NAME = os.environ.get("AWS_BUCKET","pipeline-s1")
    # REGION = os.environ.get("AWS_REGION", "eu-north-1")
    # S3_FILE_KEY = os.environ.get("S3_FILE_KEY", "resume_data.csv")
    
    AWS_ACCESS_KEY = "AKIATKZJPIZFLJJXRWF3"
    AWS_SECRET_KEY = "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"
    BUCKET_NAME ="pipeline-s1"
    REGION ="eu-north-1"
    S3_FILE_KEY ="resume_data.csv"

    if not (AWS_ACCESS_KEY and AWS_SECRET_KEY and BUCKET_NAME):
        raise RuntimeError("S3 credentials or bucket not set. Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and AWS_BUCKET.")

    # Merge dataframes
    names = pd.read_csv(os.path.join(CSV_DIR, "name.csv"))
    emails = pd.read_csv(os.path.join(CSV_DIR, "email.csv"))
    phones = pd.read_csv(os.path.join(CSV_DIR, "phone.csv"))
    skills = pd.read_csv(os.path.join(CSV_DIR, "skills.csv"))
    projects = pd.read_csv(os.path.join(CSV_DIR, "projects.csv"))

    df = names.merge(emails, on="file", how="left").merge(phones, on="file", how="left")

    def get_joined(col_df, f):
        vals = col_df[col_df["file"] == f].iloc[:, 1].dropna().tolist()
        return ", ".join(vals) if vals else None

    df["skill"] = df["file"].apply(lambda f: get_joined(skills, f))
    df["project"] = df["file"].apply(lambda f: get_joined(projects, f))
    df_out = df[["name", "email", "phone", "skill", "project"]]

    csv_file = os.path.join(CSV_DIR, "temp_resume_data.csv")
    df_out.to_csv(csv_file, index=False)
    logging.info("CSV saved successfully: %s", csv_file)

    # Upload to S3 using boto3 (import here to avoid failing on systems without it)
    import boto3
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=REGION
    )
    s3_key = "resume_data.csv"
    s3.upload_file(Filename=csv_file, Bucket=BUCKET_NAME, Key=s3_key)
    logging.info("✔ File uploaded to S3 successfully: s3://%s/%s", BUCKET_NAME, s3_key)

    # Snowflake credentials via env
    SNOWFLAKE_USER ="shakshi"
    SNOWFLAKE_PASSWORD ="Shakshi@Singh#344"
    SNOWFLAKE_ACCOUNT ="MEHWNPL-FC99159"
    SNOWFLAKE_WAREHOUSE ="COMPUTE_WH"
    SNOWFLAKE_DATABASE ="RESUME_DATA"
    SNOWFLAKE_SCHEMA =  "PUBLIC"

    if not (SNOWFLAKE_USER and SNOWFLAKE_PASSWORD and SNOWFLAKE_ACCOUNT):
        raise RuntimeError("Snowflake credentials not set. Set SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, SNOWFLAKE_ACCOUNT.")

    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )
    cur = conn.cursor()
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS resume_data (
        name STRING,
        email STRING,
        phone STRING,
        skill STRING,
        project STRING
    );
    """)
    cur.execute(f"""
    CREATE OR REPLACE STAGE s3_stage
    URL='s3://{BUCKET_NAME}/'
    CREDENTIALS=(AWS_KEY_ID='{AWS_ACCESS_KEY}' AWS_SECRET_KEY='{AWS_SECRET_KEY}')
    FILE_FORMAT=(TYPE=CSV FIELD_OPTIONALLY_ENCLOSED_BY='"' SKIP_HEADER=1 ERROR_ON_COLUMN_COUNT_MISMATCH=FALSE);
    """)
    cur.execute(f"""
    COPY INTO resume_data
    FROM @s3_stage/{s3_key}
    FILE_FORMAT=(TYPE=CSV FIELD_OPTIONALLY_ENCLOSED_BY='"' SKIP_HEADER=1)
    ON_ERROR='CONTINUE';
    """)
    cur.close()
    conn.close()
    os.remove(csv_file)
    logging.info("Temporary CSV removed and data loaded into Snowflake successfully!")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    logging.info("🔄 Generating CSVs from PDF/DOCX/TXT/JSON...")
    generate_csvs()
    sync_csv_to_mysql()          # initial MySQL sync

    # If you want to push to S3 + Snowflake, ensure env vars are set, then uncomment:
    try:
        upload_to_s3_and_snowflake()
    except Exception as e:
        logging.warning("S3/Snowflake upload skipped/failed: %s", e)

    # Start watchdog to watch CSV_DIR for changes
    event_handler = CSVChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, CSV_DIR, recursive=False)
    observer.start()
    logging.info("🔄 Watching CSV folder for changes: %s", CSV_DIR)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# ----------- Streamlit UI -------------


streamlit.title("📄 NER Resume Extractor – Multi File Support")
streamlit.write("Upload **PDF, DOCX, CSV, JSON** → Extract Name, Email, Phone, Skills → Export CSV for Power BI")

uploaded_files = streamlit.file_uploader(
    "Upload Resume Files",
    type=["pdf", "docx", "csv", "json"],
    accept_multiple_files=True
)

if streamlit.button("Process Files"):
    all_data = []

    for file in uploaded_files:
        filename = file.name

        # Identify file type
        if filename.endswith(".pdf"):
            text = extract_text_pdf(file)
        elif filename.endswith(".docx"):
            text = extract_text_docx(file)
        elif filename.endswith(".csv"):
            text = extract_text_csv(file)
        elif filename.endswith(".json"):
            text = extract_text_json(file)
        else:
            continue

        record = {
            "filename": filename,
            "name": extract_name(text),
            "email": extract_email(text),
            "phone": extract_phone(text),
            "skills": ", ".join(extract_skills(text))
        }

        all_data.append(record)

    df = pd.DataFrame(all_data)

    streamlit.success("✅ Extraction Completed Successfully!")
    streamlit.dataframe(df)

    streamlit.download_button(
        "📥 Download CSV for Power BI",
        df.to_csv(index=False),
        "ner_output.csv",
        "text/csv"
    )
