from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bs4 import BeautifulSoup
import requests
import os
import openai
import logging
import tldextract
from urllib.parse import urljoin
from dotenv import load_dotenv
import csv
import io
from google.cloud import storage
import base64
import tempfile

# Load environment variables
load_dotenv()

# Decode and configure Google Cloud credentials if using base64 env var (for Render)
creds_b64 = os.getenv("GOOGLE_CREDS_B64")
if creds_b64:
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
        temp_file.write(base64.b64decode(creds_b64).decode())
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file.name

# OpenAI API setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# FastAPI app
app = FastAPI()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# Pydantic input model
class URLInput(BaseModel):
    url: str


def is_valid_url(url):
    return url.startswith("http")


def get_domain(url):
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"


def crawl_site(url, base_domain):
    to_visit = [url]
    visited = set()
    html_content = []

    logger.info(f"🌐 Starting crawl at: {url}")

    while to_visit:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue
        visited.add(current_url)

        try:
            response = requests.get(current_url, timeout=10)
            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            html_content.append(text)

            logger.info(f"✅ Scraped: {current_url}")

            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                absolute_url = urljoin(current_url, href)
                if (
                    is_valid_url(absolute_url)
                    and get_domain(absolute_url) == base_domain
                    and absolute_url not in visited
                    and absolute_url not in to_visit
                ):
                    to_visit.append(absolute_url)

        except Exception as e:
            logger.warning(f"❌ Error visiting {current_url}: {e}")
            continue

    return "\n".join(html_content)


def extract_faqs_from_text(text):
    prompt = (
        "Extract 5 to 10 frequently asked questions (FAQs) from the following website content. "
        "Format the result strictly as a CSV file with the header: question,answer. "
        "Escape commas and quotes properly using standard CSV rules. Do NOT use markdown.\n\n"
        f"Website Content:\n{text[:8000]}"
    )

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You extract FAQs in CSV format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"❌ OpenAI API Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract FAQs using GPT.")


def safe_parse_csv(raw_csv: str):
    try:
        raw_csv = raw_csv.strip()

        if raw_csv.startswith("```"):
            raw_csv = raw_csv.strip("`").split("csv")[-1].strip()

        lines = raw_csv.splitlines()

        if not lines or "question" not in lines[0].lower() or "answer" not in lines[0].lower():
            logger.warning("⚠️ Missing or incorrect headers, attempting to fix...")
            lines.insert(0, "question,answer")

        cleaned_lines = [line for line in lines if ',' in line]

        reader = csv.DictReader(cleaned_lines)
        faqs = list(reader)

        if not faqs or "question" not in faqs[0] or "answer" not in faqs[0]:
            raise ValueError("Parsed CSV is invalid")

        return faqs
    except Exception as e:
        logger.error(f"❌ Failed to parse CSV: {e}")
        return None


def upload_to_gcs(bucket_name: str, destination_blob_name: str, source_data: str):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(source_data, content_type='text/csv')
        logger.info(f"📦 CSV uploaded to GCS: gs://{bucket_name}/{destination_blob_name}")
        return f"gs://{bucket_name}/{destination_blob_name}"
    except Exception as e:
        logger.error(f"❌ Failed to upload to GCS: {e}")
        raise HTTPException(status_code=500, detail="GCS upload failed.")


@app.post("/scrape")
def scrape_and_generate_faqs(input_data: URLInput):
    logger.info(f"📥 /scrape endpoint called with URL: {input_data.url}")

    url = input_data.url
    if not is_valid_url(url):
        raise HTTPException(status_code=400, detail="URL must start with http or https.")

    base_domain = get_domain(url)
    logger.info(f"🕸️ Crawling domain: {base_domain} (no page limit)...")

    raw_text = crawl_site(url, base_domain)
    if not raw_text:
        raise HTTPException(status_code=500, detail="Failed to crawl site or extract text.")

    logger.info("🤖 Extracting FAQs using GPT...")
    csv_text = extract_faqs_from_text(raw_text)

    faqs = safe_parse_csv(csv_text)
    if not faqs:
        raise HTTPException(status_code=500, detail="Could not parse cleaned CSV.")

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=["question", "answer"])
    writer.writeheader()
    writer.writerows(faqs)

    bucket_name = os.getenv("GCS_BUCKET_NAME")
    gcs_path = upload_to_gcs(bucket_name, "faq/faq_scraped.csv", csv_buffer.getvalue())

    return {
        "message": f"✅ Extracted {len(faqs)} FAQs.",
        "gcs_path": gcs_path
    }
