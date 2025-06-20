# from fastapi import FastAPI, HTTPException, BackgroundTasks
# from pydantic import BaseModel
# from bs4 import BeautifulSoup
# import requests
# import os
# import openai
# import logging
# import tldextract
# from urllib.parse import urljoin
# from dotenv import load_dotenv
# import csv
# import io
# from google.cloud import storage
# import base64
# import tempfile
# import uvicorn

# # Load environment variables
# load_dotenv()

# # Decode Google Cloud credentials
# creds_b64 = os.getenv("GOOGLE_CREDS_B64")
# if creds_b64:
#     with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
#         temp_file.write(base64.b64decode(creds_b64).decode())
#         os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file.name

# # OpenAI client setup
# client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("main")

# # FastAPI app
# app = FastAPI()

# class URLInput(BaseModel):
#     url: str

# def is_valid_url(url):
#     return url.startswith("http")

# def get_domain(url):
#     ext = tldextract.extract(url)
#     return f"{ext.domain}.{ext.suffix}"

# def crawl_site(url, base_domain):
#     to_visit = [url]
#     visited = set()
#     html_pages = []
#     logger.info(f"🌐 Starting crawl at: {url}")

#     while to_visit:
#         current_url = to_visit.pop(0)
#         if current_url in visited:
#             continue
#         visited.add(current_url)

#         try:
#             response = requests.get(current_url, timeout=10)
#             if response.status_code != 200:
#                 continue

#             soup = BeautifulSoup(response.text, "html.parser")
#             text = soup.get_text(separator="\n", strip=True)
#             html_pages.append((current_url, text))

#             logger.info(f"✅ Scraped: {current_url}")

#             for a_tag in soup.find_all("a", href=True):
#                 href = a_tag["href"]
#                 absolute_url = urljoin(current_url, href)
#                 if (
#                     is_valid_url(absolute_url)
#                     and get_domain(absolute_url) == base_domain
#                     and absolute_url not in visited
#                     and absolute_url not in to_visit
#                 ):
#                     to_visit.append(absolute_url)

#         except Exception as e:
#             logger.warning(f"❌ Error visiting {current_url}: {e}")
#             continue

#     return html_pages

# def extract_faqs_from_text(text, page_url):
#     prompt = (
#         "Extract 5 to 10 frequently asked questions (FAQs) from the following web page content. "
#         "Format the result strictly as a CSV file with the header: question,answer. "
#         "Escape commas and quotes properly using standard CSV rules. Do NOT use markdown.\n\n"
#         f"Page URL: {page_url}\n\n"
#         f"Content:\n{text[:8000]}"
#     )

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You extract FAQs in CSV format."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.4
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         logger.error(f"❌ OpenAI API Error for {page_url}: {e}")
#         return None

# def safe_parse_csv(raw_csv: str):
#     try:
#         raw_csv = raw_csv.strip()
#         if raw_csv.startswith("```"):
#             raw_csv = raw_csv.strip("`").split("csv")[-1].strip()

#         lines = raw_csv.splitlines()
#         if not lines or "question" not in lines[0].lower() or "answer" not in lines[0].lower():
#             logger.warning("⚠️ Missing or incorrect headers, attempting to fix...")
#             lines.insert(0, "question,answer")

#         cleaned_lines = [line for line in lines if ',' in line]
#         reader = csv.DictReader(cleaned_lines)
#         faqs = list(reader)

#         if not faqs or "question" not in faqs[0] or "answer" not in faqs[0]:
#             raise ValueError("Parsed CSV is invalid")

#         return faqs
#     except Exception as e:
#         logger.error(f"❌ Failed to parse CSV: {e}")
#         return []

# def upload_to_gcs(bucket_name: str, destination_blob_name: str, source_data: str):
#     try:
#         client = storage.Client()
#         bucket = client.bucket(bucket_name)
#         blob = bucket.blob(destination_blob_name)
#         blob.upload_from_string(source_data, content_type='text/csv')
#         logger.info(f"📦 CSV uploaded to GCS: gs://{bucket_name}/{destination_blob_name}")
#         return f"gs://{bucket_name}/{destination_blob_name}"
#     except Exception as e:
#         logger.error(f"❌ Failed to upload to GCS: {e}")
#         raise HTTPException(status_code=500, detail="GCS upload failed.")

# def scrape_and_generate_faqs_task(url: str):
#     logger.info(f"📅 Background scrape started for URL: {url}")

#     if not is_valid_url(url):
#         logger.error("❌ Invalid URL")
#         return

#     base_domain = get_domain(url)
#     pages = crawl_site(url, base_domain)

#     if not pages:
#         logger.error("❌ No pages found during crawl")
#         return

#     all_faqs = []

#     for page_url, text in pages:
#         logger.info(f"📝 Processing page: {page_url}")
#         csv_text = extract_faqs_from_text(text, page_url)
#         if csv_text:
#             faqs = safe_parse_csv(csv_text)
#             all_faqs.extend(faqs)

#     if not all_faqs:
#         logger.error("❌ No FAQs extracted")
#         return

#     csv_buffer = io.StringIO()
#     writer = csv.DictWriter(csv_buffer, fieldnames=["question", "answer"])
#     writer.writeheader()
#     writer.writerows(all_faqs)

#     bucket_name = os.getenv("GCS_BUCKET_NAME")
#     upload_to_gcs(bucket_name, "faq/faq_scraped.csv", csv_buffer.getvalue())

# @app.post("/scrape")
# def scrape_endpoint(input_data: URLInput, background_tasks: BackgroundTasks):
#     background_tasks.add_task(scrape_and_generate_faqs_task, input_data.url)
#     return {"message": "✅ Scraping started in background."}

# @app.get("/")
# def health_check():
#     return {"status": "OK"}

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)








from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from bs4 import BeautifulSoup
import requests
import os
import openai
import logging
import tldextract
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
import csv
import io
from google.cloud import storage
import base64
import tempfile
import uvicorn

# Load environment variables
load_dotenv()

# Decode Google Cloud credentials
creds_b64 = os.getenv("GOOGLE_CREDS_B64")
if creds_b64:
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
        temp_file.write(base64.b64decode(creds_b64).decode())
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file.name

# OpenAI client setup
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# FastAPI app
app = FastAPI()

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
    html_pages = []
    logger.info(f"🌐 Starting crawl at: {url}")

    parsed_root = urlparse(url)
    base_path = parsed_root.path.rstrip("/") + "/"  # Ensure trailing slash

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
            html_pages.append((current_url, text))

            logger.info(f"✅ Scraped: {current_url}")

            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                absolute_url = urljoin(current_url, href)
                parsed_url = urlparse(absolute_url)

                if (
                    is_valid_url(absolute_url)
                    and get_domain(absolute_url) == base_domain
                    and absolute_url not in visited
                    and absolute_url not in to_visit
                    and parsed_url.path.startswith(base_path)
                ):
                    to_visit.append(absolute_url)

        except Exception as e:
            logger.warning(f"❌ Error visiting {current_url}: {e}")
            continue

    return html_pages

def extract_faqs_from_text(text, page_url):
    prompt = (
        "Extract 5 to 10 frequently asked questions (FAQs) from the following web page content. "
        "Format the result strictly as a CSV file with the header: question,answer. "
        "Escape commas and quotes properly using standard CSV rules. Do NOT use markdown.\n\n"
        f"Page URL: {page_url}\n\n"
        f"Content:\n{text[:8000]}"
    )

    try:
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
        logger.error(f"❌ OpenAI API Error for {page_url}: {e}")
        return None

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
        return []

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

def scrape_and_generate_faqs_task(url: str):
    logger.info(f"📅 Background scrape started for URL: {url}")

    if not is_valid_url(url):
        logger.error("❌ Invalid URL")
        return

    base_domain = get_domain(url)
    pages = crawl_site(url, base_domain)

    if not pages:
        logger.error("❌ No pages found during crawl")
        return

    all_faqs = []

    for page_url, text in pages:
        logger.info(f"📝 Processing page: {page_url}")
        csv_text = extract_faqs_from_text(text, page_url)
        if csv_text:
            faqs = safe_parse_csv(csv_text)
            all_faqs.extend(faqs)

    if not all_faqs:
        logger.error("❌ No FAQs extracted")
        return

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=["question", "answer"])
    writer.writeheader()
    fieldnames = ["question", "answer"]  # Or whatever you used
    cleaned_faqs = [
        {key: row.get(key, "") for key in fieldnames}
        for row in all_faqs
    ]
    writer.writerows(cleaned_faqs)


    bucket_name = os.getenv("GCS_BUCKET_NAME")
    upload_to_gcs(bucket_name, "faq/faq_scraped.csv", csv_buffer.getvalue())

@app.post("/scrape")
def scrape_endpoint(input_data: URLInput, background_tasks: BackgroundTasks):
    background_tasks.add_task(scrape_and_generate_faqs_task, input_data.url)
    return {"message": "✅ Scraping started in background."}

@app.get("/")
def health_check():
    return {"status": "OK"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

