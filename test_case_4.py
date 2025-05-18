from playwright.sync_api import sync_playwright
import time
import re

# ---------------- Model loading ----------------
print("_________________________________Loading the model_________________________________")
start_time = time.time()
from model import TextGenModel
model_data = TextGenModel.get_instance()
model = model_data["model"]
tokenizer = model_data["tokenizer"]
end_time = time.time()
print(f"Model loaded in {end_time - start_time:.2f} seconds.")
print("_________________________________Model loaded successfully_________________________________")

# ---------------- Sementic Analusis Function ----------------

def get_semantics_from_llm(news_title):
    prompt = f"Given the following news title:\n\"{news_title}\"\n\nExtract its semantic meaning. Identify the topic category (e.g., Technology, Legal, Business), key entities mentioned, and summarize the context briefly. Provide the output in this structured format:\n\nTopic: <category>\nEntities: <list of entities>\nSummary: <1-2 sentence explanation>"

    messages = [
        {
            "role": "system",
            "content": "You are an expert in natural language understanding. Your task is to perform semantic analysis of news headlines. For each title, extract the topic, involved entities, and provide a short contextual summary. Respond ONLY in the given structured format."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=150)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

     # ---------------- Getting the robots.txt file so we know which routes to scrape and the rate limit we have to maintain ----------------
    page.goto("https://arstechnica.com/robots.txt")
    robots_txt = page.content()
    robots_txt = page.inner_text("body")
    disallowed_paths = re.findall(r"Disallow:\s*(\S+)", robots_txt)
    crawl_delay_match = re.search(r"Crawl-delay:\s*(\d+)", robots_txt)
    crawl_delay = int(crawl_delay_match.group(1)) if crawl_delay_match else None

    print("\n========== ROBOTS.TXT PARSING ==========\n")
    print("Disallowed Routes:")
    for path in disallowed_paths:
        print(f" - {path}")
    
    if crawl_delay is not None:
        print(f"\nCrawl Delay (Rate Limit): {crawl_delay} seconds between requests")
    else:
        print("\nNo Crawl Delay specified.")

    print("\n========================================\n")

    page.goto("https://arstechnica.com/", timeout=120000)
    page.wait_for_selector("h2 > a")
    news = page.query_selector_all("h2 > a")
    extracted_text = []
    for i , n in enumerate(news):
        print(f"{i}: " + n.inner_text())
        extracted_text.append(n.inner_text())

    
for title in extracted_text:
    analysis = get_semantics_from_llm(title)
    print(f"\nTitle: {title}\n{analysis}")