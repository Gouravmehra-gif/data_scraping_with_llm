from playwright.sync_api import sync_playwright
import time
import re

email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
phone_pattern = r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b'
name_pattern = r'[A-Z][a-z]+\s[A-Z][a-z]+'

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

# ---------------- Selector generation ----------------
def extract_selector(text):
    """Extracts a valid CSS selector from LLM output."""
    match = re.search(r"[\.\#]?[a-zA-Z][\w\-\.\#]*", text)
    return match.group(0).strip() if match else None

def get_css_selector_from_llm(parent_html, target_description):
    prompt = f"Given the following HTML:\n{parent_html}\n\nGenerate a CSS selector that selects the element described as: {target_description}. Do not include any explanation or extra text. only write valid css selector after the keyword 'selector:'."
    messages = [
        {"role": "system", "content": "You are an expert in web scraping. Your ONLY task is to return a valid and short **CSS selector** that selects the requested element from the HTML. Only return the CSS selector string, nothing else. For example, if the HTML is <div class='titleline'> and you want to select the div, just return '.titleline'."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=30)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    selector = response.replace("selector:", "").replace("Selector:", "").replace("selector", "").replace("-text", "").replace("::text", "").strip()
    if not selector.startswith(".") and not selector.startswith("#"):
        selector = "." + selector
    return selector

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
    
    # Optional clean-up if needed
    return response.strip()


# ---------------- Web scraping ----------------
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    # ---------------- Getting the robots.txt file so we know which routes to scrape and the rate limit we have to maintain ----------------
    page.goto("https://news.ycombinator.com/robots.txt")
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

    page.goto("https://news.ycombinator.com/")
    page.wait_for_selector(".title")  # Parent block

    parent_blocks = page.query_selector_all(".title")
    target_description = "title line"
    selector = get_css_selector_from_llm(parent_blocks[0].inner_html(), target_description)

    print(f"AI Generated selector: {selector}")
    news_titles = []
    for i, parent_element in enumerate(parent_blocks):
        try:
            news_element = parent_element.query_selector(selector)
            news_text = news_element.inner_text()

            print("_________________Data Anonymization in Progress_________________")
            try:
                anonymized_email = re.sub(email_pattern, '[EMAIL]', news_text)
                anonymized_phone = re.sub(phone_pattern, '[PHONE]', news_text)
                anonymized_name = re.sub(name_pattern, '[NAME]', news_text)
            except Exception as e:
                pass
            print("_________________Data Anonymization completed_________________")

            news_titles.append(news_text)
            print(f"{i}: {news_text}\n")
        except Exception as e:
            pass
    browser.close()


for title in news_titles:
    analysis = get_semantics_from_llm(title)
    print(f"\nTitle: {title}\n{analysis}")