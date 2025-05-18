from playwright.sync_api import sync_playwright
import time
import re


email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
phone_pattern = r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b'
name_pattern = r'[A-Z][a-z]+\s[A-Z][a-z]+'


print("_________________________________Loading the model_________________________________")
start_time = time.time()
from model import TextGenModel
model_data = TextGenModel.get_instance()
model = model_data["model"]
tokenizer = model_data["tokenizer"]

end_time = time.time()
print(f"Model loaded in {end_time - start_time:.2f} seconds.")
print("_________________________________Model loaded successfully_________________________________")

def extract_selector(text):
    """Extracts a valid CSS selector from LLM output."""
    match = re.search(r"[\.\#]?[a-zA-Z][\w\-\.\#]*", text)
    return match.group(0).strip() if match else None

def get_css_selector_from_llm(parent_html, target_description):
    prompt = f"Given the following HTML:\n{parent_html}\n\nGenerate a CSS selector that selects the element described as: {target_description}. Do not include any explanation or extra text. only write valid css selector after the keyword 'selector:'."
    messages = [
        {"role": "system", "content": "You are an expert in web scraping. Your ONLY task is to return a valid and short **CSS selector** that selects the requested element from the HTML. Only return the CSS selector string, nothing else."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=30
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    selector = response.replace("selector:", "").replace("Selector:", "").replace("selector", "").strip()
    if selector.startswith("."):
        pass
    else:
        selector = "." + selector
    return selector

# ---------------- Sementic Analusis Function ----------------

def get_semantics_from_llm(news_title):
    prompt = f"Given the following Quotes:\n\"{news_title}\"\n\nExtract its semantic meaning. Identify the topic category, key entities mentioned, and summarize the context briefly. Provide the output in this structured format:\n\nTopic: <category>\nEntities: <list of entities>\nSummary: <1-2 sentence explanation>"

    messages = [
        {
            "role": "system",
            "content": "You are an expert in natural language understanding. Your task is to perform semantic analysis of Quotes. For each title, extract the topic, involved entities, and provide a short contextual summary. Respond ONLY in the given structured format."
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




with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    print("""
==============================
  No Robots.txt file found
  Proceeding with scraping
  without limitations
==============================
""")

    page.goto("https://quotes.toscrape.com/js/")
    page.wait_for_selector(".quote")

    quotes = page.query_selector_all(".quote")
    Quotes = []
    for idx, parent_element in enumerate(quotes):
        parent_html = parent_element.inner_html()
        target_description = "the author's name"
        selector = get_css_selector_from_llm(parent_html, target_description)
        print(f"[{idx+1}] Generated CSS selector for author: {selector}")
        quote_text_element = parent_element.query_selector(".text")
        quote_text = quote_text_element.inner_text() if quote_text_element else "Quote not found"

        print("_________________Data Anonymization in Progress_________________")
        try:
            anonymized_email = re.sub(email_pattern, '[EMAIL]', quote_text)
            anonymized_phone = re.sub(phone_pattern, '[PHONE]', quote_text)
            anonymized_name = re.sub(name_pattern, '[NAME]', quote_text)
        except Exception as e:
            pass
        print("_________________Data Anonymization completed_________________")

        Quotes.append(quote_text)
        try:
            author_element = parent_element.query_selector(selector)
            author_text = author_element.inner_text() if author_element else "Author not found"
        except Exception as e:
            author_text = f"Error: {e}"
        print(f'"{quote_text}" - {author_text}\n')
    browser.close()


for title in Quotes:
    analysis = get_semantics_from_llm(title)
    print(f"\nTitle: {title}\n{analysis}")


