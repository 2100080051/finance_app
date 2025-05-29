import os
from openai import OpenAI

class LanguageAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"]  # Use Groq endpoint from Render env
        )

    def generate_summary(self, stock_data, news_data, doc_chunks):
        news_text = "\n".join([f"- {item['title']}" for item in news_data])
        doc_text = "\n".join(doc_chunks)

        prompt = f"""
        You're a finance assistant. Give a short spoken-style market briefing based on:

        📊 Stock Info:
        {stock_data}

        📰 News Headlines:
        {news_text}

        📄 Key Document Insights:
        {doc_text}

        Be brief, professional, and under 200 words.
        """

        response = self.client.chat.completions.create(
            model="llama3-8b-8192",  # You can also test with mixtral
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()
