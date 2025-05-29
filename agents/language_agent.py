import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Safe to include; doesn't interfere on Render

class LanguageAgent:
    def __init__(self):
        # Fetch environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

        # Validate presence
        if not api_key or not base_url:
            raise RuntimeError("❌ Missing environment variables: OPENAI_API_KEY or OPENAI_BASE_URL")

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
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

        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"❌ Failed to generate summary: {e}"


# ✅ Test this module locally (optional)
if __name__ == "__main__":
    dummy_stock = {"symbol": "AAPL", "price": 195.3, "open": 193.6}
    dummy_news = [{"title": "Apple hits all-time high"}, {"title": "Nvidia dominates AI market"}]
    dummy_docs = ["Apple's revenue reached $394B", "Strong iPhone growth seen"]

    agent = LanguageAgent()
    print(agent.generate_summary(dummy_stock, dummy_news, dummy_docs))
