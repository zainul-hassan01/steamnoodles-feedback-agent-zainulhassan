import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class SteamNoodlesFeedbackSystem:
    def __init__(self, data_file: str = "reviews.csv"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.data_file = data_file
        self.current_model = "llama3-70b-8192"
        
        if os.path.exists(self.data_file):
            self.reviews_df = pd.read_csv(self.data_file, parse_dates=['date'])
            if not pd.api.types.is_datetime64_any_dtype(self.reviews_df['date']):
                self.reviews_df['date'] = pd.to_datetime(
                    self.reviews_df['date'],
                    format='mixed',
                    errors='coerce'
                )
                self.reviews_df = self.reviews_df.dropna(subset=['date'])
        else:
            self.reviews_df = self._generate_sample_data(10)
            self._save_reviews()

    def _generate_sample_data(self, num_entries: int) -> pd.DataFrame:
        sentiments = ["positive", "negative", "neutral"]
        foods = ["ramen", "dumplings", "sushi", "fried rice"]
        services = ["friendly", "slow", "excellent", "terrible"]
        
        data = []
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(num_entries):
            date = base_date + timedelta(days=random.randint(0, 29))
            sentiment = random.choice(sentiments)
            
            if sentiment == "positive":
                text = f"The {random.choice(foods)} was amazing! Service was {random.choice(services)}."
            elif sentiment == "negative":
                text = f"Terrible {random.choice(foods)}. The service was {random.choice(services)}."
            else:
                text = f"The {random.choice(foods)} was okay. Service was fine."
            
            data.append({
                "review_id": len(data) + 1,
                "text": text,
                "sentiment": sentiment,
                "date": date.strftime("%Y-%m-%d"),
                "response": f"Sample response to {sentiment} review"
            })
        
        return pd.DataFrame(data)

    def _save_reviews(self):
        self.reviews_df.to_csv(self.data_file, index=False, date_format='%Y-%m-%d')

    def analyze_sentiment(self, review_text: str) -> str:
        try:
            response = self.client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": f"""Analyze this restaurant review and classify its sentiment as positive, negative, or neutral.
                    Examples of positive words: amazing, excellent, wonderful, delicious, loved
                    Examples of negative words: terrible, awful, disgusting, horrible, worst
                    If unsure, choose neutral.
                    
                    Review: "{review_text}"
                    
                    Respond ONLY with one word: positive, negative, or neutral"""
                }],
                model=self.current_model,
                temperature=0.1,
                max_tokens=1
            )
            result = response.choices[0].message.content.strip().lower()
            
            if result in ['positive', 'negative', 'neutral']:
                return result
            return "neutral"
                
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return "neutral"

    def generate_response(self, review_text: str, sentiment: str) -> str:
        try:
            prompt = f"""As the manager of SteamNoodles, write a professional 1-2 sentence response to this {sentiment} review.
            Tone should be: {"warm and appreciative" if sentiment == "positive" else "apologetic and constructive" if sentiment == "negative" else "polite and encouraging"}
            
            Review: "{review_text}"
            
            Response:"""
            
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.current_model,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            default_responses = {
                "positive": "Thank you for your wonderful feedback! We're delighted you enjoyed your experience.",
                "negative": "We sincerely apologize for your experience. Please contact us so we can make this right.",
                "neutral": "Thank you for your feedback. We appreciate you taking the time to share your thoughts."
            }
            return default_responses.get(sentiment, "Thank you for your feedback.")



    def add_review(self, review_text: str, date: str = None) -> dict:
        if not review_text.strip():
            return {"error": "Review text cannot be empty"}
            
        review_date = date if date else datetime.now().strftime("%Y-%m-%d")
        
        try:
            parsed_date = pd.to_datetime(review_date, format='%Y-%m-%d', errors='raise')
            review_date = parsed_date.strftime('%Y-%m-%d')
        except ValueError:
            print("Invalid date format. Using today's date instead.")
            review_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            sentiment = self.analyze_sentiment(review_text)
            response = self.generate_response(review_text, sentiment)
            
            new_review = {
                "review_id": len(self.reviews_df) + 1,
                "text": review_text,
                "sentiment": sentiment,
                "date": review_date,
                "response": response
            }
            
            new_df = pd.DataFrame([new_review])
            new_df['date'] = pd.to_datetime(new_df['date'])
            
            self.reviews_df = pd.concat([
                self.reviews_df,
                new_df
            ], ignore_index=True)
            
            self._save_reviews()
            return new_review
            
        except Exception as e:
            return {"error": str(e)}


    def sentiment_visualization_agent(self, date_range: str = "last 7 days") -> str:
        try:
            if date_range == "last 7 days":
                start_date = datetime.now() - timedelta(days=7)
            elif date_range == "last 30 days":
                start_date = datetime.now() - timedelta(days=30)
            elif "to" in date_range:
                parts = date_range.split("to")
                start_date = datetime.strptime(parts[0].strip(), "%Y-%m-%d")
                end_date = datetime.strptime(parts[1].strip(), "%Y-%m-%d")
            else:
                start_date = datetime.now() - timedelta(days=7)
            
            filtered = self.reviews_df[
                (self.reviews_df['date'] >= start_date)
            ]
            
            if "end_date" in locals():
                filtered = filtered[filtered['date'] <= end_date]
            
            if filtered.empty:
                return None
                
            daily_sentiment = filtered.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
            
            plt.figure(figsize=(10, 6))
            for sentiment in ['positive', 'neutral', 'negative']:
                if sentiment in daily_sentiment.columns:
                    plt.plot(daily_sentiment.index, daily_sentiment[sentiment], 
                            label=sentiment.capitalize(), marker='o')
            
            plt.title(f"Customer Sentiment Trends ({date_range})")
            plt.xlabel("Date")
            plt.ylabel("Number of Reviews")
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            filename = f"sentiment_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename)
            plt.close()
            return filename
            
        except Exception as e:
            print(f"Error generating visualization: {e}")
            return None

    def run_interactive(self):
        print("\n" + "="*50)
        print("🍜 SteamNoodles Feedback System")
        print(f"Using model: {self.current_model}")
        print("="*50)
        
        while True:
            print("\nMain Menu:")
            print("1. Add New Review")
            print("2. View Recent Reviews")
            print("3. Generate Sentiment Report")
            print("4. Exit System")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                self._add_review_interactive()
            elif choice == "2":
                self._view_recent_reviews()
            elif choice == "3":
                self._generate_sentiment_report()
            elif choice == "4":
                print("\nThank you for using SteamNoodles Feedback System!")
                break
            else:
                print("\nInvalid choice. Please enter 1-4.")

    def _add_review_interactive(self):
        print("\n" + "-"*50)
        print("Add New Review")
        print("(Leave review blank to cancel)")
        print("-"*50)
        
        review_text = input("\nEnter the customer review: ").strip()
        if not review_text:
            print("Review addition cancelled.")
            return
            
        date_input = input("Enter date (YYYY-MM-DD) or leave blank for today: ").strip()
        
        print("\nProcessing review...")
        result = self.add_review(review_text, date_input if date_input else None)
        
        if "error" in result:
            print(f"\n❌ Error: {result['error']}")
        else:
            print("\n✅ Review added successfully!")
            print(f"\nReview ID: {result['review_id']}")
            print(f"Date: {result['date']}")
            print(f"Sentiment: {result['sentiment'].upper()}")
            print(f"\nGenerated Response:\n{result['response']}")

    def _view_recent_reviews(self, num_reviews: int = 5):
        recent = self.reviews_df.sort_values('date', ascending=False).head(num_reviews)
        
        print("\n" + "-"*50)
        print(f"📋 Last {len(recent)} Reviews")
        print("-"*50)
        
        if recent.empty:
            print("No reviews found.")
            return
            
        for _, row in recent.iterrows():
            print(f"\nID: {row['review_id']} | Date: {row['date'].date()}")
            print(f"Sentiment: {row['sentiment'].upper()}")
            print(f"Review: {row['text']}")
            print(f"Response: {row['response']}")

    def _generate_sentiment_report(self):
        print("\n" + "-"*50)
        print("📊 Sentiment Analysis Report")
        print("-"*50)
        
        print("\nAvailable date range formats:")
        print("- 'last 7 days' (default)")
        print("- 'last 30 days'")
        print("- 'YYYY-MM-DD to YYYY-MM-DD' (custom range)")
        
        date_range = input("\nEnter date range: ").strip() or "last 7 days"
        
        print(f"\nGenerating report for {date_range}...")
        plot_file = self.sentiment_visualization_agent(date_range)
        
        if plot_file:
            print(f"\n✅ Report generated: {plot_file}")
            
            if date_range.startswith("last"):
                days = int(date_range.split()[1])
                start_date = datetime.now() - timedelta(days=days)
            elif "to" in date_range:
                start_str, end_str = date_range.split("to")
                start_date = datetime.strptime(start_str.strip(), "%Y-%m-%d")
                end_date = datetime.strptime(end_str.strip(), "%Y-%m-%d")
            else:
                start_date = datetime.now() - timedelta(days=7)
            
            filtered = self.reviews_df[self.reviews_df['date'] >= start_date]
            if "end_date" in locals():
                filtered = filtered[filtered['date'] <= end_date]
            
            if not filtered.empty:
                print("\nSentiment Distribution:")
                print(filtered['sentiment'].value_counts())
            else:
                print("No reviews in selected date range.")
        else:
            print("\n❌ Failed to generate report.")



if __name__ == "__main__":
    system = SteamNoodlesFeedbackSystem()
    system.run_interactive()