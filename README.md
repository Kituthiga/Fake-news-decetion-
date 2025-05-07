import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Sample dataset with headlines and full article content
data = {
    'headline': [
        "NASA successfully lands spacecraft on Mars",
        "World Health Organization declares end of pandemic",
        "UN signs historic climate change agreement",
        "Electric vehicle sales hit record high this year",
        "Scientists discover new planet in our solar system",
        "Aliens spotted in White House basement",
        "Drinking bleach cures COVID-19, says study",
        "Time traveler arrested in New York for future crimes",
        "Vampires confirmed to exist by NASA insider",
        "Government replaces birds with drones to spy on citizens"
    ],
    'article': [
        "NASA's latest mission to Mars successfully landed with new technology, confirming data transmission from the surface.",
        "The WHO has officially ended the pandemic status after global vaccinations proved effective in containing the virus.",
        "The United Nations has reached a groundbreaking agreement on limiting carbon emissions to fight climate change.",
        "A new report shows electric vehicle sales have reached an all-time high, especially in Europe and Asia.",
        "Astronomers using advanced telescopes discovered a new celestial body orbiting beyond Neptune.",
        "A conspiracy theory claims aliens live in the basement of the White House; no evidence has been provided.",
        "An unverified study circulating online falsely claims that bleach can cure COVID-19, which experts deny.",
        "A man claiming to be from the year 3021 was arrested in New York for predicting several events correctly.",
        "A hoax article claims that NASA admitted the existence of vampires after a leaked video appeared online.",
        "According to an internet rumor, birds are not real and have been replaced by surveillance drones by the government."
    ],
    'label': [1, 1, 1, 1, 1,   # Real
              0, 0, 0, 0, 0]   # Fake
}

# Step 2: Create a DataFrame
df = pd.DataFrame(data)

# Step 3: Preprocess the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[].*?[]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df['text'] = df['headline'] + " " + df['article']
df['text'] = df['text'].apply(clean_text)

# Step 4: NLP Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Predict user input
def predict_news(headline, article):
    combined = clean_text(headline + " " + article)
    vec = vectorizer.transform([combined])
    result = model.predict(vec)[0]
    return "REAL NEWS" if result == 1 else "FAKE NEWS"

# Step 9: Get user input
print("\n--- News Input ---")
user_headline = input("Enter the news headline: ")
user_article = input("Enter the full article content: ")

# Step 10: Show result
print("\n--- Analysis Result ---")
prediction = predict_news(user_headline, user_article)
print(f"The system detected this as: {prediction}")
