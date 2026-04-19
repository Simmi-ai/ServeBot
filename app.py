from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Data
questions = [
    "hello",
    "how are you",
    "order status",
    "refund policy",
    "contact support",
    "price of product"
]

answers = [
    "Hello! How can I assist you?",
    "I'm just a bot, but I'm here to help!",
    "Please provide your order ID.",
    "Refunds take 5-7 working days.",
    "You can email us at support@example.com",
    "Prices are listed on our website."
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(questions)

def get_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    index = np.argmax(similarity)

    if similarity[0][index] > 0.3:
        return answers[index]
    else:
        return "Sorry, I didn't understand that."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_input = request.json.get("message")
    response = get_response(user_input.lower())
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(debug=True)