from src.predict import predict_sentiment

print("🔍 Sentiment Analysis System")
print("Type 'exit' to quit\n")

while True:
    text = input("Enter review: ")
    
    if text.lower() == 'exit':
        print("👋 Exiting...")
        break
    
    result = predict_sentiment(text)
    print("👉 Sentiment:", result, "\n")