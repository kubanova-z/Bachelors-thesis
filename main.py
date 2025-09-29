from src.data_loader import load_data
from src.preprocess import prepare_data
from src.train import train_model
from src.predict import predict_single_text

if __name__ == "__main__":
    print("--- 1. Loading and Preparing Data ---")
    df = load_data("data/ecommerceDataset.csv")

    X_train, X_test, y_train, y_test, vectorizer = prepare_data(df)

    print("\n--- 2. Training Model (30 Epochs) ---")
    model, class_to_idx = train_model(X_train, y_train, X_test, y_test, epochs=30)

    # --- Initial Test of a Random Sample ---
    random_row = df.sample(n=1, random_state=42)
    sample_text = random_row["Text"].iloc[0]
    true_category = random_row["Category"].iloc[0]

    print("\n" + "#"*50)
    print("RANDOM SAMPLE TEST")
    print("#"*50)

    print(f"Sample Text: {sample_text[:100]}...")
    print(f"Actual Category: {true_category}")
    predicted_class = predict_single_text(sample_text, model, vectorizer, class_to_idx)
    print(f"Predicted Category: {predicted_class}")
    
    
    # --- Interactive Manual Input (NEW FUNCTIONALITY) ---
    print("\n" + "="*50)
    print("MANUAL CLASSIFICATION INPUT")
    print("="*50)

    # Loop to allow continuous classification until the user quits
    while True:

        user_text = input("Enter a product description (or type 'quit' to exit): ")
        
        if user_text.lower() == 'quit':
            break

        if user_text.strip() == "":
            print("Please enter some text.")
            continue
            
        # Classify the user's text
        predicted_class_manual = predict_single_text(user_text, model, vectorizer, class_to_idx)
        
        print("\n--- CLASSIFICATION RESULT ---")
        print(f"Input Text: {user_text}")
        print(f"Predicted Category: {predicted_class_manual}\n")

    print("Model testing finished.")
    # The print(model) line is left out to keep the interactive session clean, 
    # but you can add it back if you wish to inspect the layers.