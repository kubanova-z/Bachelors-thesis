from src.data_loader import load_data
from src.preprocess import prepare_data
from src.ML.train_NN import train_model
from src.predict import predict_single_text
from src.analysis import  intra_category_similarity
from src.plotting import plot_confusion_matrix, plot_learning_curve, plot_metrics_bar_chart
from src.ML.train_svm import train_SVM
from src.ML.train_rf import train_random_forest


if __name__ == "__main__":

    #load data
    print("--- 1. Loading and Preparing Data ---")
    df = load_data("/home/xkubanova_126831/bakalarka/e_com_dataset/data/ecommerceDataset.csv")


    #prepare data
    X_train, X_test, y_train, y_test, vectorizer = prepare_data(df)

    #SVM MODEL
    target_names = sorted(list(y_train.unique()))
    svm_model = train_SVM(X_train, y_train, X_test, y_test, target_names)

    #RANDOM FOREST
    rf_model = train_random_forest(X_train, y_train, X_test, y_test, target_names)


    #NN MODEL
    #print("\n--- 2. Training Model ---")
    #model, class_to_idx = train_model(X_train, y_train, X_test, y_test, epochs=30)

   
    
    # Calculate and print the average similarity within each category
    #intra_category_similarity(X_train, y_train, class_to_idx)
   
    
"""  
    # --- Interactive Manual Input  ---
    print("\n" + "="*50)
    print("MANUAL CLASSIFICATION INPUT")
    print("="*50)

    # loop for user input
    while True:

        user_text = input("Enter a product description (or type 'quit' to exit): ")
        
        if user_text.lower() == 'quit':
            break

        if user_text.strip() == "":
            print("Please enter some text.")
            continue
            
        # Classify the users text
        predicted_class_manual = predict_single_text(user_text, model, vectorizer, class_to_idx)
        
        print("\n--- CLASSIFICATION RESULT ---")
        print(f"Input Text: {user_text}")
        print(f"Predicted Category: {predicted_class_manual}\n")

    print("Model testing finished.") """
 