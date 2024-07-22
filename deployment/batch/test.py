
from predict import apply_model

input_file = 'https://raw.githubusercontent.com/Tiamz01/data_repo/main/waterQuality1.csv'
output_file = 'output/predictions.csv'

df_result = apply_model(input_file=input_file, output_file=output_file)
print(df_result)


# def load_model():
#     try:
#        with open('best_log_reg.bin', 'rb') as f_in:
#         model = pickle.load(f_in)
#     except Exception as e:
#         print("Credentials error:", e)
#         raise
#     return model

# def apply_model(input_file, output_file):
#     df = get_and_clean_data(input_file)
    
#     if 'is_safe' not in df.columns:
#         raise KeyError("'is_safe' column is missing from the data.")
    
#     X_train, X_test, y_train, y_test = feature_engineering(df)
#     model = load_model()
#     y_pred = model.predict(X_test)
#     df_result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#     # df_result['model_version'] = run_id

#     output_dir = os.path.dirname(output_file)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     df_result.to_csv(output_file, index=False)

#     return df_result
 
