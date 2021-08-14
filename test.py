from joblib import load

model = load('house_price_pred_model.pkl')
price = model.predict([[2,5,1,2]])
print(price)