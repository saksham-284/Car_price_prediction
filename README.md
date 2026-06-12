# Car_price_prediction
predicts car price
what i did
cleaned this messy data set extract numeric values from engine volume (turbo variants), mileage,doors
created binary turbo feature fro   engine volume column
identified and removed extreme outliers/impossible values (mileage in billions and single car price in million) using skewness and matplotlib
built sklearn pipeline with columntrasformer and to avoid data leakage
tuned XGBoost and Randomforest compared their results tuned hyperparameters with Randomizer {cv=5)
best cv R2:0.687
Baseline (DummyRegressor):19125

libraries
pandas,scikit learn, num py
