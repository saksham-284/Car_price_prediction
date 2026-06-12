# CAR PRICE PREDICTION PROJECT
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
df=pd.read_csv("car.csv")
print(df.shape) 
print(df.dtypes)
print(df.isnull().sum())
df=df.drop(columns=["ID"])
df["Mileage"]=df["Mileage"].str.replace("km","",case=False)
df["Mileage"]=df["Mileage"].str.extract(r'([\d.]+)').astype(float)
df["Levy"]=pd.to_numeric(df["Levy"],errors="coerce")
df=df[df["Mileage"]<2147482647]
df=df[df["Price"]<1000000]
df=df[df["Mileage"]<1000000]
df["Doors"]=df["Doors"].replace({'04-May':"4-5","02-Mar":"2-3"})
df["Turbo"]=df["Engine volume"].str.contains("Turbo",case=False,na=False).astype(int)
df["Engine volume"]=df["Engine volume"].str.extract(r'([\d.]+)')
df["Engine volume"]=df["Engine volume"].astype(float)
X=df.drop(columns=["Price"])
y=(df["Price"])
num=X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat=X.select_dtypes(include=["object"]).columns.tolist()
num_pipe=Pipeline([("impute",SimpleImputer(strategy="median"))])
cat_pipe=Pipeline([("ohe",OneHotEncoder(handle_unknown="ignore"))])
processing=ColumnTransformer([("num",num_pipe,num),("cat",cat_pipe,cat)])
model=Pipeline([("processing",processing),("model",XGBRegressor())])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
params={"model__max_depth":[3,5,7,10,15],"model__n_estimators":[100,200,300,500],"model__learning_rate}":[0.01,0.1],"model__subsample":{0.8,1.0}}
random=RandomizedSearchCV(model,params,cv=5,scoring="r2")
random.fit(X_train,y_train)
y_pred=random.predict(X_test)
y_train_pred=random.predict(X_train)
dummy=DummyRegressor()
dummy.fit(X_train,y_train)
dumm_pred=dummy.predict(X_test)
print("train",root_mean_squared_error(y_train,y_train_pred))
print("test",root_mean_squared_error(y_test,y_pred))
print("dummy",root_mean_squared_error(y_test,dumm_pred))
print(random.best_params_)
print(random.best_score_)
print(random.cv_results_["std_test_score"][random.best_index_])
