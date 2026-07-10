import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tt
from sklearn.preprocessing import StandardScaler as ss
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as msm, mean_absolute_error as mm , r2_score
import numpy as np
import joblib as jb




p = Path(r"/home/satyamchauhan/Documents/vs_code_2022/sem_2_advanced_python/lab_11/converter/employee_salary_regression.csv")
data = pd.read_csv(p)

print(data.head())
for col in ['education_level' ,   'job_role']:
    print("=" * 50)
    print(data[col].value_counts())

#sns.histplot(data = data , x = 'annual_salary_usd' , kde =True , color = 'steelblue')
#sns.histplot(data =data , x = 'city_tier' , kde = True , color ='steelblue')
#sns.heatmap(data.corr(numeric_only = True), annot = True , cmap= 'coolwarm')
print(data.select_dtypes(include='str').columns)

#data.select_dtypes(include='object')      # text columns , in newer python we use str
#data.select_dtypes(include='number')      # int + float columns
#data.select_dtypes(include='int64')       # only integers
#data.select_dtypes(include='float64')     # only floats
#data.select_dtypes(include=['int64', 'float64'])  # both, explicit
#data.select_dtypes(exclude='object')      # everything except text
dg = pd.DataFrame({
    'id':[1,2,3,4],
    'gender':['m','f','m','f'],
    'color':['b','g','y','r']
})
df_encoded = pd.get_dummies(data.drop(columns = 'employee_id'),
                            columns=['education_level','job_role'],
                            drop_first=True
                            )
print(df_encoded.head())


x =df_encoded.drop(columns=['annual_salary_usd'])
y = df_encoded['annual_salary_usd']
x_train,x_test,y_train,y_test = tt(x,y,random_state=42)
sd = ss()
x_ss_train = sd.fit_transform(x_train)
x_ss_test = sd.transform(x_test)

ld = LinearRegression()
ld.fit(x_ss_train,y_train)
y_pred = ld.predict(x_ss_test)

print(f"{r2_score(y_pred,y_test):.4f}")
print(f"{msm(y_pred,y_test):.2f}")

#plt.scatter(y_test,y_pred,alpha = 0.5)
#plt.plot([y_test.min(),y_test.max()],[y_test.min() , y_test.max()],'--',c='r')
plt.show()

#jb.dump(ld, 'salary_model.pkl')
