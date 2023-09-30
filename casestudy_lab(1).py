import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
###############################
data=pd.read_csv('insurance (1) (2).csv')
###############################
data.columns 
data.describe()
data.info()
data.isna().sum() 
sns.heatmap(data.loc[:,('age','bmi','children','smoker','charges')].corr(),annot=True)
###############################

plt.scatter(x=data['age'],y=data['charges']) 
plt.scatter(x=data['bmi'],y=data['charges']) 



sns.boxplot(x=data['sex'],y=data['charges']) 
sns.boxplot(x=data['smoker'],y=data['charges']) 
sns.boxplot(x=data['region'],y=data['charges']) 
sns.boxplot(x=data['children'],y=data['charges']) 
#################################

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
columns=['sex','smoker','region']
for column in columns:
    data[column]=encoder.fit_transform(data[column])
#################################

x=data.drop(['charges'],axis=1)
y=data['charges']
#################################


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
#################################

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(x_train,y_train)
#################################

regressor.coef_ 
regressor.intercept_ 
#################################

y_pred=regressor.predict(x_test)

from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_test, y_pred)) # 5663.358417062193
metrics.mean_absolute_error(y_test, y_pred) # 3998.2715408869726
metrics.r2_score(y_test, y_pred) # 0.7962732059725786









