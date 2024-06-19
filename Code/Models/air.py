import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('airdata.csv',encoding='ISO-8859-1')
print(df.shape)

df1 = df[['state','so2','no2','rspm','spm']]
# print(df1.head())
# print(df1.isnull().sum())

res1 = df1['rspm'].mean()
res2 = df1['so2'].mean()
res3 = df1['no2'].mean()

df1['rspm'] = df1['rspm'].fillna(0)
df1['so2'] = df1['so2'].fillna(0)
df1['no2'] = df1['no2'].fillna(0)
df1['spm'] = df1['spm'].fillna(0)
# print(df1.isnull().sum())
mini = df1['spm'].min()
maxi = df1['spm'].max()
print(mini,maxi)


q1 = df1['so2'].quantile(0.25)
q3 = df1['so2'].quantile(0.75)
iqr1 = q3-q1
low1 = q1 - 1.5 * iqr1
up1 = q3 + 1.5 * iqr1
df2 = df1[(df1['so2'] >= low1) & (df1['so2'] <= up1)]
# print(df2.describe())


q2 = df1['no2'].quantile(0.25)
q4 = df1['no2'].quantile(0.75)
iqr2 = q4-q2
low2 = q2 - 1.5 * iqr2
up2 = q4 + 1.5 * iqr2
df2 = df1[(df1['no2'] >= low2) & (df1['no2'] <= up2)]
# print(df2.describe())


q5 = df1['spm'].quantile(0.25)
q6 = df1['spm'].quantile(0.75)
iqr3 = q6-q5
low3 = q5 - 1.5 * iqr3
up3 = q6 + 1.5 * iqr3
df2 = df1[(df1['spm'] >= low3) & (df1['spm'] <= up3)]
# print(df2.describe())


q7 = df1['rspm'].quantile(0.25)
q8 = df1['rspm'].quantile(0.75)
iqr4 = q8-q7
low4 = q7 - 1.5 * iqr4
up4 = q8 + 1.5 * iqr4
df2 = df1[(df1['rspm'] >= low4) & (df1['rspm'] <= up4)]
# print(df2.describe())


# plt.figure(figsize=(30,10))
# plt.xticks(rotation = 90)
# df2.state.hist()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# plt.figure(figsize=(10,10))
# plt.xticks(rotation = 90,fontsize = 20)
# sns.barplot(x = 'state',y = 'rspm', data=df2)
# plt.xlabel('states',fontsize = 35)
# plt.ylabel('rspm',fontsize = 35)
# plt.show()



def cal_soi(so2):
    si = 0
    if(so2 <= 40):
        si = so2*(50/40)
    elif(so2 > 40 and so2 <= 80):
        si = 50 + (so2-40)*(50/40)
    elif(so2>80 and so2<=380):
        si = 100 + (so2 - 80)*(100/300)
    elif(so2>380 and so2<=800):
        si = 200 + (so2-380)*(100/420)
    elif(so2>800 and so2<=1600):
        si = 300 + (so2-800)*(100/800)
    return si

df2['soi'] = df2['so2'].apply(cal_soi)
# print(df2[['soi','so2']].head())


def cal_noi(no2):
    ni = 0
    if(no2 <= 40):
        ni = no2*(50/40)
    elif(no2 > 40 and no2 <= 80):
        ni = 50 + (no2-40)*(50/40)
    elif(no2>80 and no2<=180):
        ni = 100 + (no2 - 80)*(100/100)
    elif(no2>180 and no2<=280):
        ni = 200 + (no2-180)*(100/100)
    elif(no2>280 and no2<=400):
        ni = 300 + (no2-280)*(100/120)
    else:
        ni = 400 + (no2 - 400) * (100/120)
    return ni

df2['noi'] = df2['no2'].apply(cal_noi)
# print(df2[['noi','no2']].head())



def cal_roi(rspm):
    rpi = 0
    if(rspm <= 30):
        rpi = rspm*(50/30)
    elif(rspm > 30 and rspm <= 60):
        rpi = 50 + (rspm-30)*(50/30)
    elif(rspm>60 and rspm<=90):
        rpi = 100 + (rspm - 60)*(100/30)
    elif(rspm>90 and rspm<=120):
        rpi = 200 + (rspm-90)*(100/30)
    elif(rspm>120 and rspm<=250):
        rpi = 300 + (rspm-120)*(100/130)
    else:
        rpi = 400 + (rspm - 250) * (100/130)
    return rpi

df2['roi'] = df2['rspm'].apply(cal_roi)
# print(df2[['roi','rspm']].head())


def cal_spi(spm):
    spi = 0
    if(spm <= 50):
        spi = spm*(50/50)
    elif(spm > 50 and spm <= 100):
        spi = 50 + (spm-50)*(50/50)
    elif(spm>100 and spm<=250):
        spi = 100 + (spm - 100)*(100/150)
    elif(spm>250 and spm<=350):
        pi = 200 + (spm-250)*(100/100)
    elif(spm>350 and spm<=430):
        spi = 400 + (spm-430)*(100/430)
    return spi

df2['spmi'] = df2['spm'].apply(cal_spi)
# print(df2[['spmi','spm']].head())


def cal_aqi(si,ni,rpi,spmi):
    aqi = 0
    if(si>ni and si>rpi and si>spmi):
        aqi = si
    elif(ni>si and ni>rpi and ni>spmi):
        aqi = ni
    elif(rpi>si and rpi >ni and rpi>spmi):
        aqi = rpi
    elif(spmi>si and spmi>ni and spmi>rpi):
        aqi = spmi
    return aqi

df2 = df2.assign(aqi = df2.apply(lambda x:cal_aqi(x['soi'],x['noi'],x['roi'],x['spmi']),axis = 1))
df3 = df2[['state','soi','noi','roi','spmi','aqi']]
# print(df3.head())



from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# print(df3.columns)

x = df3[['soi','noi','roi','spmi']]
y = df3['aqi']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=50)

# train_size = len(x_train)
# test_size = len(x_test)

# sizes = [train_size, test_size]
# labels = ['Training Data (80%)', 'Testing Data (20%)']
# colors = ['skyblue', 'lightcoral']

# Plotting the pie chart
# plt.figure(figsize=(8, 8))
# plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
# plt.title('Proportion of Training and Testing Data')
# plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.show()
# # print(x_train.shape)
# print(x_test.shape)



linear = LinearRegression()
linear.fit(x_train, y_train)
linear_pred = linear.predict(x_test)
linear_score = linear.score(x_test, y_test)*100
print(f'Linear Regression R2 Score: {linear_score:.2f}')



import joblib
joblib.dump(linear,'air_quality.joblib')
import os
print(os.getcwd())


# user input
# def predict_aqi(user_input_so2, user_input_no2, user_input_rspm, user_input_spm):
#     soi = cal_soi(user_input_so2)
#     noi = cal_noi(user_input_no2)
#     roi = cal_roi(user_input_rspm)
#     spmi = cal_spi(user_input_spm)
    
#     input_data = [[soi, noi, roi, spmi]]
#     predicted_aqi = linear.predict(input_data)[0]
    
#     return predicted_aqi


# user_input_so2 = 4
# user_input_no2 = 11
# user_input_rspm = 12
# user_input_spm = 10

# predicted_aqi = predict_aqi(user_input_so2, user_input_no2, user_input_rspm, user_input_spm)
# print(f'Predicted AQI: {predicted_aqi}')
