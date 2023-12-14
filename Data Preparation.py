#!/usr/bin/env python
# coding: utf-8

# # Find publicly available data for key factors that influence US home prices nationally. Then,build a data science model that explains how these factors impacted home prices over the last 20years

# **Approach** - Fllowing are the factors choosen for the study
# - CaseShiller(House Price) , https://fred.stlouisfed.org/series/CSUSHPISA
# - Interest rates , https://fred.stlouisfed.org/series/FEDFUNDS 
# - Unemployment rates , https://fred.stlouisfed.org/series/UNRATE
# - Monthly Supply ,  https://fred.stlouisfed.org/series/DSPIC96
# - Median Income , https://fred.stlouisfed.org/series/MSACSR
# - Per Capita GDP ,https://fred.stlouisfed.org/series/A939RX0Q048SBEA
# - New Construct units , https://fred.stlouisfed.org/series/COMPUTSA
# - Construction price index (**CPI**) , https://fred.stlouisfed.org/series/WPUSI012011
# - Consumer Price index , https://fred.stlouisfed.org/series/CORESTICKM159SFRBATL
# - urban population (https://data.worldbank.org/indicator/SP.URB.TOTL.IN.ZS?end=2023&locations=US&start=2002)
# - Housing Subsidy , https://fred.stlouisfed.org/series/L312051A027NBEA
# - Working Population , https://fred.stlouisfed.org/series/LFWA64TTUSM647S
# - Total Household , https://fred.stlouisfed.org/series/TTLHH

# As **S&P** is used as a proxy for home price , Most of the data downloaded from
# - http://fred.stlouisfed.org
# - Data for all the factors has downloaded , processed and consolidated below to ceate a data set.as the frequency of all the factor varies so modification has done to get the proper dataset.
# - As we have downloaded all the data from different link  , Combined all the data with the help of advanced excel

# ### Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler ,MinMaxScaler
from sklearn.ensemble import RandomForestRegressor , ExtraTreesRegressor, AdaBoostRegressor , ExtraTreesRegressor , GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split , GridSearchCV , cross_val_score , RandomizedSearchCV
from sklearn.metrics import r2_score


# In[2]:


df=pd.read_csv("Housing Data.csv")


# In[3]:


df.tail()


# # Lets check the shape and data types

# In[4]:


df.shape


# As we can see there is 252 rows and 14 columns (1 dependent and 13 independent columns).As the Label column is having continous data hence preparing Regression model for the prediction

# In[5]:


df.dtypes


# As per observation data type for date column should be in datetime but it is in Object so let's change the same

# In[6]:


df["Date"]=pd.to_datetime(df["Date"])


# In[7]:


df.dtypes


# Now we have changed the data type of Date column

# # Checking Missing Values

# In[7]:


df.isna().sum()/len(df)


# - As Per Capita GDP data is quarterly available we can fill the null values with the help of Linear Interpolation() 
# - We will drop the missing value rows that are less then 4% which will not impact to our data . We will have the data from 2002 to 2022. 

# In[8]:


df["Per Capita Gdp"]=df["Per Capita Gdp"].interpolate()


# In[9]:


df.dropna(inplace= True)


# In[10]:


df.isnull().sum()


# Now we have removed all the null values from the data set

# # Checking Duplicate value

# In[11]:


df.duplicated().sum()


# Hence there is no duplicate value present in the data

# In[12]:


df.shape


# # Checking correlation between the factors

# In[13]:


cor=df.corr()


# In[14]:


fig,ax=plt.subplots(figsize=(10,8))
sns.heatmap(cor,ax=ax,annot=True)


# - As per above observation Unemploment has Negative correlation with many of the factors like Interest Rates , New Construction Units ,Consumer Prie index . 
# - Urban population has positive coorelation with many of the factors like Income,Per capita GDP , Construction Price index , Housing Subsidy , Working population , Total Households , House price
# - New Construct Units has negative correlation with Monthly income , Unemployment and working population as much newly construct will increase these all factors will get decreased

# # Graphical analysis

# In[15]:


y=df.pop("House Price")
x=df
# Plotting scatter plots of the CASE-SHILLER index vs features

for i in x.columns:
    plt.figure()
    plt.scatter(x = x[i], y = y)
    plt.xlabel(i)
    plt.ylabel("CASE-SHILLER")
    plt.title(f"CASE-SHILLER vs {i}")


# - As per above graphical explanation New Construct Units , unemployment , Monthly Supply , Income , Interest Rates are negatively related with House Price
# - Rest all the factor are positively related with House price

# In[16]:


sns.pairplot(data=df)


# - As per above presentation Interest Rate is having positive coorelation and Increment with all the factor exccept Unemployment , Monthly Supply and Income
# - Unemployment is getting Down in every factor.
# - Monthly Supply is getting increased as comparing from last 20 years data
# - Income has negative correlation with New Construct Units however it is having postive realtion with rest all the factor
# - Rest all the factors are having positive coorelation between each other

# In[17]:


# Let's convert other variables to numerical variable

x["Date"]=pd.to_numeric(df["Date"],errors='coerce')


# # Feature Scalling

# In[18]:


MS=MinMaxScaler()
x=MS.fit_transform(x)


# In[19]:


def my_model(model,x,y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
    model.fit(x_train,y_train)
    pred=model.predict(x_test)
    R2_score=r2_score(y_test,pred)
    print(f"{model} R2 Score is : {R2_score}")
    
    score=cross_val_score(model,x,y,cv=5)
    print(score)
    print("The Average cross validation Score is " ,score.mean())
    print("The difference between cross validation score and R2 score is ",score.mean()-R2_score)


# In[20]:


models=[RandomForestRegressor(),LinearRegression(),AdaBoostRegressor(),ExtraTreesRegressor(),DecisionTreeRegressor(),
        GradientBoostingRegressor()]
for model in models:
    my_model(model,x,y)
    print("*"*50)


# As per above Model validation We have choosen LinearRegression as the best Model

# # Hyper Parameter Tunning

# In[21]:


param={"fit_intercept":[True],
    "copy_X":[True],
    "n_jobs":[None],
    "positive":[False]}


# In[22]:


grid=RandomizedSearchCV(estimator=LinearRegression(),param_distributions=param,n_iter=20,verbose=True)


# In[23]:


grid.fit(x,y)


# # Saving Model

# In[24]:


model=LinearRegression(copy_X=[True],
                        fit_intercept= [True],
                            n_jobs=[None], 
                       positive= [False])


# In[25]:


model.fit(x,y)


# In[26]:


import joblib
joblib.dump(model,"House price Prediction")


# In[27]:


data=joblib.load("House price Prediction")


# In[28]:


predict=data.fit(x,y)


# In[29]:


predict


# In[31]:


r=np.array(y_test)
data=pd.DataFrame()
data["Original"]=r
data["Prediction"]=prediction
data


# In[ ]:




