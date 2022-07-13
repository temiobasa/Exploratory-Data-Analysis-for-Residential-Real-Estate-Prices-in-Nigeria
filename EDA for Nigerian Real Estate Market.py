#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis for the Nigerian Property Market
# ### Author: Temitope Obasa

# ## Introduction

# #### This Exploratory Data Analysis project was carried out to gain insights on the Nigerian Property Market.
# #### The Federal Republic of Nigeria, is a country in West Africa and the most populous country in Africa with a population of over 216 million. Nigeria borders Niger in the north, Chad in the northeast, Cameroon in the east, and Benin in the west. Nigeria consists of 36 Federal states and the Federal Capital Territory, Abuja. Lagos is the largest city in Nigeria and the second-largest in Africa.

# #### The dataset used was obtained from Kaggle (https://www.kaggle.com/datasets/abdullahiyunus/nigeria-houses-and-prices-dataset). It contains data on Lagos Property prices scrapped by Abdullahi Yunus ( https://www.kaggle.com/abdullahiyunus).

# ## Business Problem

# #### In order gain a clear understanding of the Nigerian Real Estate market, an Exploratory Data Analysis project is necessary. This EDA project focuses on;
# #### - property prices in Nigeria.
# #### - property types in the Nigerian Real Estate Market
# #### - common features of properties in the Nigerian Real Estate Market.
# #### - factors that influence property prices in the Nigerian Real Estate Market.

# ### Categorical Features

# #### Bedrooms - Number of bedrooms
# #### Toilets - Number of toilets
# #### Bathrooms - Number of bathrooms
# #### State - Federal State
# #### Title - Property type
# #### Town - Town in a federal state

# ### Importing Libraries

# In[2]:


# Import essential libraries for Data Analysis

import pandas as pd
import numpy as np
import statistics as st

# Import essential library for splitting datetime
import datetime as dt

# Importing essential library for downloading the dataset from Kaggle
# import opendatasets as od

# Import essential libraries for Exploratory Data Analysis and Visualization
import matplotlib as ploty
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mode
from scipy.stats import iqr
import plotly.express as px

# Ignore Warnings

import warnings
warnings.filterwarnings("ignore")


# ## Exploring the Dataset

# ### Loading the Dataset

# In[3]:


# This loads the dataset and prints the first five rows in the dataframe
df = pd.read_csv("/Users/temi/Datasets/nigeria_houses_data.csv")
df.head(5)


# In[4]:


df.tail()


# In[5]:


df.shape
print(f"It consists of", df.shape[1], "columns and", df.shape[0], "rows.")


# In[6]:


df.info()


# In[7]:


df.describe()


# ### Distribution of Features 

# In[8]:


# Plotting histogram

df.hist(figsize=(16, 16), xrot=90)

plt.show()


# ### Checking Columns

# In[9]:


df.columns


# ### Categorical features

# In[10]:


# Summarize the categorical features

df.describe(include=["object"])


# ### Federal States in Nigeria

# In[11]:


# Identify the states represented in the dataframe

df["state"].unique()


# #### Although there are 36 states (including the national capital) in Nigeria, the dataset only contains data on 25 states.

# ### Towns

# In[12]:


# Identify the towns represented in the dataframe

df["town"].unique()


# ### Property Types

# In[13]:


df["title"].unique()


# ## Data Cleaning 
# 
# #### Data cleaning involves the process of identifying inaccurate, incomplete or irrelevant parts of the data. This is carried out to prepare the data for analysis or storing in a database. It includes deleting missing values to removing irrelevant rows or columns.

# ### Check for duplicates

# In[14]:


df.duplicated().sum()


# ### Remove the duplicates

# In[15]:


df.drop_duplicates(keep="first", inplace=True)
df.shape

print(f"The dataframe now consists of", df.shape[1],
      "columns and", df.shape[0], "rows.")


# #### By removing the duplicates, 42% of the datapoints are lost. This could be an indication of inaccurate and inefficient data collation techniques.

# ### Checking for missing values

# In[16]:


df.isnull().sum()


# #### There are no missing values in the dataframe.

# ## Outliers

# ### Identifying Outliers
# 
# #### The target column "price" will be checked for potential outliers.

# In[17]:


df[["price"]].describe()


# #### Visualise

# In[18]:


# Plotting a boxplot to visualise the outliers in price column

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df["price"], y=df["state"],  palette="Set2",)
sns.stripplot(x=df["price"], y=df["state"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# #### The box plot above show majority of the dataoiints on the left side, however, there some datapoints on the that appear distant from the other datapoints as you begin to scan towards the right part of the chart. These point indicate the presence of Outliers. These outliers are properties with significantly higher prices when compared to the property prices of other federal states within Nigeria. It shows that "Lagos" and "Abuja" appear to have these outliers. 

# ### InterQuantileRange (IQR)

# In[19]:


# Calculating q1 and q3 for price
Q1 = df.price.quantile(0.25)
Q3 = df.price.quantile(0.75)
Q1, Q3


# In[20]:


# Calculating InterQuantileRange (IQR) for price
IQR = Q3 - Q1

print("The Interquartile Range for price is", IQR)


# ### Lower and Upper Limit

# In[21]:


# For price
lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
lower_limit, upper_limit


# ### Identifying the Outliers

# In[22]:


price_outliers = df[(df.price < lower_limit)
                    | (df.price > upper_limit)]

price_outliers.shape


# ### Removing the Outliers

# #### Check the shape before removing the outliers

# In[23]:


Nigeria = df.shape
Nigeria


# #### Remove the outliers using the IQR

# In[24]:


df_outlier_free = df[(df.price > lower_limit)
                     & (df.price < upper_limit)]
df_outlier_free.shape


# #### Visualise to validate changes 

# In[25]:


# Plotting scatter plot for living space

plt.scatter(x=df_outlier_free["price"],
            y=df_outlier_free["price"], color="green",)
plt.show()


# In[26]:


# Plotting a boxplot to validate changes

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df_outlier_free["price"],
            y=df_outlier_free["state"],  palette="Set2",)
sns.stripplot(x=df_outlier_free["price"],
              y=df_outlier_free["state"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# #### There appear to be no outliers left.

# #### Re-plotting the box plot indicates the absence of the outliers initially spotted. However, this chart reveals that "Abuja" and "Lagos" have properties with significantly higher prices than other states within Nigeria. For this reason, the dataset should split into subsets per state and the outliers should be removed per state. This will provide a more accurate analysis.

# ### REAL ESTATE MARKET TREND IN NIGERIA

# ### Distribution of Features of Properties in Nigeria

# In[27]:


# Plotting histogram

df_outlier_free.hist(figsize=(16, 16), xrot=90)

plt.show()


# ### Skewness

# In[28]:


df_outlier_free.skew(axis=1, skipna=True)


# #### The price chart shows that the dataset is positively skewed i.e, there are more properties with extreme prices in the dataset. This raises the average property price.

# ### Average property price in Nigeria

# In[29]:


average_price = st.mean(df_outlier_free["price"])

print("The average property price in Nigeria is " +
      str(round(average_price, 2)) + " Naira")


# #### Visualise

# In[30]:


# Using Groupby to get mean base rent for each state

average_price = df_outlier_free.groupby("state")["price"].mean()
average_price = average_price.rename("mean")

# Combine using concat
compare_states = pd.concat([average_price], axis=1)

# Plot line charts using Plotly library

fig = px.line(compare_states, title="Mean Property Price per State in Nigeria",
              color_discrete_sequence=["red"])

fig.update_layout(yaxis_title="price", legend_title="state", font_size=12)

fig.update_yaxes(rangemode="tozero")

fig.show()


# In[31]:


# Plotting a bar chart

average_price = df_outlier_free.groupby(["state"])["price"]                    .mean().sort_index()

fig = px.histogram(x=df_outlier_free["state"].
                   value_counts().sort_index().index,
                   y=average_price,
                   color=df_outlier_free["state"].
                   value_counts().sort_index().index)

fig.update_xaxes(title="state")
fig.update_yaxes(title="Average Property Price per state in Nigeria")
fig.show()


# #### The above chart shows that the following federal states have the highest property prices:
# #### - Abuja
# #### - Anambara
# #### - Bayelsa
# #### - Borno
# #### - Enugu
# #### - Imo
# #### - Lagos
# #### - Rivers
# #### Borno appears to have the highest property prices. 

# ### Most Common Property type

# In[32]:


most_common = st.mode(df_outlier_free["title"])

print("The most common property type in Nigeria is " +
      str(most_common))


# ### Correlation
# 
# #### Correlation (corr()) represents the relationship between columns in a dataframe. The results are measured in a range; -1 to 1.
#  #### When the result is 1, it indicates a perfect correlation. This means that the values in these columns increase and decrease simultaneously and are interdependent.
#  #### A result of 0.9 indicates a good relationship also. This works as when the result is 1.
#  #### A result of -0.9 also indicate a good relationship. However, in this case, when the values in one column rises, the value in the other column reduces.
#  #### A result of 0.01 indicates a bad relationship. This means that the values in one column cannot be predicted based on the rise or fall of values in the other column.
#  
# #### This would reveal features that are independent and interdependent on other features.

# In[33]:


df_outlier_free.corr()


# In[34]:


# Visualizing the correlations

# Set the size
plt.figure(figsize=(15, 15))

# Plotting a heatmap
sns.heatmap(df_outlier_free.corr()*100, annot=True, fmt='.0f')


# #### Usually in real estate, the price is dependent on certian factors such as location, size, furnishing and features of the property. The heat map shows that the "number of bedrooms", "number of toilets", "parking space", and "number of bathrooms" correlates with the property price. 

# ## EXPLORING EACH FEDERAL STATE
# 
# 
# #### Although the original dataframe has been cleaned and analysed, a deeper look into the dataframe is necessary. This is because the property prices vary vastly between certain federal states such as Lagos and Ogun. For more accurate analysis, it will be beneficial to explore each state individually.

# ### ABIA STATE

# In[35]:


# Selecting data on only Abia

df_abi = df[df["state"] == "Abia"].reset_index(drop=True)
df_abi.head(3)


# In[36]:


df_abi.shape
print(f"The new dataframe has", df_abi.shape[0],
      "rows and", df_abi.shape[1], "columns.")


# ### ABUJA

# In[37]:


# Selecting data on only Abuja

df_abj = df[df["state"] == "Abuja"].reset_index(drop=True)
df_abj.head(3)


# In[38]:


df_abj.shape
print(f"The new dataframe has", df_abj.shape[0],
      "rows and", df_abj.shape[1], "columns.")


# ### AKWA IBOM

# In[39]:


# Selecting data on only Akwa Ibom

df_akw = df[df["state"] == "Akwa Ibom"].reset_index(drop=True)
df_akw.head(3)


# In[40]:


df_akw.shape
print(f"The new dataframe has", df_akw.shape[0],
      "rows and", df_akw.shape[1], "columns.")


# ### ANAMBARA
# #### Describe it 

# In[41]:


# Selecting data on only Anambara

df_ana = df[df["state"] == "Anambara"].reset_index(drop=True)
df_ana.head(3)


# In[42]:


df_ana.shape
print(f"The new dataframe has", df_ana.shape[0],
      "rows and", df_ana.shape[1], "columns.")


# ### BAYELSA

# In[43]:


# Selecting data on only Bayelsa

df_bay = df[df["state"] == "Bayelsa"].reset_index(drop=True)
df_bay.head(3)


# In[44]:


df_bay.shape
print(f"The new dataframe has", df_bay.shape[0],
      "rows and", df_bay.shape[1], "columns.")


# ### BORNO

# In[45]:


# Selecting data on only Borno

df_bor = df[df["state"] == "Borno"].reset_index(drop=True)
df_bor.head(3)


# In[46]:


df_bor.shape
print(f"The new dataframe has", df_bor.shape[0],
      "rows and", df_bor.shape[1], "columns.")


# ### CROSS RIVER

# In[47]:


# Selecting data on only Cross River

df_cro = df[df["state"] == "Cross River"].reset_index(drop=True)
df_cro.head(3)


# In[48]:


df_cro.shape
print(f"The new dataframe has", df_cro.shape[0],
      "rows and", df_cro.shape[1], "columns.")


# ### DELTA

# In[49]:


# Selecting data on only Delta

df_del = df[df["state"] == "Delta"].reset_index(drop=True)
df_del.head(3)


# In[50]:


df_del.shape
print(f"The new dataframe has", df_del.shape[0],
      "rows and", df_del.shape[1], "columns.")


# ### EDO

# In[51]:


# Selecting data on only Edo

df_edo = df[df["state"] == "Edo"].reset_index(drop=True)
df_edo.head(3)


# In[52]:


df_edo.shape
print(f"The new dataframe has", df_edo.shape[0],
      "rows and", df_edo.shape[1], "columns.")


# ### EKITI

# In[53]:


# Selecting data on only Ekiti

df_eki = df[df["state"] == "Ekiti"].reset_index(drop=True)
df_eki.head(3)


# In[54]:


df_eki.shape
print(f"The new dataframe has", df_eki.shape[0],
      "rows and", df_eki.shape[1], "columns.")


# ### ENUGU

# In[55]:


# Selecting data on only Enugu

df_enu = df[df["state"] == "Enugu"].reset_index(drop=True)
df_enu.head(3)


# In[56]:


df_enu.shape
print(f"The new dataframe has", df_enu.shape[0],
      "rows and", df_enu.shape[1], "columns.")


# ### IMO

# In[57]:


# Selecting data on only Imo

df_imo = df[df["state"] == "Imo"].reset_index(drop=True)
df_imo.head(3)


# In[58]:


df_imo.shape
print(f"The new dataframe has", df_imo.shape[0],
      "rows and", df_imo.shape[1], "columns.")


# ### KADUNA

# In[59]:


# Selecting data on only Kaduna

df_kad = df[df["state"] == "Kaduna"].reset_index(drop=True)
df_kad.head(3)


# In[60]:


df_kad.shape
print(f"The new dataframe has", df_kad.shape[0],
      "rows and", df_kad.shape[1], "columns.")


# ### KANO

# In[61]:


# Selecting data on only Kano

df_kan = df[df["state"] == "Kaduna"].reset_index(drop=True)
df_kan.head(3)


# In[62]:


df_kan.shape
print(f"The new dataframe has", df_kan.shape[0],
      "rows and", df_kan.shape[1], "columns.")


# ### KASTINA

# In[63]:


# Selecting data on only Kastina

df_kas = df[df["state"] == "Kastina"].reset_index(drop=True)
df_kas.head(3)


# In[64]:


df_kas.shape
print(f"The new dataframe has", df_kas.shape[0],
      "rows and", df_kas.shape[1], "columns.")


# ### KOGI

# In[65]:


# Selecting data on only Kogi

df_kog = df[df["state"] == "Kogi"].reset_index(drop=True)
df_kog.head(3)


# In[66]:


df_kog.shape
print(f"The new dataframe has", df_kog.shape[0],
      "rows and", df_kog.shape[1], "columns.")


# ### KWARA

# In[67]:


# Selecting data on only Kwara

df_kwa = df[df["state"] == "kwara"].reset_index(drop=True)
df_kwa.head(3)


# In[68]:


df_kwa.shape
print(f"The new dataframe has", df_kwa.shape[0],
      "rows and", df_kwa.shape[1], "columns.")


# ### LAGOS

# In[69]:


# Selecting data on only Lagos

df_lag = df[df["state"] == "Lagos"].reset_index(drop=True)
df_lag.head(3)


# In[70]:


df_lag.shape
print(f"The new dataframe has", df_lag.shape[0],
      "rows and", df_lag.shape[1], "columns.")


# ### NASARAWA

# In[71]:


# Selecting data on only Nasarawa

df_nas = df[df["state"] == "Nasarawa"].reset_index(drop=True)
df_nas.head(3)


# In[72]:


df_nas.shape
print(f"The new dataframe has", df_nas.shape[0],
      "rows and", df_nas.shape[1], "columns.")


# ### NIGER

# In[73]:


# Selecting data on only Niger

df_nig = df[df["state"] == "Niger"].reset_index(drop=True)
df_nig.head(3)


# In[74]:


df_nig.shape
print(f"The new dataframe has", df_nig.shape[0],
      "rows and", df_nig.shape[1], "columns.")


# ### OGUN

# In[75]:


# Selecting data on only Ogun

df_ogu = df[df["state"] == "Ogun"].reset_index(drop=True)
df_ogu.head(3)


# In[76]:


df_ogu.shape
print(f"The new dataframe has", df_ogu.shape[0],
      "rows and", df_ogu.shape[1], "columns.")


# ### OSUN

# In[77]:


# Selecting data on only Osun

df_osu = df[df["state"] == "Osun"].reset_index(drop=True)
df_osu.head(3)


# In[78]:


df_osu.shape
print(f"The new dataframe has", df_osu.shape[0],
      "rows and", df_osu.shape[1], "columns.")


# ### OYO

# In[79]:


# Selecting data on only Oyo

df_oyo = df[df["state"] == "Oyo"].reset_index(drop=True)
df_oyo.head(3)


# In[80]:


df_oyo.shape
print(f"The new dataframe has", df_oyo.shape[0],
      "rows and", df_oyo.shape[1], "columns.")


# ### PLATEAU

# In[81]:


# Selecting data on only Plateau

df_pla = df[df["state"] == "Plateau"].reset_index(drop=True)
df_pla.head(3)


# In[82]:


df_pla.shape
print(f"The new dataframe has", df_pla.shape[0],
      "rows and", df_pla.shape[1], "columns.")


# ### RIVERS

# In[83]:


# Selecting data on only Rivers

df_riv = df[df["state"] == "Rivers"].reset_index(drop=True)
df_riv.head(3)


# In[84]:


df_riv.shape
print(f"The new dataframe has", df_riv.shape[0],
      "rows and", df_riv.shape[1], "columns.")


# #### Having explored each federal state only states with a dataframe of 50+ rows will be considered for data analysis. 
# #### States with 50+ rows; Abuja, Anambara, Delta, Edo, Enugu, Imo, Lagos, Ogun, Oyo, and Rivers.
# #### States with less than 50+ rows; Abia, Akwa Ibom, Bayelsa, Borno, Cross River, Ekiti, Kaduna, Kano, Kastina, Kogi, Kwara, Nasarawa, Niger, Ogun and Plateau

# ## ABUJA STATE (Federal Capital Territory)

# In[85]:


# Selecting data on only Abuja

df_abj = df[df["state"] == "Abuja"].reset_index(drop=True)
df_abj.head(5)


# In[86]:


df_abj.shape
print(f"The new dataframe has", df_abj.shape[0],
      "rows and", df_abj.shape[1], "columns.")


# In[87]:


df_abj.describe()


# ### Categorical features

# In[88]:


# Summarize the categorical features

df_abj.describe(include=["object"])


# #### There are 

# ### Towns in Abuja

# In[89]:


df_abj["town"].unique()


# ### Property Types

# In[90]:


df_abj["title"].unique()


# ## Outliers

# In[91]:


# Visualise

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df_abj["price"], y=df_abj["town"],  palette="Set2",)
sns.stripplot(x=df_abj["price"], y=df_abj["town"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# ### InterQuantileRange (IQR)

# In[92]:


# Calculating q1 and q3 for price
abj_Q1 = df_abj.price.quantile(0.25)
abj_Q3 = df_abj.price.quantile(0.75)
abj_Q1, abj_Q3


# In[93]:


# Calculating InterQuantileRange (IQR) for price
abj_IQR = abj_Q3 - abj_Q1

print("The Interquartile Range for the price is", abj_IQR)


# ### Lower and Upper Limit

# In[94]:


# For price
abj_lower_limit = Q1 - 1.5*abj_IQR
abj_upper_limit = Q3 + 1.5*abj_IQR
abj_lower_limit, abj_upper_limit


# ### Identifying the Outliers

# In[95]:


abj_price_outliers = df_abj[(df_abj.price < abj_lower_limit)
                            | (df_abj.price > abj_upper_limit)]

abj_price_outliers.shape


# ### Removing the Outliers

# In[96]:


# Check the shape before removing the outliers

Abuja = df_abj.shape
Abuja


# In[97]:


# Remove the outliers using the IQR

df_abj_outlier_free = df_abj[(df_abj.price > abj_lower_limit)
                             & (df_abj.price < abj_upper_limit)]
df_abj_outlier_free.shape


# In[98]:


# Visualise to validate changes

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df_abj_outlier_free["price"],
            y=df_abj_outlier_free["town"],  palette="Set2",)
sns.stripplot(x=df_abj_outlier_free["price"],
              y=df_abj_outlier_free["town"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# ### ABUJA REAL ESTATE MARKET TREND (Data Analysis)

# ### Distribution of Features of Properties in Abuja

# In[99]:


# Plotting histogram

df_abj_outlier_free.hist(figsize=(16, 16), xrot=90)

plt.show()


# ### Average property price in Abuja

# In[100]:


average_price_abj = st.mean(df_abj_outlier_free["price"])

print("The average property price in Abuja is " +
      str(round(average_price_abj, 2)) + " Naira")


# #### Visualise

# In[101]:


# Plotting a bar chart for average price per town

average_price_abj = df_abj_outlier_free.groupby(["town"])["price"]                    .mean().sort_index()

fig = px.histogram(x=df_abj_outlier_free["town"].
                   value_counts().sort_index().index,
                   y=average_price_abj,
                   color=df_abj_outlier_free["town"].
                   value_counts().sort_index().index)

fig.update_xaxes(title="Town")
fig.update_yaxes(title="Average Property Price per town in Abuja State")
fig.show()


# #### Maitama District, Asokoro District, Wuse and Wuse 2 appear to be the most expensive towns to live in Abuja State.

# ### Most Common Property Type

# In[102]:


abj_most_common = st.mode(df_abj_outlier_free["title"])

print("The most common property type in Abuja State is " +
      str(abj_most_common))


# ## Anambara

# In[103]:


# Selecting data on only Anambara

df_ana = df[df["state"] == "Anambara"].reset_index(drop=True)
df_ana.head(3)


# In[104]:


df_ana.shape
print(f"The new dataframe has", df_ana.shape[0],
      "rows and", df_ana.shape[1], "columns.")


# ### Towns in Anambara

# In[105]:


df_ana["town"].unique()


# #### A quick Google search shows that these towns do not exist in Anambara state. This indicative of wrongly collated data. The data on Anambara state cannot be used for analysis.

# ### DELTA

# In[106]:


# Selecting data on only Delta

df_del = df[df["state"] == "Delta"].reset_index(drop=True)
df_del.head(5)


# In[107]:


df_del.shape
print(f"The new dataframe has", df_del.shape[0],
      "rows and", df_del.shape[1], "columns.")


# In[108]:


df_del.describe()


# ### Categorical features

# In[109]:


# Summarize the categorical features

df_del.describe(include=["object"])


# ### Towns in Delta

# In[110]:


df_del["town"].unique()


# ### Property Types

# In[111]:


df_del["title"].unique()


# ### InterQuantileRange (IQR)

# In[112]:


# Plotting a boxplot to visualise the outliers in price column

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df_del["price"], y=df_del["town"],  palette="Set2",)
sns.stripplot(x=df_del["price"], y=df_del["town"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# ### InterQuantileRange (IQR)

# In[113]:


# Calculating q1 and q3 for price
del_Q1 = df_del.price.quantile(0.25)
del_Q3 = df_del.price.quantile(0.75)
del_Q1, del_Q3


# In[114]:


# Calculating InterQuantileRange (IQR) for price
del_IQR = del_Q3 - del_Q1

print("The Interquartile Range for the price is", del_IQR)


# In[115]:


# Lower and Upper Limit

del_lower_limit = Q1 - 1.5*del_IQR
del_upper_limit = Q3 + 1.5*del_IQR
del_lower_limit, del_upper_limit


# In[116]:


# Identifying the Outliers

del_price_outliers = df_del[(df_del.price < del_lower_limit)
                            | (df_del.price > del_upper_limit)]

del_price_outliers.shape


# ### Removing the Outliers

# In[117]:


# Check the shape before removing the outliers

Delta = df_del.shape
Delta


# In[118]:


# Remove the outliers using the IQR

df_del_outlier_free = df_del[(df_del.price > del_lower_limit)
                             & (df_del.price < del_upper_limit)]
df_del_outlier_free.shape


# In[119]:


# Plotting a boxplot to validate the changes

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df_del_outlier_free["price"],
            y=df_del_outlier_free["town"],  palette="Set2",)
sns.stripplot(x=df_del_outlier_free["price"],
              y=df_del_outlier_free["town"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# ### DELTA REAL ESTATE MARKET TREND (Data Analysis)

# ### Distribution of Features of Properties in Delta

# In[120]:


# Plotting histogram

df_del_outlier_free.hist(figsize=(16, 16), xrot=90)

plt.show()


# ### Average property price in Delta

# In[121]:


del_average_price = st.mean(df_del_outlier_free["price"])

print("The average property price in Delta is " +
      str(round(del_average_price, 2)) + " Naira")


# In[122]:


# Visualise

# Plotting a bar chart

del_average_price = df_del_outlier_free.groupby(["town"])["price"]                    .mean().sort_index()

fig = px.histogram(x=df_del_outlier_free["town"].
                   value_counts().sort_index().index,
                   y=del_average_price,
                   color=df_del_outlier_free["town"].
                   value_counts().sort_index().index)

fig.update_xaxes(title="Town")
fig.update_yaxes(title="Average Property Price per town in Delta State")
fig.show()


# #### Warri and Asaba appear to be the most expensive towns to live in Delta State.

# ### Most Common Property Type

# In[123]:


del_most_common = st.mode(df_del_outlier_free["title"])

print("The most common property type in Delta State is " +
      str(del_most_common))


# ### EDO

# In[124]:


# Selecting data on only Edo

df_edo = df[df["state"] == "Edo"].reset_index(drop=True)
df_edo.head(5)


# In[125]:


df_edo.shape
print(f"The new dataframe has", df_edo.shape[0],
      "rows and", df_edo.shape[1], "columns.")


# In[126]:


df_edo.describe()


# ### Categorical features

# In[127]:


# Summarize the categorical features

df_edo.describe(include=["object"])


# ### Towns in Edo

# In[128]:


df_edo["town"].unique()


# ### Property Types

# In[129]:


df_edo["title"].unique()


# ## Outliers

# In[130]:


# Plotting a boxplot to visualise the outliers in price column

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df_edo["price"], y=df_edo["town"],  palette="Set2",)
sns.stripplot(x=df_edo["price"], y=df_edo["town"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# ### InterQuantileRange (IQR)

# In[131]:


# Calculating q1 and q3 for price

edo_Q1 = df_edo.price.quantile(0.25)
edo_Q3 = df_edo.price.quantile(0.75)
edo_Q1, edo_Q3


# In[132]:


# Calculating InterQuantileRange (IQR) for price

edo_IQR = edo_Q3 - edo_Q1

print("The Interquartile Range for the price is", edo_IQR)


# In[133]:


# Lower and Upper Limit

edo_lower_limit = Q1 - 1.5*edo_IQR
edo_upper_limit = Q3 + 1.5*edo_IQR
edo_lower_limit, edo_upper_limit


# In[134]:


# Identifying the Outliers
edo_price_outliers = df_edo[(df_edo.price < edo_lower_limit)
                            | (df_edo.price > edo_upper_limit)]

edo_price_outliers.shape


# ### Removing the Outliers

# In[135]:


# Check the shape before removing the outliers

Edo = df_edo.shape
Edo


# In[136]:


# Remove the outliers using the IQR

df_edo_outlier_free = df_edo[(df_edo.price > edo_lower_limit)
                             & (df_edo.price < edo_upper_limit)]
df_edo_outlier_free.shape


# In[137]:


# Plotting a boxplot to validate changes

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df_edo_outlier_free["price"],
            y=df_edo_outlier_free["town"],  palette="Set2",)
sns.stripplot(x=df_edo_outlier_free["price"],
              y=df_edo_outlier_free["town"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# ### EDO REAL ESTATE MARKET TREND (Data Analysis) 

# ### Distribution of Features of Properties in Edo

# In[138]:


# Plotting histogram

df_edo_outlier_free.hist(figsize=(16, 16), xrot=90)

plt.show()


# ### Average property price in Edo

# In[139]:


edo_average_price = st.mean(df_edo_outlier_free["price"])

print("The average property price in Edo is " +
      str(round(edo_average_price, 2)) + " Naira")


# In[140]:


# Visualise

# Plotting a bar chart

edo_average_price = df_edo_outlier_free.groupby(["town"])["price"]                    .mean().sort_index()

fig = px.histogram(x=df_edo_outlier_free["town"].
                   value_counts().sort_index().index,
                   y=edo_average_price,
                   color=df_edo_outlier_free["town"].
                   value_counts().sort_index().index)

fig.update_xaxes(title="Town")
fig.update_yaxes(title="Average Property Price per town in Edo State")
fig.show()


# #### Oredo appears to be the most expensive town to live in Edo State.

# ### Most Common Property Type

# In[141]:


edo_most_common = st.mode(df_edo_outlier_free["title"])

print("The most common property type in Edo State is " +
      str(edo_most_common))


# ### ENUGU

# In[142]:


# Selecting data on only Enugu

df_enu = df[df["state"] == "Enugu"].reset_index(drop=True)
df_enu.head(3)


# In[143]:


df_enu.shape
print(f"The new dataframe has", df_enu.shape[0],
      "rows and", df_enu.shape[1], "columns.")


# In[144]:


df_enu.describe()


# In[145]:


# Summarize the categorical features

df_enu.describe(include=["object"])


# ### Towns in Enugu

# In[146]:


df_enu["town"].unique()


# #### The dataset contains data on only Enugu town.

# ### Property Types

# In[147]:


df_enu["title"].unique()


# ## Outliers

# In[148]:


# Plotting a boxplot to visualise the outliers in price column

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df_enu["price"], y=df_enu["town"],  palette="Set2",)
sns.stripplot(x=df_enu["price"], y=df_enu["town"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# ### InterQuantileRange (IQR)

# In[149]:


# Calculating q1 and q3 for price
enu_Q1 = df_enu.price.quantile(0.25)
enu_Q3 = df_enu.price.quantile(0.75)
enu_Q1, enu_Q3


# In[150]:


# Calculating InterQuantileRange (IQR) for price
enu_IQR = enu_Q3 - enu_Q1

print("The Interquartile Range for the price is", enu_IQR)


# In[151]:


# Lower and Upper Limit

enu_lower_limit = Q1 - 1.5*enu_IQR
enu_upper_limit = Q3 + 1.5*enu_IQR
enu_lower_limit, enu_upper_limit


# In[152]:


# Identifying the Outliers

enu_price_outliers = df_enu[(df_enu.price < enu_lower_limit)
                            | (df_enu.price > enu_upper_limit)]
enu_price_outliers.shape


# ### Removing the Outliers

# In[153]:


# Check the shape before removing the outliers

Enugu = df_enu.shape
Enugu


# In[154]:


# Remove the outliers using the IQR

df_enu_outlier_free = df_enu[(df_enu.price > enu_lower_limit)
                             & (df_enu.price < enu_upper_limit)]
df_enu_outlier_free.shape


# In[155]:


# Plotting a boxplot to validate the changes

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df_enu_outlier_free["price"],
            y=df_enu_outlier_free["town"],  palette="Set2",)
sns.stripplot(x=df_enu_outlier_free["price"],
              y=df_enu_outlier_free["town"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# ### ENUGU REAL ESTATE MARKET TREND (Data Analysis)

# #### Distribution of Features of Properties in Enugu

# In[156]:


# Plotting histogram

df_enu_outlier_free.hist(figsize=(16, 16), xrot=90)

plt.show()


# ### Average property price in Enugu

# In[157]:


enu_average_price = st.mean(df_enu_outlier_free["price"])

print("The average property price in Enugu is " +
      str(round(enu_average_price, 2)) + " Naira")


# ### Most Common Property Type

# In[158]:


enu_most_common = st.mode(df_enu_outlier_free["title"])

print("The most common property type in Enugu State is " +
      str(enu_most_common))


# ### IMO

# In[159]:


# Selecting data on only Imo

df_imo = df[df["state"] == "Imo"].reset_index(drop=True)
df_imo.head(5)


# In[160]:


df_imo.shape
print(f"The new dataframe has", df_imo.shape[0],
      "rows and", df_imo.shape[1], "columns.")


# In[161]:


df_imo.describe()


# In[162]:


# Summarize the categorical features

df_imo.describe(include=["object"])


# In[ ]:





# ### Towns in Imo

# In[163]:


df_imo["town"].unique()


# ### Property Types

# In[164]:


df_imo["title"].unique()


# ## Outliers

# In[165]:


# Plotting a boxplot to visualise the outliers in price column

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df_imo["price"], y=df_imo["town"],  palette="Set2",)
sns.stripplot(x=df_imo["price"], y=df_imo["town"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# ### InterQuantileRange (IQR)

# In[166]:


# Calculating q1 and q3 for price

imo_Q1 = df_imo.price.quantile(0.25)
imo_Q3 = df_imo.price.quantile(0.75)
imo_Q1, imo_Q3


# In[167]:


# Calculating InterQuantileRange (IQR) for price
imo_IQR = imo_Q3 - imo_Q1

print("The Interquartile Range for the price is", imo_IQR)


# In[168]:


# Lower and Upper Limit

imo_lower_limit = Q1 - 1.5*imo_IQR
imo_upper_limit = Q3 + 1.5*imo_IQR
imo_lower_limit, imo_upper_limit


# In[169]:


# Identifying the Outliers

imo_price_outliers = df_imo[(df_imo.price < imo_lower_limit)
                            | (df_imo.price > imo_upper_limit)]

imo_price_outliers.shape


# ### Removing the Outliers

# In[170]:


# Check the shape before removing the outliers

Imo = df_imo.shape
Imo


# In[171]:


# Remove the outliers using the IQR

df_imo_outlier_free = df_imo[(df_imo.price > imo_lower_limit)
                             & (df_imo.price < imo_upper_limit)]
df_imo_outlier_free.shape


# In[172]:


# Plotting a boxplot to validate the changes

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df_imo_outlier_free["price"],
            y=df_imo_outlier_free["town"],  palette="Set2",)
sns.stripplot(x=df_imo_outlier_free["price"],
              y=df_imo_outlier_free["town"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# ### IMO REAL ESTATE MARKET TREND (Data Analysis)

# ### Distribution of Features of Properties in Imo

# In[173]:


# Plotting histogram

df_imo_outlier_free.hist(figsize=(16, 16), xrot=90)

plt.show()


# ### Average property price in Imo

# In[174]:


imo_average_price = st.mean(df_imo_outlier_free["price"])

print("The average property price in Imo is " +
      str(round(imo_average_price, 2)) + " Naira")


# #### Visualise

# In[175]:


# Plotting a bar chart

imo_average_price = df_imo_outlier_free.groupby(["town"])["price"]                    .mean().sort_index()

fig = px.histogram(x=df_imo_outlier_free["town"].
                   value_counts().sort_index().index,
                   y=imo_average_price,
                   color=df_imo_outlier_free["town"].
                   value_counts().sort_index().index)

fig.update_xaxes(title="Town")
fig.update_yaxes(title="Average Property Price per town in Imo State")
fig.show()


# #### Owerri West appears to be the most expensive town to live in Imo State.

# ### Most Common Property Type

# In[176]:


imo_most_common = st.mode(df_imo_outlier_free["title"])

print("The most common property type in Imo State is " +
      str(imo_most_common))


# ## LAGOS STATE

# In[177]:


# Selecting data on only Lagos

df_lag = df[df["state"] == "Lagos"].reset_index(drop=True)
df_lag.head(3)


# In[178]:


df_lag.shape
print(f"The new dataframe has", df_lag.shape[0],
      "rows and", df_lag.shape[1], "columns.")


# In[179]:


df_lag.describe()


# ### Categorical features

# In[180]:


# Summarize the categorical features

df_lag.describe(include=["object"])


# ### Towns in Lagos

# In[181]:


df_lag["town"].unique()


# ### Property Types

# In[182]:


df_lag["title"].unique()


# ## Outliers

# In[183]:


# Plotting a boxplot to visualise the outliers in price column

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df_lag["price"], y=df_lag["town"],  palette="Set2",)
sns.stripplot(x=df_lag["price"], y=df_lag["town"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# #### The box plot above show majority of the dataoiints on the left side, however, there some datapoints on the that appear separated from the other datapoints as you begin to scan towards the right part of the chart. These point indicate the presence of Outliers. These outliers are properties with significantly higher prices when compared to the property prices within Lagos State. The "lekki" and "Ikoyi" appear to have more of these outliers. 

# ### InterQuantileRange (IQR)

# In[184]:


# Calculating q1 and q3 for price
lag_Q1 = df_lag.price.quantile(0.25)
lag_Q3 = df_lag.price.quantile(0.75)
lag_Q1, lag_Q3


# In[185]:


# Calculating InterQuantileRange (IQR) for price
lag_IQR = lag_Q3 - lag_Q1

print("The Interquartile Range for the price is", lag_IQR)


# In[186]:


# Lower and Upper Limit

lag_lower_limit = Q1 - 1.5*lag_IQR
lag_upper_limit = Q3 + 1.5*lag_IQR
lag_lower_limit, lag_upper_limit


# ### Identifying the Outliers

# In[187]:


lag_price_outliers = df_lag[(df_lag.price < lag_lower_limit)
                            | (df_lag.price > lag_upper_limit)]

lag_price_outliers.shape


# ### Removing the Outliers

# #### Check the shape before removing the outliers

# In[188]:


Lagos = df_lag.shape
Lagos


# #### Remove the outliers using the IQR

# In[189]:


df_lag_outlier_free = df_lag[(df_lag.price > lag_lower_limit)
                             & (df_lag.price < lag_upper_limit)]
df_lag_outlier_free.shape


# #### Visualise to validate changes 

# In[190]:


# Plotting a boxplot to validate changes

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df_lag_outlier_free["price"],
            y=df_lag_outlier_free["town"],  palette="Set2",)
sns.stripplot(x=df_lag_outlier_free["price"],
              y=df_lag_outlier_free["town"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# #### Re-plotting the box plot indicates the absence of the outliers initially spotted. However, this chart reveals that certain towns such as "IKoyi", "Lekki", "Victoria Island", "Magodo", "Apapa", "Ikeja" and "Lagos Island" have properties with significantly higher prices than other towns within Lagos.

# ### LAGOS REAL ESTATE MARKET TREND (Data Analysis)

# ### Distribution of Features of Properties in Lagos

# In[191]:


# Plotting histogram

df_lag_outlier_free.hist(figsize=(16, 16), xrot=90)

plt.show()


# ### Average property price in Lagos

# In[192]:


average_price_lag = st.mean(df_lag_outlier_free["price"])

print("The average property price in Lagos is " +
      str(round(average_price_lag, 2)) + " Naira")


# ### Visualise

# In[193]:


# Plotting a bar chart of average price per town

average_price_lag = df_lag_outlier_free.groupby(["town"])["price"]                    .mean().sort_index()

fig = px.histogram(x=df_lag_outlier_free["town"].
                   value_counts().sort_index().index,
                   y=average_price_lag,
                   color=df_lag_outlier_free["town"].
                   value_counts().sort_index().index)

fig.update_xaxes(title="Town")
fig.update_yaxes(title="Average Property Price per town in Lagos State")
fig.show()


# #### Ikoyi, Victoria Island (VI) and Lekki appear to be the most expensive towns to live in Lagos State.

# ### Most Common Property Type

# In[194]:


lag_most_common = st.mode(df_lag_outlier_free["title"])

print("The most common property type in Lagos State is " +
      str(lag_most_common))


# #### Write a conclusion

# ### OGUN

# In[195]:


# Selecting data on only Ogun

df_ogu = df[df["state"] == "Ogun"].reset_index(drop=True)
df_ogu.head(5)


# In[196]:


df_ogu.shape
print(f"The new dataframe has", df_ogu.shape[0],
      "rows and", df_ogu.shape[1], "columns.")


# In[197]:


df_ogu.describe()


# ### Categorical features

# In[198]:


# Summarize the categorical features

df_ogu.describe(include=["object"])


# ### Towns in Ogun State

# In[199]:


df_ogu["town"].unique()


# ### Property Types

# In[200]:


df_ogu["title"].unique()


# ## Outliers

# In[201]:


# Plotting a boxplot to visualise the outliers in price column

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df_ogu["price"], y=df_ogu["town"],  palette="Set2",)
sns.stripplot(x=df_ogu["price"], y=df_ogu["town"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# ### InterQuantileRange (IQR)

# In[202]:


# Calculating q1 and q3 for price
ogu_Q1 = df_ogu.price.quantile(0.25)
ogu_Q3 = df_ogu.price.quantile(0.75)
ogu_Q1, ogu_Q3


# In[203]:


# Calculating InterQuantileRange (IQR) for price

ogu_IQR = ogu_Q3 - ogu_Q1

print("The Interquartile Range for the price is", ogu_IQR)


# In[204]:


# Lower and Upper Limit

ogu_lower_limit = Q1 - 1.5*ogu_IQR
ogu_upper_limit = Q3 + 1.5*ogu_IQR
ogu_lower_limit, ogu_upper_limit


# In[205]:


# Identifying the Outliers

ogu_price_outliers = df_ogu[(df_ogu.price < ogu_lower_limit)
                            | (df_ogu.price > ogu_upper_limit)]

ogu_price_outliers.shape


# ### Removing the Outliers

# In[206]:


# Check the shape before removing the outliers

Ogun = df_ogu.shape
Ogun


# In[207]:


# Remove the outliers using the IQR

df_ogu_outlier_free = df_ogu[(df_ogu.price > ogu_lower_limit)
                             & (df_ogu.price < ogu_upper_limit)]
df_ogu_outlier_free.shape


# In[208]:


# Plotting a boxplot to validate changes

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df_ogu_outlier_free["price"],
            y=df_ogu_outlier_free["town"],  palette="Set2",)
sns.stripplot(x=df_ogu_outlier_free["price"],
              y=df_ogu_outlier_free["town"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# ### OGUN REAL ESTATE MARKET TREND (Data Analysis)

# ### Distribution of Features of Properties in Ogun

# In[209]:


# Plotting histogram

df_ogu_outlier_free.hist(figsize=(16, 16), xrot=90)

plt.show()


# ### Average property price in Ogun

# In[210]:


ogu_average_price = st.mean(df_ogu_outlier_free["price"])

print("The average property price in Ogun is " +
      str(round(ogu_average_price, 2)) + " Naira")


# In[211]:


# Visualise

# Plotting a bar chart

ogu_average_price = df_ogu_outlier_free.groupby(["town"])["price"]                    .mean().sort_index()

fig = px.histogram(x=df_ogu_outlier_free["town"].
                   value_counts().sort_index().index,
                   y=ogu_average_price,
                   color=df_ogu_outlier_free["town"].
                   value_counts().sort_index().index)

fig.update_xaxes(title="Town")
fig.update_yaxes(title="Average Property Price per town in Ogun State")
fig.show()


# #### Ewekoro, Ijoko, and Isheri North appear to be the most expensive towns to live in Ogun State.

# ### Most Common Property Type

# In[212]:


ogu_most_common = st.mode(df_ogu_outlier_free["title"])

print("The most common property type in Ogun State is " +
      str(ogu_most_common))


# ### OYO STATE

# In[213]:


# Selecting data on only Oyo

df_oyo = df[df["state"] == "Oyo"].reset_index(drop=True)
df_oyo.head(5)


# In[214]:


df_oyo.shape
print(f"The new dataframe has", df_oyo.shape[0],
      "rows and", df_oyo.shape[1], "columns.")


# In[215]:


df_oyo.describe()


# ### Categorical features

# In[216]:


# Summarize the categorical features

df_oyo.describe(include=["object"])


# ### Towns in Oyo

# In[217]:


df_oyo["town"].unique()


# ### Property Types

# In[218]:


df_oyo["title"].unique()


# ## Outliers

# In[219]:


# Plotting a boxplot to visualise the outliers in price column

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df_oyo["price"], y=df_oyo["town"],  palette="Set2",)
sns.stripplot(x=df_oyo["price"], y=df_oyo["town"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# ### InterQuantileRange (IQR)

# In[220]:


# Calculating q1 and q3 for price
oyo_Q1 = df_oyo.price.quantile(0.25)
oyo_Q3 = df_oyo.price.quantile(0.75)
oyo_Q1, oyo_Q3


# In[221]:


# Calculating InterQuantileRange (IQR) for price

oyo_IQR = oyo_Q3 - oyo_Q1

print("The Interquartile Range for the price is", oyo_IQR)


# In[222]:


# Lower and Upper Limit

oyo_lower_limit = Q1 - 1.5*oyo_IQR
oyo_upper_limit = Q3 + 1.5*oyo_IQR
oyo_lower_limit, oyo_upper_limit


# In[223]:


# Identifying the Outliers

oyo_price_outliers = df_oyo[(df_oyo.price < oyo_lower_limit)
                            | (df_oyo.price > oyo_upper_limit)]

oyo_price_outliers.shape


# ### Removing the Outliers

# In[224]:


# Check the shape before removing the outliers
Oyo = df_oyo.shape
Oyo


# In[225]:


# Remove the outliers using the IQR

df_oyo_outlier_free = df_oyo[(df_oyo.price > oyo_lower_limit)
                             & (df_oyo.price < oyo_upper_limit)]
df_oyo_outlier_free.shape


# In[226]:


# Plotting a boxplot to validate the changes

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df_oyo_outlier_free["price"],
            y=df_oyo_outlier_free["town"],  palette="Set2",)
sns.stripplot(x=df_oyo_outlier_free["price"],
              y=df_oyo_outlier_free["town"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# ### OYO RESIDENTIAL REAL ESTATE MARKET TREND (Data Analysis)

# ### Distribution of Features of Properties in Oyo

# In[227]:


# Plotting histogram

df_oyo_outlier_free.hist(figsize=(16, 16), xrot=90)

plt.show()


# #### Comment here 

# ### Average property price in Oyo

# In[228]:


oyo_average_price = st.mean(df_oyo_outlier_free["price"])

print("The average property price in Oyo is " +
      str(round(oyo_average_price, 2)) + " Naira")


# In[229]:


# Visualise

# Plotting a bar chart

oyo_average_price = df_oyo_outlier_free.groupby(["town"])["price"]                    .mean().sort_index()

fig = px.histogram(x=df_oyo_outlier_free["town"].
                   value_counts().sort_index().index,
                   y=oyo_average_price,
                   color=df_oyo_outlier_free["town"].
                   value_counts().sort_index().index)

fig.update_xaxes(title="Town")
fig.update_yaxes(title="Average Property Price per town in Oyo State")
fig.show()


# #### Ibadan South-West, Ibadan and Oluyole appear to be the most expensive towns to live in Oyo State.

# ### Most Common Property Type

# In[230]:


oyo_most_common = st.mode(df_oyo_outlier_free["title"])

print("The most common property type in Oyo State is " +
      str(oyo_most_common))


# ### RIVERS STATE

# In[231]:


# Selecting data on only Rivers

df_riv = df[df["state"] == "Rivers"].reset_index(drop=True)
df_riv.head(5)


# In[232]:


df_riv.shape
print(f"The new dataframe has", df_riv.shape[0],
      "rows and", df_riv.shape[1], "columns.")


# In[233]:


df_riv.describe()


# ### Categorical features

# In[234]:


# Summarize the categorical features

df_riv.describe(include=["object"])


# ### Towns in Rivers

# In[235]:


df_riv["town"].unique()


# ### Property Types

# In[236]:


df_riv["title"].unique()


# ## Outliers

# In[237]:


# Plotting a boxplot to visualise the outliers in price column

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df_riv["price"], y=df_riv["town"],  palette="Set2",)
sns.stripplot(x=df_riv["price"], y=df_riv["town"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# ### InterQuantileRange (IQR)

# In[238]:


# Calculating q1 and q3 for price

riv_Q1 = df_riv.price.quantile(0.25)
riv_Q3 = df_riv.price.quantile(0.75)
riv_Q1, riv_Q3


# In[239]:


# Calculating InterQuantileRange (IQR) for price

riv_IQR = riv_Q3 - riv_Q1

print("The Interquartile Range for the price is", riv_IQR)


# In[240]:


# Lower and Upper Limit

riv_lower_limit = Q1 - 1.5*riv_IQR
riv_upper_limit = Q3 + 1.5*riv_IQR
riv_lower_limit, riv_upper_limit


# In[241]:


# Identifying the Outliers

riv_price_outliers = df_riv[(df_riv.price < riv_lower_limit)
                            | (df_riv.price > riv_upper_limit)]

riv_price_outliers.shape


# ### Removing the Outliers

# In[242]:


# Check the shape before removing the outliers

Rivers = df_riv.shape
Rivers


# In[243]:


# Remove the outliers using the IQR

df_riv_outlier_free = df_riv[(df_riv.price > riv_lower_limit)
                             & (df_riv.price < riv_upper_limit)]
df_riv_outlier_free.shape


# In[244]:


# Plotting a boxplot to validate the changes

# Styling the plot
sns.set_theme(style="whitegrid")

# Scaling the plot
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)

# Plotting
sns.boxplot(x=df_riv_outlier_free["price"],
            y=df_riv_outlier_free["town"],  palette="Set2",)
sns.stripplot(x=df_riv_outlier_free["price"],
              y=df_riv_outlier_free["town"], color=".25",
              jitter=True,
              marker='o',
              alpha=0.8)


# ### RIVERS STATE REAL ESTATE MARKET TREND (Data Analysis)

# ### Distribution of Features of Properties in Rivers

# In[245]:


# Plotting histogram

df_riv_outlier_free.hist(figsize=(16, 16), xrot=90)

plt.show()


# ### Average property price in Rivers

# In[246]:


riv_average_price = st.mean(df_riv_outlier_free["price"])

print("The average property price in Rivers State is " +
      str(round(riv_average_price, 2)) + " Naira")


# In[247]:


# Visualise

# Plotting a bar chart

riv_average_price = df_riv_outlier_free.groupby(["town"])["price"]                    .mean().sort_index()

fig = px.histogram(x=df_riv_outlier_free["town"].
                   value_counts().sort_index().index,
                   y=riv_average_price,
                   color=df_riv_outlier_free["town"].
                   value_counts().sort_index().index)

fig.update_xaxes(title="Town")
fig.update_yaxes(title="Average Property Price per town in Rivers State")
fig.show()


# #### Port Harcourt appears to be the most expensive town to live in Rivers State.

# ### Most Common Property Type

# In[248]:


riv_most_common = st.mode(df_riv_outlier_free["title"])

print("The most common property type in Rivers State is " +
      str(riv_most_common))


# ## CONCLUSIONS AND RECOMMENDATIONS
# 
# #### The following conclusions were drawn based on the findings;
# 
# #### Usually in real estate, the price is dependent on certain factors such as location, size, furnishing and features of the property. This seemed to be the case with the Nigerian real estate market. 
# 
# #### Borno appeared to have the highest property prices which is rather unusual. This is because over the past decade, Borno state as well as it neighbouring states have been severely affected by series of attacks by Boko Haram. This would most likely lead to a decrease in property value or a stall in property price increase. This finding questions the quality of the dataset used for this project and the property market in Borno State. 
# 
# #### As the project progressed, it became clear that the columns labelled "title" and "bedrooms" were misleading in some entries. For instance, a survey of the website where the dataset was scraped from revealed that it is uncommon for a block of flats in Nigeria to have only 4 bedrooms. Rather, a block of flats would have 4 or more flats, with each flat having 1 to 4 bedrooms. For clarity, a block of flats with 4 flats and 2 bedrooms each would have 8 bedrooms in total. Therefore an entry in the dataset with 4 bedrooms and titled "block of flats" could either mean a block of flats with 4 one-bedroom flats or a specific 4-bedrooms flat in a block of flats. This leads to inaccurate analysis, as the "price" could be the value of the entire block of flats or one flat in a block of flats. 
# 
# #### Next, a quick Google search showed that the towns tagged as "Anambara" in the "state" column do not exist in Anambara State. The towns appear to be in Lagos, Abuja and Ondo state respectively. This is indicative of wrongly collated data. This further reduced the quality of the dataset.
# 
# #### In addition, asides from features like number of bedrooms, toilets, type of property, town and state; there are not much information available on the properties. This does not provide sufficient information on the property and could make price prediction a difficult task.
# 
# #### Conclusively, to get more accurate insights on the Nigerian property market, care needs to be taken when scrapping the data from property websites and attention to details is necessary when labelling the data in a table format.

# ## References

# #### - Nigeria: https://en.wikipedia.org/wiki/Nigeria
# #### - Anambara: https://en.wikipedia.org/wiki/Anambra_State#Cities_and_administrative_divisions 
