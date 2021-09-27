import pandas as pd
import numpy as np

df = pd.read_csv('data/LasVegasTripAdvisorReviews-Dataset.csv', sep=';', lineterminator='\n')
df.columns = df.columns.str.replace("\r", "")
df['Review weekday'] = df['Review weekday'].str.replace('\r', '')
print(df.head(3))

# Defining Amenities column with the n. of amenities per hotel.
# Defining Month variable with 3-letter format
df["Amenities"] = df.eq('YES').sum(axis=1)
df["Month"] = df['Review month'].str[:3]

print(df.describe(exclude=[np.object]))
print(df.describe(include=[np.object]))

# Checking for missing values
print(df.isna().sum())

# Visualization samples
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
internet_box = sns.boxplot(x='Amenities', y='Score', data=df, palette='husl')
plt.show()

# Median values
print('Median score by', df.groupby(['Amenities'])['Score'].median())
mydata = df.groupby(['Amenities'])['Hotel name'].nunique()
mylabels = df['Amenities'].unique()

plt.figure(figsize=(8, 6))
plt.pie(mydata, labels=mydata, autopct='%1.1f%%')
plt.title('Hotels grouped by the amount of amenities', fontsize=20)
plt.legend(mylabels, loc=2)
plt.show()

sns.set_style('whitegrid')
pool_box = sns.boxplot(x='Pool', y='Score', data=df, palette='husl', order=('NO', 'YES'))
plt.show()
print('Median score by', df.groupby(['Pool'])['Score'].median())

df.groupby(['Traveler type', 'Period of stay']).size().unstack().plot(kind='bar', stacked=True)
plt.title('Type of travelers by period of time', fontsize=20)
plt.show()

# Pivot table - Hotel name and amenities score
df_filtered = df[['Hotel name', 'Amenities', 'Nr. rooms', 'Score']]
print(df_filtered.pivot_table(index=['Hotel name']))

### Web Scraping - HTML Holiday Database
# Fetching 2015 holidays from www.timeanddate.com/holdays/us/2015
import requests
from bs4 import BeautifulSoup

headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '3600',
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
}

url = "https://www.timeanddate.com/holidays/us/2015"
req = requests.get(url, headers)
soup = BeautifulSoup(req.content, 'html.parser')
soup.encode("utf8")
table = soup.find("table", id="holidays-table")
data = table.find_all("tr")
hmonth = []
hdayn = []
hday = []
hname = []
htype = []
hdetails = []

for td in data[3:]:
    th = td.find_all("th")
    if th:
        th_data = [col.text.strip('\n') for col in th]
        date_str = th_data[0].split()
        hmonth.append(date_str[0])
        hdayn.append(date_str[1])
        td = td.find_all('td')
        row = [i.text.replace('\n', '') for i in td]
        hday.append(row[0])
        hname.append(row[1])
        htype.append(row[2])
        hdetails.append(row[3])

df_holidays = pd.DataFrame(list(zip(hmonth, hdayn, hname, htype, hdetails)),
                           columns=['Month', 'Day', 'Name', 'Type', 'Details'])
print(df_holidays.head())

### CSV file - Weather Database
# Weather database from www.ncei.noaa.gov for 2015 Las Vegas weather
df_weather = pd.read_csv('data/LasVegasTemp.csv', sep=',', lineterminator='\n')
df_weather.drop(columns=['NAME'], inplace=True)
df_weather.drop(columns=['YEAR'], inplace=True)
df_weather.columns = df_weather.columns.str.replace("\r", "")
print(df_weather.head())

print(df_weather.groupby('PERIOD')['C_TAVG'].mean())


# Adjusting monthly group periods to adequate to the main TripAdvisor df
def period_group(month):
    if month == 'Jan' or month == 'Feb' or month == 'Dec':
        return ('Dec-Feb')
    elif month == 'Mar' or month == 'Apr' or month == 'May':
        return ('Mar-May')
    elif month == 'Jun' or month == 'Jul' or month == 'Aug':
        return ('Jun-Aug')
    elif month == 'Sep' or month == 'Oct' or month == 'Nov':
        return ('Sep-Nov')


df_weather['PERIOD'] = df_weather['MONTH'].apply(period_group)  ## Converting to corresponding period of df
print(df_weather.head())

# New average temperatures for each period
print(df_weather.groupby('PERIOD')['C_TAVG'].mean())

# Creating PERIOD variable corresponding to df
df_holidays['PERIOD'] = df_holidays['Month'].apply(period_group)
print(df_holidays.head())

# Amount of holidays per month
print(df_holidays.groupby('Month')['Name'].nunique())

### SQL - Merging DataFrames with sqlite3
# Creating databse to store tables
import sqlite3

conn = sqlite3.connect('reviews.db')


def is_open(connection):
    try:
        connection.cursor()
        return True
    except:
        print('Connection closed. Cannot operate on a closed database.')
        return False


print(is_open(conn))

cursor = conn.cursor()
tables = ['reviews', 'holidays', 'weather']
for table in tables:
    sql = "DROP TABLE IF EXISTS {}".format(table, table)
    console = cursor.execute(sql)
df.to_sql(name='reviews', con=conn)
df_holidays.to_sql(name='holidays', con=conn)
df_weather.to_sql(name='weather', con=conn)

# Checking created tables
for table in tables:
    sql = "SELECT * FROM {} LIMIT 1".format(table)
    console = cursor.execute(sql)
    columns = cursor.execute(sql).description
    print('Columns on table {}:\n'.format(table))
    for column in columns:
        print(column[0])
    for data in console:
        print('\nFirst row on table {}:\n'.format(table), data, '\n----------------')

# SQL Mean scores per period
sql = """
SELECT R.'Period of stay', ROUND(AVG(Score),2) FROM reviews AS R
GROUP BY R.'Period of stay'
"""
console = cursor.execute(sql)
for row in console:
    print(row)

# Equivalent on python
print(round(df.groupby(['Period of stay'])['Score'].mean(), 2))

# SQL Mean scores and average temperatures per period
sql = """
SELECT R.'Period of stay', ROUND(AVG(R.Score),2), ROUND(AVG(W.C_TAVG),2) FROM reviews AS R
INNER JOIN weather AS W ON W.PERIOD = R.'Period of stay'
GROUP BY R.'Period of stay'
"""
console = cursor.execute(sql)
for row in console:
    print(row)

# SQL Mean scores, average temperatures and amount of holidays per period
sql = """
SELECT R.'Period of stay', ROUND(AVG(R.Score),2), ROUND(AVG(W.C_TAVG),2), COUNT(DISTINCT H.Name) FROM reviews AS R
INNER JOIN weather AS W ON W.PERIOD = R.'Period of stay'
INNER JOIN holidays AS H ON H.PERIOD = R.'Period of stay'
GROUP BY R.'Period of stay'
"""
console = cursor.execute(sql)
row_data = []
column_names = ['Period', 'Score_Avg', 'Temperature_Avg', 'Holidays_Amount']
for row in console:
    row_data.append(row)
print(pd.DataFrame(row_data, columns=column_names))

sql = """
SELECT R.Month, R.'Hotel name', ROUND(AVG(R.Score),2), ROUND(AVG(W.C_TAVG),2), R.Amenities, COUNT(DISTINCT H.Name) FROM reviews AS R
INNER JOIN weather AS W ON W.Month = R.Month
INNER JOIN holidays AS H ON H.MONTH = R.Month
GROUP BY R.Month, R.'Hotel name'
"""
console = cursor.execute(sql)
row_data = []
column_names = ['Month', 'Hotel', 'Score_Avg', 'Temperature_Avg', 'Amenities', 'Holidays_Amount']
for row in console:
    row_data.append(row)
df_master = pd.DataFrame(row_data, columns=column_names)
print(df_master.head())

df_master.to_csv()

### Linear Regression Model
# Score vs Temperature / Holidays / Amenities
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

regr = linear_model.LinearRegression()
x = df_master[['Temperature_Avg', 'Holidays_Amount', 'Amenities']]
y = df_master['Score_Avg']

x2 = sm.add_constant(x)
reg = sm.OLS(y, x2)
reg2 = reg.fit()
print(reg2.summary())

# Linear Regression Model 2 - Score vs Amenities
x = df_master['Amenities']
y = df_master['Score_Avg']

x2 = sm.add_constant(x)
reg = sm.OLS(y, x2)
reg2 = reg.fit()
print(reg2.summary())

# References
# "Temperature for Las Vegas, NV in 2015", Retrieved from https://www.ncei.noaa.gov/
# "Las Vegas Strip Data Set", Retrieving from https://archive.ics.uci.edu/ml/datasets/Las+Vegas+Strip#
# "Holidays in USA for 2015", Retrieved from https://www.timeanddate.com/holidays/us/201
