import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv('cost_revenue_dirty.csv')

df.shape  # df rows and columns
df.isna().values.any()  # df check for NA values
df.duplicated().values.any()  # df check for duplicated values
df.dtypes  # data types of the columns
chars_to_remove = [',', '$']
columns_to_clean = ['USD_Production_Budget',
                    'USD_Worldwide_Gross',
                    'USD_Domestic_Gross']
# remove the , & $ from the columns to clean and convert to number
for col in columns_to_clean:
    for char in chars_to_remove:
        # Replace each character with an empty string
        df[col] = df[col].astype(str).str.replace(char, "")
    # Convert column to a numeric data type
    df[col] = pd.to_numeric(df[col])

df.Release_Date = pd.to_datetime(df.Release_Date)  # convert release date to datetime
df.dtypes

avg_prod_cost = df.USD_Production_Budget.mean().round(0)  # average production budget pandas
avg_prod_cost_2 = np.average(df.USD_Production_Budget).round(0)  # average production budget numpy

avg_worldwide_gross = np.average(df.USD_Worldwide_Gross).round(0)  # average USD_Worldwide_gross

min_worldwide_gross = np.min(df.USD_Worldwide_Gross).round(0)

min_domestic_gross = np.min(df.USD_Domestic_Gross).round(0)

df.describe()  # the above numpy formulas can be answered by describe()

df[df.USD_Production_Budget == 1100.00]  # which movie has the lowest budget
df[df.USD_Production_Budget == 425000000.00]  # which movie had the largest budget

zero_dom_gross = df[df.USD_Domestic_Gross == 0]
len(zero_dom_gross)
zero_dom_gross.sort_values('USD_Production_Budget', ascending=False)

zero_intl_gross = df[df.USD_Worldwide_Gross == 0]
len(zero_intl_gross)
zero_intl_gross.sort_values('USD_Production_Budget', ascending=False)

a = df.query(
    "USD_Domestic_Gross == 0 and USD_Worldwide_Gross > 0")  # Create a subset for international releases that had some worldwide gross revenue, but made zero revenue in the United States
a

b = df.query(
    "Release_Date > '2018-05-01'")  # Identify which films were not released yet as of the time of data collection (May 1st, 2018).
b

# what is the true percentage of films where the costs exceed the worldwide gross revenue?
data_clean = df.drop(b.index)
data_clean_cnt = len(data_clean)

c = data_clean.query("USD_Production_Budget > USD_Worldwide_Gross")
c_cnt = len(c)

c_cnt / data_clean_cnt

# scatter plot

plt.figure(figsize=(8, 4), dpi=200)

ax = sns.scatterplot(data=data_clean,
                     x='USD_Production_Budget',
                     y='USD_Worldwide_Gross')

ax.set(ylim=(0, 3000000000),
       xlim=(0, 450000000),
       ylabel='Revenue in $ billions',
       xlabel='Budget in $100 millions')

plt.show()

# bubble chart
plt.figure(figsize=(8, 4), dpi=200)
ax = sns.scatterplot(data=data_clean,
                     x='USD_Production_Budget',
                     y='USD_Worldwide_Gross',
                     hue='USD_Worldwide_Gross',  # colour
                     size='USD_Worldwide_Gross', )  # dot size

ax.set(ylim=(0, 3000000000),
       xlim=(0, 450000000),
       ylabel='Revenue in $ billions',
       xlabel='Budget in $100 millions', )

plt.show()

# using with

plt.figure(figsize=(8, 4), dpi=200)

# set styling on a single chart
with sns.axes_style('darkgrid'):
    ax = sns.scatterplot(data=data_clean,
                         x='USD_Production_Budget',
                         y='USD_Worldwide_Gross',
                         hue='USD_Worldwide_Gross',
                         size='USD_Worldwide_Gross')

    ax.set(ylim=(0, 3000000000),
           xlim=(0, 450000000),
           ylabel='Revenue in $ billions',
           xlabel='Budget in $100 millions')

# chart to show growth of USD Production Budget
with sns.axes_style('darkgrid'):
    ax = sns.scatterplot(data=data_clean,
                         x='Release_Date',
                         y='USD_Production_Budget',
                         hue='USD_Worldwide_Gross',
                         size='USD_Worldwide_Gross')

    ax.set(ylim=(0, 450000000),
           xlim=(data_clean.Release_Date.min(), data_clean.Release_Date.max()),
           ylabel='Budget in $100 millions',
           xlabel='Year')

# small budgets: few releases, more releases: growing budget


# using floor divison covert years to decades
dt_index = pd.DatetimeIndex(data_clean.Release_Date)
years = dt_index.year

decades = years // 10 * 10
data_clean['Decade'] = decades
data_clean.head()

old_films = data_clean[data_clean.Decade <= 1960]
new_films = data_clean[data_clean.Decade > 1960]

old_films.describe()  # finding count of old films
old_films.sort_values('USD_Production_Budget',
                      ascending=False).head()  # finding most expensive production prior to 1970

# linear regression - relationship between movie budget and worldwide revenue for old films
plt.figure(figsize=(8, 4), dpi=200)
with sns.axes_style("whitegrid"):
    sns.regplot(data=old_films,
                x='USD_Production_Budget',
                y='USD_Worldwide_Gross',
                scatter_kws={'alpha': 0.4},
                line_kws={'color': 'black'})

# linear regression - relationship between movie budget and worldwide revenue for old films
plt.figure(figsize=(8, 4), dpi=200)
with sns.axes_style("darkgrid"):
    ay = sns.regplot(data=new_films,
                     x='USD_Production_Budget',
                     y='USD_Worldwide_Gross',
                     color='#2f4b7c',
                     scatter_kws={'alpha': 0.3},
                     line_kws={'color': '#ff7c43'})

    ay.set(ylim=(0, new_films.USD_Worldwide_Gross.max()),
           xlim=(0, new_films.USD_Production_Budget.max()),
           ylabel='Revenue in $ billions',
           xlabel='Budget in $ millions')

regression = LinearRegression()
# Explanatory Variable(s) or Feature(s)
X = pd.DataFrame(new_films, columns=['USD_Production_Budget'])

# Response Variable or Target
y = pd.DataFrame(new_films, columns=['USD_Worldwide_Gross'])

regression.fit(X, y)

# theta zero
regression.intercept_
# Literally, means that if a movie budget is $0, the estimated movie revenue is -$8.65 million.

# theta one
regression.coef_
# The slope tells us that for every extra $1 in the budget, movie revenue increases by $3.1

# R-squared
regression.score(X, y)
# This means that our model explains about 56% of the variance in movie revenue.


# Explanatory Variable(s) or Feature(s)
R = pd.DataFrame(old_films, columns=['USD_Production_Budget'])

# Response Variable or Target
s = pd.DataFrame(old_films, columns=['USD_Worldwide_Gross'])

regression.fit(R, s)
rr = regression.intercept_  # The intercept is telling us that is the movie budget is $0, the estimated movie revenue is $22.82 million
ss = regression.coef_  # The slope is telling us that for every extra dollar spend in the budget then the movie revenue increaes by $1.65
regression.score(R, s)  # regression fit is ~3%

# For example, how much global revenue does our model estimate for a film with a budget of $350 million?
budget = 350000000

revenue_estimate = rr + ss * budget
revenue_estimate = revenue_estimate.astype(int)