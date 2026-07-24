import pandas as pd

# Load the real football dataset
df = pd.read_csv('results.csv')

# See the first 5 rows
print(df.head())

# See how many rows and columns
print(df.shape)

# See all column names
print(df.columns)

# How many unique teams are there?
print(df['home_team'].nunique())

# What tournaments are in the data?
print(df['tournament'].unique())

# Filter only World Cup matches
wc = df[df['tournament'] == 'FIFA World Cup']
print(wc.shape)
