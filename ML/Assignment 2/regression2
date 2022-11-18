import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('temperatures.csv')
print(df)

print(df.head()) #printing first 5 rows

#input data
x = df['YEAR']

#output data
y = df['ANNUAL']

sns.regplot(x='YEAR', y='ANNUAL', data=df)
plt.show()