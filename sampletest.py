import pandas as pd

df = pd.read_csv('train.csv') #read file

print(df['Survived'].count()) #prints total number of data with a value for count
print(df['Survived'].value_counts()) #prints number who are dead and who are alive
print(df.groupby('Survived').count())  #prints number of dead and alive for each feature
print(df.corr()) #prints correlation in relation of each feature to each other
print(df['PassengerId'].corr(df['Survived'])) #prints correlation between PassengerId and Survival
print(df.groupby('Sex').Survived.value_counts().unstack()) #prints table of correlation between gender and survival
print(df.groupby(['Sex', 'Survived']).size().unstack()) #does the same thing as above
print(df.count()) #prints the count of each feature
survival_table = df.groupby(['Sex', 'Survived']).size().unstack() #table between sex and survived
survival_table.rename(index=str, columns={0: 'Dead', 1: 'Alive'}) #renaes columns
survival_table['Total'] = survival_table[0] + survival_table[1] #creates a new cikumn of totals
survival_table['Percentages'] = survival_table[1] / survival_table['Total'] * 100 #creates a new percent column
print(survival_table)
df['Letter'] = df['Cabin'].str[0] #creates a new column of just the first letter of the cabin
print(df.groupby(['Letter', 'Survived']).size().unstack()) #print relation between letter and survival