#Feature Selection


#--------------Visualization for Feature Selection--------------#

#Histograms,Barplot, Density Graphs

df3['quality'].value_counts()


df3.hist(column='quality')


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = df3)


import matplotlib.pyplot as plt
df3['fixed acidity'].plot.density(color='green')
plt.xlabel('Fixed Acidity')
plt.show()


df3.hist(column='fixed acidity')


df3['fixed acidity'].describe()



fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = df3)


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = df3)




fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'residual sugar', data = df3)


df3.hist(column='residual sugar')



df3['residual sugar'].plot.density(color='green')
plt.xlabel('Residual Sugar')
plt.show()



fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'chlorides', data = df3)



df3.hist(column='chlorides')




fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = df3)


df3.hist(column='free sulfur dioxide')



df3['free sulfur dioxide'].plot.density(color='green')
plt.xlabel('Free Sulfur Dioxide')
plt.show()





fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = df3)

df3.hist(column='total sulfur dioxide')

df3['total sulfur dioxide'].plot.density(color='green')
plt.xlabel('Total Sulfur Dioxide')
plt.show()




fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'density', data = df3)


df3['density'].plot.density(color='green')
plt.xlabel('Density')
plt.show()

df3.hist(column='density')




fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'pH', data = df3)

df3['pH'].plot.density(color='green')
plt.xlabel('pH')
plt.show()


df3.hist(column='pH')




fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'sulphates', data = df3)


df3['sulphates'].plot.density(color='green')
plt.xlabel('sulphates')
plt.show()

df3.hist(column='sulphates')

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'alcohol', data = df3)



#Correlation Matrix and Heatmap


corr_df=df3.corr('pearson')
corr_df

correlation = df3.corr(method='pearson')
columns = correlation.nlargest(6, 'quality').index
columns


correlation_map = np.corrcoef(df3[columns].values.T)
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlation_map, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=columns.values, xticklabels=columns.values)
plt.show()



#-------Select K Best Feature Selection--------#

a=df3.loc[:, df3. columns != 'quality']
b=df3['quality']


from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k =5)
selector.fit(a,b)
a.columns[selector.get_support()]
