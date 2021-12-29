# WineQuality

**_Abstract_**

This project centers around creating a regression ML model based on data of specific physiochecmical properties of Portuguese white and red wines to predict the quality of each wine on a 0-10 scale. In order to create the best model for this specific data, the main objective was to generate a model in Python using Scikit-Learn with Pandas, Numpy, and Scipy for data preprocessing and to determine the most important features for modeling via data visualization using Matplotlib and Seaborn and feature selection tools. After preprocessing and creating models, the data was executed through 6 different machine learning regression models and parameters were hypertuned to determine which model performs the best and is most appropriate for this data.

**_Methodology:_** Data

Both red and wine data was sourced from the University of California, Irvine Machine Learning Repository. Attributes included from physiochemical tests:
* Fixed Acidity
* Volatile Acidity
* Citric Acid 
* Residual Sugar 
* Chlorides
* Free Sulfur Dioxide
* Total Sulfur Dioxide
* Density
* pH
* Sulphates
* Alcohol
* Quality: Score given between 0 and 10 by testers based on sensory data



**_Methodology:_** Preprocessing


* **_Removing Missing (NA) Values:_**
Both CSV files had 0 missing values
```python
df3.isnull().sum()
```

* **_Removing Duplicate Values:_**
Duplicate values were dropped. 

```python
df1.drop_duplicates(inplace=True)
df2.drop_duplicates(inplace=True)
df3 = df1.append(df2, ignore_index=True)
```

* **_Removing Outlier Values:_**
Outlier detection using z-score method was employed and records with a z-score greater than 3 were dropped.

```python
z_scores = stats.zscore(df3)
abs_z_scores = np.abs(z_scores)
filtered = (abs_z_scores < 3).all(axis=1)
df4 = pd.DataFrame(df3[filtered])
df4.shape
```

**_Methodology:_** Feature Selection 

* **_Visualization:_**
  * Histograms 



  * Barplots 




  * Density Plots 



  * Correlation Matrix & Heatmap

* **_Feature Selection:_**
Using Scikit-Learn's Select K Best algorithm, the top 5 most important features in relation to the target variable were returned. 

include code here

**_Methodology:_** Machine Learning  




