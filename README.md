print('running')
import pandas as pd
df=pd.read_csv('employee.dataset.csv')
## Confirm the dataset is loaded successfully
print("Dataset loaded successfully!")

# info how many coloumn and row
df.info()
# we want to read all column
df = pd.DataFrame(df)
df.head()
## Check Missing Data
df.isnull().sum()
column=list(df.columns)
column
# drop column that is not needed
df.drop(columns='Marital_Status',inplace=True) 
##  DATA cleaning
# check n/a values
df.isna().sum()
# dropping n/a values and check again
df  =df.dropna(subset=["Monthly_Income"])
df.isna().sum()
# check duplicate
df.duplicated('Monthly_Income').sum()
# drop duplicate
df.drop_duplicates(subset='Monthly_Income',inplace=True)
df.duplicated('Monthly_Income').sum()
# again check
df.duplicated('Monthly_Income').sum()
# description
df.describe()
# if we want to seperate things by commas
df['Department'].str.split(',',expand=True)
## data agregation
# Applying aggregation across all the columns 
df.aggregate(['sum', 'min'])
## We are going to find aggregation for these columns 
This function lets you perform several calculations at once, like getting the total, average, and maximum of a column in one go.
df.aggregate({"Department":['sum', 'min'], 
              "Job_Role":['max', 'min'], 
             }) 
## Common Aggregations with groupby():
Think of it like sorting your data into groups (e.g., by category) and then calculating things like totals or averages for each group.
df.groupby('Years_at_Company').agg({'Years_at_Company': ['sum', 'mean', 'max', 'min']})
## Used to summarize and aggregate data with multiple dimensions.
It’s like an Excel pivot table. You can reorganize your data to show summaries by different categories and calculations.
df.pivot_table(values='Years_at_Company', index=["Department", "Job_Role"], aggfunc='sum')
## apply() – Custom Aggregation Function
Allows applying custom functions to each group.
Lets you write your own function to apply on each row or group of data.
df.groupby("Department")['Years_at_Company'].apply(lambda x: x.max() - x.min())
## sum() – Add up values
# add up all the numbers in the column
df["Department"].sum()
## mean() – Calculating Average
# Finds the average of a numerical column.
df['Monthly_Income'].mean()
## count() – Counting Rows
Counts the number of non-null values.
df['Work_Life_Balance'].count()
## median() – Median Value
Calculates the median (middle value) of a column.
df['Monthly_Income'].median()
##  std() – Standard Deviation
Computes the standard deviation of a numerical column.
df['Monthly_Income'].std()
## var() – Variance Calculation
Computes the variance of a numerical column.
df['Monthly_Income'].var()
## nunique() – Counting Unique Values
Finds the number of unique values in a column.
df['Years_at_Company'].nunique()
## Data preprocessing
## Handling Missing Values
isnull() / notnull() – Check for missing values
for missing values
df.isnull().sum() 
# for non missing values
df.notnull().sum()
## dropna() – Remove missing values
Removes rows or columns with missing values.
# Remove rows with any missing value
df.dropna()
# Remove columns with missing values
df.dropna(axis=1)
## fillna() – Fill missing values
Fills missing values with a specified value or strategy.
# replace missing values with 0
df.fillna(0)
## Handling Duplicates
duplicated() – Identify duplicate rows
Checks for duplicate rows in the DataFrame.
# Count duplicate rows
df.duplicated().sum()
# remove duplicate rows
df.drop_duplicates()
## Data Transformation
astype() – Convert data types
Converts the data type of columns to a specific type.
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

## rename() – Rename columns
Renames column names for better understanding.
df.rename(columns={'Gender': 'name'}, inplace=True)

## Dealing with Outliers
clip() – Limit values within a range
Restrict values to a specified range.
df['Job_Level'] = df['Job_Level'].clip(lower=1, upper=5)

q1 = df['Job_Level'].quantile(0.25)
q3 = df['Job_Level'].quantile(0.75)
iqr = q3 - q1
print(f"Q1 (25th percentile): {q1}")
print(f"Q3 (75th percentile): {q3}")
print(f"IQR (Interquartile Range): {iqr}")

## Data Encoding
get_dummies() – One-hot encoding
Converts categorical data into dummy variables.
pd.get_dummies(df, columns=['name'])

 ## factorize() – Convert categorical values to numeric codes
Encodes labels into numeric values.
df['Years_at_Company'], unique_values = pd.factorize(df['Years_at_Company'])
## converting data type
df['Years_at_Company'] = df['Years_at_Company'].astype(float)
print(df.dtypes)
## data plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df.groupby('Job_Role')['Years_at_Company'].sum().plot(kind='bar', title='Total Years at Company by Job Role')

plt.ylabel('Total Years at Company')
plt.xlabel('Job Role')
plt.show()

import matplotlib.pyplot as plt

# Plotting a histogram for the 'Monthly_Income' column
df['Monthly_Income'].plot(kind='hist', bins=5, title='Monthly Income Distribution', color='skyblue', edgecolor='black')

# Adding axis labels
plt.xlabel('Monthly Income')
plt.ylabel('Frequency')

# Display the plot
plt.show()


import matplotlib.pyplot as plt

# Plotting a pie chart for the 'Department' column distribution
df['Department'].value_counts().plot(
    kind='pie', 
    autopct='%1.1f%%', 
    title='Department Distribution', 
    startangle=90,  # Rotates the chart for better orientation
    colors=plt.cm.Paired.colors  # Adds a visually appealing color palette
)

# Remove the default y-label
plt.ylabel('')

# Display the plot
plt.show()

df['Department'].value_counts().plot(kind="bar")
df['Department'].value_counts().plot(kind="area")
df['Department'].value_counts().plot(kind="hist")
df['Department'].value_counts().plot(kind="kde")
df['Department'].value_counts().plot(kind="line")
df['Department'].value_counts().plot(kind="box")
df['Department'].value_counts().plot(kind="area")
## saving data
df = pd.read_csv("employee.dataset.csv")
df
df.to_excel("newdata.xlsx", sheet_name="newsheet")
