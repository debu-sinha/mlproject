# Databricks notebook source
# MAGIC %md
# MAGIC ## Student Performance Indicator

# COMMAND ----------

# MAGIC %md
# MAGIC #### Life cycle of Machine learning Project
# MAGIC 
# MAGIC - Understanding the Problem Statement
# MAGIC - Data Collection
# MAGIC - Data Checks to perform
# MAGIC - Exploratory data analysis
# MAGIC - Data Pre-Processing
# MAGIC - Model Training
# MAGIC - Choose best model

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1) Problem statement
# MAGIC - This project understands how the student's performance (test scores) is affected by other variables such as Gender, Ethnicity, Parental level of education, Lunch and Test preparation course.
# MAGIC 
# MAGIC 
# MAGIC ### 2) Data Collection
# MAGIC - Dataset Source - https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977
# MAGIC - The data consists of 8 column and 1000 rows.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Import Data and Required Packages
# MAGIC ####  Importing Pandas, Numpy, Matplotlib, Seaborn and Warings Library.

# COMMAND ----------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Import the CSV Data as Pandas DataFrame

# COMMAND ----------

df = pd.read_csv('data/stud.csv')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Show Top 5 Records

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Shape of the dataset

# COMMAND ----------

df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Dataset information

# COMMAND ----------

# MAGIC %md
# MAGIC - gender : sex of students  -> (Male/female)
# MAGIC - race/ethnicity : ethnicity of students -> (Group A, B,C, D,E)
# MAGIC - parental level of education : parents' final education ->(bachelor's degree,some college,master's degree,associate's degree,high school)
# MAGIC - lunch : having lunch before test (standard or free/reduced) 
# MAGIC - test preparation course : complete or not complete before test
# MAGIC - math score
# MAGIC - reading score
# MAGIC - writing score

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Data Checks to perform
# MAGIC 
# MAGIC - Check Missing values
# MAGIC - Check Duplicates
# MAGIC - Check data type
# MAGIC - Check the number of unique values of each column
# MAGIC - Check statistics of data set
# MAGIC - Check various categories present in the different categorical column

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Check Missing values

# COMMAND ----------

df.isna().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC #### There are no missing values in the data set

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Check Duplicates

# COMMAND ----------

df.duplicated().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC #### There are no duplicates  values in the data set

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Check data types

# COMMAND ----------

# Check Null and Dtypes
df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Checking the number of unique values of each column

# COMMAND ----------

df.nunique()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 Check statistics of data set

# COMMAND ----------

df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Insight
# MAGIC - From above description of numerical data, all means are very close to each other - between 66 and 68.05;
# MAGIC - All standard deviations are also close - between 14.6 and 15.19;
# MAGIC - While there is a minimum score  0 for math, for writing minimum is much higher = 10 and for reading myet higher = 17

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.7 Exploring Data

# COMMAND ----------

df.head()

# COMMAND ----------

print("Categories in 'gender' variable:     ",end=" " )
print(df['gender'].unique())

print("Categories in 'race_ethnicity' variable:  ",end=" ")
print(df['race_ethnicity'].unique())

print("Categories in'parental level of education' variable:",end=" " )
print(df['parental_level_of_education'].unique())

print("Categories in 'lunch' variable:     ",end=" " )
print(df['lunch'].unique())

print("Categories in 'test preparation course' variable:     ",end=" " )
print(df['test_preparation_course'].unique())

# COMMAND ----------

# define numerical & categorical columns
numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

# print columns
print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))

# COMMAND ----------

df.head(2)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.8 Adding columns for "Total Score" and "Average"

# COMMAND ----------

df['total score'] = df['math_score'] + df['reading_score'] + df['writing_score']
df['average'] = df['total score']/3
df.head()

# COMMAND ----------

reading_full = df[df['reading_score'] == 100]['average'].count()
writing_full = df[df['writing_score'] == 100]['average'].count()
math_full = df[df['math_score'] == 100]['average'].count()

print(f'Number of students with full marks in Maths: {math_full}')
print(f'Number of students with full marks in Writing: {writing_full}')
print(f'Number of students with full marks in Reading: {reading_full}')

# COMMAND ----------

reading_less_20 = df[df['reading_score'] <= 20]['average'].count()
writing_less_20 = df[df['writing_score'] <= 20]['average'].count()
math_less_20 = df[df['math_score'] <= 20]['average'].count()

print(f'Number of students with less than 20 marks in Maths: {math_less_20}')
print(f'Number of students with less than 20 marks in Writing: {writing_less_20}')
print(f'Number of students with less than 20 marks in Reading: {reading_less_20}')

# COMMAND ----------

# MAGIC %md
# MAGIC #####  Insights
# MAGIC  - From above values we get students have performed the worst in Maths 
# MAGIC  - Best performance is in reading section

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Exploring Data ( Visualization )
# MAGIC #### 4.1 Visualize average score distribution to make some conclusion. 
# MAGIC - Histogram
# MAGIC - Kernel Distribution Function (KDE)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.1.1 Histogram & KDE

# COMMAND ----------

fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplot(121)
sns.histplot(data=df,x='average',bins=30,kde=True,color='g')
plt.subplot(122)
sns.histplot(data=df,x='average',kde=True,hue='gender')
plt.show()

# COMMAND ----------

fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplot(121)
sns.histplot(data=df,x='total score',bins=30,kde=True,color='g')
plt.subplot(122)
sns.histplot(data=df,x='total score',kde=True,hue='gender')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #####  Insights
# MAGIC - Female students tend to perform well then male students.

# COMMAND ----------

plt.subplots(1,3,figsize=(25,6))
plt.subplot(141)
sns.histplot(data=df,x='average',kde=True,hue='lunch')
plt.subplot(142)
sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='lunch')
plt.subplot(143)
sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='lunch')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #####  Insights
# MAGIC - Standard lunch helps perform well in exams.
# MAGIC - Standard lunch helps perform well in exams be it a male or a female.

# COMMAND ----------

plt.subplots(1,3,figsize=(25,6))
plt.subplot(141)
ax =sns.histplot(data=df,x='average',kde=True,hue='parental level of education')
plt.subplot(142)
ax =sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='parental level of education')
plt.subplot(143)
ax =sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='parental level of education')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #####  Insights
# MAGIC - In general parent's education don't help student perform well in exam.
# MAGIC - 2nd plot shows that parent's whose education is of associate's degree or master's degree their male child tend to perform well in exam
# MAGIC - 3rd plot we can see there is no effect of parent's education on female students.

# COMMAND ----------

plt.subplots(1,3,figsize=(25,6))
plt.subplot(141)
ax =sns.histplot(data=df,x='average',kde=True,hue='race/ethnicity')
plt.subplot(142)
ax =sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='race/ethnicity')
plt.subplot(143)
ax =sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='race/ethnicity')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #####  Insights
# MAGIC - Students of group A and group B tends to perform poorly in exam.
# MAGIC - Students of group A and group B tends to perform poorly in exam irrespective of whether they are male or female

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.2 Maximumum score of students in all three subjects

# COMMAND ----------


plt.figure(figsize=(18,8))
plt.subplot(1, 4, 1)
plt.title('MATH SCORES')
sns.violinplot(y='math score',data=df,color='red',linewidth=3)
plt.subplot(1, 4, 2)
plt.title('READING SCORES')
sns.violinplot(y='reading score',data=df,color='green',linewidth=3)
plt.subplot(1, 4, 3)
plt.title('WRITING SCORES')
sns.violinplot(y='writing score',data=df,color='blue',linewidth=3)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Insights
# MAGIC - From the above three plots its clearly visible that most of the students score in between 60-80 in Maths whereas in reading and writing most of them score from 50-80

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.3 Multivariate analysis using pieplot

# COMMAND ----------

plt.rcParams['figure.figsize'] = (30, 12)

plt.subplot(1, 5, 1)
size = df['gender'].value_counts()
labels = 'Female', 'Male'
color = ['red','green']


plt.pie(size, colors = color, labels = labels,autopct = '.%2f%%')
plt.title('Gender', fontsize = 20)
plt.axis('off')



plt.subplot(1, 5, 2)
size = df['race/ethnicity'].value_counts()
labels = 'Group C', 'Group D','Group B','Group E','Group A'
color = ['red', 'green', 'blue', 'cyan','orange']

plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%')
plt.title('Race/Ethnicity', fontsize = 20)
plt.axis('off')



plt.subplot(1, 5, 3)
size = df['lunch'].value_counts()
labels = 'Standard', 'Free'
color = ['red','green']

plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%')
plt.title('Lunch', fontsize = 20)
plt.axis('off')


plt.subplot(1, 5, 4)
size = df['test preparation course'].value_counts()
labels = 'None', 'Completed'
color = ['red','green']

plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%')
plt.title('Test Course', fontsize = 20)
plt.axis('off')


plt.subplot(1, 5, 5)
size = df['parental level of education'].value_counts()
labels = 'Some College', "Associate's Degree",'High School','Some High School',"Bachelor's Degree","Master's Degree"
color = ['red', 'green', 'blue', 'cyan','orange','grey']

plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%')
plt.title('Parental Education', fontsize = 20)
plt.axis('off')


plt.tight_layout()
plt.grid()

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #####  Insights
# MAGIC - Number of Male and Female students is almost equal
# MAGIC - Number students are greatest in Group C
# MAGIC - Number of students who have standard lunch are greater
# MAGIC - Number of students who have not enrolled in any test preparation course is greater
# MAGIC - Number of students whose parental education is "Some College" is greater followed closely by "Associate's Degree"

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.4 Feature Wise Visualization
# MAGIC #### 4.4.1 GENDER COLUMN
# MAGIC - How is distribution of Gender ?
# MAGIC - Is gender has any impact on student's performance ?

# COMMAND ----------

# MAGIC %md
# MAGIC #### UNIVARIATE ANALYSIS ( How is distribution of Gender ? )

# COMMAND ----------

f,ax=plt.subplots(1,2,figsize=(20,10))
sns.countplot(x=df['gender'],data=df,palette ='bright',ax=ax[0],saturation=0.95)
for container in ax[0].containers:
    ax[0].bar_label(container,color='black',size=20)
    
plt.pie(x=df['gender'].value_counts(),labels=['Male','Female'],explode=[0,0.1],autopct='%1.1f%%',shadow=True,colors=['#ff4d4d','#ff8000'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Insights 
# MAGIC - Gender has balanced data with female students are 518 (48%) and male students are 482 (52%) 

# COMMAND ----------

# MAGIC %md
# MAGIC #### BIVARIATE ANALYSIS ( Is gender has any impact on student's performance ? ) 

# COMMAND ----------

gender_group = df.groupby('gender').mean()
gender_group

# COMMAND ----------

plt.figure(figsize=(10, 8))

X = ['Total Average','Math Average']


female_scores = [gender_group['average'][0], gender_group['math score'][0]]
male_scores = [gender_group['average'][1], gender_group['math score'][1]]

X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, male_scores, 0.4, label = 'Male')
plt.bar(X_axis + 0.2, female_scores, 0.4, label = 'Female')
  
plt.xticks(X_axis, X)
plt.ylabel("Marks")
plt.title("Total average v/s Math average marks of both the genders", fontweight='bold')
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Insights 
# MAGIC - On an average females have a better overall score than men.
# MAGIC - whereas males have scored higher in Maths.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.4.2 RACE/EHNICITY COLUMN
# MAGIC - How is Group wise distribution ?
# MAGIC - Is Race/Ehnicity has any impact on student's performance ?

# COMMAND ----------

# MAGIC %md
# MAGIC #### UNIVARIATE ANALYSIS ( How is Group wise distribution ?)

# COMMAND ----------

f,ax=plt.subplots(1,2,figsize=(20,10))
sns.countplot(x=df['race/ethnicity'],data=df,palette = 'bright',ax=ax[0],saturation=0.95)
for container in ax[0].containers:
    ax[0].bar_label(container,color='black',size=20)
    
plt.pie(x = df['race/ethnicity'].value_counts(),labels=df['race/ethnicity'].value_counts().index,explode=[0.1,0,0,0,0],autopct='%1.1f%%',shadow=True)
plt.show()   

# COMMAND ----------

# MAGIC %md
# MAGIC #### Insights 
# MAGIC - Most of the student belonging from group C /group D.
# MAGIC - Lowest number of students belong to groupA.

# COMMAND ----------

# MAGIC %md
# MAGIC #### BIVARIATE ANALYSIS ( Is Race/Ehnicity has any impact on student's performance ? )

# COMMAND ----------

Group_data2=df.groupby('race/ethnicity')
f,ax=plt.subplots(1,3,figsize=(20,8))
sns.barplot(x=Group_data2['math score'].mean().index,y=Group_data2['math score'].mean().values,palette = 'mako',ax=ax[0])
ax[0].set_title('Math score',color='#005ce6',size=20)

for container in ax[0].containers:
    ax[0].bar_label(container,color='black',size=15)

sns.barplot(x=Group_data2['reading score'].mean().index,y=Group_data2['reading score'].mean().values,palette = 'flare',ax=ax[1])
ax[1].set_title('Reading score',color='#005ce6',size=20)

for container in ax[1].containers:
    ax[1].bar_label(container,color='black',size=15)

sns.barplot(x=Group_data2['writing score'].mean().index,y=Group_data2['writing score'].mean().values,palette = 'coolwarm',ax=ax[2])
ax[2].set_title('Writing score',color='#005ce6',size=20)

for container in ax[2].containers:
    ax[2].bar_label(container,color='black',size=15)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Insights 
# MAGIC - Group E students have scored the highest marks. 
# MAGIC - Group A students have scored the lowest marks. 
# MAGIC - Students from a lower Socioeconomic status have a lower avg in all course subjects

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.4.3 PARENTAL LEVEL OF EDUCATION COLUMN
# MAGIC - What is educational background of student's parent ?
# MAGIC - Is parental education has any impact on student's performance ?

# COMMAND ----------

# MAGIC %md
# MAGIC #### UNIVARIATE ANALYSIS ( What is educational background of student's parent ? )

# COMMAND ----------

plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('fivethirtyeight')
sns.countplot(df['parental level of education'], palette = 'Blues')
plt.title('Comparison of Parental Education', fontweight = 30, fontsize = 20)
plt.xlabel('Degree')
plt.ylabel('count')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Insights 
# MAGIC - Largest number of parents are from some college.

# COMMAND ----------

# MAGIC %md
# MAGIC #### BIVARIATE ANALYSIS ( Is parental education has any impact on student's performance ? )

# COMMAND ----------

df.groupby('parental level of education').agg('mean').plot(kind='barh',figsize=(10,10))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Insights 
# MAGIC - The score of student whose parents possess master and bachelor level education are higher than others.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.4.4 LUNCH COLUMN 
# MAGIC - Which type of lunch is most common amoung students ?
# MAGIC - What is the effect of lunch type on test results?

# COMMAND ----------

# MAGIC %md
# MAGIC #### UNIVARIATE ANALYSIS ( Which type of lunch is most common amoung students ? )

# COMMAND ----------

plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('seaborn-talk')
sns.countplot(df['lunch'], palette = 'PuBu')
plt.title('Comparison of different types of lunch', fontweight = 30, fontsize = 20)
plt.xlabel('types of lunch')
plt.ylabel('count')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Insights 
# MAGIC - Students being served Standard lunch was more than free lunch

# COMMAND ----------

# MAGIC %md
# MAGIC #### BIVARIATE ANALYSIS (  Is lunch type intake has any impact on student's performance ? )

# COMMAND ----------

f,ax=plt.subplots(1,2,figsize=(20,8))
sns.countplot(x=df['parental level of education'],data=df,palette = 'bright',hue='test preparation course',saturation=0.95,ax=ax[0])
ax[0].set_title('Students vs test preparation course ',color='black',size=25)
for container in ax[0].containers:
    ax[0].bar_label(container,color='black',size=20)
    
sns.countplot(x=df['parental level of education'],data=df,palette = 'bright',hue='lunch',saturation=0.95,ax=ax[1])
for container in ax[1].containers:
    ax[1].bar_label(container,color='black',size=20)   

# COMMAND ----------

# MAGIC %md
# MAGIC #### Insights 
# MAGIC - Students who get Standard Lunch tend to perform better than students who got free/reduced lunch

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.4.5 TEST PREPARATION COURSE COLUMN 
# MAGIC - Which type of lunch is most common amoung students ?
# MAGIC - Is Test prepration course has any impact on student's performance ?

# COMMAND ----------

# MAGIC %md
# MAGIC #### BIVARIATE ANALYSIS ( Is Test prepration course has any impact on student's performance ? )

# COMMAND ----------

plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
sns.barplot (x=df['lunch'], y=df['math score'], hue=df['test preparation course'])
plt.subplot(2,2,2)
sns.barplot (x=df['lunch'], y=df['reading score'], hue=df['test preparation course'])
plt.subplot(2,2,3)
sns.barplot (x=df['lunch'], y=df['writing score'], hue=df['test preparation course'])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Insights  
# MAGIC - Students who have completed the Test Prepration Course have scores higher in all three categories than those who haven't taken the course

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.4.6 CHECKING OUTLIERS

# COMMAND ----------

plt.subplots(1,4,figsize=(16,5))
plt.subplot(141)
sns.boxplot(df['math score'],color='skyblue')
plt.subplot(142)
sns.boxplot(df['reading score'],color='hotpink')
plt.subplot(143)
sns.boxplot(df['writing score'],color='yellow')
plt.subplot(144)
sns.boxplot(df['average'],color='lightgreen')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.4.7 MUTIVARIATE ANALYSIS USING PAIRPLOT

# COMMAND ----------

sns.pairplot(df,hue = 'gender')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Insights
# MAGIC - From the above plot it is clear that all the scores increase linearly with each other.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Conclusions
# MAGIC - Student's Performance is related with lunch, race, parental level education
# MAGIC - Females lead in pass percentage and also are top-scorers
# MAGIC - Student's Performance is not much related with test preparation course
# MAGIC - Finishing preparation course is benefitial.
