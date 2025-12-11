import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


df = pd.read_csv("abtest.csv")


print("\n==========HEAD============")
print(df.head())
print("\n==========TAIL============")
print(df.tail())
print("\n==========SHAPE & INFO & DESCRIBE============")
print(df.shape)
print(df.info())
print(df.describe())


print(df.isnull().sum())      
df = df.drop_duplicates()     

# Univariate Analysis

# Time spent histogram
plt.figure(figsize=(8,5))
sns.histplot(df['time_spent_on_the_page'], bins=10, kde=True, color='skyblue')
plt.title("Distribution of Time Spent on Landing Page")
plt.xlabel("Time Spent (minutes)")
plt.ylabel("Count")
plt.show()

# Conversion count
plt.figure(figsize=(6,4))
sns.countplot(x='converted', data=df, palette='orange')
plt.title("Conversion Count")
plt.show()

# Landing page distribution
plt.figure(figsize=(6,4))
sns.countplot(x='landing_page', data=df, palette='purple')
plt.title("Landing Page Distribution")
plt.show()

# Language distribution
plt.figure(figsize=(6,4))
sns.countplot(x='language_preferred', data=df, palette='green')
plt.title("Preferred Language Distribution")
plt.show()

#Bivariate Analysis

# Time spent vs Landing Page
plt.figure(figsize=(8,5))
sns.boxplot(x='landing_page', y='time_spent_on_the_page', data=df, palette='lightgreen')
plt.title("Time Spent vs Landing Page")
plt.show()

# Conversion vs Landing Page
plt.figure(figsize=(6,4))
sns.histplot(data=df, x='landing_page', hue='converted', multiple='fill', palette='pastel')
plt.title("Conversion Proportion by Landing Page")
plt.show()

# Conversion vs Language
plt.figure(figsize=(6,4))
sns.histplot(data=df, x='language_preferred', hue='converted', multiple='fill', palette='pastel')
plt.title("Conversion Proportion by Language")
plt.show()

#Hypothesis Testing

# ---- Q1: Do users spend more time on new landing page? ----
new_time = df[df['landing_page']=="new"]['time_spent_on_the_page']
old_time = df[df['landing_page']=="old"]['time_spent_on_the_page']

# Normality check
print(stats.shapiro(new_time))
print(stats.shapiro(old_time))

# Use t-test if normal
t_stat, p_val = stats.ttest_ind(new_time, old_time, alternative='greater')
print("T-test result:", t_stat, p_val)

# ---- Q2: Is conversion rate higher for new page? ----
conv_table = pd.crosstab(df['landing_page'], df['converted'])
print(conv_table)

# Proportion test
success = np.array([conv_table.loc['new',1], conv_table.loc['old',1]])
nobs = np.array([conv_table.loc['new'].sum(), conv_table.loc['old'].sum()])
stat, pval = stats.proportions_ztest(success, nobs, alternative='larger')
print("Proportion z-test result:", stat, pval)

# ---- Q3: Does conversion depend on preferred language? ----
lang_table = pd.crosstab(df['language_preferred'], df['converted'])
chi2, p, dof, expected = stats.chi2_contingency(lang_table)
print("Chi-square test result:", chi2, p)

# ---- Q4: Is time spent same across languages for new page users? ----
new_df = df[df['landing_page']=="new"]
groups = [group['time_spent_on_the_page'].values for name, group in new_df.groupby('language_preferred')]

