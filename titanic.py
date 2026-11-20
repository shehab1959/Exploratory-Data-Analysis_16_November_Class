# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# others
import os
import random
import datetime



import pandas as pd

# If the CSV is in the same project folder:
df = pd.read_csv("test.csv")

# Display the first 5 rows to verify it loaded
print(df.head())


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
combine = [train_df, test_df]
train_df.tail()





print(train_df.columns.values)




train_df.info()




# Check for missing values
print(train_df.isnull().sum())




print(train_df.describe())





##Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Plot class distribution
sns.countplot(x='Survived', hue='Survived', data=train_df, palette=["blue", "green"])
plt.xticks([0, 1], ["Not Survived", "Survived"])
plt.title("Survival Distribution")
plt.show()
