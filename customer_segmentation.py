# * Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)

# * LOADING DATA
# * Data is receive from https://www.kaggle.com/
data = pd.read_csv("marketing_campaign.csv",
                   sep="\t", parse_dates=['Dt_Customer'])
data.head()
data.info()

# * Dealing with missing values
# Check for nulls
print("Sum of the null values:")
print(data.isna().sum())
print("Ratio of the null values:")
print(data.isna().sum() / len(data))

# Income col has missing values
# To remove the NA values
data = data.dropna()
print(len(data))  # after removing NA values print data total number

# * Created a feature "Customer_For" in order to store customer's enrollment in days
# Calculate difference between customer and newest customer's enrollment and put in days array
days = []
dMax = data["Dt_Customer"].max()
for i in data["Dt_Customer"]:
    delta = dMax - i
    days.append(delta)

# add total days column in dataframe and then convert it to make numeric for column's dtype
data["Customer_For"] = days
data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")

data["Marital_Status"].value_counts()
data["Education"].value_counts()

# Feature Extraction
# age = current year - year birth
data["Age"] = 2022 - data["Year_Birth"]

# Total spending on various items
data["Spent"] = data["MntWines"] + data["MntFruits"] + data["MntMeatProducts"] + data["MntFishProducts"] + data[
    "MntSweetProducts"] + data["MntGoldProds"]

# Living situation [ partner - Alone ]
data["Living_With"] = data["Marital_Status"].replace({"Married": "Partner", "Together": "Partner", "Absurd": "Partner",
                                                      "Widow": "Alone", "YOLO": "Alone", "Divorced": "Alone",
                                                      "Single": "Alone"})

# Feature indicating total children living in the household
data["Children"] = data["Kidhome"] + data["Teenhome"]

# Total member family
data["Family_Size"] = data["Living_With"].replace(
    {"Alone": 1, "Partner": 2}) + data["Children"]

# Feature : this customer is parent according to # children
data["Is_Parent"] = np.where(data.Children > 0, 1, 0)

# Segmenting education levels
# Segmenting education levels in three groups -> Undergraduate, Graduate, Postgraduate
data["Education"] = data["Education"].replace(
    {"Basic": "Undergraduate", "2n Cycle": "Undergraduate", "Graduation": "Graduate", "Master": "Postgraduate",
     "PhD": "Postgraduate"})

# Rename the some columns name for understanding clearly
data = data.rename(
    columns={"MntWines": "Wines", "MntFruits": "Fruits", "MntMeatProducts": "Meat", "MntFishProducts": "Fish",
             "MntSweetProducts": "Sweets", "MntGoldProds": "Gold"})

# Dropping some of the redundant features
to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID",
           "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "AcceptedCmp1", "AcceptedCmp2", "Complain", "Response"]

data = data.drop(to_drop, axis=1)
data.head()

# Descriptive Analysis
describe = data.describe()

# Plot the Age and Income column's spread
# * Age
sns.boxplot(y="Age", data=data)
plt.savefig('age_boxPlot.png')
plt.show()

# * Income
sns.boxplot(y="Income", data=data)
plt.savefig('income_boxPlot.png')
plt.show()

# Age and Income outliers are dropped
data = data[(data["Age"] < 90)]
data = data[(data["Income"] < 600000)]
print(len(data))  # after dropping, new length of the data

# * CORRELATION MATRIX
corrmat = data.corr()

# Plot the correlation matrix
plt.figure(figsize=(15, 15))
sns.heatmap(corrmat, annot=True)
plt.savefig('corrMatrix_heatmap.png')
plt.show()

# Pre-processing of the Data
# Get all categorical fields which are object dtype and then put in a list
categorical_col = [col for col in data.columns if data[col].dtype == "object"]

# Label Encoding the object dtypes ( categorical field) and use LabelEncoder()
LE = LabelEncoder()
for i in categorical_col:
    data[i] = data[[i]].apply(LE.fit_transform)

# Scaling ( scale 0-1 ) for standardization
# After scale the data, put in new dataframe that is scaled_data
scaler = StandardScaler()
scaler.fit(data)
scaled_data = pd.DataFrame(scaler.transform(data), columns=data.columns)
scaled_data.head()

# * PCA - dimension reduction -> use to reduce complexity of the data
pca = PCA(n_components=3)
pca.fit(scaled_data)
X = pd.DataFrame(pca.transform(scaled_data),
                 columns=['col-1', 'col-2', 'col-3'])
print(pca.explained_variance_ratio_)

# * Find number of clusters with Elbow method
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(X)
Elbow_M.show(outpath='Elbow_M.png')

# * MODELLING
# * CLUSTERING
# * Use K-Means algorithm
# n-cluster is determined 4 by Elbow method
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)

labels = kmeans.fit_predict(X)
# add labels to data and X (new transformed data after PCA)
data["Clusters"] = labels
X["Clusters"] = labels

# add X's columns in series seperately to easily catch the data
x = X['col-1']
y = X['col-2']
z = X['col-3']

# Plotting the clusters in 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=40, c=X["Clusters"], cmap='plasma')
plt.savefig('3d_Cluster')
plt.show()

# * Plotting countplot of clusters
# Identify color for plot
pal = ["#541690", "#FF4949", "#FF8D29", "#FFCD38"]

pl = sns.countplot(x=data["Clusters"], palette=pal)
pl.set_title("Distribution Of The Clusters")
plt.savefig('Cluster_bar_plot_png')
plt.show()


# Create a scatter plot through seaborn -> Spent - Income
pl = sns.scatterplot(x=data["Spent"], y=data["Income"],
                     hue=data["Clusters"], data=data, palette=pal)
pl.set_title("The Relationship Between Spent and Income for Clusters")
plt.legend()
plt.savefig('Cluster_Scatter_Spent-Income.png')
plt.show()

# Personal property - Analyze clusters' atttributes
Personal = ["Kidhome", "Teenhome", "Customer_For", "Age",
            "Children", "Family_Size", "Is_Parent", "Education", "Living_With"]

for i in Personal:
    sns.jointplot(data=data, x=data[i], y=data["Spent"],
                  hue=data["Clusters"], palette=pal)
    plt.savefig(f"Cluster_Plot_{i}-Spent.png")
    plt.show()
