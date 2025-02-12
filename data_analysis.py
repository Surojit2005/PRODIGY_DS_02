import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic Titanic dataset
n_samples = 1000
data = {
    "PassengerId": np.arange(1, n_samples + 1),
    "Survived": np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]),  # 60% did not survive, 40% survived
    "Pclass": np.random.choice([1, 2, 3], size=n_samples, p=[0.2, 0.3, 0.5]),  # Class distribution
    "Name": [f"Passenger {i}" for i in range(1, n_samples + 1)],
    "Sex": np.random.choice(["male", "female"], size=n_samples, p=[0.5, 0.5]),
    "Age": np.random.normal(loc=30, scale=10, size=n_samples).clip(1, 80),  # Age distribution (clipped to 1-80)
    "SibSp": np.random.randint(0, 5, size=n_samples),  # Siblings/Spouses aboard
    "Parch": np.random.randint(0, 4, size=n_samples),  # Parents/Children aboard
    "Ticket": np.random.randint(100000, 999999, size=n_samples).astype(str),  # Random ticket numbers
    "Fare": np.round(np.random.uniform(10, 200, size=n_samples), 2),  # Fare prices
    "Cabin": np.random.choice([np.nan, "C23", "D45", "E31", "B20"], size=n_samples, p=[0.7, 0.1, 0.1, 0.05, 0.05]),  # 70% missing values
    "Embarked": np.random.choice(["C", "Q", "S", np.nan], size=n_samples, p=[0.4, 0.2, 0.35, 0.05])  # Some missing embarkation points
}

# Create DataFrame
df = pd.DataFrame(data)

# Save dataset as CSV
file_path = r"C:\Users\SUROJIT PAUL\OneDrive\Desktop\PRODIGY INFOTECH\task 2\titanic_synthetic.csv"
df.to_csv(file_path, index=False)

# Display first few rows
df.head()



import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_style("whitegrid")

# Check for missing values
missing_values = df.isnull().sum()

# Summary statistics
summary_stats = df.describe()

# Visualizing missing data
plt.figure(figsize=(8, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Plot survival rate by class
plt.figure(figsize=(6, 4))
sns.barplot(x="Pclass", y="Survived", data=df, ci=None, palette="viridis")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.show()

# Plot survival rate by gender
plt.figure(figsize=(6, 4))
sns.barplot(x="Sex", y="Survived", data=df, ci=None, palette="magma")
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.show()

# Age distribution plot
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=30, kde=True, color="blue")
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Display missing values and summary stats
missing_values, summary_stats
