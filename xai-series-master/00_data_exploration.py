# %% Imports
import matplotlib.pyplot as plt
from utils import DataLoader

# %% Load data
data_loader = DataLoader()
data_loader.load_dataset()
data = data_loader.data

# %% Show head
print(data.shape)
data.head()

# %% Show general statistics
data.info()

# %% Show histogram for all columns
columns = data.columns
for col in columns:
    print("col: ", col)
    data[col].hist()
    plt.show()


# %% Show preprocessed dataframe
#For Figure 5 & Figure 6
data_loader.preprocess_data()
data_loader.data.head()

import seaborn as sns
import matplotlib.pyplot as plt

# 1. Missingness Plot (Shows the missing BMI data)
plt.figure(figsize=(8, 4))
sns.heatmap(data_loader.data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title("Missing Data Visualization (Yellow = Missing)")
plt.tight_layout()
plt.show()

# 2. Selected Comparison by Stroke Status (Age vs Stroke)
plt.figure(figsize=(8, 4))
sns.boxplot(x='stroke', y='age', data=data_loader.data, palette='Set2')
plt.title("Comparison: Patient Age by Stroke Status")
plt.xlabel("Stroke (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.tight_layout()
plt.show()
# %%
