import pandas as pd
import matplotlib.pyplot as plt

# --- Extract ---
data = pd.DataFrame({
    'Department': ['IT', 'HR', 'Sales', 'Marketing', 'Finance'],
    'Employees': [50, 30, 40, 25, 35],
    'Revenue': [500000, 200000, 800000, 350000, 600000],
    'Satisfaction': [4.2, 3.8, 4.0, 4.5, 3.9]
})
print("Extracted Data:\n", data)

# --- Transform ---
data['Revenue_Per_Employee'] = data['Revenue'] / data['Employees']
print("\nTransformed Data:\n", data)

# --- Load (Save to CSV) ---
data.to_csv('etl_output.csv', index=False)
print("\nData saved to etl_output.csv")

# --- Visualize ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('ETL Data Visualization', fontsize=16)

# Bar chart - Revenue by Department
axes[0, 0].bar(data['Department'], data['Revenue'], color='steelblue')
axes[0, 0].set_title('Revenue by Department')
axes[0, 0].set_ylabel('Revenue')

# Pie chart - Employee Distribution
axes[0, 1].pie(data['Employees'], labels=data['Department'], autopct='%1.1f%%')
axes[0, 1].set_title('Employee Distribution')

# Line chart - Satisfaction Score
axes[1, 0].plot(data['Department'], data['Satisfaction'], marker='o', color='green')
axes[1, 0].set_title('Satisfaction Score')
axes[1, 0].set_ylabel('Score')

# Bar chart - Revenue Per Employee
axes[1, 1].barh(data['Department'], data['Revenue_Per_Employee'], color='coral')
axes[1, 1].set_title('Revenue Per Employee')

plt.tight_layout()
plt.savefig('etl_visualization.png')
plt.show()
print("Visualization saved as etl_visualization.png")
