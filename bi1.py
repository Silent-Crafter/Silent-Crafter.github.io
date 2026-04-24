import pandas as pd
import sqlite3

# --- Source 1: Excel ---
excel_data = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'Salary': [50000, 60000, 70000, 55000]
})
excel_data.to_excel('sample_data.xlsx', index=False)
df_excel = pd.read_excel('sample_data.xlsx')

# --- Source 2: CSV ---
excel_data.to_csv('sample_data.csv', index=False)
df_csv = pd.read_csv('sample_data.csv')

# --- Source 3: JSON ---
excel_data.to_json('sample_data.json', orient='records')
df_json = pd.read_json('sample_data.json')

# --- Source 4: MySQL ---
# conn = pymssql.connect(
#     server='localhost',
#     user='root',
#     password='root',
#     database='test'
# )
# cursor = conn.cursor()

# query = "SELECT * FROM users"
# cursor.execute(query)
# df_sql = pd.read_sql(query, conn)
# conn.close()

print(f"""
------ EXCEL -----
{df_excel}

------ CSV -----
{df_csv}

------ JSON -----
{df_json}
""")

# ------ SQL -----
# {df_sql}
# """)

# --- Load into Target System (SQLite Database) ---
conn = sqlite3.connect('target_database.db')
df_excel.to_sql('employees', conn, if_exists='replace', index=False)
print("\nData loaded into SQLite database successfully!")

# Verify loaded data
result = pd.read_sql('SELECT * FROM employees', conn)
print("\nData in Target DB:\n", result)
conn.close()
