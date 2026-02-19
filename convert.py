import pandas as pd

# 1. Load the file
# 'dtype=str' is the most important part—it stops Excel-style "auto-corrections"
file_path = "C:\\Users\\dziug\\AI analyst\\Online_retail(example to work with)\\online_retail_II.xlsx"
df = pd.read_excel(file_path, dtype=str)

# 2. Export to CSV
# 'index=False' prevents an extra column of numbers from being added
# 'encoding=utf-8-sig' ensures currency symbols and special characters stay intact
df.to_csv('final_sales_data.csv', index=False, encoding='utf-8-sig')

print("Conversion complete! Your data integrity is preserved.")