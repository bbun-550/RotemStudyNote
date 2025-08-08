import pandas as pd

csv_file = r'C:\Users\acorn\Documents\samsung_data.csv'
ss_stock = pd.read_csv(csv_file, encoding='utf-8', header=0, index_col=0)

print(ss_stock.head(3))