import pandas as pd
data = pd.read_csv('marketing_campaign.csv', sep="\t")
print(data.columns)
data.info()

data['Year_Birth']