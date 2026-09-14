from entsoe import EntsoePandasClient
import pandas as pd, os
from dotenv import load_dotenv
load_dotenv()

client = EntsoePandasClient(api_key=os.getenv("ENTSOE_API_KEY"))
start = pd.Timestamp("2024-12-01", tz="Europe/Brussels")
end   = pd.Timestamp("2024-12-08", tz="Europe/Brussels")
prices = client.query_day_ahead_prices("DE_LU", start=start, end=end)
print(prices.describe())