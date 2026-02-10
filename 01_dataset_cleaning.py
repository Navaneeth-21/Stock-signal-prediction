import pandas as pd

df = pd.read_csv("data/NIFTY50_raw_data.csv")

df = df.rename(columns={
    'Date': 'date',
    'Price': 'close',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Vol.': 'volume',
    'Change %': 'change_pct'
})


df['date'] = pd.to_datetime(df['date'], dayfirst=True)

df = df.sort_values('date')
df = df.reset_index(drop=True)

price_cols = ['open', 'high', 'low', 'close']

for col in price_cols:
    df[col] = df[col].str.replace(',', '')
    df[col] = df[col].astype(float)


def clean_volume(x):
    if pd.isna(x):
        return None

    x = str(x).strip()

    if x.endswith('B'):
        return float(x.replace('B', '')) * 1_000_000_000
    elif x.endswith('M'):
        return float(x.replace('M', '')) * 1_000_000
    elif x.endswith('K'):
        return float(x.replace('K', '')) * 1_000
    else:
        return float(x.replace(',', ''))

df['volume'] = df['volume'].apply(clean_volume)

df['change_pct'] = df['change_pct'].str.replace('%', '').astype(float)

df = df.dropna()

df = df.drop_duplicates(subset='date')
 
df.to_csv("NIFTY50_cleaned.csv", index=False)


print(df.head())


