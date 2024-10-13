import dask.dataframe as dd

df = dd.read_csv('house_0.csv')

df.compute()