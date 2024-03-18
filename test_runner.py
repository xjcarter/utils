import pandas
from indicators import Runner

runner = Runner(3)

df = pandas.read_csv('/home/jcarter/work/trading/data/SPY.csv')
for i, row in df.iterrows():
    v = runner.push(row['Close'])
    print(row['Date'], row['Close'], v)
