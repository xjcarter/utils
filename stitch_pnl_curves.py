
## stitching pnl curves 
import pandas

tf = None
for i in range(20,110,10):
    fn = f'pnl.{i}.csv'
    df = pandas.read_csv(fn)
    df = df[['Date','Equity']]
    df.rename(columns={'Equity':f'h_{i}'}, inplace=True)

    if tf is None:
        tf = df.copy()
    else:
        tf = tf.merge(df,on='Date')
        print(tf.head(3))

tf.to_csv('pnl.ALL.csv',index=False)



