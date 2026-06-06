import pandas as pd, numpy as np
CACHE='data/cache/H1_5ers_eet'
MAJ=['EURUSD','GBPUSD','AUDUSD','USDJPY']
JPY=['EURJPY','GBPJPY','AUDJPY','EURGBP']
PAIRS=MAJ+JPY
def watr(h,l,c,n=14):
    pc=c.shift(1); tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()
def load(p):
    d=pd.read_parquet(f'{CACHE}/{p}.parquet')
    return pd.DataFrame({'o':(d.open_bid+d.open_ask)/2,'h':(d.high_bid+d.high_ask)/2,
                         'l':(d.low_bid+d.low_ask)/2,'c':(d.close_bid+d.close_ask)/2,
                         'spr':(d.close_ask-d.close_bid)},index=d.index)
rows=[]
for p in PAIRS:
    df=load(p); atr=watr(df.h,df.l,df.c).shift(1)
    lon=df.index.tz_convert('Europe/London')
    fixbar=(lon.hour==15)               # 15:00-16:00 London bar (fix at 16:00 close)
    push=(df.c-df.o)/atr                 # body of the approach-to-fix bar
    # post-fix reversion over k next H1 bars (from this bar's close)
    for k in (1,2,3):
        df[f'fwd{k}']=(df.c.shift(-k)-df.c)/atr
    cost_atr=(df.spr*1.5)/atr            # ~1-leg spread cost in ATR (rough, ex(no commission/slip))
    m=fixbar & (df.index>=pd.Timestamp('2010-01-01',tz='UTC')) & (df.index<=pd.Timestamp('2020-12-31',tz='UTC'))
    d=pd.DataFrame({'push':push,'fwd1':df.fwd1,'fwd2':df.fwd2,'fwd3':df.fwd3,
                    'cost':cost_atr,'year':df.index.year},index=df.index)[m].dropna()
    d['pair']=p; rows.append(d)
A=pd.concat(rows)
print(f"fix-bar obs (IS, all days, 10 pairs): n={len(A)}  (~{len(A)/11/8:.0f}/pair/yr)")
print(f"mean 1-leg spread cost ~{A.cost.mean():.4f} ATR (RT ~{2*A.cost.mean():.4f}; +commission/slip on top)")
for k in (1,2,3):
    print(f"\n--- post-fix horizon k={k} bars ---")
    print(f"corr(push, fwd{k}) = {A['push'].corr(A[f'fwd{k}']):.4f}  (reversion => NEGATIVE)")
    for thr in (0.5,1.0):
        up=A[A.push>=thr]; dn=A[A.push<=-thr]
        fade=pd.concat([-up[f'fwd{k}'], dn[f'fwd{k}']])  # fade big push: short up-push/long down-push
        print(f"  push>={thr}: n={len(fade)} fade_drift={fade.mean():.4f} ATR median={fade.median():.4f} frac+={(fade>0).mean():.3f}")
# per-year (push>=0.5, k=2 fade)
up=A[A.push>=0.5].assign(f=lambda x:-x.fwd2); dn=A[A.push<=-0.5].assign(f=lambda x:x.fwd2)
F=pd.concat([up,dn])
yy=F.groupby('year').f.agg(['mean','count'])
print("\nper-year fade drift (push>=0.5,k=2):"); print(yy.round(4).to_string())
print(f"2015={yy.loc[2015,'mean']:.4f} 2016={yy.loc[2016,'mean']:.4f} 2018={yy.loc[2018,'mean']:.4f}")
