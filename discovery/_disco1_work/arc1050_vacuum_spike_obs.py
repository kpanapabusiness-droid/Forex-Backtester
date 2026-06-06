import pandas as pd, numpy as np
CACHE='data/cache/H4_5ers_eet'
MAJ=['EURUSD','GBPUSD','AUDUSD','NZDUSD','USDCAD','USDCHF','USDJPY']
JPY=['EURJPY','GBPJPY','AUDJPY']
PAIRS=MAJ+JPY

def wilder_atr(h,l,c,n=14):
    pc=c.shift(1)
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def load(p):
    df=pd.read_parquet(f'{CACHE}/{p}.parquet')
    o=(df.open_bid+df.open_ask)/2; h=(df.high_bid+df.high_ask)/2
    l=(df.low_bid+df.low_ask)/2; c=(df.close_bid+df.close_ask)/2
    return pd.DataFrame({'o':o,'h':h,'l':l,'c':c},index=df.index)

rows=[]
K=6  # fade horizon (bars)
for p in PAIRS:
    df=load(p)
    atr=wilder_atr(df.h,df.l,df.c).shift(1)
    tr=pd.concat([df.h-df.l,(df.h-df.c.shift(1)).abs(),(df.l-df.c.shift(1)).abs()],axis=1).max(axis=1)
    prior_range_atr=(tr.shift(1)/atr)          # prior bar's TR / ATR  (vacuum if small)
    disp=(df.c-df.o)/atr                        # current bar body displacement in ATR
    # forward move over next K bars (close[i+K]-close[i])/atr
    fwd=(df.c.shift(-K)-df.c)/atr
    # IS window only
    mask=(df.index>=pd.Timestamp('2010-01-01',tz='UTC'))&(df.index<=pd.Timestamp('2020-12-31',tz='UTC'))
    d=pd.DataFrame({'prior_range_atr':prior_range_atr,'disp':disp,'fwd':fwd,
                    'year':df.index.year},index=df.index)[mask].dropna()
    d['pair']=p
    rows.append(d)
A=pd.concat(rows)
print(f"total bars (IS, all pairs): {len(A)}")

# vacuum = prior bar range in bottom tercile (per-pair would be cleaner; pooled for scan)
vac_thr=A.prior_range_atr.quantile(0.33)
print(f"vacuum prior_range_atr<= {vac_thr:.3f} (33pct)")
for SPIKE in [0.75,1.0,1.5]:
    # spike UP -> fade short -> fade return = -fwd ; spike DOWN -> fade long -> fade return = +fwd
    up=A[(A.prior_range_atr<=vac_thr)&(A.disp>=SPIKE)]
    dn=A[(A.prior_range_atr<=vac_thr)&(A.disp<=-SPIKE)]
    fade_up=-up.fwd   # short the up-spike
    fade_dn= dn.fwd   # long the down-spike
    fade=pd.concat([fade_up,fade_dn])
    n=len(fade)
    print(f"\nSPIKE>={SPIKE}: n={n}  fade_mean_drift={fade.mean():.4f} ATR  median={fade.median():.4f}  frac+={(fade>0).mean():.3f}")
    # control: same spike WITHOUT vacuum
    up2=A[(A.prior_range_atr>vac_thr)&(A.disp>=SPIKE)]; dn2=A[(A.prior_range_atr>vac_thr)&(A.disp<=-SPIKE)]
    ctrl=pd.concat([-up2.fwd,dn2.fwd])
    print(f"   NON-vacuum control: n={len(ctrl)} fade_mean={ctrl.mean():.4f} median={ctrl.median():.4f} frac+={(ctrl>0).mean():.3f}")

print("\n=== per-pair & per-year (SPIKE>=1.0, vacuum) fade drift ===")
SPIKE=1.0
sub=A[(A.prior_range_atr<=vac_thr)]
up=sub[sub.disp>=SPIKE].assign(fade=lambda x:-x.fwd)
dn=sub[sub.disp<=-SPIKE].assign(fade=lambda x: x.fwd)
F=pd.concat([up,dn])
pp=F.groupby('pair').fade.agg(['mean','count'])
print("per-pair:"); print(pp.round(4).to_string())
print(f"per-pair >0.50-equiv (mean>0): {(pp['mean']>0).sum()}/{len(pp)}")
yy=F.groupby('year').fade.agg(['mean','count'])
print("\nper-year:"); print(yy.round(4).to_string())
print(f"\n2015={yy.loc[2015,'mean']:.4f} 2016={yy.loc[2016,'mean']:.4f} 2018={yy.loc[2018,'mean']:.4f}")
