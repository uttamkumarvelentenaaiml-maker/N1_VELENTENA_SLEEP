import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATHS={
 'baseline':'../results/baseline/summary_metrics.csv',
 'transformer':'../results/transformer/summary_metrics.csv',
 'attention':'../results/attention/summary_metrics.csv',
 'n1_special':'../results/n1_special/summary_metrics.csv',
 'n1_v2':'../results/n1_special/v2/summary_metrics.csv'}

rows=[]
for _,rel in MODEL_PATHS.items():
 p=os.path.normpath(os.path.join(BASE_DIR,rel))
 if os.path.exists(p): rows.append(pd.read_csv(p).iloc[0].to_dict())
if not rows: raise SystemExit('No summary files found')

df=pd.DataFrame(rows)
metrics=['accuracy','macro_f1','weighted_f1','n1_precision','n1_recall','n1_f1']
cols=['model']+[m for m in metrics if m in df.columns]
df=df[cols]
df.to_csv(os.path.join(RESULTS_DIR,'comparison_table.csv'),index=False)

# styled table image
fig, ax = plt.subplots(figsize=(12,2+0.5*len(df)))
ax.axis('off')
tbl=ax.table(cellText=df.round(4).values, colLabels=df.columns, loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1,1.5)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,'comparison_table.png'), dpi=220)
plt.close()

# radar-like separate bars
for metric in metrics:
 if metric in df.columns:
  vals=df[metric].astype(float)
  plt.figure(figsize=(9,4.5))
  bars=plt.bar(df['model'], vals)
  for b,v in zip(bars,vals): plt.text(b.get_x()+b.get_width()/2,v,f'{v:.3f}',ha='center',va='bottom',fontsize=9)
  plt.title(f'Model Comparison: {metric}')
  plt.ylabel(metric)
  plt.xticks(rotation=20, ha='right')
  plt.tight_layout()
  plt.savefig(os.path.join(RESULTS_DIR,f'{metric}_chart.png'), dpi=220)
  plt.close()

# narrative report
best=[]
for m in metrics:
 if m in df.columns:
  i=df[m].astype(float).idxmax(); best.append(f'{m}: {df.loc[i,"model"]} ({df.loc[i,m]:.4f})')
open(os.path.join(RESULTS_DIR,'best_model_summary.txt'),'w').write('BEST MODELS\n===========\n\n'+'\n'.join(best))
open(os.path.join(RESULTS_DIR,'comparison_report.txt'),'w').write('COMPARISON REPORT\n=================\n\n'+df.to_string(index=False)+'\n\nBest by metric:\n'+'\n'.join(best))
print('Saved beautiful outputs in comparison/results/')
