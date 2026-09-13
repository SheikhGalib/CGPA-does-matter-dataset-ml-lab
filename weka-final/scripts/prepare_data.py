import sys, os, re, json, contextlib, hashlib, shutil
from pathlib import Path
os.environ.setdefault('MPLBACKEND','Agg')
import pandas as pd
import numpy as np
PROJ=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent.parent/'evidence'
sys.path.insert(0,str(PROJ/'teacher_focused_final/scripts'))
import tf_common as tc
df=tc.load_clean()
assert len(df)==1108 and df.recent_sgpa_imputed.sum()==6
# Reexecute only the existing cleaning and harmonisation portion, with outputs redirected.
src=(PROJ/'notebooks/pipeline_v2_source.py').read_text(encoding='utf-8')
src=src[:src.index('# ### 5.10')]
src=src.replace('ROOT = Path.cwd()',f'ROOT = Path({str(PROJ)!r})')
check_dir=OUT/'cleaning_check'
shutil.rmtree(check_dir,ignore_errors=True)
src=src.replace('OUT_DIR   = ROOT / "cleaned-dataset" / "ours" / "merged"',f'OUT_DIR = Path({str(check_dir)!r})')
src=src.replace('FIG_DIR   = ROOT / "docs" / "figures" / "merged"','FIG_DIR = OUT_DIR')
src=src.replace('DOC_DIR   = ROOT / "docs"','DOC_DIR = OUT_DIR')
src=src.replace('plt.show()','plt.close("all")')
env={'display':lambda *a,**kw:None}
with (OUT/'cleaning_verification.log').open('w',encoding='utf-8') as f, contextlib.redirect_stdout(f):
 exec(compile(src,'cleaning_prefix','exec'),env)
rebuilt=env['clean']
checkcols=[f+'_code' for f in tc.TREE_FEATURES]+['recent_sgpa_code','target_code','recent_sgpa_imputed','university']
for c in checkcols:
 assert rebuilt[c].astype(str).tolist()==df[c].astype(str).tolist(),f'raw-to-clean mismatch: {c}'
shutil.rmtree(check_dir,ignore_errors=True)
eligible=df.loc[~df.recent_sgpa_imputed].copy()
split=PROJ/'teacher_focused_final/splits'
tr=pd.read_csv(split/'train_indices.csv').query("split_for == 'cgpa_experiments_B_C'").row_index.tolist()
te=pd.read_csv(split/'test_indices.csv').query("split_for == 'cgpa_experiments_B_C'").row_index.tolist()
assert len(tr)==881 and len(te)==221 and not set(tr)&set(te)
assert set(tr+te)==set(eligible.index)
def arff(key,frame,features,target):
 text='@RELATION '+key+'\n\n'+''.join('@ATTRIBUTE '+c+' NUMERIC\n' for c in features)+'@ATTRIBUTE target {C0,C1,C2,C3}\n\n@DATA\n'
 for _,r in frame.iterrows():
  text+=','.join(str(int(r[tc.code_col(c)])) for c in features)+','+r[target]+'\n'
 (OUT/(key+'.arff')).write_text(text,encoding='utf-8')
for key,target,features in [('sgpa','sgpa_class',tc.TREE_FEATURES),('questionnaire','cgpa_class',tc.TREE_FEATURES),('main','cgpa_class',tc.TREE_FEATURES+['recent_sgpa'])]:
 for suffix,frame in [('all',eligible),('train',df.loc[tr]),('test',df.loc[te])]: arff(key+'_'+suffix,frame,features,target)
stats={}
for feat in tc.TREE_FEATURES+['recent_sgpa']:
 col=tc.code_col(feat); opts=tc.options_for(feat)
 counts=df[col].value_counts().reindex(range(len(opts)),fill_value=0)
 mix=pd.crosstab(df[col],df.cgpa_class).reindex(index=range(len(opts)),columns=tc.CLASSES,fill_value=0)
 stats[feat]={'label':tc.FEATURE_LABEL[feat],'options':opts,'counts':counts.tolist(),'mix':mix.values.tolist()}
from scipy.stats import spearmanr
relations=[{'feature':f,'label':tc.FEATURE_LABEL[f],'cgpa':float(spearmanr(eligible[tc.code_col(f)],eligible.target_code).statistic),'sgpa':float(spearmanr(eligible[tc.code_col(f)],eligible.recent_sgpa_code).statistic)} for f in tc.TREE_FEATURES]
data={'stats':stats,'relations':relations,'classes':tc.CLASSES,'ranges':['< 3.20','3.20-3.49','3.50-3.74','>= 3.75'],'cgpa_counts':df.cgpa_class.value_counts().reindex(tc.CLASSES).tolist(),'sgpa_counts':df.sgpa_class.value_counts().reindex(tc.CLASSES).tolist(),'universities':df.university.value_counts().to_dict(),'features':tc.TREE_FEATURES,'options':tc.ORDINAL_MAPS}
(OUT/'data.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
pd.DataFrame({'row_index':tr+te,'split':['train']*len(tr)+['test']*len(te)}).to_csv(OUT/'split_indices.csv',index=False)
hashes={str(p.relative_to(PROJ)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJ/'raw-data').glob('*.csv')}
(OUT/'data_provenance.json').write_text(json.dumps({'raw_sha256':hashes,'raw_to_clean_columns_verified':checkcols,'n_descriptive':1108,'n_model':1102,'train':881,'test':221,'split':'existing CGPA stratified split, seed 42, used for all three targets','preprocessing_limitation':'Questionnaire missing-value modes were fitted before evaluation; results remain exploratory.'},indent=2))
print('Raw-to-clean match verified. Exported 9 ARFF files, descriptive data, and split provenance.')
