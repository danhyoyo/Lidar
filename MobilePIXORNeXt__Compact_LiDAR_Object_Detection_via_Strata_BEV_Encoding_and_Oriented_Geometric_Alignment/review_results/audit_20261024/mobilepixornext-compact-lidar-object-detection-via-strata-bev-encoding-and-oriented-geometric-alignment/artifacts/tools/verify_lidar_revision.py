from pathlib import Path
import re,json,csv,statistics,subprocess,difflib
P=Path('/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment')
W=next((P/'review_results/audit_20261024').iterdir()); A=W/'artifacts';O=A/'original'
checks=[]
def check(name,ok,details=''):
    checks.append({'check':name,'passed':bool(ok),'details':details})
def cells(line):
    line=re.sub(r'\\(?:textbf|underline)\{([0-9.]+)\}',r'\1',line)
    return line.split('&')
def numbers(line):
    return [float(n) for n in re.findall(r'(?<![A-Za-z])\d+\.\d+',line)]
with open('Result/artifacts/ablation_analysis_20261012/runs_summary.csv',encoding='utf-8-sig') as f:
    runs={r['run_id']:r for r in csv.DictReader(f)}
sources={}
for filename in ['paper_06_Loss_IQA.csv','paper_02_Encoding.csv']:
    with open('Result/artifacts/ablation_analysis_20261012/'+filename,encoding='utf-8-sig') as f:
        sources[filename]={r['Run']:r for r in csv.DictReader(f)}
for run in ['P1','P2','P3','P4','P11']:
    row=next(l for l in (P/'tables/loss.tex').read_text().splitlines() if l.startswith(run+':'))
    vals=numbers(row); src=sources['paper_06_Loss_IQA.csv'][run]
    wanted=[round(int(runs[run]['parameters'])/1e6,3)]+[round(float(src[k]),2) for k in ['Car_Moderate','Pedestrian_Moderate','Cyclist_Moderate','mAP_Moderate']]
    check('loss row '+run,vals==wanted,{'table':vals,'source_rounded':wanted})
for name,run in [('Binary35','B1'),('Strata-BEV8','E8'),('\\textbf{Strata-BEV14','E14')]:
    row=next(l for l in (P/'tables/encoding.tex').read_text().splitlines() if l.startswith(name))
    vals=numbers(row);src=sources['paper_02_Encoding.csv'][run]
    expected=[round(float(src[k]),2) for k in ['Car_Moderate','Pedestrian_Moderate','Cyclist_Moderate','mAP_Moderate','AP9','Model speed (Hz)']]
    check('encoding row '+run,vals[-6:]==expected,{'table':vals[-6:],'source_rounded':expected})
for run in ['B0','B1','E8','E14','G16','O32','O32G16','H32','S32','P1','P4']:
    row=next(l for l in (P/'tables/components.tex').read_text().splitlines() if (' & '+run+' & ') in l)
    vals=numbers(row);want=[round(float(runs[run][k]),2) for k in ['map_moderate_percent','mean_ap_9_percent']]
    check('configuration AP '+run,vals[:2]==want,{'table':vals[:2],'source':want})
books=json.loads((A/'data/workbooks.json').read_text())
def book_cells(book,sheet):
    out={}
    for row in books[book][sheet]:
        for cell in re.split(r' \| (?=[A-Z]+[0-9]+=)',row):
            key,val=cell.split('=',1);out[key]=val
    return out
seed=book_cells('Result/mobilepixor_vs_ours_seed_summary.xlsx','Theo seed')
for rownum,run in [(7,'Final42'),(9,'Final43'),(11,'Final44'),(13,'Base42'),(15,'Base43'),(17,'Base44')]:
    vals=[float(seed[col+str(rownum)]) for col in ['J','M','P']]
    macro=float(seed['G'+str(rownum)]); check('seed macro '+run,abs(statistics.mean(vals)-macro)<1e-10,macro)
# Summary statistics recomputed from the source workbook, not from the printed means.
rows=[7,9,11]; vals=[float(seed['G'+str(n)]) for n in rows]
check('final seed summary',round(statistics.mean(vals),2)==70.94 and round(statistics.stdev(vals),2)==.89,{'mean':statistics.mean(vals),'sample_sd':statistics.stdev(vals)})
rows=[13,15,17]; vals=[float(seed['G'+str(n)]) for n in rows]
check('baseline seed summary',round(statistics.mean(vals),2)==47.21 and round(statistics.stdev(vals),2)==5.06,{'mean':statistics.mean(vals),'sample_sd':statistics.stdev(vals)})
v7='Result/benchmark_T4_20261018/Tables_bai_bao_LiDAR_v7_SpeedDetail.xlsx'
eff=book_cells(v7,'B10_Speed_Detail')
for prefix,n in [('PIXOR~',7),('MobilePIXOR (L4)',8),('PointPillars~',9),('SECOND~',10),('PointRCNN~',11),('Part-A$^2$-Anchor~',12),('PV-RCNN~',13),('\\textbf{MobilePIXORNeXt',14)]:
    line=next(l for l in (P/'tables/efficiency.tex').read_text().splitlines() if l.startswith(prefix))
    vals=numbers(line)[-7:]
    wanted=[round(float(eff[col+str(n)]),2) for col in ['E','I','H','J','K','L','M']]
    check('timing source '+prefix,vals==wanted,{'table':vals,'source_rounded':wanted})
for file in ['sections/methodology.tex']:
    old=(O/file).read_text();new=(P/file).read_text()
    pattern=r'\\begin\{(equation|align|gather|multline)\*?\}.*?\\end\{\1\*?\}'
    eq=lambda t:[m.group() for m in re.finditer(pattern,t,re.S)]
    check('displayed method equations preserved',eq(old)==eq(new),f'{len(eq(new))} environments')
oldlabels=set();newlabels=set()
for file in [*sorted((O/'sections').glob('*.tex')),*sorted((O/'tables').glob('*.tex'))]:
    rel=file.relative_to(O);oldlabels.update(re.findall(r'\\label\{([^}]+)\}',file.read_text()));newlabels.update(re.findall(r'\\label\{([^}]+)\}',(P/rel).read_text()))
check('section/table labels preserved',oldlabels==newlabels,{'removed':sorted(oldlabels-newlabels),'added':sorted(newlabels-oldlabels)})
check('single abstract source',r'\input{sections/abstract}' in (P/'main.tex').read_text() and (P/'MobilePIXORNeXt.tex').read_text().strip().endswith(r'\input{main}'))
# Run the skill checks, retaining stdout and exit status verbatim.
for name,cmd in [
('tables',['python3','-B','/home/duyennh/.agents/skills/latex-paper-en/scripts/check_tables.py',str(P/'main.tex')]),
('experiment_results',['python3','-B','/home/duyennh/.agents/skills/latex-paper-en/scripts/analyze_experiment.py',str(P/'main.tex'),'--section','results']),
('method_logic',['python3','-B','/home/duyennh/.agents/skills/latex-paper-en/scripts/analyze_logic.py',str(P/'main.tex'),'--section','methods']),
]:
    result=subprocess.run(cmd,text=True,capture_output=True);(A/'data'/f'check_{name}.txt').write_text(result.stdout+result.stderr)
    check('script executed '+name,result.returncode==0,{'command':cmd,'exit_code':result.returncode,'note':'Heuristic findings require reading; exit 0 means executed, not semantic clearance.'})
result={'checks':checks,'all_checks_passed':all(c['passed'] for c in checks),'scope':'Source consistency/build support only; not a KITTI rerun or methodological acceptance.'}
(A/'data/verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
for c in checks:
    if not c['passed']:print('FAIL',c['check'],c['details'])
print(f"Checks: {sum(c['passed'] for c in checks)}/{len(checks)} passed")
raise SystemExit(0 if result['all_checks_passed'] else 1)
