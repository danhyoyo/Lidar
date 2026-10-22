"""Local-only, deterministic arithmetic and claim-candidate collection."""
import hashlib
import json
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
PAPER = ROOT / 'paper'
sys.path.insert(0, '/home/duyennh/.agents/skills/paper-audit/scripts')
from tex_loader import assemble
from build_claim_map import build_claim_candidate, split_sentences


def save(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


doc = assemble(PAPER/'MobilePIXORNeXt.tex')
assert not doc.missing, doc.missing
files = sorted({src for src,_ in doc.origins})
save('include_inventory.json', {'entry':'paper/MobilePIXORNeXt.tex','requested_main_exists':(PAPER/'main.tex').exists(),
                               'files':files,'missing':doc.missing,'assembled_lines':len(doc.lines)})
lexemes = re.compile(r'\b(?:outperform\w*|match\w*|superior|significan\w*|leading|best|highest|lowest|fastest|real.time|robust\w*|eliminat\w*|ensur\w*|prevent\w*|without|all|no |never|always|controlled|caus\w*|converg\w*|improv\w*|reduc\w*|gain\w*|preserv\w*)\b', re.I)
candidates=[]
for src in files:
    for line_no,line in enumerate((PAPER/src).read_text().splitlines(),1):
        active=re.sub(r'(?<!\\)%.*','',line).strip()
        if active and (lexemes.search(active) or re.search(r'\d+(?:\.\d+)?\\%',active)):
            candidates.append({'source':src,'line':line_no,'text':active,'matches':[x.group() for x in lexemes.finditer(active)]})
save('source_candidate_inventory.json',candidates)
skill_candidates=[]
for section in ['abstract','introduction','methodology','experiments','results','conclusion','related_work']:
    text=(PAPER/f'sections/{section}.tex').read_text()
    text='\n'.join(re.sub(r'(?<!\\)%.*','',line) for line in text.splitlines())
    for i,sentence in enumerate(split_sentences(text)):
        if lexemes.search(sentence) or re.search(r'\d+(?:\.\d+)?\\%',sentence):
            skill_candidates.append(build_claim_candidate(sentence,section,i))
save('skill_claim_candidates.json',skill_candidates)

D=Decimal
def fmt(v,places=2):return str(v.quantize(D(1).scaleb(-places),rounding=ROUND_HALF_UP))
calc=[]
def subtract(name,a,b,claimed=None,unit='pp'):
    result=D(a)-D(b)
    if claimed is not None: assert result==D(claimed),(name,result,claimed)
    calc.append({'name':name,'a':a,'b':b,'operation':'subtract','exact':str(result),'unit':unit,
                 'claimed':claimed,'arithmetic_matches':claimed is None or result==D(claimed)})

for values in [
 ('ours-PointPillars Moderate','71.46','70.13','1.33'),('ours-SECOND Moderate','71.46','72.09','-0.63'),
 ('ours-MobilePIXOR Moderate','71.46','41.40','30.06'),('ours-MobilePIXOR mAP9','73.42','42.27','31.15'),
 ('Pedestrian vs PointPillars','60.75','56.01','4.74'),('Pedestrian vs SECOND','60.75','56.56','4.19'),
 ('Pedestrian vs MobilePIXOR','60.75','28.16','32.59'),('Car vs PIXOR','86.90','72.02','14.88'),
 ('Car vs MobilePIXOR','86.90','75.52','11.38'),('backbone A0-A1','47.60','41.40','6.20'),
 ('encoding vs TriBand','50.14','44.42','5.72'),('encoding vs Binary35','50.14','47.60','2.54'),
 ('group head width','60.22','51.30','8.92'),('PC-GTA Moderate','66.25','60.22','6.03'),
 ('PC-GTA mAP9','68.45','61.11','7.34'),('fusion width','68.12','66.25','1.87'),
 ('SG-FPN isolated progression','67.38','68.12','-0.74'),('width+neck combined','67.38','66.25','1.13'),
 ('OGA+IQA','70.58','67.38','3.20'),('SimAM Moderate final','71.46','70.58','0.88'),
 ('SimAM mAP9 final','73.42','72.66','0.76'),('SimAM vs stage2','71.46','69.44','2.02'),
 ('SimAM vs stage3','71.46','69.80','1.66'),('SimAM vs stage4','71.46','69.36','2.10'),
 ('OGA alone','68.97','67.38','1.59'),('IQA alone','70.31','67.38','2.93'),
 ('OGA added with IQA','70.58','70.31','0.27'),('IQA added with OGA','70.58','68.97','1.61'),
 ('factorial interaction','3.20','4.52','-1.32'),('Q-OGA vs OGA','69.20','68.97','0.23'),
 ('Q-OGA vs baseline','69.20','67.38','1.82'),('quality head time','8.19','6.94','1.25','ms'),
 ('attention time','8.67','8.19','0.48','ms'),('checkpoint E2E','81.23','60.96','20.27','ms'),
 ('near Pedestrian','65.23','30.45','34.78'),('near Cyclist','76.32','25.55','50.77'),
 ('mid Pedestrian','11.49','3.75','7.74'),('mid Cyclist','44.30','12.61','31.69'),
 ('far Pedestrian','0.80','0.99','-0.19'),('far Cyclist','14.94','0.56','14.38'),
 ('context R37 Moderate conflict','71.50','71.46','0.04'),('context R37 mAP9 conflict','73.45','73.42','0.03'),
 ('context R37 attention delta','71.50','70.58','0.92'),('context R37 mAP9 attention delta','73.45','72.66','0.79')]:
    subtract(*values)

logs={n:json.loads((ROOT/f'Result/benchmark_T4_20261018/sources/{n}.json').read_text())
      for n in ['ours_best_ap','ours_best_loss','pointpillars','second']}
params={n:j['model']['parameters'] for n,j in logs.items()}
reductions={}
for name in ['pointpillars','second']:
    actual=(1-D(params['ours_best_ap'])/D(params[name]))*100
    rounded=(1-D('.754')/D('4.835' if name=='pointpillars' else '5.326'))*100
    reductions[name]={'baseline_count':params[name],'ours_count':params['ours_best_ap'],
        'exact_count_percent':str(actual),'raw_rounded_1dp':fmt(actual,1),'raw_rounded_2dp':fmt(actual,2),
        'rounded_table_percent':str(rounded),'table_rounded_1dp':fmt(rounded,1),
        'parameter_ratio':str(D(params[name])/D(params['ours_best_ap']))}
save('arithmetic_checks.json',{'subtractions':calc,'parameter_reductions':reductions,
    'relative_percent_examples':{'PointPillars_margin':str(D('1.33')/D('70.13')*100),
                                  'PC_GTA_gain':str(D('6.03')/D('60.22')*100)},
    'payload':{'binary35_bytes':800*704*35*4,'hist14_bytes':800*704*14*4,'reduction_percent':60},
    'fps_from_ms':{ms:str(D(1000)/D(ms)) for ms in ['60.96','17.88','8.67']}})

pr_text=(OUT/'pr_all_difficulties.txt').read_text()
pdf_ap={name:[D(x) for x in re.findall(re.escape(name)+r' \(AP ([0-9.]+)\)',pr_text)]
        for name in ['PointPillars','SECOND','PIXOR','Block-SimAM C3-C5 / F32 (our)']}
assert len(pdf_ap['PointPillars'])==9 and len(pdf_ap['SECOND'])==9
matched={}
for name in ['PointPillars','SECOND','Block-SimAM C3-C5 / F32 (our)']:
    # pdftotext -layout emits row-major difficulty rows; columns are class panels.
    vals=pdf_ap[name]
    # Each model's per-panel occurrences are Easy Car/Ped/Cyc, Moderate Car/Ped/Cyc, Hard Car/Ped/Cyc.
    moderate=vals[3:6]
    mean=sum(moderate)/3
    matched[name]={'pdf_legend_APs':[str(x) for x in vals],
                   'moderate':[str(x) for x in moderate],'moderate_mean_displayed_APs':str(mean),
                   'mean_rounded_2dp':fmt(mean)}
save('pr_figure_checks.json',{'scope':'Arithmetic from readable PDF legend values only; not re-evaluation',
                            'matched_protocol_panel_values':matched,
                            'paper_reference_values':{'PointPillars':['88.02','56.01','66.34'],'SECOND':['88.56','56.56','71.16']}})

models=[('MobilePIXOR','.598','41.40','85.92'),('PointPillars','4.835','70.13','63.84'),
        ('SECOND','5.326','72.09','93.70'),('Ours','.754','71.46','60.96'),
        ('PointRCNN','4.039','73.38','194.50'),('Part-A2','63.815','76.28','133.75'),
        ('PV-RCNN','13.116','74.44','188.60')]
frontiers={}
for key,column in [('parameter',1),('latency',3)]:
    frontier=[]
    for row in models:
        dominated=any((D(other[column])<=D(row[column]) and D(other[2])>=D(row[2])
                       and (D(other[column])<D(row[column]) or D(other[2])>D(row[2]))) for other in models)
        if not dominated:frontier.append(row[0])
    frontiers[key]=frontier
save('pareto_checks.json',{'scope':'Descriptive arithmetic from reported table values; mixed AP protocols and MobilePIXOR L4 timing retained',
                         'non_dominated_2D':frontiers,'note':'No statistical equivalence or convex-hull claim'})
before=json.loads((OUT/'source_hashes_before.json').read_text())
changed=[s for s,h in before.items() if hashlib.sha256((ROOT/s).read_bytes()).hexdigest()!=h]
assert not changed,changed
save('source_integrity.json',{'checked_files':len(before),'changed_files':changed})
print(json.dumps({'included_files':len(files),'candidate_lines':len(candidates),'skill_candidates':len(skill_candidates),
                  'subtractions':len(calc),'raw_parameter_reductions':reductions,
                  'PR_matched_local_rounded_means':{n:x['mean_rounded_2dp'] for n,x in matched.items()},
                  'source_changes':changed},ensure_ascii=False,indent=2))
