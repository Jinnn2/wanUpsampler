"""Exploratory single-rater audit; does not change frozen consensus rules."""
import sys, json, statistics
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from UNIV_adaptor.scripts.data.acceleration_blind_audit import load_plan, load_package, read, session, csv_write, write, DIMENSIONS
root = Path(sys.argv[1])
plan = load_plan(root)
package = load_package(root, plan)
data = read(root/'private/ratings/rater1.json')
assert data['plan_sha256'] == plan['plan_sha256']
assert data['package_sha256'] == package['package_sha256']
mapping = {p['id']:p for p in session(plan, package, data['participant'])}
public = {p['id']:p for p in package['pairs']}
payload = read(root/'private/presented_scores.json')
assert payload['package_sha256'] == package['package_sha256']
scores = {r['clip_id']:r for r in payload['scores']}
rows = []
for p in plan['pairs']:
    if p['id'] not in data['answers']: continue
    pp = public[p['id']]
    a,b = scores[pp['a']], scores[pp['b']]
    for clip in (pp['a'],pp['b']):
        assert scores[clip]['video_sha256'] == package['clips'][clip]['sha256']
    trans = p['transform_b'] or {}
    row = dict(pair=p['id'],kind=p['kind'],source=p['source'],cluster=p['cluster'],cell=p['cell'],
               action_a=p['a']['action'],action_b=p['b']['action'],transform=trans.get('kind',''),level=trans.get('level',''))
    for dim in DIMENSIONS:
        choice = data['answers'][p['id']][dim]
        if mapping[p['id']]['swap'] and choice in ('A','B'): choice = 'B' if choice == 'A' else 'A'
        row[dim] = choice
    for m,eps in plan['config']['metric_epsilon'].items():
        delta = b[m]-a[m]
        row['delta_'+m] = delta
        row['pred_'+m] = 'tie' if abs(delta)<=eps else 'B' if delta>0 else 'A'
    rows.append(row)
def summarize(items):
    result = {'n':len(items),'clusters':len({r['cluster'] for r in items}),
              'votes':{d:dict(Counter(r[d] for r in items)) for d in DIMENSIONS},'agreement':{}}
    for d in DIMENSIONS:
        directional = [r for r in items if r[d] in ('A','B')]
        result['agreement'][d] = {}
        for m in plan['config']['metric_epsilon']:
            result['agreement'][d][m] = dict(n=len(directional),correct=sum(r[d]==r['pred_'+m] for r in directional),
                metric_tie=sum(r['pred_'+m]=='tie' for r in directional),
                reversed=sum(r['pred_'+m] not in ('tie',r[d]) for r in directional))
    result['mean_delta_vbench5'] = statistics.mean(r['delta_vbench5'] for r in items)
    return result
groups = defaultdict(list)
for r in rows:
    groups[(r['kind'],r['source'],r['transform'],str(r['level']))].append(r)
result = {'participant':data['participant'],'completed':len(rows),'planned':len(plan['pairs']),
          'all':summarize(rows),'real':summarize([r for r in rows if r['kind']=='real']),
          'groups':[dict(group=list(k),**summarize(v)) for k,v in sorted(groups.items())],
          'display_overall':dict(Counter(v['overall'] for v in data['answers'].values()))}
write(root/'analysis/single_rater_exploratory.json',result)
csv_write(root/'analysis/single_rater_pairs.csv',rows)
print(json.dumps(result,ensure_ascii=False,indent=2))
