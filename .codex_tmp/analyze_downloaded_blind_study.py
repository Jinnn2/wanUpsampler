"""Local integrity and score-response audit, with no human-quality assumptions."""
import csv
import json
from pathlib import Path
import statistics as st
import sys
import zipfile
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from UNIV_adaptor.scripts.data.acceleration_blind_audit import read, load_plan, load_package, file_hash, write, csv_write

root = Path(sys.argv[1])
archive_path = Path(sys.argv[2])
with zipfile.ZipFile(archive_path) as archive:
    failure = archive.testzip()
    if failure:
        raise ValueError(f"Archive CRC failure: {failure}")
plan = load_plan(root)
package = load_package(root, plan)
payload = read(root / 'private/presented_scores.json')
assert payload['package_sha256'] == package['package_sha256']
scores = {r['clip_id']: r for r in payload['scores']}
with (root / 'private/presented_scores.csv').open(encoding='utf-8-sig') as f:
    csv_rows = list(csv.DictReader(f))
assert len(scores) == len(payload['scores']) == len(csv_rows) == len(package['clips'])
assert {r['clip_id'] for r in csv_rows} == set(scores) == set(package['clips'])
for row in csv_rows:
    clip = row['clip_id']
    assert file_hash(root / f'media/{clip}.mp4') == row['video_sha256'] == scores[clip]['video_sha256'] == package['clips'][clip]['sha256']
    for metric in plan['config']['metric_epsilon']:
        assert float(row[metric]) == scores[clip][metric]
public = {p['id']: p for p in package['pairs']}
metrics = list(plan['config']['metric_epsilon'])
rows = []
for p in plan['pairs']:
    pp = public[p['id']]
    a, b = scores[pp['a']], scores[pp['b']]
    trans = p['transform_b'] or {}
    row = {'pair': p['id'], 'kind': p['kind'], 'source': p['source'], 'cell': p['cell'],
           'cluster': p['cluster'], 'action_a': p['a']['action'], 'action_b': p['b']['action'],
           'transform': trans.get('kind', ''), 'level': trans.get('level', ''),
           'quality_a': a['vbench5'], 'quality_b': b['vbench5'],
           **{f'delta_{m}': b[m] - a[m] for m in metrics}}
    if p['kind'] != 'synthetic':
        row['original_delta_vbench5'] = p['b']['scores']['vbench5'] - p['a']['scores']['vbench5']
        row['presentation_delta_shift'] = row['delta_vbench5'] - row['original_delta_vbench5']
    else:
        row['original_delta_vbench5'] = row['presentation_delta_shift'] = ''
    rows.append(row)
out = root / 'analysis'
out.mkdir(exist_ok=True)
csv_write(out / 'metric_response_pairs.csv', rows)
groups = defaultdict(list)
for r in rows:
    key = (r['kind'], r['source'], r['transform'], str(r['level']))
    groups[key].append(r)
summary = []
for key, items in sorted(groups.items()):
    vals = [r['delta_vbench5'] for r in items]
    summary.append(dict(zip(('kind','source','transform','level'), key)) | {
        'n':len(items), 'prompt_clusters':len({r['cluster'] for r in items}),
        'mean_delta':st.mean(vals), 'median_delta':st.median(vals),
        'b_higher_gt_001':sum(v > .001 for v in vals), 'within_001':sum(abs(v) <= .001 for v in vals),
        'b_lower_gt_001':sum(v < -.001 for v in vals),
        **{f'mean_delta_{m}':st.mean(r[f'delta_{m}'] for r in items) for m in metrics}})
def sign(x):
    return 0 if abs(x) <= .001 else 1 if x > 0 else -1
real = [r for r in rows if r['kind'] == 'real']
stable = [r for r in real if sign(r['delta_vbench5']) and sign(r['original_delta_vbench5'])]
result = {'integrity': 'ZIP CRC, manifest identities, 121 video SHA256s and score CSV/JSON checked',
          'videos':len(scores), 'pairs':len(rows), 'participants':read(out/'report.json')['participants'],
          'real_pair_presentation_shift_mean_abs':st.mean(abs(r['presentation_delta_shift']) for r in real),
          'real_pair_sign_reversals_excluding_ties':sum(sign(r['delta_vbench5']) != sign(r['original_delta_vbench5']) for r in stable),
          'real_pairs_directional_in_both':len(stable), 'summary':summary,
          'caveat':'Synthetic response is not human quality. Only four synthetic source prompts; no human labels yet.'}
write(out / 'metric_response_audit.json', result)
lines = ['# 本地评分诊断（2026-10-01）', '',
         f"已检查 ZIP CRC、计划/媒体清单身份、{len(scores)} 个视频 SHA256，以及评分 CSV 与 JSON 的一致性。",
         f"共 {len(rows)} 对，当前人工评价者 {result['participants']} 人。尚不能判断指标是否符合人工质量判断。", '',
         '表中差值均为 B 减 A；合成退化的 B 为派生视频，A 为同源基线。正值表示派生视频得分更高，不等于其感知质量更好。', '',
         '| 类型/来源 | 退化 | 强度 | 对数 | VBench5 平均差值 | B更高/接近/B更低（阈值0.001） |',
         '|---|---|---:|---:|---:|---|']
for r in summary:
    lines.append(f"| {r['kind']}/{r['source']} | {r['transform']} | {r['level']} | {r['n']} | {r['mean_delta']:+.6f} | {r['b_higher_gt_001']}/{r['within_001']}/{r['b_lower_gt_001']} |")
lines += ['', f"展示归一化前后，48 对真实比较的质量差值平均绝对变化为 {result['real_pair_presentation_shift_mean_abs']:.6f}；两次均非近似平局的 {len(stable)} 对中，{result['real_pair_sign_reversals_excluding_ties']} 对方向翻转。", '',
          '这是转码、显示归一化及重新评分的合并影响，不能全部归因于指标噪声。人工评价应与展示副本评分比较。', '',
          '合成退化仅来自混元的四个 prompt，每类每档四对；它们不是独立的大样本。冻结静态场景可能没有明显影响。不同 seed 对照也没有被预设为同质量。', '',
          '论文主张状态：发现可检验的指标响应现象；尚未建立人类偏好一致性，也未证明新指标或 prompt 路由收益。']
(out / 'LOCAL_METRIC_AUDIT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps(result, indent=2, ensure_ascii=False))
