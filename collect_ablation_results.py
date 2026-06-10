"""汇总所有消融实验结果 → 对比表"""
import json, os, glob

variants = ['full', 'no_struct', 'no_physico', 'no_chembias', 'no_divreg', 'no_natreg', 'no_mlm']
labels = {
    'full': '完整模型', 'no_struct': '无结构轨', 'no_physico': '无理化编码',
    'no_chembias': '无化学偏置', 'no_divreg': '无多样正则', 'no_natreg': '无天然正则',
    'no_mlm': '无MLM辅助',
}
descs = {
    'full': '所有组件',
    'no_struct': '移除StructureTrack',
    'no_physico': '移除PhysicoEncoder',
    'no_chembias': '移除ChemBias',
    'no_divreg': '移除熵+边际+重复+均匀',
    'no_natreg': '移除二肽KL+k-mer',
    'no_mlm': '移除MLM辅助任务',
}

results = {}
for v in variants:
    path = f'logs_ablation/{v}/eval_result.json'
    if os.path.exists(path):
        with open(path) as f:
            results[v] = json.load(f)

if not results:
    print('未找到消融结果。请先运行各变体。')
    exit(1)

# 打印对比表
full = results.get('full', {})
sep = '-' * 108
print(f'\n{"="*108}')
print('  消融实验结果汇总')
print(f'{"="*108}')
print(f'{"变体":<16} {"天然度":>8} {"vsFull":>8} {"理化":>6} {"多样性":>6} {"综合":>6} {"移除组件"}')
print(sep)
for v in variants:
    if v not in results: continue
    r = results[v]
    nat = r['naturalness']
    phys = r['physico_mean']
    div = r['diversity']
    comp = r['composite']
    rem = descs.get(v, '?')

    if 'naturalness' in full and nat != full['naturalness']:
        delta = (nat - full['naturalness'])
        vs_str = f'{delta:+.3f}'
    else:
        vs_str = '  —'

    print(f'{labels[v]:<16} {nat:>8.4f} {vs_str:>8} {phys:>6.3f} {div:>6.3f} {comp:>6.3f}  {rem}')
print(sep)

# 找出最大贡献者
if 'full' in results:
    full_nat = results['full']['naturalness']
    print(f'\n天然度贡献分析 (相对完整模型):')
    for v in variants:
        if v == 'full' or v not in results: continue
        delta = results[v]['naturalness'] - full_nat
        label = labels[v]
        bar = '█' * abs(int(delta * 50)) if delta < 0 else ''
        pct = f'{delta*100:.1f}%'
        print(f'  {label:<16} {delta:+.4f} ({pct:>6}) {bar}')
