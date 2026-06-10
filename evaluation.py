"""
FoldPath-LLM: Model Evaluation & Benchmark
纯序列评估系统 — 不依赖结构预测，直接评估生成序列质量

三项核心指标:
1. 理化合理性 (Physicochemical Score) — 序列本身的理化性质是否合理
2. 序列多样性 (Diversity Score) — 生成是否多样化，是否发生模式崩塌
3. 天然相似度 (Naturalness Score) — 与天然 Cytochrome b 的相似程度
"""

import numpy as np
from collections import Counter
from scipy import stats
import os
import json

from config import AMINO_ACIDS, AA_TO_IDX, IDX_TO_AA
from physicochemical import PHYSICO_MATRIX, PHYSICO_MATRIX_NORM, PHYSICO_MEAN, PHYSICO_STD

# ============================================================
# 氨基酸理化查找表
# ============================================================
AA_LIST = list(AMINO_ACIDS)
HYDRO_IDX = 0    # 疏水指数
VOLUME_IDX = 1   # 侧链体积
CHARGE_IDX = 2   # 电荷
FLEX_IDX = 3     # 柔性
HB_DONOR_IDX = 4 # 氢键供体
HB_ACCEPT_IDX = 5 # 氢键受体
HELIX_IDX = 6    # 螺旋偏好
SHEET_IDX = 7    # 折叠偏好
TURN_IDX = 8     # 转角偏好
AROMATIC_IDX = 9 # 芳香性
DISULFIDE_IDX = 10 # 二硫键
PKA_IDX = 11     # pKa

# 氨基酸分子量 (Da)
AA_MASS = {
    'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13, 'F': 165.19,
    'G': 75.07, 'H': 155.16, 'I': 131.17, 'K': 146.19, 'L': 131.17,
    'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
    'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19,
}


# ============================================================
# 1. 序列级理化合理性评估
# ============================================================

def _get_prop_array(sequence):
    """将氨基酸序列转换为理化矩阵 [L, 12]"""
    indices = [AA_TO_IDX.get(aa, 0) for aa in sequence if aa in AA_TO_IDX]
    return PHYSICO_MATRIX[indices] if indices else np.zeros((0, 12))


def _sigmoid(x, center=0.0, sharpness=8.0):
    """软化阶跃函数"""
    z = np.clip(sharpness * (x - center), -20, 20)
    return 1.0 / (1.0 + np.exp(-z))


class SequencePhysicochemicalScorer:
    """纯序列理化评分 — 不依赖结构预测"""

    def score(self, sequence):
        """返回各分项得分和综合理化分数 [0, 1]"""
        if len(sequence) < 10:
            return {'total': 0.0, 'detail': {}}

        props = _get_prop_array(sequence)
        L = len(sequence)
        hydro = props[:, HYDRO_IDX]
        charge = props[:, CHARGE_IDX]
        volume = props[:, VOLUME_IDX]
        aromatic = props[:, AROMATIC_IDX]
        helix = props[:, HELIX_IDX]
        sheet = props[:, SHEET_IDX]
        turn = props[:, TURN_IDX]
        hb_donor = props[:, HB_DONOR_IDX]
        hb_acceptor = props[:, HB_ACCEPT_IDX]
        disulfide = props[:, DISULFIDE_IDX]
        flex = props[:, FLEX_IDX]

        scores = {}

        # 1) 疏水比例: 天然蛋白 30%-55%
        hydro_ratio = (hydro > 1.0).mean()
        if 0.30 <= hydro_ratio <= 0.55:
            scores['hydro_ratio'] = 1.0
        elif hydro_ratio < 0.20 or hydro_ratio > 0.65:
            scores['hydro_ratio'] = 0.3
        else:
            scores['hydro_ratio'] = 0.7

        # 2) 电荷平衡: 正负电荷比例各 ≤28%
        pos_ratio = (charge > 0.5).mean()
        neg_ratio = (charge < -0.5).mean()
        charge_balance = 1.0 - abs(pos_ratio - neg_ratio)
        scores['charge_balance'] = float(np.clip(charge_balance, 0.2, 1.0))

        # 3) 净电荷密度: |net charge|/L ≤ 0.25
        net_charge_density = abs(charge.sum()) / L
        scores['net_charge'] = float(np.clip(1.0 - net_charge_density / 0.25, 0.2, 1.0))

        # 4) 芳香残基: 天然 6%-12%
        aro_ratio = (aromatic > 0.5).mean()
        if 0.04 <= aro_ratio <= 0.14:
            scores['aromatic'] = 1.0
        elif aro_ratio < 0.02 or aro_ratio > 0.20:
            scores['aromatic'] = 0.3
        else:
            scores['aromatic'] = 0.6

        # 5) Pro含量: ≤8% (过多Pro破坏二级结构)
        pro_count = sequence.count('P')
        pro_ratio = pro_count / L
        scores['proline'] = float(np.clip(1.0 - pro_ratio / 0.08, 0.1, 1.0))

        # 6) Gly含量: 5%-12%
        gly_count = sequence.count('G')
        gly_ratio = gly_count / L
        if 0.03 <= gly_ratio <= 0.15:
            scores['glycine'] = 1.0
        else:
            scores['glycine'] = 0.6

        # 7) 疏水分布: 连续疏水 ≤5
        max_consecutive = 0
        current = 0
        for h in hydro:
            if h > 1.0:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        scores['hydro_clustering'] = float(np.clip(1.0 - (max_consecutive - 3) / 7.0, 0.1, 1.0))

        # 8) 氢键能力: 供体+受体比例 ≥40%
        hb_capable = ((hb_donor > 0) | (hb_acceptor > 0)).mean()
        scores['hb_capacity'] = float(np.clip(hb_capable / 0.5, 0.2, 1.0))

        # 9) 二级结构偏好多样性: 螺旋/折叠/转角不应全偏向一种
        ss_prefs = np.array([
            helix.mean(), sheet.mean(), turn.mean()
        ])
        ss_entropy = -np.sum(ss_prefs * np.log(ss_prefs + 1e-8)) / np.log(3)
        scores['ss_diversity'] = float(np.clip(ss_entropy / 0.8, 0.3, 1.0))

        # 10) Cys 含量: ≤5% (避免不配对半胱氨酸)
        cys_ratio = sequence.count('C') / L
        scores['cysteine'] = float(np.clip(1.0 - cys_ratio / 0.05, 0.1, 1.0))

        # 11) 等电点合理性: 略低于7 (线粒体内膜蛋白偏碱性)
        # 简化计算: 统计 Arg+Lys vs Asp+Glu
        basic = sequence.count('R') + sequence.count('K') + sequence.count('H') * 0.5
        acidic = sequence.count('D') + sequence.count('E')
        if L > 0:
            charge_index = (basic - acidic) / L
            # 膜蛋白 charge_index 典型 -0.02~0.08
            scores['pi_estimate'] = float(np.clip(1.0 - abs(charge_index) / 0.15, 0.3, 1.0))

        # 12) 柔性: 不宜过高或过低
        flex_mean = flex.mean()
        scores['flexibility'] = float(np.clip(1.0 - abs(flex_mean - 0.42) / 0.15, 0.3, 1.0))

        # 加权综合
        weights = {
            'hydro_ratio': 0.15, 'charge_balance': 0.10, 'net_charge': 0.08,
            'aromatic': 0.06, 'proline': 0.07, 'glycine': 0.05,
            'hydro_clustering': 0.12, 'hb_capacity': 0.08, 'ss_diversity': 0.06,
            'cysteine': 0.08, 'pi_estimate': 0.07, 'flexibility': 0.05,
        }
        total = sum(weights[k] * scores[k] for k in weights if k in scores)
        return {'total': float(total), 'detail': scores}

    def batch_score(self, sequences):
        """批量评分，返回汇总统计"""
        individual = [self.score(s) for s in sequences]
        totals = [s['total'] for s in individual]
        return {
            'mean': float(np.mean(totals)),
            'std': float(np.std(totals)),
            'min': float(np.min(totals)),
            'max': float(np.max(totals)),
            'median': float(np.median(totals)),
            'individual': individual,
        }


# ============================================================
# 2. 序列多样性评估
# ============================================================

class DiversityScorer:
    """评估生成序列的多样性，检测模式崩塌"""

    def score(self, sequences):
        """返回多样性分数 [0, 1]"""
        n = len(sequences)
        if n < 2:
            return {'total': 0.0, 'detail': {}}

        scores = {}

        # 1) 平均两两序列 identity (越低越好)
        identities = []
        for i in range(min(n, 100)):  # 限制计算量
            for j in range(i + 1, min(n, 100)):
                if len(sequences[i]) > 0 and len(sequences[j]) > 0:
                    min_len = min(len(sequences[i]), len(sequences[j]))
                    matches = sum(1 for k in range(min_len)
                                  if sequences[i][k] == sequences[j][k])
                    identities.append(matches / min_len if min_len > 0 else 0)

        if identities:
            avg_identity = np.mean(identities)
            # 天然同源蛋白 identity ~20-90%, 生成应 >30%
            # identity 过低 = 随机噪声, 过高 = 模式崩塌
            if 0.15 <= avg_identity <= 0.70:
                scores['pairwise_identity'] = 1.0
            elif avg_identity > 0.85:
                scores['pairwise_identity'] = 0.1  # 严重崩塌
            elif avg_identity > 0.70:
                scores['pairwise_identity'] = 0.4
            else:
                scores['pairwise_identity'] = 0.5  # 太随机
        else:
            scores['pairwise_identity'] = 0.5

        # 2) 氨基酸组成熵 (跨序列)
        all_aa_freqs = []
        for seq in sequences:
            counter = Counter(seq)
            total = len(seq) if len(seq) > 0 else 1
            freqs = np.array([counter.get(aa, 0) / total for aa in AA_LIST])
            all_aa_freqs.append(freqs)
        avg_freqs = np.mean(all_aa_freqs, axis=0)
        # 加入拉普拉斯平滑
        avg_freqs_smooth = (avg_freqs * n + 1.0/20) / (n + 1)
        aa_entropy = -np.sum(avg_freqs_smooth * np.log(avg_freqs_smooth))
        max_entropy = np.log(20)
        scores['composition_entropy'] = float(np.clip(aa_entropy / max_entropy, 0.3, 1.0))

        # 3) 长度多样性
        lengths = np.array([len(s) for s in sequences])
        length_cv = np.std(lengths) / (np.mean(lengths) + 1e-8)
        scores['length_diversity'] = float(np.clip(length_cv / 0.3, 0.0, 1.0))

        # 4) 唯一 k-mer 比例 (k=5)
        all_kmers = set()
        total_kmers = 0
        for seq in sequences:
            for i in range(len(seq) - 4):
                all_kmers.add(seq[i:i+5])
                total_kmers += 1
        unique_ratio = len(all_kmers) / max(total_kmers, 1)
        scores['unique_kmer'] = float(np.clip(unique_ratio * 5, 0.0, 1.0))

        # 5) 重复序列检测
        unique_seqs = set(sequences)
        scores['unique_ratio'] = len(unique_seqs) / n

        # 加权综合
        weights = {
            'pairwise_identity': 0.30, 'composition_entropy': 0.25,
            'length_diversity': 0.10, 'unique_kmer': 0.20, 'unique_ratio': 0.15,
        }
        total = sum(weights[k] * scores[k] for k in weights if k in scores)
        return {'total': float(total), 'detail': scores}


# ============================================================
# 3. 天然相似度评估
# ============================================================

class NaturalnessComparator:
    """与天然 Cytochrome b 序列比较"""

    def __init__(self, reference_fasta=None):
        """
        Args:
            reference_fasta: 天然序列 FASTA 文件路径
        """
        self.reference_seqs = []
        self.ref_aa_freq = None
        self.ref_dipep_freq = None
        self.ref_kmer_index = None

        if reference_fasta and os.path.exists(reference_fasta):
            self._load_reference(reference_fasta)

    def _load_reference(self, fasta_path):
        """加载参考序列并计算统计特征"""
        from dataset import load_fasta
        self.reference_seqs = load_fasta(fasta_path, max_seqs=5000)
        if not self.reference_seqs:
            return

        # 氨基酸频率
        all_aa = ''.join(self.reference_seqs)
        counter = Counter(all_aa)
        total = sum(counter.values())
        self.ref_aa_freq = np.array([counter.get(aa, 0) / total for aa in AA_LIST])

        # 二肽频率
        dipep_counter = Counter()
        for seq in self.reference_seqs:
            for i in range(len(seq) - 1):
                dipep_counter[seq[i:i+2]] += 1
        dipep_total = sum(dipep_counter.values())
        self.ref_dipep_freq = {dp: c / dipep_total for dp, c in dipep_counter.items()
                              if dp[0] in AA_TO_IDX and dp[1] in AA_TO_IDX}

        # k-mer 索引 (k=7, 用于最近邻匹配)
        self.ref_kmer_index = {}
        for seq in self.reference_seqs:
            for i in range(len(seq) - 6):
                kmer = seq[i:i+7]
                if kmer not in self.ref_kmer_index:
                    self.ref_kmer_index[kmer] = 0
                self.ref_kmer_index[kmer] += 1

        print(f"[Eval] 已加载 {len(self.reference_seqs)} 条天然参考序列")

    def score(self, sequence):
        """计算单条序列的天然相似度 [0, 1]"""
        if not self.reference_seqs:
            return {'total': 0.5, 'detail': {}}

        L = len(sequence)
        if L < 10:
            return {'total': 0.0, 'detail': {}}

        scores = {}

        # 1) AA 组成 JS 散度 (越低越相似)
        counter = Counter(sequence)
        gen_freq = np.array([counter.get(aa, 0) / L for aa in AA_LIST])
        # Jensen-Shannon divergence
        M = (gen_freq + self.ref_aa_freq) / 2
        kl_gen = np.sum(gen_freq * np.log((gen_freq + 1e-9) / (M + 1e-9)))
        kl_ref = np.sum(self.ref_aa_freq * np.log((self.ref_aa_freq + 1e-9) / (M + 1e-9)))
        js_div = (kl_gen + kl_ref) / 2
        scores['aa_js'] = float(np.clip(1.0 - js_div / 0.5, 0.0, 1.0))

        # 2) 二肽频率 Pearson 相关
        gen_dipep = Counter()
        for i in range(L - 1):
            dp = sequence[i:i+2]
            if dp[0] in AA_TO_IDX and dp[1] in AA_TO_IDX:
                gen_dipep[dp] += 1
        dipep_total = sum(gen_dipep.values())
        if dipep_total > 0:
            common_dipeps = set(self.ref_dipep_freq.keys()) & set(gen_dipep.keys())
            if len(common_dipeps) > 50:
                gen_vec = [gen_dipep.get(dp, 0) / dipep_total for dp in common_dipeps]
                ref_vec = [self.ref_dipep_freq[dp] for dp in common_dipeps]
                corr, _ = stats.pearsonr(gen_vec, ref_vec)
                scores['dipep_corr'] = float(np.clip((corr + 1) / 2, 0.0, 1.0))
            else:
                scores['dipep_corr'] = 0.3
        else:
            scores['dipep_corr'] = 0.3

        # 3) k-mer 重叠率: 生成序列中有多少 7-mer 出现在天然库中
        if self.ref_kmer_index:
            kmers_found = 0
            kmers_total = 0
            for i in range(L - 6):
                kmer = sequence[i:i+7]
                kmers_total += 1
                if kmer in self.ref_kmer_index:
                    kmers_found += 1
            scores['kmer_recall'] = kmers_found / max(kmers_total, 1)
        else:
            scores['kmer_recall'] = 0.5

        # 4) 长度合理性 (与参考集比较)
        ref_lengths = [len(s) for s in self.reference_seqs]
        ref_mean = np.mean(ref_lengths)
        ref_std = np.std(ref_lengths)
        length_z = abs(L - ref_mean) / (ref_std + 1e-8)
        scores['length_naturalness'] = float(np.clip(1.0 - length_z / 4.0, 0.0, 1.0))

        # 5) 疏水模式周期性 (跨膜蛋白特征: 每3.6残基一个疏水峰)
        props = _get_prop_array(sequence)
        hydro = props[:, HYDRO_IDX]
        if len(hydro) > 30:
            # 自相关检测螺旋周期性
            autocorr = np.correlate(hydro - hydro.mean(), hydro - hydro.mean(), mode='same')
            center = len(autocorr) // 2
            period_3_5 = autocorr[center + 3:center + 5].mean() if center + 5 < len(autocorr) else 0
            period_random = autocorr[center + 6:center + 10].mean() if center + 10 < len(autocorr) else 1e-8
            periodicity = period_3_5 / (abs(period_random) + 1e-8)
            scores['helix_periodicity'] = float(np.clip(_sigmoid(periodicity, 1.5, 2.0), 0.0, 1.0))
        else:
            scores['helix_periodicity'] = 0.5

        # 加权综合
        weights = {
            'aa_js': 0.25, 'dipep_corr': 0.20, 'kmer_recall': 0.25,
            'length_naturalness': 0.15, 'helix_periodicity': 0.15,
        }
        total = sum(weights[k] * scores[k] for k in weights if k in scores)
        return {'total': float(total), 'detail': scores}

    def batch_score(self, sequences):
        """批量评分"""
        individual = [self.score(s) for s in sequences]
        totals = [s['total'] for s in individual]
        return {
            'mean': float(np.mean(totals)),
            'std': float(np.std(totals)),
            'min': float(np.min(totals)),
            'max': float(np.max(totals)),
            'median': float(np.median(totals)),
            'individual': individual,
        }


# ============================================================
# 4. 综合评测系统 (替代 P 值)
# ============================================================

class FoldPathBenchmark:
    """
    FoldPath-LLM 综合评测基准

    三项核心指标:
    - Physicochemical Score (理化分): 序列本身的理化合理性 [0-1], 目标 ≥0.65
    - Diversity Score (多样性分): 生成多样性, 防崩塌 [0-1], 目标 ≥0.60
    - Naturalness Score (天然度分): 与天然 Cytochrome b 的相似度 [0-1], 目标 ≥0.45

    综合分 = 0.35 * 理化 + 0.25 * 多样性 + 0.40 * 天然度

    不再使用 P 值 (Next-Token Accuracy) 作为评价指标。
    """

    def __init__(self, reference_fasta=None):
        self.physico_scorer = SequencePhysicochemicalScorer()
        self.diversity_scorer = DiversityScorer()
        self.naturalness = NaturalnessComparator(reference_fasta)

    def evaluate(self, sequences, verbose=True):
        """
        对生成序列进行综合评测

        Args:
            sequences: list of str, 氨基酸序列
            verbose: 是否打印详细报告
        Returns:
            dict: {
                'physico': {...}, 'diversity': {...}, 'naturalness': {...},
                'composite': float, 'grade': str
            }
        """
        if not sequences:
            return {'error': 'No sequences provided'}

        n = len(sequences)
        lengths = [len(s) for s in sequences]

        # 三项评估
        physico_result = self.physico_scorer.batch_score(sequences)
        diversity_result = self.diversity_scorer.score(sequences)
        naturalness_result = self.naturalness.batch_score(sequences)

        # 综合分
        composite = (
            0.35 * physico_result['mean'] +
            0.25 * diversity_result['total'] +
            0.40 * naturalness_result['mean']
        )

        # 评级
        if composite >= 0.70:
            grade = 'A (优秀)'
        elif composite >= 0.55:
            grade = 'B (良好)'
        elif composite >= 0.40:
            grade = 'C (一般)'
        elif composite >= 0.25:
            grade = 'D (较差)'
        else:
            grade = 'F (不可用)'

        result = {
            'physico': physico_result,
            'diversity': diversity_result,
            'naturalness': naturalness_result,
            'composite': float(composite),
            'grade': grade,
            'num_sequences': n,
            'length_stats': {
                'mean': float(np.mean(lengths)),
                'std': float(np.std(lengths)),
                'min': int(np.min(lengths)),
                'max': int(np.max(lengths)),
            }
        }

        if verbose:
            self._print_report(result)

        return result

    def _print_report(self, result):
        """打印可视化评测报告"""
        print("\n" + "=" * 62)
        print("  FoldPath-LLM 综合评测报告")
        print("=" * 62)

        n = result['num_sequences']
        ls = result['length_stats']
        print(f"\n  生成序列数: {n}")
        print(f"  长度: {ls['mean']:.0f} ± {ls['std']:.0f} (范围 {ls['min']}-{ls['max']})")

        # ── 理化分 ──
        p = result['physico']
        self._print_section("理化合理性", p['mean'], p['std'], 0.65,
                           p.get('individual', []))

        # ── 多样性分 ──
        d = result['diversity']
        self._print_section("序列多样性", d['total'], None, 0.60,
                           detail=d.get('detail', {}))

        # ── 天然度分 ──
        n_res = result['naturalness']
        self._print_section("天然相似度", n_res['mean'], n_res['std'], 0.45,
                           n_res.get('individual', []))

        # ── 综合 ──
        print(f"\n  {'─' * 56}")
        bar_len = int(result['composite'] * 28)
        bar = "█" * bar_len + "░" * (28 - bar_len)
        print(f"  ★ 综合评分: [{bar}] {result['composite']:.0%}")
        print(f"  ★ 等级: {result['grade']}")
        print(f"\n  评分标准: 理化(35%) + 多样性(25%) + 天然度(40%)")
        print(f"  注意: P 值 (Next-Token Accuracy) 不再作为评价指标")
        print("=" * 62 + "\n")

    def _print_section(self, name, mean_val, std_val, threshold, individual=None, detail=None):
        """打印单个评估维度"""
        bar_len = int(mean_val * 28)
        bar = "█" * bar_len + "░" * (28 - bar_len)
        status = "✓" if mean_val >= threshold else "✗"
        std_str = f" ± {std_val:.3f}" if std_val is not None else ""
        print(f"\n  {status} {name}: [{bar}] {mean_val:.3f}{std_str}  (阈值≥{threshold:.2f})")

        if detail and isinstance(detail, dict):
            # 打印子项
            key_labels = {
                'pairwise_identity': '平均两两identity',
                'composition_entropy': 'AA组成熵',
                'length_diversity': '长度多样性',
                'unique_kmer': '唯一k-mer比',
                'unique_ratio': '唯一序列比',
            }
            for key, label in key_labels.items():
                if key in detail:
                    v = detail[key]
                    s = "✓" if v >= 0.5 else "✗"
                    print(f"     {s} {label}: {v:.3f}")

    def benchmark_checkpoint(self, generator, num_samples=50, save_path=None):
        """
        对指定 checkpoint 的模型进行标准评测

        Args:
            generator: ProteinGenerator 实例
            num_samples: 生成序列数量
            save_path: 结果保存路径 (可选)
        Returns:
            dict: 评测结果
        """
        print(f"\n{'='*50}")
        print(f"  开始标准评测 ({num_samples} 条序列)")
        print(f"{'='*50}")

        from config import GenerateConfig
        config = GenerateConfig()
        config.num_samples = num_samples
        config.temperature = 1.0
        config.top_k = 50
        config.top_p = 0.92
        config.use_physico_filter = True

        sequences, _ = generator.generate(config)
        sequences = [s for s in sequences if len(s) >= 20]  # 过滤过短序列

        result = self.evaluate(sequences, verbose=True)

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            serializable = {
                'physico_mean': result['physico']['mean'],
                'physico_std': result['physico']['std'],
                'diversity': result['diversity']['total'],
                'naturalness_mean': result['naturalness']['mean'],
                'naturalness_std': result['naturalness']['std'],
                'composite': result['composite'],
                'grade': result['grade'],
                'num_sequences': result['num_sequences'],
                'length_stats': result['length_stats'],
            }
            with open(save_path, 'w') as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            print(f"  结果已保存: {save_path}")

        return result


# ============================================================
# 5. 训练过程中的快速评估
# ============================================================

class TrainingEvaluator:
    """
    训练过程中的轻量评估器 — 在每个 epoch 结束后调用。
    不依赖生成（训练中模型可能未收敛），只评估训练日志和验证集统计。
    """

    @staticmethod
    def evaluate_training_state(history, val_sequences=None):
        """
        基于训练历史评估当前状态

        Args:
            history: dict, trainer.history
            val_sequences: list of str, 验证集序列 (可选)

        Returns:
            dict: 训练状态评估
        """
        result = {}

        # 1) Loss 收敛趋势
        if len(history.get('val_loss', [])) >= 3:
            recent_val = history['val_loss'][-3:]
            if recent_val[-1] < recent_val[0] * 0.95:
                result['convergence'] = 'still_improving'
            elif recent_val[-1] >= recent_val[0]:
                result['convergence'] = 'stalled'
            else:
                result['convergence'] = 'slowing'

        # 2) 过拟合检测
        if len(history.get('train_loss', [])) >= 3 and len(history.get('val_loss', [])) >= 3:
            train_trend = history['train_loss'][-3:]
            val_trend = history['val_loss'][-3:]
            train_improvement = (train_trend[0] - train_trend[-1]) / max(train_trend[0], 1e-8)
            val_improvement = (val_trend[0] - val_trend[-1]) / max(val_trend[0], 1e-8)
            if val_improvement < -0.05 and train_improvement > 0.02:
                result['overfit_warning'] = True
            else:
                result['overfit_warning'] = False

        # 3) P 值趋势 (仅作参考标注)
        if len(history.get('precision', [])) >= 2:
            recent_p = history['precision'][-3:]
            result['p_value'] = float(np.mean(recent_p))
            if result['p_value'] > 0.75:
                result['p_warning'] = 'P值偏高(>0.75)，可能是ESM双向嵌入贡献或数据保守性高。不应作为核心指标。'

        # 4) 结构损失是否崩塌
        if len(history.get('train_struct_loss', [])) >= 3:
            struct_vals = history['train_struct_loss'][-3:]
            if np.mean(struct_vals) < 0.01:
                result['struct_collapse'] = True

        return result


# ============================================================
# 命令行入口
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='FoldPath-LLM Model Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型 checkpoint 路径')
    parser.add_argument('--reference', type=str, default='data/train_sequences.fasta',
                        help='天然参考序列 FASTA')
    parser.add_argument('--num-samples', type=int, default=50, help='生成序列数量')
    parser.add_argument('--no-esm', action='store_true', help='禁用 ESM 基座')
    parser.add_argument('--save', type=str, default=None, help='结果保存路径')
    args = parser.parse_args()

    from generate import ProteinGenerator

    use_esm = not args.no_esm
    generator = ProteinGenerator(
        checkpoint_path=args.checkpoint,
        use_esm=use_esm,
    )

    benchmark = FoldPathBenchmark(reference_fasta=args.reference)
    result = benchmark.benchmark_checkpoint(
        generator,
        num_samples=args.num_samples,
        save_path=args.save,
    )

    return result


if __name__ == "__main__":
    main()
