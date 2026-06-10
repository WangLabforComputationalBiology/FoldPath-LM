"""
FoldPath-LLM: Physicochemical Property Encoder & 15-Indicator Evaluation
折叠路径引导的蛋白质设计大语言模型 - 理化性质编码与15项评估模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import AMINO_ACIDS, AA_TO_IDX

# ============================================================
# 20种氨基酸的理化性质数据库
# ============================================================
# 每个氨基酸12维特征:
# [疏水指数, 侧链体积, 电荷(pH7), 柔性, 氢键供体数, 氢键受体数,
#  螺旋偏好, 折叠偏好, 转角偏好, 芳香性(0/1), 二硫键能力(0/1), pKa]

PHYSICOCHEMICAL_TABLE = {
    'A': [ 1.8,  88.6,  0.0, 0.36, 0, 0, 1.42, 0.83, 0.66, 0, 0, 0.0],
    'C': [ 2.5, 108.5,  0.0, 0.35, 1, 0, 0.70, 1.19, 0.77, 0, 1, 8.3],
    'D': [-3.5, 111.1, -1.0, 0.51, 1, 2, 1.01, 0.54, 1.46, 0, 0, 3.9],
    'E': [-3.5, 138.4, -1.0, 0.50, 1, 2, 1.51, 0.37, 0.74, 0, 0, 4.1],
    'F': [ 2.8, 189.9,  0.0, 0.31, 0, 0, 1.13, 1.38, 0.60, 1, 0, 0.0],
    'G': [-0.4,  60.1,  0.0, 0.54, 0, 0, 0.57, 0.75, 1.56, 0, 0, 0.0],
    'H': [-3.2, 153.2,  0.5, 0.32, 2, 1, 1.00, 0.87, 0.95, 1, 0, 6.0],
    'I': [ 4.5, 166.7,  0.0, 0.46, 0, 0, 1.08, 1.60, 0.47, 0, 0, 0.0],
    'K': [-3.9, 168.6,  1.0, 0.47, 2, 1, 1.16, 0.74, 1.01, 0, 0, 10.5],
    'L': [ 3.8, 166.7,  0.0, 0.37, 0, 0, 1.21, 1.30, 0.59, 0, 0, 0.0],
    'M': [ 1.9, 162.9,  0.0, 0.30, 0, 0, 1.45, 1.05, 0.60, 0, 0, 0.0],
    'N': [-3.5, 114.1,  0.0, 0.46, 2, 2, 0.67, 0.89, 1.56, 0, 0, 0.0],
    'P': [-1.6, 112.7,  0.0, 0.51, 0, 0, 0.57, 0.55, 1.52, 0, 0, 0.0],
    'Q': [-3.5, 143.8,  0.0, 0.46, 2, 2, 1.11, 1.10, 0.98, 0, 0, 0.0],
    'R': [-4.5, 173.4,  1.0, 0.53, 3, 1, 0.98, 0.93, 0.95, 0, 0, 12.5],
    'S': [-0.8,  89.0,  0.0, 0.51, 1, 1, 0.77, 0.75, 1.43, 0, 0, 0.0],
    'T': [-0.7, 116.1,  0.0, 0.44, 1, 1, 0.83, 1.19, 0.96, 0, 0, 0.0],
    'V': [ 4.2, 140.0,  0.0, 0.39, 0, 0, 1.06, 1.70, 0.50, 0, 0, 0.0],
    'W': [-0.9, 227.8,  0.0, 0.31, 1, 0, 1.08, 1.37, 0.96, 1, 0, 0.0],
    'Y': [-1.3, 193.6,  0.0, 0.42, 1, 1, 0.69, 1.47, 1.14, 1, 0, 10.1],
}

PHYSICO_MATRIX = np.array([PHYSICOCHEMICAL_TABLE[aa] for aa in AMINO_ACIDS], dtype=np.float32)
PHYSICO_MEAN = PHYSICO_MATRIX.mean(axis=0)
PHYSICO_STD = PHYSICO_MATRIX.std(axis=0) + 1e-8
PHYSICO_MATRIX_NORM = (PHYSICO_MATRIX - PHYSICO_MEAN) / PHYSICO_STD


class PhysicochemicalEncoder(nn.Module):
    """将氨基酸的理化性质编码为固定维度的向量"""
    def __init__(self, raw_dim=12, embed_dim=32):
        super().__init__()
        self.register_buffer('physico_table',
            torch.tensor(PHYSICO_MATRIX_NORM, dtype=torch.float32))
        self.encoder = nn.Sequential(
            nn.Linear(raw_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.finetune_table = nn.Parameter(
            torch.zeros(20, raw_dim), requires_grad=True
        )
        self.use_finetune = False

    def forward(self, aa_indices):
        """aa_indices: (B, L) -> (B, L, physico_dim)"""
        physico_feat = self.physico_table[aa_indices]
        if self.use_finetune:
            physico_feat = physico_feat + self.finetune_table[aa_indices]
        return self.encoder(physico_feat)


class ChemicalInteractionBias(nn.Module):
    """计算残基对之间的理化交互偏置，注入注意力机制（显存优化版）"""
    def __init__(self, physico_dim=32, struct_latent_dim=64, bias_dim=16):
        super().__init__()
        self.physico_dim = physico_dim
        self.struct_latent_dim = struct_latent_dim
        self.bias_dim = bias_dim
        self.physico_proj = nn.Linear(physico_dim, bias_dim)
        self.struct_proj = nn.Linear(struct_latent_dim, bias_dim) if struct_latent_dim > 0 else None
        self.out_proj = nn.Sequential(nn.GELU(), nn.Linear(bias_dim, 1))

    def forward(self, physico_embed, struct_latent=None):
        """physico_embed: (B, L, D) -> bias: (B, 1, L, L)
        Memory-efficient: project to bias_dim first, then pairwise add"""
        B, L, _ = physico_embed.shape
        pi = self.physico_proj(physico_embed)
        if struct_latent is not None and self.struct_proj is not None:
            si = self.struct_proj(struct_latent)
            combined = pi + si
        else:
            combined = pi
        ci = combined.unsqueeze(2)
        cj = combined.unsqueeze(1)
        pair = ci + cj
        bias = self.out_proj(pair).squeeze(-1)
        return bias.unsqueeze(1)


class PhysicochemicalEvaluator:
    """蛋白质理化性质15项评估系统"""

    def __init__(self, device='cpu'):
        self.device = device
        self.physico_table = torch.tensor(PHYSICO_MATRIX, dtype=torch.float32, device=device)

    def _get_prop(self, sequence, prop_idx):
        indices = [AA_TO_IDX.get(aa, 0) for aa in sequence]
        return self.physico_table[indices, prop_idx]

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -20, 20)
        return 1.0 / (1.0 + np.exp(-x))

    def get_status(self, score):
        if score >= 0.7: return "✅"
        elif score >= 0.4: return "⚠️"
        else: return "❌"

    # --- 第一类: 内部稳定性 ---

    def eval_hydrophobic_core(self, sequence, exposure_scores):
        if len(sequence) == 0: return 0.0
        hydro = self._get_prop(sequence, 0).cpu().numpy()
        exposure = exposure_scores.cpu().numpy() if torch.is_tensor(exposure_scores) else np.array(exposure_scores)
        core_mask = exposure < 0.3
        surface_mask = exposure > 0.7
        core_hydro = hydro[core_mask].mean() if core_mask.sum() > 0 else 0
        surface_hydro = hydro[surface_mask].mean() if surface_mask.sum() > 0 else 0
        gradient = core_hydro - surface_hydro
        if len(hydro) > 2 and np.std(hydro) > 0 and np.std(exposure) > 0:
            correlation = -np.corrcoef(hydro, exposure)[0, 1]
        else:
            correlation = 0.0
        score = 0.4 * self._sigmoid(gradient / 5.0) + 0.3 * self._sigmoid(correlation) + 0.3 * self._sigmoid(core_hydro / 5.0)
        return float(score)

    def eval_packing_density(self, sequence, distance_matrix):
        L = len(sequence)
        if L < 3: return 0.5
        dist = distance_matrix.cpu().numpy() if torch.is_tensor(distance_matrix) else np.array(distance_matrix)
        neighbors = np.sum(dist < 8.0, axis=1) - 1
        avg_neighbors = neighbors.mean()
        density_score = self._sigmoid((avg_neighbors - 5) / 10.0)
        min_neighbors = neighbors.min()
        void_penalty = max(0, (3 - min_neighbors) / 3.0) * 0.2
        return float(density_score - void_penalty)

    def eval_steric_clash(self, sequence, distance_matrix):
        L = len(sequence)
        if L < 2: return 1.0
        dist = distance_matrix.cpu().numpy() if torch.is_tensor(distance_matrix) else np.array(distance_matrix)
        volumes = self._get_prop(sequence, 1).cpu().numpy()
        min_dist = (volumes[:, None] ** (1/3) + volumes[None, :] ** (1/3)) * 0.8
        seq_sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
        mask = seq_sep > 2
        clashes = (dist < min_dist) & mask & (dist > 0)
        clash_rate = clashes.sum() / max(mask.sum(), 1)
        return float(1.0 - clash_rate)

    def eval_hydrogen_bonds(self, sequence, distance_matrix, angle_matrix=None):
        L = len(sequence)
        if L < 4: return 0.5
        dist = distance_matrix.cpu().numpy() if torch.is_tensor(distance_matrix) else np.array(distance_matrix)
        donors = self._get_prop(sequence, 4).cpu().numpy()
        acceptors = self._get_prop(sequence, 5).cpu().numpy()
        hbond_count = 0
        for i in range(L):
            for j in range(i + 3, L):
                if 0 < dist[i, j] < 3.5:
                    if (donors[i] > 0 and acceptors[j] > 0) or (donors[j] > 0 and acceptors[i] > 0):
                        hbond_count += 1
        hbond_density = hbond_count / max(L, 1)
        return float(self._sigmoid((hbond_density - 0.3) / 0.5))

    # --- 第二类: 静电与电荷 ---

    def eval_salt_bridges(self, sequence, distance_matrix):
        L = len(sequence)
        if L < 4: return 0.5
        dist = distance_matrix.cpu().numpy() if torch.is_tensor(distance_matrix) else np.array(distance_matrix)
        charges = self._get_prop(sequence, 2).cpu().numpy()
        salt_bridges = 0
        repulsions = 0
        for i in range(L):
            for j in range(i + 3, L):
                if 0 < dist[i, j] < 6.0:
                    if charges[i] * charges[j] < 0: salt_bridges += 1
                    elif charges[i] * charges[j] > 0: repulsions += 1
        bridge_score = salt_bridges / max(L / 20.0, 1)
        repulsion_penalty = repulsions / max(L / 10.0, 1)
        return float(self._sigmoid(bridge_score - repulsion_penalty))

    def eval_surface_electrostatics(self, sequence, exposure_scores):
        if len(sequence) == 0: return 0.5
        charges = self._get_prop(sequence, 2).cpu().numpy()
        exposure = exposure_scores.cpu().numpy() if torch.is_tensor(exposure_scores) else np.array(exposure_scores)
        surface_mask = exposure > 0.5
        if surface_mask.sum() == 0: return 0.5
        surface_charges = charges[surface_mask]
        pos_ratio = (surface_charges > 0).sum() / max(len(surface_charges), 1)
        neg_ratio = (surface_charges < 0).sum() / max(len(surface_charges), 1)
        balance = 1.0 - abs(pos_ratio - neg_ratio)
        return float(self._sigmoid(balance * 2 - 0.5))

    def eval_ph_dependence(self, sequence, exposure_scores, target_pH=7.0):
        if len(sequence) == 0: return 0.5
        pKa_values = self._get_prop(sequence, 11).cpu().numpy()
        charges = self._get_prop(sequence, 2).cpu().numpy()
        exposure = exposure_scores.cpu().numpy() if torch.is_tensor(exposure_scores) else np.array(exposure_scores)
        protonated = np.zeros(len(sequence))
        for i, pKa in enumerate(pKa_values):
            if pKa > 0:
                if charges[i] > 0:
                    protonated[i] = 1.0 / (1.0 + 10 ** (target_pH - pKa))
                elif charges[i] < 0:
                    protonated[i] = 1.0 / (1.0 + 10 ** (pKa - target_pH))
        surface_mask = exposure > 0.5
        if surface_mask.sum() > 0:
            surface_protonation = protonated[surface_mask].mean()
            score = self._sigmoid(surface_protonation * 2 - 0.3)
        else:
            score = 0.5
        return float(score)

    # --- 第三类: 骨架与构象 ---

    def eval_rama_compliance(self, sequence, ss_types):
        if len(sequence) == 0: return 0.5
        ss = ss_types.cpu().numpy() if torch.is_tensor(ss_types) else np.array(ss_types)
        helix_pref = self._get_prop(sequence, 6).cpu().numpy()
        sheet_pref = self._get_prop(sequence, 7).cpu().numpy()
        coil_pref = self._get_prop(sequence, 8).cpu().numpy()
        compliance = 0.0
        for i in range(len(sequence)):
            if ss[i] == 0: compliance += helix_pref[i]
            elif ss[i] == 1: compliance += sheet_pref[i]
            else: compliance += coil_pref[i]
        avg_compliance = compliance / max(len(sequence), 1)
        return float(self._sigmoid((avg_compliance - 0.7) / 0.5))

    def eval_special_residue_position(self, sequence, ss_types):
        if len(sequence) == 0: return 0.5
        ss = ss_types.cpu().numpy() if torch.is_tensor(ss_types) else np.array(ss_types)
        score_sum = 0.0
        count = 0
        for i, aa in enumerate(sequence):
            if aa == 'P':
                count += 1
                if ss[i] == 0: score_sum += 0.2
                elif ss[i] == 2: score_sum += 1.0
                else: score_sum += 0.5
            elif aa == 'G':
                count += 1
                if ss[i] == 2: score_sum += 1.0
                elif ss[i] == 0: score_sum += 0.4
                else: score_sum += 0.6
        if count == 0: return 0.7
        return float(self._sigmoid((score_sum / count - 0.3) / 0.5))

    def eval_sidechain_conformation(self, sequence, distance_matrix):
        L = len(sequence)
        if L < 2: return 0.8
        dist = distance_matrix.cpu().numpy() if torch.is_tensor(distance_matrix) else np.array(distance_matrix)
        volumes = self._get_prop(sequence, 1).cpu().numpy()
        large_indices = np.where(volumes > 150)[0]
        if len(large_indices) < 2: return 0.8
        feasible = 0
        total = 0
        for i_idx in range(len(large_indices)):
            for j_idx in range(i_idx + 1, len(large_indices)):
                i, j = large_indices[i_idx], large_indices[j_idx]
                if abs(i - j) > 2 and dist[i, j] > 0:
                    total += 1
                    if dist[i, j] > 4.5: feasible += 1
        if total == 0: return 0.8
        return float(feasible / total)

    # --- 第四类: 特异性相互作用 ---

    def eval_aromatic_stacking(self, sequence, distance_matrix):
        L = len(sequence)
        if L < 2: return 0.5
        dist = distance_matrix.cpu().numpy() if torch.is_tensor(distance_matrix) else np.array(distance_matrix)
        aromatic = self._get_prop(sequence, 9).cpu().numpy()
        aromatic_indices = np.where(aromatic > 0.5)[0]
        if len(aromatic_indices) < 2: return 0.5
        stacking_count = 0
        for i_idx in range(len(aromatic_indices)):
            for j_idx in range(i_idx + 1, len(aromatic_indices)):
                i, j = aromatic_indices[i_idx], aromatic_indices[j_idx]
                if 0 < dist[i, j] < 7.0: stacking_count += 1
        max_possible = len(aromatic_indices) * (len(aromatic_indices) - 1) / 2
        stacking_ratio = stacking_count / max(max_possible, 1)
        return float(self._sigmoid((stacking_ratio - 0.1) / 0.4))

    def eval_disulfide_bonds(self, sequence, distance_matrix):
        L = len(sequence)
        if L < 2: return 0.5
        dist = distance_matrix.cpu().numpy() if torch.is_tensor(distance_matrix) else np.array(distance_matrix)
        disulfide_capable = self._get_prop(sequence, 10).cpu().numpy()
        cys_indices = np.where(disulfide_capable > 0.5)[0]
        if len(cys_indices) < 2: return 0.7
        good_pairs = 0
        total_pairs = 0
        for i_idx in range(len(cys_indices)):
            for j_idx in range(i_idx + 1, len(cys_indices)):
                i, j = cys_indices[i_idx], cys_indices[j_idx]
                if abs(i - j) > 2:
                    total_pairs += 1
                    if 3.6 < dist[i, j] < 6.5: good_pairs += 1
        if total_pairs == 0: return 0.7
        return float(good_pairs / total_pairs)

    def eval_cation_pi(self, sequence, distance_matrix):
        L = len(sequence)
        if L < 2: return 0.5
        dist = distance_matrix.cpu().numpy() if torch.is_tensor(distance_matrix) else np.array(distance_matrix)
        charges = self._get_prop(sequence, 2).cpu().numpy()
        aromatic = self._get_prop(sequence, 9).cpu().numpy()
        cation_indices = np.where(charges > 0.5)[0]
        aromatic_indices = np.where(aromatic > 0.5)[0]
        if len(cation_indices) == 0 or len(aromatic_indices) == 0: return 0.5
        catpi_count = 0
        for i in cation_indices:
            for j in aromatic_indices:
                if abs(i - j) > 2 and 0 < dist[i, j] < 6.0: catpi_count += 1
        max_possible = len(cation_indices) * len(aromatic_indices)
        catpi_ratio = catpi_count / max(max_possible, 1)
        return float(self._sigmoid((catpi_ratio - 0.05) / 0.2))

    # --- 第五类: 全局性质 ---

    def eval_hydrophobic_moment(self, sequence, ss_types):
        if len(sequence) == 0: return 0.5
        ss = ss_types.cpu().numpy() if torch.is_tensor(ss_types) else np.array(ss_types)
        hydro = self._get_prop(sequence, 0).cpu().numpy()
        helix_regions = []
        in_helix = False
        start = 0
        for i, s in enumerate(ss):
            if s == 0 and not in_helix:
                start = i; in_helix = True
            elif s != 0 and in_helix:
                if i - start >= 5: helix_regions.append((start, i))
                in_helix = False
        if in_helix and len(ss) - start >= 5:
            helix_regions.append((start, len(ss)))
        if len(helix_regions) == 0: return 0.5
        amphipathic_count = 0
        for start, end in helix_regions:
            h = hydro[start:end]
            n = len(h)
            angle_step = 100 * np.pi / 180
            sin_sum = sum(h[i] * np.sin(i * angle_step) for i in range(n))
            cos_sum = sum(h[i] * np.cos(i * angle_step) for i in range(n))
            moment = np.sqrt(sin_sum**2 + cos_sum**2) / n
            if moment > 1.5: amphipathic_count += 1
        ratio = amphipathic_count / max(len(helix_regions), 1)
        return float(self._sigmoid((ratio - 0.2) / 0.5))

    def eval_aggregation_tendency(self, sequence, exposure_scores):
        if len(sequence) == 0: return 0.5
        hydro = self._get_prop(sequence, 0).cpu().numpy()
        exposure = exposure_scores.cpu().numpy() if torch.is_tensor(exposure_scores) else np.array(exposure_scores)
        surface_hydrophobic = (exposure > 0.5) & (hydro > 1.0)
        patch_size = 0; max_patch = 0
        for h in surface_hydrophobic:
            if h: patch_size += 1; max_patch = max(max_patch, patch_size)
            else: patch_size = 0
        patch_score = self._sigmoid(1.0 - (max_patch - 3) / 8.0)
        surface_hydro_avg = hydro[exposure > 0.5].mean() if (exposure > 0.5).any() else 0
        surface_score = self._sigmoid(1.0 - surface_hydro_avg / 3.0)
        return float(0.6 * patch_score + 0.4 * surface_score)

    # --- 综合评估 ---

    def evaluate_all(self, sequence, exposure_scores, distance_matrix, ss_types, target_pH=7.0):
        results = {}
        results['hydrophobic_core'] = self.eval_hydrophobic_core(sequence, exposure_scores)
        results['packing_density'] = self.eval_packing_density(sequence, distance_matrix)
        results['steric_clash'] = self.eval_steric_clash(sequence, distance_matrix)
        results['hydrogen_bonds'] = self.eval_hydrogen_bonds(sequence, distance_matrix)
        results['salt_bridges'] = self.eval_salt_bridges(sequence, distance_matrix)
        results['surface_electrostatics'] = self.eval_surface_electrostatics(sequence, exposure_scores)
        results['ph_dependence'] = self.eval_ph_dependence(sequence, exposure_scores, target_pH)
        results['rama_compliance'] = self.eval_rama_compliance(sequence, ss_types)
        results['special_residue'] = self.eval_special_residue_position(sequence, ss_types)
        results['sidechain_conformation'] = self.eval_sidechain_conformation(sequence, distance_matrix)
        results['aromatic_stacking'] = self.eval_aromatic_stacking(sequence, distance_matrix)
        results['disulfide_bonds'] = self.eval_disulfide_bonds(sequence, distance_matrix)
        results['cation_pi'] = self.eval_cation_pi(sequence, distance_matrix)
        results['hydrophobic_moment'] = self.eval_hydrophobic_moment(sequence, ss_types)
        results['aggregation'] = self.eval_aggregation_tendency(sequence, exposure_scores)

        weights = {
            'hydrophobic_core': 0.1429, 'packing_density': 0.0762, 'steric_clash': 0.0952,
            'hydrogen_bonds': 0.1048, 'salt_bridges': 0.0571, 'surface_electrostatics': 0.0381,
            'ph_dependence': 0.0286, 'rama_compliance': 0.0667, 'special_residue': 0.0571,
            'sidechain_conformation': 0.0476, 'aromatic_stacking': 0.0667, 'disulfide_bonds': 0.0286,
            'cation_pi': 0.0190, 'hydrophobic_moment': 0.0762, 'aggregation': 0.0952,
        }
        results['total_score'] = sum(weights[k] * results[k] for k in weights)
        results['category_scores'] = {
            '内部稳定性': np.mean([results['hydrophobic_core'], results['packing_density'], results['steric_clash'], results['hydrogen_bonds']]),
            '静电与电荷': np.mean([results['salt_bridges'], results['surface_electrostatics'], results['ph_dependence']]),
            '骨架与构象': np.mean([results['rama_compliance'], results['special_residue'], results['sidechain_conformation']]),
            '特异性相互作用': np.mean([results['aromatic_stacking'], results['disulfide_bonds'], results['cation_pi']]),
            '全局性质': np.mean([results['hydrophobic_moment'], results['aggregation']]),
        }
        return results

    def format_report(self, results):
        category_names = {
            'hydrophobic_core': ('疏水核心分布', '🏠'), 'packing_density': ('包装密度', '🏠'),
            'steric_clash': ('空间位阻', '🏠'), 'hydrogen_bonds': ('氢键网络', '🏠'),
            'salt_bridges': ('盐桥配对', '⚡'), 'surface_electrostatics': ('表面静电', '⚡'),
            'ph_dependence': ('pH依赖', '⚡'),
            'rama_compliance': ('Ramachandran', '🧬'), 'special_residue': ('Pro/Gly位置', '🧬'),
            'sidechain_conformation': ('侧链构象', '🧬'),
            'aromatic_stacking': ('芳香堆积', '🔗'), 'disulfide_bonds': ('二硫键', '🔗'),
            'cation_pi': ('阳离子-π', '🔗'),
            'hydrophobic_moment': ('两亲性', '🎯'), 'aggregation': ('聚集倾向', '🎯'),
        }
        cat_full = {'🏠': '内部稳定性', '⚡': '静电与电荷', '🧬': '骨架与构象', '🔗': '特异性相互作用', '🎯': '全局性质'}
        lines = []
        lines.append("=" * 55)
        lines.append("       蛋白质理化体检报告 (Physico Checkup)")
        lines.append("=" * 55)
        current_cat = ""
        for key, (name, cat) in category_names.items():
            if cat != current_cat:
                current_cat = cat
                lines.append(f"\n  {cat} {cat_full[cat]}")
                lines.append("  " + "-" * 40)
            score = results[key]
            status = self.get_status(score)
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"  ├─ {name:12s} {bar} {score:.0%} {status}")
        lines.append("")
        lines.append(f"  📊 综合评分: {results['total_score']:.0%}/100%")
        lines.append("=" * 55)
        return "\n".join(lines)


class PhysicochemicalLoss(nn.Module):
    """可微的理化评估损失函数（确定性数学约束版 - 无可学习参数，杜绝loss collapse）
    
    改进说明 (v2):
    - 移除 hydro_exposure_net 和 charge_pair_net (可学习评分器会作弊)
    - 疏水-暴露: 直接用 Pearson 相关性 (鼓励负相关 → 疏水核心低暴露)
    - 电荷配对: 统计正负电荷对在距离<6Å的比例 (鼓励正负靠近)
    - 空间位阻: 保留确定性 relu(min_dist - dist) 惩罚 (天然无作弊路径)
    """
    def __init__(self, physico_dim=32):
        super().__init__()
        self.register_buffer('physico_table',
            torch.tensor(PHYSICO_MATRIX_NORM, dtype=torch.float32))

    def forward(self, physico_embed, aa_indices, exposure_pred, distance_pred):
        """
        Args:
            physico_embed: [B, L, physico_dim] 编码后的理化嵌入 (保留兼容，未使用)
            aa_indices: [B, L] 氨基酸索引 (0-19)
            exposure_pred: [B, L] 预测的溶剂暴露度
            distance_pred: [B, L, L] 预测的距离矩阵
        """
        B, L = aa_indices.shape
        
        # 从原始理化表获取真实的理化属性值
        physico_raw = self.physico_table[aa_indices]  # [B, L, 12]
        
        # ── 1. 疏水-暴露一致性: Pearson 相关性 (确定性，无作弊路径) ──
        # 生物物理原理: 疏水残基应埋在蛋白质核心 (低暴露度)
        # → 疏水指数与暴露度应呈负相关
        hydro_feat = physico_raw[:, :, 0]  # [B, L] 疏水指数
        
        # 逐样本计算 Pearson 相关系数
        hydro_mean = hydro_feat.mean(dim=1, keepdim=True)
        hydro_std = hydro_feat.std(dim=1, keepdim=True) + 1e-6
        exp_mean = exposure_pred.mean(dim=1, keepdim=True)
        exp_std = exposure_pred.std(dim=1, keepdim=True) + 1e-6
        
        hydro_norm = (hydro_feat - hydro_mean) / hydro_std
        exp_norm = (exposure_pred - exp_mean) / exp_std
        pearson_corr = (hydro_norm * exp_norm).mean(dim=1)  # [B]
        
        # 期望 pearson_corr < -0.3 (显著负相关)
        # relu(pearson_corr + 0.3) > 0 表示负相关性不够强
        loss_hydro = F.relu(pearson_corr + 0.3).mean()

        # ── 2. 电荷配对 + 空间位阻: 采样 K 个位置 ──
        K = min(64, L)
        if L > K:
            idx = torch.randperm(L, device=aa_indices.device)[:K]
            charge_s = physico_raw[:, idx, 2]   # 电荷 (第2维)
            dist_s = distance_pred[:, idx][:, :, idx]
            vol_s = physico_raw[:, idx, 1]      # 侧链体积 (第1维)
        else:
            charge_s = physico_raw[:, :, 2]
            dist_s = distance_pred
            vol_s = physico_raw[:, :, 1]
            K = L
        
        # 衰减系数防止远距离残基对损失毫无贡献
        decay = torch.exp(-dist_s * 0.3)  # λ=0.3, 在10Å处衰减至~5%
        
        # ── 2a. 电荷配对评估 (确定性) ──
        ci = charge_s.unsqueeze(2)  # [B, K, 1]
        cj = charge_s.unsqueeze(1)  # [B, 1, K]
        
        # 两者都带电 (|charge| > 0.1)
        is_charged = (charge_s.abs() > 0.1).float()
        both_charged = is_charged.unsqueeze(2) * is_charged.unsqueeze(1)  # [B, K, K]
        
        # 异号电荷对 (ci * cj < 0)
        is_opposite = ((ci * cj < -0.01) & both_charged.bool()).float()
        # 同号电荷对 (ci * cj > 0)
        is_same = ((ci * cj > 0.01) & both_charged.bool()).float()
        
        # 序列间隔 ≥ 3 的 mask
        seq_mask = torch.triu(torch.ones(K, K, device=aa_indices.device), diagonal=3)
        
        # 加权计数: 近距离衰减 × 序列mask
        favorable = (is_opposite * decay * seq_mask).sum()
        unfavorable = (is_same * decay * seq_mask).sum()
        total_charged = both_charged.sum() * seq_mask.sum() / (K * K)  # 归一化估计
        
        # charge_score: favorable / (favorable + unfavorable), 期望 > 0.5
        charge_score = (favorable + 1.0) / (favorable + unfavorable + 2.0)
        loss_charge = F.relu(0.5 - charge_score)

        # ── 2b. 空间位阻评估 (确定性，保持不变) ──
        vi = vol_s.unsqueeze(2)
        vj = vol_s.unsqueeze(1)
        min_dist = (vi.abs().clamp(min=1e-6) ** (1/3) + vj.abs().clamp(min=1e-6) ** (1/3)) * 0.8
        clash_penalty = F.relu(min_dist - dist_s) * decay * seq_mask
        loss_clash = clash_penalty.sum() / seq_mask.sum().clamp(min=1)

        total = loss_hydro + 0.4 * loss_charge + 0.6 * loss_clash
        # NaN 保护
        total = torch.nan_to_num(total, nan=0.0, posinf=10.0, neginf=0.0)
        return total
