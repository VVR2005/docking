import mdtraj as md
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

print("Loading trajectory...")
traj = md.load('/home/vr/TRP_CAGE/production_chunk_100pct.xtc', top='/home/vr/TRP_CAGE/solvated.pdb')
protein = traj.topology.select('protein and chainid 0')
ca = traj.topology.select('protein and name CA and chainid 0')

com_all = np.zeros((traj.n_frames, 3))
for i in range(traj.n_frames):
    com_all[i] = traj.xyz[i, ca].mean(axis=0)
box_center = traj.unitcell_lengths[0] / 2
for i in range(traj.n_frames):
    traj.xyz[i] += (box_center - com_all[i])
traj.xyz = traj.xyz % traj.unitcell_lengths[:, np.newaxis, :]

time_ns = traj.time / 1000
prot_res = [r for r in traj.topology.residues if r.chain.index == 0]
resnames = [(r.name, r.resSeq) for r in prot_res]
n_res = len(resnames)
print(f"Trajectory: {traj.n_frames} frames, {traj.time[-1]/1000:.1f} ns")

print("Computing RMSD...")
rmsd = md.rmsd(traj, traj, 0, protein) * 10

print("Computing RMSF...")
rmsf = np.zeros(n_res)
for j, ri in enumerate([r.index for r in prot_res]):
    atoms = [a.index for a in traj.topology.residue(ri).atoms]
    coords = traj.xyz[:, atoms, :]
    rmsf[j] = np.sqrt(((coords - coords.mean(axis=0))**2).sum(axis=2)).mean(axis=0).mean() * 10

print("Computing Rg...")
rg = np.zeros(traj.n_frames)
for i in range(traj.n_frames):
    rg[i] = np.sqrt(((traj.xyz[i, ca] - traj.xyz[i, ca].mean(axis=0))**2).sum(axis=1).mean()) * 10

print("Computing DSSP...")
ss_all = md.compute_dssp(traj, simplified=True)
res_idx = [r.index for r in prot_res]
ss_prot = ss_all[:, res_idx]
helix_frac = np.array([(ss_prot[:, j] == 'H').sum() / traj.n_frames * 100 for j in range(n_res)])

print("Computing Trp6-Pro distances...")
trp_idx = None
pro_indices = {}
for r in traj.topology.residues:
    if r.chain.index == 0:
        if r.name == 'TRP':
            trp_idx = r.index
        elif r.name == 'PRO':
            pro_indices[r.resSeq] = r.index

trp_pro_dists = {}
for ps in [12, 17, 18, 19]:
    d = np.zeros(traj.n_frames)
    for i in range(traj.n_frames):
        tc = traj.xyz[i, [a.index for a in traj.topology.residue(trp_idx).atoms], :].mean(axis=0)
        pc = traj.xyz[i, [a.index for a in traj.topology.residue(pro_indices[ps]).atoms], :].mean(axis=0)
        d[i] = np.linalg.norm(tc - pc) * 10
    trp_pro_dists[ps] = d

print("Computing native contacts...")
ca_atoms = [a.index for a in traj.topology.atoms if a.name == 'CA' and a.residue.chain.index == 0]
ca_coords_0 = traj.xyz[0, ca_atoms, :]
dm0 = squareform(pdist(ca_coords_0))
native_pairs = [(i, j) for i, j in zip(*np.where((dm0 < 0.8) & (dm0 > 0))) if i < j]
print(f"  {len(native_pairs)} native pairs")

nQ = np.zeros(traj.n_frames)
for t in range(traj.n_frames):
    dm_t = squareform(pdist(traj.xyz[t, ca_atoms, :]))
    q = sum(1.0 / (1.0 + np.exp(5.0 * (dm_t[i, j] - dm0[i, j]))) for i, j in native_pairs)
    nQ[t] = q / len(native_pairs)

print("Computing PCA...")
ca_coords_all = traj.xyz[:, ca_atoms, :]
X = ca_coords_all.reshape(traj.n_frames, -1)
X -= X.mean(axis=0)
pca = PCA(n_components=3)
proj = pca.fit_transform(X)
print(f"  PC1: {pca.explained_variance_ratio_[0]*100:.1f}%, PC2: {pca.explained_variance_ratio_[1]*100:.1f}%, PC3: {pca.explained_variance_ratio_[2]*100:.1f}%")

print("Computing clusters...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(proj[:, :2])
for c in range(4):
    print(f"  Cluster {c}: {(labels == c).sum() / len(labels) * 100:.1f}%")

print("Computing H-bonds...")
hb_counts = np.zeros(traj.n_frames)
hb_pairs_count = {}
for t in range(0, traj.n_frames, 5):
    hbonds = md.baker_hubbard(traj[t:t+1], freq=0, periodic=True)
    count = 0
    for hb in hbonds:
        a1 = traj.topology.atom(hb[0])
        a2 = traj.topology.atom(hb[2])
        if a1.residue.chain.index == 0 and a2.residue.chain.index == 0:
            if a1.name in ['N', 'O', 'NE1', 'OH'] and a2.name in ['N', 'O', 'NE1', 'OH']:
                count += 1
                pair = (f"{a1.residue.name}{a1.residue.resSeq}", f"{a2.residue.name}{a2.residue.resSeq}")
                hb_pairs_count[pair] = hb_pairs_count.get(pair, 0) + 1
    hb_counts[t] = count
top_hbonds = sorted(hb_pairs_count.items(), key=lambda x: -x[1])[:10]

print("Computing SASA...")
sasa = md.shrake_rupley(traj, mode='residue')
sasa_prot = sasa[:, [r.index for r in prot_res]] * 100

print("Computing free energy landscape...")
hist, xedges, yedges = np.histogram2d(proj[:, 0], proj[:, 1], bins=50)
hist = hist.T
prob = hist / hist.sum()
prob[prob == 0] = np.nan
fel = -0.593 * np.log(prob)
fel -= np.nanmin(fel)

print("Generating PDF...")
pdf_path = '/home/vr/TRP_CAGE/TRP_CAGE_100ns_FullAnalysis.pdf'
window = 20
cl_colors = ['#1565C0', '#C62828', '#2E7D32', '#F57F17']

with PdfPages(pdf_path) as pdf:

    # PAGE 1: Title
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.text(0.5, 0.8, 'Trp-Cage Miniprotein', fontsize=24, fontweight='bold', ha='center', va='top', color='#1a1a2e')
    ax.text(0.5, 0.65, '100 ns MD - Full Analysis Report', fontsize=16, ha='center', va='top', color='#4a4a6a')
    ax.text(0.5, 0.45, 'PDB: 1L2Y  |  NLYIQWLKDGGLSSGRPPPS  |  AMBER14  |  TIP3P', fontsize=11, ha='center', va='top', color='#666')
    ax.text(0.5, 0.35, '20 residues  |  1499 waters  |  9 ions  |  36.6 A box  |  1000 frames', fontsize=11, ha='center', va='top', color='#666')
    ax.text(0.5, 0.15, '11 analyses: RMSD, RMSF, Rg, DSSP, Trp6-Pro distances,\nNative contacts, PCA, Clustering, H-bonds, SASA, Free-energy landscape',
            fontsize=10, ha='center', va='top', color='#888')
    ax.axhline(y=0.25, xmin=0.15, xmax=0.85, color='#1a1a2e', linewidth=2)
    pdf.savefig(fig, dpi=150)
    plt.close()

    # PAGE 2: RMSD + Rg
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))
    ax1.plot(time_ns, rmsd, color='#2196F3', linewidth=0.8, alpha=0.5)
    ma = np.convolve(rmsd, np.ones(window)/window, mode='valid')
    ax1.plot(time_ns[window-1:], ma, color='#d32f2f', linewidth=2, label=f'{window}-frame MA')
    ax1.axhline(y=rmsd[-100:].mean(), color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('RMSD (A)')
    ax1.set_title('A) Backbone RMSD', fontsize=12, fontweight='bold')
    ax1.text(0.98, 0.95, f'Mean: {rmsd.mean():.2f}+/-{rmsd.std():.2f} A\nLast 10ns: {rmsd[-100:].mean():.2f}+/-{rmsd[-100:].std():.2f} A',
             transform=ax1.transAxes, fontsize=8, va='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.set_xlim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.plot(time_ns, rg, color='#4CAF50', linewidth=0.8, alpha=0.5)
    ma_rg = np.convolve(rg, np.ones(window)/window, mode='valid')
    ax2.plot(time_ns[window-1:], ma_rg, color='#E65100', linewidth=2, label=f'{window}-frame MA')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Rg (A)')
    ax2.set_title('B) Radius of Gyration', fontsize=12, fontweight='bold')
    ax2.text(0.98, 0.95, f'Mean: {rg.mean():.2f}+/-{rg.std():.2f} A',
             transform=ax2.transAxes, fontsize=8, va='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.set_xlim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout(pad=2)
    pdf.savefig(fig, dpi=150)
    plt.close()

    # PAGE 3: RMSF + Helix
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))
    x_pos = np.arange(n_res)
    ss_colors = ['#4CAF50' if hf > 50 else '#FF9800' if hf > 10 else '#f44336' for hf in helix_frac]

    ax1.bar(x_pos, rmsf, color=ss_colors, edgecolor='white', linewidth=0.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{rn}\n{rs}' for rn, rs in resnames], fontsize=7, rotation=45, ha='right')
    ax1.set_ylabel('RMSF (A)')
    ax1.set_title('C) Per-Residue RMSF', fontsize=12, fontweight='bold')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylim(0, 1.3)
    ax1.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(rmsf):
        ax1.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=6)

    ax2.bar(x_pos, helix_frac, color=ss_colors, edgecolor='white', linewidth=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{rn}\n{rs}' for rn, rs in resnames], fontsize=7, rotation=45, ha='right')
    ax2.set_ylabel('Helix %')
    ax2.set_title('D) Per-Residue Helix Content (DSSP)', fontsize=12, fontweight='bold')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 105)
    ax2.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(helix_frac):
        ax2.text(i, v + 1, f'{v:.0f}%', ha='center', fontsize=6)
    fig.tight_layout(pad=2)
    pdf.savefig(fig, dpi=150)
    plt.close()

    # PAGE 4: DSSP heatmap (first/last 20 ns)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))
    n_show = min(200, traj.n_frames)
    ss_num = np.zeros((n_show, n_res))
    for j in range(n_res):
        ss_num[:, j] = (ss_prot[:n_show, j] == 'H').astype(float)

    ax1.imshow(ss_num.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Residue')
    ax1.set_title('E) DSSP Timeline (first 20 ns)', fontsize=12, fontweight='bold')
    ax1.set_yticks(range(n_res))
    ax1.set_yticklabels([f'{rn}{rs}' for rn, rs in resnames], fontsize=7)
    ax1.legend(handles=[Patch(facecolor='#4CAF50', label='Helix'), Patch(facecolor='#f44336', label='Coil')], loc='lower right', fontsize=8)

    ss_num2 = np.zeros((n_show, n_res))
    for j in range(n_res):
        ss_num2[:, j] = (ss_prot[-n_show:, j] == 'H').astype(float)
    ax2.imshow(ss_num2.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Residue')
    ax2.set_title('F) DSSP Timeline (last 20 ns)', fontsize=12, fontweight='bold')
    ax2.set_yticks(range(n_res))
    ax2.set_yticklabels([f'{rn}{rs}' for rn, rs in resnames], fontsize=7)
    ax2.legend(handles=[Patch(facecolor='#4CAF50', label='Helix'), Patch(facecolor='#f44336', label='Coil')], loc='lower right', fontsize=8)
    fig.tight_layout(pad=2)
    pdf.savefig(fig, dpi=150)
    plt.close()

    # PAGE 5: Trp6-Pro distances + Native contacts
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))
    colors_d = ['#1565C0', '#C62828', '#2E7D32', '#F57F17']
    labels_d = ['Trp6-Pro12', 'Trp6-Pro17', 'Trp6-Pro18', 'Trp6-Pro19']
    for idx, ps in enumerate([12, 17, 18, 19]):
        ax1.plot(time_ns, trp_pro_dists[ps], color=colors_d[idx], linewidth=1.2, alpha=0.7, label=labels_d[idx])
    ax1.axhline(y=5.5, color='gray', linestyle='--', alpha=0.4, label='Fold cutoff (5.5 A)')
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Distance (A)')
    ax1.set_title('G) Trp6-Proline Cage Distances', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_xlim(0, 105)
    ax1.set_ylim(2, 14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(time_ns, nQ, color='#6A1B9A', linewidth=1.0, alpha=0.7)
    ma_nq = np.convolve(nQ, np.ones(window)/window, mode='valid')
    ax2.plot(time_ns[window-1:], ma_nq, color='#E65100', linewidth=2, label=f'{window}-frame MA')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Native Contacts (Q)')
    ax2.set_title('H) Native Contact Fraction', fontsize=12, fontweight='bold')
    ax2.text(0.98, 0.95, f'Mean: {nQ.mean():.3f}\nLast 10ns: {nQ[-100:].mean():.3f}',
             transform=ax2.transAxes, fontsize=8, va='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.set_xlim(0, 105)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout(pad=2)
    pdf.savefig(fig, dpi=150)
    plt.close()
print("Pages 4-5 done")
with PdfPages(pdf_path) as pdf:

    # PAGE 6: PCA
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    sc = ax1.scatter(proj[:, 0], proj[:, 1], c=time_ns, cmap='viridis', s=5, alpha=0.6)
    plt.colorbar(sc, ax=ax1, label='Time (ns)')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_title('I) PCA (colored by time)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    sc2 = ax2.scatter(proj[:, 0], proj[:, 1], c=nQ, cmap='RdYlGn', s=5, alpha=0.6, vmin=0.3, vmax=0.7)
    plt.colorbar(sc2, ax=ax2, label='Native Contacts (Q)')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax2.set_title('PCA (colored by Q)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    fig.tight_layout(pad=2)
    pdf.savefig(fig, dpi=150)
    plt.close()

    # PAGE 7: Clusters
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    for c in range(4):
        mask = labels == c
        pct = mask.sum() / len(labels) * 100
        ax1.scatter(proj[mask, 0], proj[mask, 1], c=cl_colors[c], s=5, alpha=0.6, label=f'C{c} ({pct:.0f}%)')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_title('J) K-Means Clustering (k=4)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, markerscale=3)
    ax1.grid(True, alpha=0.3)

    for c in range(4):
        mask = labels == c
        rmsd_c = rmsd[mask]
        ax2.hist(rmsd_c, bins=20, alpha=0.5, color=cl_colors[c], label=f'C{c}', density=True)
    ax2.set_xlabel('RMSD (A)')
    ax2.set_ylabel('Density')
    ax2.set_title('RMSD Distribution per Cluster', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout(pad=2)
    pdf.savefig(fig, dpi=150)
    plt.close()

    # PAGE 8: H-bonds + SASA
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))

    hb_names = [f"{p[0]}-{p[1]}" for p, c in top_hbonds]
    hb_counts_vals = [c for p, c in top_hbonds]
    ax1.barh(range(len(hb_names)), hb_counts_vals, color='#1565C0', edgecolor='white')
    ax1.set_yticks(range(len(hb_names)))
    ax1.set_yticklabels(hb_names, fontsize=8)
    ax1.set_xlabel('Count (sampled every 5th frame)')
    ax1.set_title('K) Top 10 Protein-Protein H-Bonds', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, axis='x', alpha=0.3)

    for j in range(n_res):
        ax2.plot(time_ns, sasa_prot[:, j], linewidth=0.6, alpha=0.5, label=f'{resnames[j][0]}{resnames[j][1]}')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('SASA (A^2)')
    ax2.set_title('L) Solvent-Accessible Surface Area per Residue', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 105)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout(pad=2)
    pdf.savefig(fig, dpi=150)
    plt.close()

    # PAGE 9: Free Energy Landscape
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    X_grid, Y_grid = np.meshgrid(x_centers, y_centers)
    cf = ax1.contourf(X_grid, Y_grid, fel, levels=20, cmap='RdYlBu_r')
    plt.colorbar(cf, ax=ax1, label='Free Energy (kcal/mol)')
    ax1.scatter(proj[0, 0], proj[0, 1], c='white', s=100, marker='*', zorder=5, edgecolors='black', label='Start')
    ax1.scatter(proj[-1, 0], proj[-1, 1], c='red', s=100, marker='*', zorder=5, edgecolors='black', label='End')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_title('M) Free Energy Landscape', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2.plot(time_ns, rmsd, color='#2196F3', linewidth=0.6, alpha=0.5)
    ax2.scatter(time_ns, rmsd, c=labels, cmap='Set1', s=3, alpha=0.6)
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('RMSD (A)')
    ax2.set_title('N) RMSD colored by cluster', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 105)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout(pad=2)
    pdf.savefig(fig, dpi=150)
    plt.close()

print(f"PDF saved: {pdf_path}")
