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
import csv

OUT = '/home/vr/TRP_CAGE'
print("Loading trajectory...")
traj = md.load(f'{OUT}/production_chunk_100pct.xtc', top=f'{OUT}/solvated.pdb')
protein = traj.topology.select('protein and chainid 0')
ca = traj.topology.select('protein and name CA and chainid 0')
traj.superpose(traj, 0, atom_indices=ca)
prot_res = [r for r in traj.topology.residues if r.chain.index == 0]
resnames = [(r.name, r.resSeq) for r in prot_res]
n_res = len(resnames)
time_ns = traj.time / 1000
print(f"Loaded: {traj.n_frames} frames, {traj.time[-1]/1000:.1f} ns")

print("1/11 RMSD..."); rmsd = md.rmsd(traj, traj, 0, protein) * 10
print("2/11 RMSF..."); rmsf = np.zeros(n_res)
for j, ri in enumerate([r.index for r in prot_res]):
    atoms = [a.index for a in traj.topology.residue(ri).atoms]
    coords = traj.xyz[:, atoms, :]
    rmsf[j] = np.sqrt(((coords - coords.mean(axis=0))**2).sum(axis=2)).mean(axis=0).mean() * 10
print("3/11 Rg..."); rg = np.zeros(traj.n_frames)
for i in range(traj.n_frames):
    rg[i] = np.sqrt(((traj.xyz[i, ca] - traj.xyz[i, ca].mean(axis=0))**2).sum(axis=1).mean()) * 10
print("4/11 DSSP...")
ss_all = md.compute_dssp(traj, simplified=True)
res_idx = [r.index for r in prot_res]
ss_prot = ss_all[:, res_idx]
helix_frac = np.array([(ss_prot[:, j] == 'H').sum() / traj.n_frames * 100 for j in range(n_res)])
print("5/11 Trp6-Pro...")
trp_idx = None; pro_indices = {}
for r in traj.topology.residues:
    if r.chain.index == 0:
        if r.name == 'TRP': trp_idx = r.index
        elif r.name == 'PRO': pro_indices[r.resSeq] = r.index
trp_pro_dists = {}
for ps in [12, 17, 18, 19]:
    d = np.zeros(traj.n_frames)
    for i in range(traj.n_frames):
        tc = traj.xyz[i, [a.index for a in traj.topology.residue(trp_idx).atoms], :].mean(axis=0)
        pc = traj.xyz[i, [a.index for a in traj.topology.residue(pro_indices[ps]).atoms], :].mean(axis=0)
        d[i] = np.linalg.norm(tc - pc) * 10
    trp_pro_dists[ps] = d
print("6/11 Native contacts...")
ca_atoms = [a.index for a in traj.topology.atoms if a.name == 'CA' and a.residue.chain.index == 0]
ca0 = traj.xyz[0, ca_atoms, :]; dm0 = squareform(pdist(ca0))
native_pairs = [(i, j) for i, j in zip(*np.where((dm0 < 0.8) & (dm0 > 0))) if i < j]
nQ = np.zeros(traj.n_frames)
for t in range(traj.n_frames):
    dm_t = squareform(pdist(traj.xyz[t, ca_atoms, :]))
    q = sum(1.0 / (1.0 + np.exp(5.0 * (dm_t[i, j] - dm0[i, j]))) for i, j in native_pairs)
    nQ[t] = q / len(native_pairs)
print(f"  {len(native_pairs)} native pairs")
print("7/11 PCA...")
X = traj.xyz[:, ca_atoms, :].reshape(traj.n_frames, -1); X -= X.mean(axis=0)
pca = PCA(n_components=10); proj = pca.fit_transform(X)
print(f"  PC1:{pca.explained_variance_ratio_[0]*100:.1f}% PC2:{pca.explained_variance_ratio_[1]*100:.1f}%")
print("8/11 Clusters...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10); labels = kmeans.fit_predict(proj[:, :2])
print("9/11 H-bonds...")
hb_counts = np.zeros(traj.n_frames); hb_pairs_count = {}
for t in range(0, traj.n_frames, 5):
    hbonds = md.baker_hubbard(traj[t:t+1], freq=0, periodic=True); count = 0
    for hb in hbonds:
        a1 = traj.topology.atom(hb[0]); a2 = traj.topology.atom(hb[2])
        if a1.residue.chain.index == 0 and a2.residue.chain.index == 0:
            if a1.name in ['N','O','NE1','OH'] and a2.name in ['N','O','NE1','OH']:
                count += 1
                pair = (f"{a1.residue.name}{a1.residue.resSeq}", f"{a2.residue.name}{a2.residue.resSeq}")
                hb_pairs_count[pair] = hb_pairs_count.get(pair, 0) + 1
    hb_counts[t] = count
top_hbonds = sorted(hb_pairs_count.items(), key=lambda x: -x[1])[:10]
print("10/11 SASA...")
sasa = md.shrake_rupley(traj, mode='residue')
sasa_prot = sasa[:, [r.index for r in prot_res]] * 100
print("11/11 FEL...")
hist, xedges, yedges = np.histogram2d(proj[:, 0], proj[:, 1], bins=50)
hist = hist.T; prob = hist / hist.sum(); prob[prob == 0] = np.nan
fel = -0.593 * np.log(prob); fel -= np.nanmin(fel)
print("ALL COMPUTATIONS DONE")

print("\n===== EXPORTING CSVs =====")
def wc(fn, rows):
    with open(f'{OUT}/{fn}', 'w', newline='') as f:
        w = csv.writer(f)
        for row in rows: w.writerow(row)

wc('data_rmsd.csv', [['Time_ns','RMSD_A']]+[[f'{time_ns[i]:.1f}',f'{rmsd[i]:.4f}'] for i in range(traj.n_frames)])
print("  data_rmsd.csv")
wc('data_rmsf.csv', [['Residue','ResNum','RMSF_A','Helix_Pct']]+[[rn,rs,f'{rmsf[j]:.4f}',f'{helix_frac[j]:.1f}'] for j,(rn,rs) in enumerate(resnames)])
print("  data_rmsf.csv")
wc('data_rg.csv', [['Time_ns','Rg_A']]+[[f'{time_ns[i]:.1f}',f'{rg[i]:.4f}'] for i in range(traj.n_frames)])
print("  data_rg.csv")
header = ['Time_ns'] + [f'{rn}{rs}' for rn, rs in resnames]
rows = [header] + [[f'{time_ns[i]:.1f}'] + list(ss_prot[i]) for i in range(traj.n_frames)]
wc('data_dssp.csv', rows)
print("  data_dssp.csv")
wc('data_trp_pro_dist.csv', [['Time_ns','Trp6_Pro12','Trp6_Pro17','Trp6_Pro18','Trp6_Pro19']]+
   [[f'{time_ns[i]:.1f}',f'{trp_pro_dists[12][i]:.3f}',f'{trp_pro_dists[17][i]:.3f}',f'{trp_pro_dists[18][i]:.3f}',f'{trp_pro_dists[19][i]:.3f}'] for i in range(traj.n_frames)])
print("  data_trp_pro_dist.csv")
wc('data_native_contacts.csv', [['Time_ns','NativeQ','Count']]+
   [[f'{time_ns[i]:.1f}',f'{nQ[i]:.4f}',f'{int(nQ[i]*len(native_pairs))}'] for i in range(traj.n_frames)])
print("  data_native_contacts.csv")
wc('data_pca.csv', [['Time_ns','PC1','PC2','PC3','Cluster','RMSD_A','NativeQ','Rg_A']]+
   [[f'{time_ns[i]:.1f}',f'{proj[i,0]:.4f}',f'{proj[i,1]:.4f}',f'{proj[i,2]:.4f}',labels[i],f'{rmsd[i]:.4f}',f'{nQ[i]:.4f}',f'{rg[i]:.4f}'] for i in range(traj.n_frames)])
print("  data_pca.csv")
wc('data_hbonds.csv', [['Donor','Acceptor','Count']]+[[d,a,c] for (d,a),c in top_hbonds])
print("  data_hbonds.csv")
header = ['Time_ns'] + [f'{rn}{rs}_SASA' for rn, rs in resnames]
rows = [header] + [[f'{time_ns[i]:.1f}'] + [f'{sasa_prot[i,j]:.2f}' for j in range(n_res)] for i in range(traj.n_frames)]
wc('data_sasa.csv', rows)
print("  data_sasa.csv")
wc('data_clusters.csv', [['Cluster','Count','Pct','Mean_RMSD','Mean_Rg','Mean_Q']]+
   [[c, int((labels==c).sum()), f'{(labels==c).sum()/len(labels)*100:.1f}', f'{rmsd[labels==c].mean():.3f}', f'{rg[labels==c].mean():.3f}', f'{nQ[labels==c].mean():.4f}'] for c in range(4)])
print("  data_clusters.csv")
cum = 0
rows = [['PC','Variance_Pct','Cumulative_Pct']]
for i in range(10):
    cum += pca.explained_variance_ratio_[i] * 100
    rows.append([f'PC{i+1}', f'{pca.explained_variance_ratio_[i]*100:.2f}', f'{cum:.2f}'])
wc('data_pca_variance.csv', rows)
print("  data_pca_variance.csv")
rows = [['PC1_edge','PC2_edge','FE_kcal_mol']]
for i in range(fel.shape[0]):
    for j in range(fel.shape[1]):
        if not np.isnan(fel[i,j]):
            rows.append([f'{xedges[j]:.3f}', f'{yedges[i]:.3f}', f'{fel[i,j]:.4f}'])
wc('data_fel.csv', rows)
print("  data_fel.csv")

print("\n===== GENERATING PDF =====")
pdf_path = f'{OUT}/TRP_CAGE_100ns_FullAnalysis.pdf'
w = 20; cc = ['#1565C0','#C62828','#2E7D32','#F57F17']
with PdfPages(pdf_path) as pdf:
    # P1: Title
    fig = plt.figure(figsize=(11,8)); ax = fig.add_subplot(111); ax.axis('off')
    ax.text(0.5,0.8,'Trp-Cage Miniprotein',fontsize=24,fontweight='bold',ha='center',va='top',color='#1a1a2e')
    ax.text(0.5,0.65,'100 ns MD - Full Analysis Report',fontsize=16,ha='center',va='top',color='#4a4a6a')
    ax.text(0.5,0.45,'PDB: 1L2Y | NLYIQWLKDGGLSSGRPPPS | AMBER14 | TIP3P',fontsize=11,ha='center',va='top',color='#666')
    ax.text(0.5,0.35,'20 residues | 1499 waters | 9 ions | 36.6 A box | 1000 frames',fontsize=11,ha='center',va='top',color='#666')
    ax.text(0.5,0.15,'11 analyses: RMSD, RMSF, Rg, DSSP, Trp6-Pro, Native contacts,\nPCA, Clustering, H-bonds, SASA, Free-energy landscape',fontsize=10,ha='center',va='top',color='#888')
    ax.axhline(y=0.25,xmin=0.15,xmax=0.85,color='#1a1a2e',linewidth=2)
    pdf.savefig(fig,dpi=150); plt.close()
    # P2: RMSD + Rg
    fig,(a1,a2) = plt.subplots(2,1,figsize=(11,9))
    a1.plot(time_ns,rmsd,color='#2196F3',lw=0.8,alpha=0.5)
    ma=np.convolve(rmsd,np.ones(w)/w,mode='valid'); a1.plot(time_ns[w-1:],ma,color='#d32f2f',lw=2,label=f'{w}-frame MA')
    a1.axhline(y=rmsd[-100:].mean(),color='gray',ls='--',alpha=0.5)
    a1.set(xlabel='Time (ns)',ylabel='RMSD (A)',title='A) Backbone RMSD',xlim=(0,105))
    a1.text(0.98,0.95,f'Mean: {rmsd.mean():.2f}+/-{rmsd.std():.2f} A\nLast 10ns: {rmsd[-100:].mean():.2f}+/-{rmsd[-100:].std():.2f} A',transform=a1.transAxes,fontsize=8,va='top',ha='right',bbox=dict(boxstyle='round',fc='wheat',alpha=0.5))
    a1.grid(True,alpha=0.3); a1.legend(fontsize=8)
    a2.plot(time_ns,rg,color='#4CAF50',lw=0.8,alpha=0.5)
    mr=np.convolve(rg,np.ones(w)/w,mode='valid'); a2.plot(time_ns[w-1:],mr,color='#E65100',lw=2,label=f'{w}-frame MA')
    a2.set(xlabel='Time (ns)',ylabel='Rg (A)',title='B) Radius of Gyration',xlim=(0,105))
    a2.text(0.98,0.95,f'Mean: {rg.mean():.2f}+/-{rg.std():.2f} A',transform=a2.transAxes,fontsize=8,va='top',ha='right',bbox=dict(boxstyle='round',fc='wheat',alpha=0.5))
    a2.grid(True,alpha=0.3); a2.legend(fontsize=8)
    fig.tight_layout(pad=2); pdf.savefig(fig,dpi=150); plt.close()

with PdfPages(pdf_path) as pdf:
    # P3: RMSF + Helix
    fig,(a1,a2) = plt.subplots(2,1,figsize=(11,9))
    xp = np.arange(n_res)
    sc = ['#4CAF50' if h>50 else '#FF9800' if h>10 else '#f44336' for h in helix_frac]
    a1.bar(xp,rmsf,color=sc,edgecolor='white',lw=0.5)
    a1.set_xticks(xp); a1.set_xticklabels([f'{rn}\n{rs}' for rn,rs in resnames],fontsize=7,rotation=45,ha='right')
    a1.set(ylabel='RMSF (A)',title='C) Per-Residue RMSF',ylim=(0,max(rmsf)*1.2))
    a1.axhline(y=1.0,color='gray',ls='--',alpha=0.5); a1.grid(True,axis='y',alpha=0.3)
    for i,v in enumerate(rmsf): a1.text(i,v+0.02,f'{v:.2f}',ha='center',fontsize=6)
    a2.bar(xp,helix_frac,color=sc,edgecolor='white',lw=0.5)
    a2.set_xticks(xp); a2.set_xticklabels([f'{rn}\n{rs}' for rn,rs in resnames],fontsize=7,rotation=45,ha='right')
    a2.set(ylabel='Helix %',title='D) Per-Residue Helix Content',ylim=(0,105))
    a2.axhline(y=50,color='gray',ls='--',alpha=0.5); a2.grid(True,axis='y',alpha=0.3)
    for i,v in enumerate(helix_frac): a2.text(i,v+1,f'{v:.0f}%',ha='center',fontsize=6)
    fig.tight_layout(pad=2); pdf.savefig(fig,dpi=150); plt.close()
    # P4: DSSP heatmaps
    fig,(a1,a2) = plt.subplots(2,1,figsize=(11,9))
    ns=min(200,traj.n_frames)
    for ax,sl,ti in [(a1,slice(None,ns),'E) DSSP (first 20 ns)'),(a2,slice(-ns,None),'F) DSSP (last 20 ns)')]:
        im=np.zeros((ns,n_res))
        for j in range(n_res): im[:,j]=(ss_prot[sl,j]=='H').astype(float)
        ax.imshow(im.T,aspect='auto',cmap='RdYlGn',vmin=0,vmax=1,interpolation='nearest')
        ax.set(xlabel='Frame',ylabel='Residue',title=ti)
        ax.set_yticks(range(n_res)); ax.set_yticklabels([f'{rn}{rs}' for rn,rs in resnames],fontsize=7)
        ax.legend(handles=[Patch(fc='#4CAF50',label='Helix'),Patch(fc='#f44336',label='Coil')],loc='lower right',fontsize=8)
    fig.tight_layout(pad=2); pdf.savefig(fig,dpi=150); plt.close()
    # P5: Trp6-Pro + NativeQ
    fig,(a1,a2) = plt.subplots(2,1,figsize=(11,9))
    cd=['#1565C0','#C62828','#2E7D32','#F57F17']
    for idx,ps in enumerate([12,17,18,19]):
        a1.plot(time_ns,trp_pro_dists[ps],color=cd[idx],lw=1.2,alpha=0.7,label=f'Trp6-Pro{ps}')
    a1.axhline(y=5.5,color='gray',ls='--',alpha=0.4,label='Cutoff 5.5 A')
    a1.set(xlabel='Time (ns)',ylabel='Distance (A)',title='G) Trp6-Proline Distances',xlim=(0,105),ylim=(2,14))
    a1.legend(fontsize=8); a1.grid(True,alpha=0.3)
    a2.plot(time_ns,nQ,color='#6A1B9A',lw=1.0,alpha=0.7)
    mq=np.convolve(nQ,np.ones(w)/w,mode='valid'); a2.plot(time_ns[w-1:],mq,color='#E65100',lw=2,label=f'{w}-frame MA')
    a2.set(xlabel='Time (ns)',ylabel='Q',title='H) Native Contact Fraction',xlim=(0,105),ylim=(0,1))
    a2.text(0.98,0.95,f'Mean: {nQ.mean():.3f}\nLast 10ns: {nQ[-100:].mean():.3f}',transform=a2.transAxes,fontsize=8,va='top',ha='right',bbox=dict(boxstyle='round',fc='wheat',alpha=0.5))
    a2.grid(True,alpha=0.3); a2.legend(fontsize=8)
    fig.tight_layout(pad=2); pdf.savefig(fig,dpi=150); plt.close()

with PdfPages(pdf_path) as pdf:
    # P6: PCA
    fig,(a1,a2) = plt.subplots(1,2,figsize=(11,5))
    s1=a1.scatter(proj[:,0],proj[:,1],c=time_ns,cmap='viridis',s=5,alpha=0.6)
    plt.colorbar(s1,ax=a1,label='Time (ns)')
    a1.set(xlabel=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',ylabel=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',title='I) PCA (time)')
    a1.grid(True,alpha=0.3)
    s2=a2.scatter(proj[:,0],proj[:,1],c=nQ,cmap='RdYlGn',s=5,alpha=0.6,vmin=0.3,vmax=0.7)
    plt.colorbar(s2,ax=a2,label='Native Q')
    a2.set(xlabel=f'PC1',ylabel=f'PC2',title='PCA (native contacts)')
    a2.grid(True,alpha=0.3)
    fig.tight_layout(pad=2); pdf.savefig(fig,dpi=150); plt.close()
    # P7: Clusters
    fig,(a1,a2) = plt.subplots(1,2,figsize=(11,5))
    for c in range(4):
        m=labels==c; a1.scatter(proj[m,0],proj[m,1],color=cc[c],s=5,alpha=0.6,label=f'C{c} ({m.sum()/len(labels)*100:.0f}%)')
    a1.set(xlabel='PC1',ylabel='PC2',title='J) K-Means Clusters (k=4)')
    a1.legend(fontsize=8,markerscale=3); a1.grid(True,alpha=0.3)
    for c in range(4):
        m=labels==c; a2.hist(rmsd[m],bins=20,alpha=0.5,color=cc[c],label=f'C{c}',density=True)
    a2.set(xlabel='RMSD (A)',ylabel='Density',title='RMSD per Cluster')
    a2.legend(fontsize=8); a2.grid(True,alpha=0.3)
    fig.tight_layout(pad=2); pdf.savefig(fig,dpi=150); plt.close()
    # P8: H-bonds + SASA
    fig,(a1,a2) = plt.subplots(2,1,figsize=(11,9))
    hn=[f"{p[0]}-{p[1]}" for p,c in top_hbonds]; hv=[c for p,c in top_hbonds]
    a1.barh(range(len(hn)),hv,color='#1565C0',edgecolor='white')
    a1.set_yticks(range(len(hn))); a1.set_yticklabels(hn,fontsize=8)
    a1.set(xlabel='Count',title='K) Top 10 H-Bonds')
    a1.invert_yaxis(); a1.grid(True,axis='x',alpha=0.3)
    for j in range(n_res):
        a2.plot(time_ns,sasa_prot[:,j],lw=0.6,alpha=0.5,label=f'{resnames[j][0]}{resnames[j][1]}')
    a2.set(xlabel='Time (ns)',ylabel='SASA (A^2)',title='L) SASA per Residue',xlim=(0,105))
    a2.grid(True,alpha=0.3)
    fig.tight_layout(pad=2); pdf.savefig(fig,dpi=150); plt.close()
    # P9: FEL
    fig,(a1,a2) = plt.subplots(1,2,figsize=(11,5))
    xc=(xedges[:-1]+xedges[1:])/2; yc=(yedges[:-1]+yedges[1:])/2
    Xg,Yg=np.meshgrid(xc,yc)
    cf=a1.contourf(Xg,Yg,fel,levels=20,cmap='RdYlBu_r')
    plt.colorbar(cf,ax=a1,label='FE (kcal/mol)')
    a1.scatter(proj[0,0],proj[0,1],c='white',s=100,marker='*',zorder=5,edgecolors='black',label='Start')
    a1.scatter(proj[-1,0],proj[-1,1],c='red',s=100,marker='*',zorder=5,edgecolors='black',label='End')
    a1.set(xlabel='PC1',ylabel='PC2',title='M) Free Energy Landscape')
    a1.legend(fontsize=8); a1.grid(True,alpha=0.3)
    a2.scatter(time_ns,rmsd,c=labels,cmap='Set1',s=3,alpha=0.6)
    a2.set(xlabel='Time (ns)',ylabel='RMSD (A)',title='N) RMSD by cluster',xlim=(0,105))
    a2.grid(True,alpha=0.3)
    fig.tight_layout(pad=2); pdf.savefig(fig,dpi=150); plt.close()

print(f"\nPDF saved: {pdf_path}")
print("ALL DONE")
