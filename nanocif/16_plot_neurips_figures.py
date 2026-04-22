#!/usr/bin/env python3
"""
NanoCIF / NeurIPS — All Publication Figures (UPDATED)
Usage: python3 16_plot_neurips_figures.py --base .
Saves to figures-neurips/ as pdf + png + svg at 1200 DPI.
"""
import argparse
import json
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7, 'legend.title_fontsize': 7.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'figure.dpi': 1200, 'savefig.dpi': 1200,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    'axes.grid': False, 'legend.frameon': False,
})

COLORS = {
    'perovskite': '#0072B2', 'heusler': '#D55E00', 'hydride': '#009E73',
    'gen': '#FF8C00', 'test': '#1E88E5', 'train': '#E53935',
    'valid': '#2E7D32', 'invalid': '#AD1457',
    'before': '#AB47BC', 'after': '#00897B',
    'gray': '#999999',
}
C4 = {'gen': '#E53935', 'test': '#1E88E5'}
C5 = {'gen': '#F4511E', 'test': '#00838F'}

CLASS_NAMES = {'perovskite': 'Perovskite', 'heusler': 'Heusler', 'hydride': 'Hydride'}
DC = 180 / 25.4
SC = 88 / 25.4


def add_panel_label(ax, label, x=-0.14, y=1.06):
    ax.text(x, y, f'({label})', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')

def label_inside_left(ax, label):
    ax.text(0.05, 0.95, f'({label})', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')

def label_inside_right(ax, label):
    ax.text(0.95, 0.95, f'({label})', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='right')

def save(fig, name, outdir):
    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(outdir / f"{name}.{fmt}", format=fmt)
    plt.close(fig)
    print(f"  Saved: {name}")

def parse_nanocif(text):
    result = {'valid': False, 'formula': None, 'cls': None, 'radius': None,
              'atoms': [], 'coords': []}
    lines = text.strip().split('\n')
    in_loop = False
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith('data_'): result['formula'] = line[5:]
        elif line.startswith('_class '): result['cls'] = line[7:]
        elif line.startswith('_radius '):
            try: result['radius'] = int(line[8:])
            except: pass
        elif line.startswith('loop_'): in_loop = True
        elif line.startswith('_atom_type'): continue
        elif in_loop:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    result['atoms'].append(parts[0])
                    result['coords'].append([float(parts[1]), float(parts[2]), float(parts[3])])
                except: continue
    n = len(result['atoms'])
    if result['formula'] and result['cls'] in ['perovskite','heusler','hydride'] and result['radius'] in [5,6] and n >= 3:
        result['valid'] = True
    result['n_atoms'] = n
    result['coords'] = np.array(result['coords']) if result['coords'] else np.array([])
    return result

def compute_metrics(parsed):
    if not parsed['valid'] or len(parsed['coords']) < 2: return None
    coords = parsed['coords']
    from scipy.spatial.distance import cdist
    dists = cdist(coords, coords)
    np.fill_diagonal(dists, np.inf)
    nn = dists.min(axis=1)
    com = coords.mean(axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - com)**2, axis=1)))
    return {'n_atoms': len(coords), 'nn_mean': nn.mean(), 'nn_min': nn.min(),
            'nn_std': nn.std(), 'rg': rg, 'phys_valid': nn.min() > 0.8,
            'cls': parsed['cls'], 'formula': parsed['formula']}


# ================================================================
# FIG 1: Pipeline (unchanged)
# ================================================================
def fig1_representation(outdir):
    fig, ax = plt.subplots(figsize=(DC, DC * 0.55))
    ax.set_xlim(0, 100); ax.set_ylim(0, 65); ax.axis('off')
    C = {'box':'#37474F','nanocif':'#1B5E20','model':'#0D47A1',
         'output':'#E65100','arrow':'#888888','annot':'#555555','code':'#263238'}
    def box(x,y,w,h,title,sub='',color='#333'):
        b = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.25",facecolor=color,edgecolor='white',linewidth=0.6,zorder=2)
        ax.add_patch(b)
        if sub:
            ax.text(x+w/2,y+h/2+1.2,title,ha='center',va='center',fontsize=7.5,fontweight='600',color='white',zorder=3)
            ax.text(x+w/2,y+h/2-1.2,sub,ha='center',va='center',fontsize=5.5,color='white',alpha=0.9,zorder=3)
        else:
            ax.text(x+w/2,y+h/2,title,ha='center',va='center',fontsize=7.5,fontweight='600',color='white',zorder=3)
    def arr(x1,y1,x2,y2):
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color=C['arrow'],lw=1.0),zorder=1)
    ax.text(50,63,'NanoCIF: Text-based Nanoparticle Representation & Generation',ha='center',fontsize=9,fontweight='600')
    ex_x,ex_y=2,18
    ax.add_patch(FancyBboxPatch((ex_x,ex_y),30,40,boxstyle="round,pad=0.3",facecolor=C['code'],edgecolor='#455A64',linewidth=0.6,zorder=2))
    ax.text(ex_x+15,ex_y+38,'NanoCIF Format',ha='center',fontsize=7,fontweight='600',color='#81C784',zorder=3)
    code_lines=[('data_','BaTiO3','#81C784','#E8F5E9'),('_class',' perovskite','#64B5F6','#BBDEFB'),
        ('_radius',' 6','#64B5F6','#BBDEFB'),('_natoms',' 54','#64B5F6','#BBDEFB'),
        ('_elements',' Ba Ti O','#64B5F6','#BBDEFB'),('_composition',' Ba:0.20 Ti:0.20 O:0.60','#64B5F6','#BBDEFB'),
        ('loop_','','#FFF176','#FFF176'),('_atom_type',' _x _y _z','#FFF176','#FFF176'),
        ('Ba','  0.00  0.00  0.00','#EF9A9A','#FFCDD2'),('Ti',' -1.95  0.00  0.00','#EF9A9A','#FFCDD2'),
        ('O','  -0.97 -0.97  0.00','#EF9A9A','#FFCDD2'),('...',' (sorted core\u2192surface)','#9E9E9E','#9E9E9E')]
    for i,(key,val,kcol,vcol) in enumerate(code_lines):
        ly=ex_y+34.5-i*2.8
        ax.text(ex_x+2,ly,key,fontsize=5,fontfamily='monospace',color=kcol,fontweight='600',zorder=3)
        ax.text(ex_x+12,ly,val,fontsize=5,fontfamily='monospace',color=vcol,zorder=3)
    px=42
    box(px,50,24,7,'NanoGenLM Dataset','2,488 relaxed NPs','#546E7A')
    arr(px+24,53.5,px+28,53.5)
    box(px+28,50,24,7,'NanoCIF Converter','Cartesian, radial sort',C['nanocif'])
    arr(px+40,50,px+40,46)
    box(px+16,36,28,8,'Autoregressive GPT','5.1M params, 256-dim, 6 layers',C['model'])
    ax.text(px+58,42,'Data Augmentation',fontsize=6,fontweight='600',color=C['annot'])
    ax.text(px+58,40,'\u25b8 10\u00d7 random rotation',fontsize=5,color=C['annot'])
    ax.text(px+58,38.5,'\u25b8 21,340 training seqs',fontsize=5,color=C['annot'])
    arr(px+30,36,px+30,32)
    box(px+10,22,18,8,'Generated\nNanoCIF',color=C['output'])
    box(px+34,22,18,8,'DFTB+\nRelaxation',color='#D55E00')
    arr(px+28,26,px+34,26)
    arr(px+43,22,px+43,18)
    box(px+22,8,30,8,'Validated NP Structures','Physically plausible & relaxed','#2E7D32')
    mx,my=px+58,28
    ax.text(mx,my,'Generation Metrics',fontsize=6,fontweight='600',color=C['annot'])
    for i,t in enumerate(['Validity: 100%','Physical: 80%','Unique: 71%','NN mean: 2.14 \u00c5','Rg: 4.19 \u00c5']):
        ax.text(mx+0.5,my-1.7*(i+1),f'\u25b8 {t}',fontsize=5,color=C['annot'])
    save(fig,'fig1_nanocif_pipeline',outdir)


# ================================================================
# FIG 2: Training — labels inside, Final gap upper left
# ================================================================
def fig2_training(base, outdir):
    hist_path = base/"model"/"training_history.json"
    if not hist_path.exists():
        print("  WARNING: training_history.json not found, skipping fig2"); return
    with open(hist_path) as f: history = json.load(f)
    epochs=[h['epoch'] for h in history]; train_loss=[h['train_loss'] for h in history]; val_loss=[h['val_loss'] for h in history]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(DC,DC*0.35))
    ax1.plot(epochs,train_loss,'-',color=COLORS['train'],linewidth=1.0,label='Train',alpha=0.8)
    ax1.plot(epochs,val_loss,'-',color=COLORS['test'],linewidth=1.0,label='Validation',alpha=0.8)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Cross-entropy loss')
    ax1.legend(loc='upper right')
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2)); ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    best_idx=np.argmin(val_loss)
    ax1.axvline(epochs[best_idx],color=COLORS['gray'],linewidth=0.5,linestyle=':',alpha=0.5)
    ax1.text(epochs[best_idx]+1,max(val_loss)*0.95,f'Best: {val_loss[best_idx]:.3f}\n(epoch {epochs[best_idx]})',fontsize=5.5,color=COLORS['gray'])
    label_inside_left(ax1,'a')
    gap=[v-t for v,t in zip(val_loss,train_loss)]
    ax2.plot(epochs,gap,'-',color='#7B2D8E',linewidth=1.0,alpha=0.8)
    ax2.axhline(0,color=COLORS['gray'],linewidth=0.5,linestyle='--',alpha=0.5)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Generalization gap (val \u2212 train)')
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2)); ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.text(0.95,0.05,f'Final gap: {gap[-1]:.3f}',transform=ax2.transAxes,fontsize=7,ha='right',va='bottom',
             bbox=dict(boxstyle='round,pad=0.3',facecolor='white',edgecolor=COLORS['gray'],alpha=0.9,linewidth=0.5))
    label_inside_left(ax2,'b')
    fig.tight_layout(w_pad=3); save(fig,'fig2_training_curves',outdir)


# ================================================================
# FIG 3: Generation Summary — labels inside right, legend left, vivid
# ================================================================
def fig3_generation_summary(gen_parsed, test_parsed, outdir):
    fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(DC,DC*0.35))
    c_valid='#00C853'; c_phys='#00BFA5'; c_test='#2979FF'; c_gen='#FF6D00'
    c_unique='#FFD600'; c_novel='#00E676'; c_total='#448AFF'

    n_total=len(gen_parsed)
    n_valid=sum(1 for p in gen_parsed if p['valid'])
    n_phys=sum(1 for p in gen_parsed if p['valid'] and compute_metrics(p) is not None and compute_metrics(p)['phys_valid'])
    categories=['Format\nvalid','Structurally\nplausible']; values=[n_valid/n_total*100,n_phys/n_total*100]
    bars=ax1.bar(categories,values,color=[c_valid,c_phys],width=0.5,edgecolor='white',linewidth=0.3)
    for bar,val in zip(bars,values):
        ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.5,f'{val:.0f}%',ha='center',fontsize=7,fontweight='500')
    ax1.set_ylim(0,115); ax1.set_ylabel('Percentage (%)'); ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    label_inside_right(ax1,'a')

    classes=['Perovskite','Heusler','Hydride']
    gen_cls=Counter(p['cls'] for p in gen_parsed if p['valid'])
    test_cls=Counter(p['cls'] for p in test_parsed if p['valid'])
    gen_total=sum(gen_cls.values()); test_total=sum(test_cls.values())
    gen_fracs=[gen_cls.get(c.lower(),0)/gen_total*100 for c in classes]
    test_fracs=[test_cls.get(c.lower(),0)/test_total*100 for c in classes]
    x=np.arange(len(classes)); w=0.3
    ax2.bar(x-w/2,test_fracs,w,label='Test set',color=c_test,edgecolor='white',linewidth=0.3,alpha=0.85)
    ax2.bar(x+w/2,gen_fracs,w,label='Generated',color=c_gen,edgecolor='white',linewidth=0.3,alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(classes); ax2.set_ylabel('Fraction (%)')
    ax2.legend(loc='upper left',fontsize=5.5,handletextpad=0.3,borderpad=0.3); ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    label_inside_right(ax2,'b')

    valid_parsed=[p for p in gen_parsed if p['valid']]
    unique_formulas=set(p['formula'] for p in valid_parsed)
    n_unique=len(unique_formulas); n_novel=0
    eval_path=Path(outdir).parent/"generated"/"evaluation_summary.json"
    if eval_path.exists():
        with open(eval_path) as f: ev=json.load(f)
        n_unique=ev.get('n_unique_formulas',n_unique); n_novel=ev.get('n_novel_formulas',0)
    cats=['Total\nvalid','Unique\nformulas','Novel\nformulas']; vals=[len(valid_parsed),n_unique,n_novel]
    bars=ax3.bar(cats,vals,color=[c_total,c_unique,c_novel],width=0.5,edgecolor='white',linewidth=0.3)
    for bar,val in zip(bars,vals):
        ax3.text(bar.get_x()+bar.get_width()/2,bar.get_height()+8,str(val),ha='center',fontsize=7,fontweight='500')
    ax3.set_ylabel('Count'); ax3.set_ylim(0,580); ax3.yaxis.set_minor_locator(AutoMinorLocator(2))
    label_inside_right(ax3,'c')
    fig.tight_layout(w_pad=2.5); save(fig,'fig3_generation_summary',outdir)


# ================================================================
# FIG 4: Structural Distributions — labels inside left, vivid
# ================================================================
def fig4_structural_comparison(gen_metrics, test_metrics, outdir):
    fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=(DC,DC*0.65))
    def hist_compare(ax,gen_vals,test_vals,xlabel,label):
        all_vals=np.concatenate([gen_vals,test_vals])
        bins=np.linspace(np.percentile(all_vals,1),np.percentile(all_vals,99),35)
        ax.hist(test_vals,bins=bins,alpha=0.65,color=C4['test'],label='Test set',edgecolor='white',linewidth=0.3,density=True)
        ax.hist(gen_vals,bins=bins,alpha=0.55,color=C4['gen'],label='Generated',edgecolor='white',linewidth=0.3,density=True)
        ax.set_xlabel(xlabel); ax.set_ylabel('Density')
        ax.legend(loc='upper right',fontsize=6)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        label_inside_left(ax,label)
    hist_compare(ax1,[m['nn_mean'] for m in gen_metrics],[m['nn_mean'] for m in test_metrics],'Mean NN distance (\u00c5)','a')
    hist_compare(ax2,[m['nn_min'] for m in gen_metrics],[m['nn_min'] for m in test_metrics],'Min NN distance (\u00c5)','b')
    hist_compare(ax3,[m['rg'] for m in gen_metrics],[m['rg'] for m in test_metrics],'Radius of gyration (\u00c5)','c')
    hist_compare(ax4,[m['n_atoms'] for m in gen_metrics],[m['n_atoms'] for m in test_metrics],'Number of atoms','d')
    fig.tight_layout(w_pad=2.5,h_pad=3); save(fig,'fig4_structural_distributions',outdir)


# ================================================================
# FIG 5: Per-class NN — labels inside right, vivid different
# ================================================================
def fig5_class_comparison(gen_metrics, test_metrics, outdir):
    fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(DC,DC*0.35))
    classes=['perovskite','heusler','hydride']; axes=[ax1,ax2,ax3]
    for idx,(cls,ax) in enumerate(zip(classes,axes)):
        gm=[m['nn_mean'] for m in gen_metrics if m['cls']==cls]
        tm=[m['nn_mean'] for m in test_metrics if m['cls']==cls]
        if not gm or not tm: continue
        all_vals=np.concatenate([gm,tm])
        bins=np.linspace(all_vals.min()-0.2,all_vals.max()+0.2,25)
        ax.hist(tm,bins=bins,alpha=0.65,color=C5['test'],label=f'Test (n={len(tm)})',edgecolor='white',linewidth=0.3,density=True)
        ax.hist(gm,bins=bins,alpha=0.55,color=C5['gen'],label=f'Gen (n={len(gm)})',edgecolor='white',linewidth=0.3,density=True)
        ax.set_xlabel('\u27e8d\u2099\u2099\u27e9 (\u00c5)')
        if idx==0: ax.set_ylabel('Density')
        ax.set_title(CLASS_NAMES[cls],fontweight='500',pad=8)
        ax.legend(loc='upper left',fontsize=5.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        label_inside_right(ax,chr(ord('a')+idx))
    fig.tight_layout(w_pad=2.5); save(fig,'fig5_class_nn_comparison',outdir)


# ================================================================
# FIG 6: Post-Processing — labels in good spots, y-axis extended
# ================================================================
def fig6_postprocessing(base, outdir):
    pp_path=base/"generated"/"postprocess_results.json"
    if not pp_path.exists():
        print("  WARNING: postprocess_results.json not found, skipping fig6"); return
    with open(pp_path) as f: results=json.load(f)
    passed=[r for r in results if r['status']=='passed']
    if not passed:
        print("  WARNING: no passed structures, skipping fig6"); return

    fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(DC,DC*0.35))

    # (a) Status — y extended
    status_counts=Counter(r['status'] for r in results)
    labels=list(status_counts.keys()); values=list(status_counts.values())
    colors_pie={'passed':'#2E7D32','error':'#C62828','timeout':'#F57F17','no_convergence':'#78909C'}
    cols=[colors_pie.get(l,COLORS['gray']) for l in labels]
    bars=ax1.bar(range(len(labels)),values,color=cols,width=0.6,edgecolor='white',linewidth=0.3)
    ax1.set_xticks(range(len(labels))); ax1.set_xticklabels([l.replace('_','\n') for l in labels],fontsize=6.5)
    ax1.set_ylabel('Count'); ax1.set_ylim(0,max(values)*1.25)
    for bar,val in zip(bars,values):
        ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+3,str(val),ha='center',fontsize=6.5,fontweight='500')
    label_inside_right(ax1,'a')

    # (b) NN before vs after
    pre_nn=[r.get('pre_nn_min') for r in passed if r.get('pre_nn_min') is not None]
    post_nn=[r.get('nn_min') for r in passed if r.get('pre_nn_min') is not None]
    if pre_nn and post_nn:
        n=min(len(pre_nn),len(post_nn)); pre_nn=pre_nn[:n]; post_nn=post_nn[:n]
        ax2.scatter(pre_nn,post_nn,s=20,alpha=0.6,color='#00897B',edgecolors='none',zorder=2)
        lims=[0,max(max(pre_nn),max(post_nn))+0.5]
        ax2.plot(lims,lims,'--',color=COLORS['gray'],linewidth=0.6,alpha=0.5)
        ax2.axhline(0.8,color='black',linewidth=0.6,linestyle=':',alpha=0.4)
        ax2.axvline(0.8,color='black',linewidth=0.6,linestyle=':',alpha=0.4)
        ax2.set_xlabel(r'$d_{\mathrm{NN,min}}$ before relax ($\AA$)'); ax2.set_ylabel(r'$d_{\mathrm{NN,min}}$ after relax ($\AA$)')
        ax2.text(0.05,0.95,f'n = {n}',transform=ax2.transAxes,fontsize=7,va='top',
                 bbox=dict(boxstyle='round,pad=0.3',facecolor='white',edgecolor=COLORS['gray'],alpha=0.9,linewidth=0.5))
    ax2.text(0.95,0.08,'(b)',transform=ax2.transAxes,fontsize=10,fontweight='bold',va='bottom',ha='right')

    # (c) Physical validity by class — y extended
    cls_data={}
    for r in passed:
        cls=r.get('class','unknown')
        if cls not in cls_data: cls_data[cls]={'total':0,'phys':0}
        cls_data[cls]['total']+=1
        if r.get('physically_valid',False): cls_data[cls]['phys']+=1
    if cls_data:
        classes=sorted(cls_data.keys()); x=np.arange(len(classes))
        totals=[cls_data[c]['total'] for c in classes]
        phys=[cls_data[c]['phys'] for c in classes]
        phys_pct=[p/t*100 if t>0 else 0 for p,t in zip(phys,totals)]
        cls_colors=[{'perovskite':'#1565C0','heusler':'#E65100','hydride':'#2E7D32'}.get(c,'#999') for c in classes]
        ax3.bar(x,phys_pct,color=cls_colors,width=0.5,edgecolor='white',linewidth=0.3)
        ax3.set_xticks(x); ax3.set_xticklabels([CLASS_NAMES.get(c,c) for c in classes])
        ax3.set_ylabel('Physical validity (%)'); ax3.set_ylim(0,120)
        for i,(pct,tot) in enumerate(zip(phys_pct,totals)):
            ax3.text(i,pct+3,f'{pct:.0f}%\n(n={tot})',ha='center',fontsize=5.5)
    ax3.text(0.50,0.95,'(c)',transform=ax3.transAxes,fontsize=10,fontweight='bold',va='top',ha='center')

    fig.tight_layout(w_pad=2.5); save(fig,'fig6_postprocessing',outdir)


# ================================================================
# MAIN
# ================================================================
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--base",default=".")
    args=parser.parse_args(); base=Path(args.base)
    outdir=base/"figures-neurips"; outdir.mkdir(exist_ok=True)

    gen_path=base/"generated"/"generated_nanocifs.txt"
    if not gen_path.exists(): print(f"ERROR: {gen_path} not found"); return
    gen_texts=[s.strip() for s in gen_path.read_text().split("\n\n") if s.strip()]
    gen_parsed=[parse_nanocif(t) for t in gen_texts]
    print(f"Loaded {len(gen_parsed)} generated structures")

    test_path=base/"nanocif"/"test.txt"
    if test_path.exists():
        test_texts=[s.strip() for s in test_path.read_text().split("\n\n") if s.strip()]
        test_parsed=[parse_nanocif(t) for t in test_texts]
        print(f"Loaded {len(test_parsed)} test structures")
    else: test_parsed=[]; print("  WARNING: test.txt not found")

    gen_metrics=[compute_metrics(p) for p in gen_parsed if p['valid']]
    gen_metrics=[m for m in gen_metrics if m is not None]
    test_metrics=[compute_metrics(p) for p in test_parsed if p['valid']]
    test_metrics=[m for m in test_metrics if m is not None]
    print(f"Gen metrics: {len(gen_metrics)}, Test metrics: {len(test_metrics)}")

    print("\n--- Fig 1: NanoCIF representation ---"); fig1_representation(outdir)
    print("\n--- Fig 2: Training curves ---"); fig2_training(base,outdir)
    print("\n--- Fig 3: Generation summary ---"); fig3_generation_summary(gen_parsed,test_parsed,outdir)
    print("\n--- Fig 4: Structural distributions ---"); fig4_structural_comparison(gen_metrics,test_metrics,outdir)
    print("\n--- Fig 5: Per-class comparison ---"); fig5_class_comparison(gen_metrics,test_metrics,outdir)
    print("\n--- Fig 6: Post-processing ---"); fig6_postprocessing(base,outdir)
    print(f"\nAll figures saved to {outdir}/")

if __name__=="__main__": main()
