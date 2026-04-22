#!/usr/bin/env python3
"""
NanoCIF — Timeout Analysis & Improved Post-Processing
Usage:
  Step 1 (analysis):  python3 17_analyze_failures.py --base . --mode analyze
  Step 2 (improved):  python3 17_analyze_failures.py --base . --mode rerun --workers 10
  Step 3 (compare):   python3 17_analyze_failures.py --base . --mode compare

Diagnoses timeout causes and implements:
  - Adaptive timeout (scaled by atom count)
  - Pre-relaxation distance filter
  - Pre-relaxation coordinate repair (push apart close atoms)
  - SCC parameter tuning for problematic compositions
"""
import argparse
import json
import subprocess
import numpy as np
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


def load_results(base):
    """Load original post-processing results."""
    pp_path = base / "generated" / "postprocess_results.json"
    with open(pp_path) as f:
        return json.load(f)


# ================================================================
# STEP 1: ANALYZE
# ================================================================
def analyze(base):
    results = load_results(base)
    
    print("=" * 70)
    print("TIMEOUT & FAILURE ANALYSIS")
    print("=" * 70)
    
    # Categorize
    passed = [r for r in results if r['status'] == 'passed']
    timeout = [r for r in results if r['status'] == 'timeout']
    error = [r for r in results if r['status'] in ('error', 'no_convergence')]
    
    print(f"\nOverall: {len(passed)} passed, {len(timeout)} timeout, {len(error)} error")
    
    # --- 1. Atom count analysis ---
    print(f"\n{'─'*50}")
    print("1. ATOM COUNT ANALYSIS")
    print(f"{'─'*50}")
    
    passed_atoms = [r['n_atoms'] for r in passed]
    timeout_atoms = [r['n_atoms'] for r in timeout]
    error_atoms = [r['n_atoms'] for r in error]
    
    print(f"  Passed:  mean={np.mean(passed_atoms):.1f}, median={np.median(passed_atoms):.0f}, "
          f"min={np.min(passed_atoms)}, max={np.max(passed_atoms)}")
    print(f"  Timeout: mean={np.mean(timeout_atoms):.1f}, median={np.median(timeout_atoms):.0f}, "
          f"min={np.min(timeout_atoms)}, max={np.max(timeout_atoms)}")
    if error_atoms:
        print(f"  Error:   mean={np.mean(error_atoms):.1f}, median={np.median(error_atoms):.0f}")
    
    # Atom count bins
    bins = [(0, 30), (30, 50), (50, 80), (80, 200)]
    print(f"\n  Atom count distribution:")
    print(f"  {'Range':<12s} {'Passed':>8s} {'Timeout':>8s} {'Error':>8s} {'Pass%':>8s}")
    for lo, hi in bins:
        p = sum(1 for r in passed if lo <= r['n_atoms'] < hi)
        t = sum(1 for r in timeout if lo <= r['n_atoms'] < hi)
        e = sum(1 for r in error if lo <= r['n_atoms'] < hi)
        total = p + t + e
        pct = p / total * 100 if total > 0 else 0
        print(f"  {lo:>3d}-{hi:<3d}     {p:>8d} {t:>8d} {e:>8d} {pct:>7.1f}%")
    
    # --- 2. Class analysis ---
    print(f"\n{'─'*50}")
    print("2. CLASS ANALYSIS")
    print(f"{'─'*50}")
    
    for cls in ['perovskite', 'heusler', 'hydride']:
        p = sum(1 for r in passed if r.get('class') == cls)
        t = sum(1 for r in timeout if r.get('class') == cls)
        e = sum(1 for r in error if r.get('class') == cls)
        total = p + t + e
        print(f"  {cls:<12s}: {p:>3d} passed, {t:>3d} timeout, {e:>3d} error "
              f"(pass rate: {p/total*100:.1f}%)")
    
    # --- 3. Initial d_NN,min analysis ---
    print(f"\n{'─'*50}")
    print("3. INITIAL d_NN,min ANALYSIS")
    print(f"{'─'*50}")
    
    passed_pre = [r.get('pre_nn_min', None) for r in passed]
    passed_pre = [v for v in passed_pre if v is not None]
    timeout_pre = [r.get('pre_nn_min', None) for r in timeout]
    timeout_pre = [v for v in timeout_pre if v is not None]
    
    if passed_pre:
        print(f"  Passed  pre d_NN,min: mean={np.mean(passed_pre):.3f}, "
              f"median={np.median(passed_pre):.3f} Å")
    if timeout_pre:
        print(f"  Timeout pre d_NN,min: mean={np.mean(timeout_pre):.3f}, "
              f"median={np.median(timeout_pre):.3f} Å")
    
    # d_NN,min bins
    bins_d = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 3.0)]
    print(f"\n  Pre-relax d_NN,min distribution (passed+timeout with known pre_nn_min):")
    print(f"  {'Range (Å)':<14s} {'Passed':>8s} {'Timeout':>8s} {'Pass%':>8s}")
    for lo, hi in bins_d:
        p = sum(1 for r in passed if r.get('pre_nn_min') is not None and lo <= r['pre_nn_min'] < hi)
        t = sum(1 for r in timeout if r.get('pre_nn_min') is not None and lo <= r['pre_nn_min'] < hi)
        total = p + t
        pct = p / total * 100 if total > 0 else 0
        print(f"  {lo:.1f}-{hi:.1f}       {p:>8d} {t:>8d} {pct:>7.1f}%")
    
    # --- 4. Combined risk factors ---
    print(f"\n{'─'*50}")
    print("4. COMBINED RISK FACTORS")
    print(f"{'─'*50}")
    
    high_risk = sum(1 for r in timeout if r['n_atoms'] > 50)
    low_risk = sum(1 for r in timeout if r['n_atoms'] <= 50)
    print(f"  Timeout with >50 atoms: {high_risk}/{len(timeout)} ({high_risk/len(timeout)*100:.1f}%)")
    print(f"  Timeout with ≤50 atoms: {low_risk}/{len(timeout)} ({low_risk/len(timeout)*100:.1f}%)")
    
    # --- 5. Suggested adaptive timeout ---
    print(f"\n{'─'*50}")
    print("5. ADAPTIVE TIMEOUT RECOMMENDATION")
    print(f"{'─'*50}")
    
    print("  Current: flat 600s for all structures")
    print("  Proposed: timeout = 300 + 15 × n_atoms")
    print(f"  {'n_atoms':<10s} {'Current':>10s} {'Proposed':>10s}")
    for n in [20, 40, 60, 80, 100, 150]:
        proposed = 300 + 15 * n
        print(f"  {n:<10d} {600:>10d}s {proposed:>10d}s")
    
    # --- 6. Pre-filter recommendation ---
    print(f"\n{'─'*50}")
    print("6. PRE-FILTER RECOMMENDATION")
    print(f"{'─'*50}")
    
    # Count structures with d_NN,min < 0.3 that timeout
    very_short = sum(1 for r in timeout if r.get('pre_nn_min') is not None and r['pre_nn_min'] < 0.3)
    print(f"  Timeout structures with pre d_NN,min < 0.3 Å: {very_short}")
    print(f"  Recommendation: push apart atoms closer than 0.5 Å before DFTB+")
    
    # Save analysis
    analysis = {
        'passed_count': len(passed),
        'timeout_count': len(timeout),
        'error_count': len(error),
        'passed_mean_atoms': float(np.mean(passed_atoms)),
        'timeout_mean_atoms': float(np.mean(timeout_atoms)),
        'passed_pre_nn_min': float(np.mean(passed_pre)) if passed_pre else None,
        'timeout_pre_nn_min': float(np.mean(timeout_pre)) if timeout_pre else None,
        'timeout_high_risk_pct': high_risk / len(timeout) * 100,
    }
    
    out_path = base / "generated" / "failure_analysis.json"
    with open(out_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {out_path}")


# ================================================================
# PRE-RELAXATION COORDINATE REPAIR
# ================================================================
def repair_coordinates(atoms, coords, min_dist=0.5, push_dist=1.2, max_iter=50):
    """Push apart atoms that are too close before DFTB+ relaxation."""
    from scipy.spatial.distance import cdist
    
    coords = coords.copy()
    
    for iteration in range(max_iter):
        dists = cdist(coords, coords)
        np.fill_diagonal(dists, np.inf)
        nn_min = dists.min()
        
        if nn_min >= min_dist:
            break
        
        # Find closest pair
        i, j = np.unravel_index(dists.argmin(), dists.shape)
        
        # Push apart along their connecting vector
        vec = coords[j] - coords[i]
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            vec = np.random.randn(3)
            norm = np.linalg.norm(vec)
        
        direction = vec / norm
        displacement = (push_dist - norm) / 2.0
        
        coords[i] -= direction * displacement
        coords[j] += direction * displacement
    
    return coords


def parse_nanocif(text):
    """Parse NanoCIF text."""
    result = {'formula': None, 'cls': None, 'radius': None, 'atoms': [], 'coords': []}
    lines = text.strip().split('\n')
    in_loop = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('data_'):
            result['formula'] = line[5:]
        elif line.startswith('_class '):
            result['cls'] = line[7:]
        elif line.startswith('_radius '):
            try: result['radius'] = int(line[8:])
            except: pass
        elif line.startswith('loop_'):
            in_loop = True
        elif line.startswith('_atom_type'):
            continue
        elif in_loop:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    result['atoms'].append(parts[0])
                    result['coords'].append([float(parts[1]), float(parts[2]), float(parts[3])])
                except:
                    continue
    result['coords'] = np.array(result['coords']) if result['coords'] else np.array([])
    result['n_atoms'] = len(result['atoms'])
    return result


def write_gen_file(filepath, atoms, coords):
    """Write DFTB+ .gen format."""
    n = len(atoms)
    unique_elements = list(dict.fromkeys(atoms))
    type_map = {el: i + 1 for i, el in enumerate(unique_elements)}
    lines = [f"{n} C"]
    lines.append(" ".join(unique_elements))
    for i, (el, (x, y, z)) in enumerate(zip(atoms, coords)):
        lines.append(f"{i+1} {type_map[el]} {x:.6f} {y:.6f} {z:.6f}")
    Path(filepath).write_text("\n".join(lines) + "\n")
    return unique_elements


def write_dftb_input(workdir, unique_elements, sk_path, temperature=600, 
                      mixing=0.2, max_scc=1500, scc_tol=1e-5):
    """Write dftb_in.hsd with configurable parameters."""
    ang_mom = {
        'H':'s','He':'s','Li':'p','Be':'p','B':'p','C':'p','N':'p','O':'p','F':'p','Ne':'p',
        'Na':'p','Mg':'p','Al':'p','Si':'p','P':'p','S':'p','Cl':'p','Ar':'p',
        'K':'p','Ca':'d','Sc':'d','Ti':'d','V':'d','Cr':'d','Mn':'d',
        'Fe':'d','Co':'d','Ni':'d','Cu':'d','Zn':'d',
        'Ga':'p','Ge':'p','As':'p','Se':'p','Br':'p','Kr':'p',
        'Rb':'p','Sr':'d','Y':'d','Zr':'d','Nb':'d','Mo':'d',
        'Tc':'d','Ru':'d','Rh':'d','Pd':'d','Ag':'d','Cd':'d',
        'In':'p','Sn':'p','Sb':'p','Te':'p','I':'p','Xe':'p',
        'Cs':'p','Ba':'d','La':'d','Lu':'d',
        'Hf':'d','Ta':'d','W':'d','Re':'d','Os':'d','Ir':'d',
        'Pt':'d','Au':'d','Hg':'d','Tl':'p','Pb':'p','Bi':'p',
    }
    
    d_block = {'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
               'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
               'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','La','Lu'}
    is_metallic = any(el in d_block for el in unique_elements)
    temp = 1000 if is_metallic else temperature
    
    max_ang_lines = "\n".join(f'    {el} = "{ang_mom.get(el, "p")}"' for el in unique_elements)
    
    hsd = f"""Geometry = GenFormat {{
  <<< "input.gen"
}}

Hamiltonian = DFTB {{
  Scc = Yes
  MaxSccIterations = {max_scc}
  SccTolerance = {scc_tol}
  Mixer = Broyden {{
    MixingParameter = {mixing}
  }}
  MaxAngularMomentum {{
{max_ang_lines}
  }}
  SlaterKosterFiles = Type2FileNames {{
    Prefix = "{sk_path}/"
    Separator = "-"
    Suffix = ".skf"
  }}
  Filling = Fermi {{
    Temperature [K] = {temp}
  }}
}}

Driver = ConjugateGradient {{
  MaxSteps = 2000
  LatticeOpt = No
  MaxForceComponent = 1.0e-3
  AppendGeometries = No
}}

Options {{
  WriteDetailedOut = Yes
}}
"""
    (workdir / "dftb_in.hsd").write_text(hsd)


def run_improved_relaxation(args_tuple):
    """Run improved DFTB+ relaxation with pre-repair and adaptive timeout."""
    idx, nanocif_text, base_dir, sk_path, dftb_binary, strategy = args_tuple
    
    parsed = parse_nanocif(nanocif_text)
    if not parsed['atoms'] or len(parsed['atoms']) < 3:
        return {'idx': idx, 'status': 'parse_error', 'formula': None, 'strategy': strategy}
    
    formula = parsed['formula'] or f'unknown_{idx}'
    cls = parsed['cls'] or 'unknown'
    radius = parsed['radius'] or 0
    n_atoms = parsed['n_atoms']
    
    workdir = base_dir / f"gen_{idx:04d}_{formula}_R{radius}"
    workdir.mkdir(parents=True, exist_ok=True)
    
    result = {
        'idx': idx, 'formula': formula, 'class': cls, 'radius': radius,
        'n_atoms': n_atoms, 'status': 'failed', 'strategy': strategy,
    }
    
    coords = parsed['coords']
    atoms = parsed['atoms']
    
    # Pre-relaxation d_NN,min
    from scipy.spatial.distance import cdist
    pre_dists = cdist(coords, coords)
    np.fill_diagonal(pre_dists, np.inf)
    pre_nn_min = pre_dists.min()
    result['pre_nn_min'] = float(pre_nn_min)
    
    # --- STRATEGY: Pre-repair ---
    if strategy in ('repair', 'full'):
        if pre_nn_min < 0.5:
            coords = repair_coordinates(atoms, coords, min_dist=0.5, push_dist=1.2)
            # Recompute
            rep_dists = cdist(coords, coords)
            np.fill_diagonal(rep_dists, np.inf)
            result['repaired_nn_min'] = float(rep_dists.min())
    
    # --- STRATEGY: Adaptive timeout ---
    if strategy in ('adaptive', 'full'):
        timeout = max(600, 300 + 15 * n_atoms)
    else:
        timeout = 600
    result['timeout_used'] = timeout
    
    # --- STRATEGY: SCC tuning for problematic compositions ---
    mixing = 0.2
    max_scc = 1500
    scc_tol = 1e-5
    if strategy in ('scc_tune', 'full'):
        if cls == 'perovskite' and n_atoms > 60:
            mixing = 0.1  # More conservative mixing for large perovskites
            max_scc = 2000
        if cls == 'hydride':
            scc_tol = 5e-5  # Slightly relaxed tolerance for hydrides
    
    try:
        unique_elements = write_gen_file(workdir / "input.gen", atoms, coords)
        write_dftb_input(workdir, unique_elements, sk_path, mixing=mixing, 
                         max_scc=max_scc, scc_tol=scc_tol)
        
        proc = subprocess.run(
            [dftb_binary], cwd=str(workdir),
            capture_output=True, text=True, timeout=timeout,
        )
        
        geo_end = workdir / "geo_end.gen"
        detailed = workdir / "detailed.out"
        
        if geo_end.exists():
            energy = None
            if detailed.exists():
                for line in detailed.read_text().split('\n'):
                    if 'Total energy' in line and 'eV' in line:
                        try:
                            energy = float(line.split()[-2])
                        except:
                            pass
            
            with open(geo_end) as f:
                gen_lines = f.readlines()
            n_relaxed = int(gen_lines[0].split()[0])
            relaxed_coords = []
            for i in range(2, 2 + n_relaxed):
                parts = gen_lines[i].split()
                relaxed_coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
            relaxed_coords = np.array(relaxed_coords)
            
            dists = cdist(relaxed_coords, relaxed_coords)
            np.fill_diagonal(dists, np.inf)
            nn_dists = dists.min(axis=1)
            
            result['status'] = 'passed'
            result['energy_total'] = energy
            result['energy_per_atom'] = energy / n_relaxed if energy else None
            result['nn_mean'] = float(nn_dists.mean())
            result['nn_min'] = float(nn_dists.min())
            result['nn_std'] = float(nn_dists.std())
            result['physically_valid'] = bool(nn_dists.min() > 0.8)
        else:
            result['status'] = 'no_convergence'
            
    except subprocess.TimeoutExpired:
        result['status'] = 'timeout'
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)[:200]
    
    return result


# ================================================================
# STEP 2: RERUN WITH IMPROVEMENTS
# ================================================================
def rerun(base, workers=10, strategy='full'):
    """Re-run post-processing with improvements on previously failed structures."""
    
    # Load original results
    orig_results = load_results(base)
    
    # Find DFTB+ and SK files
    dftb_binary = None
    for candidate in ["/home/mkurban/dftbplus-install/bin/dftb+", "dftb+"]:
        if Path(candidate).exists():
            dftb_binary = candidate
            break
    
    sk_path = None
    for candidate in ["/mnt/d/DFTB_shared/skfiles/ParameterSets/ptbp/complete_set"]:
        if Path(candidate).exists():
            sk_path = str(candidate)
            break
    
    if not dftb_binary or not sk_path:
        print("ERROR: dftb+ or SK files not found")
        return
    
    # Load generated NanoCIF texts
    gen_path = base / "generated" / "generated_nanocifs.txt"
    texts = [s.strip() for s in gen_path.read_text().split("\n\n") if s.strip()]
    
    # Find failed indices
    failed_indices = []
    for r in orig_results:
        if r['status'] in ('timeout', 'error', 'no_convergence'):
            failed_indices.append(r['idx'])
    
    print(f"Re-running {len(failed_indices)} previously failed structures")
    print(f"Strategy: {strategy}")
    print(f"Workers: {workers}")
    
    # Work directory
    work_dir = base / "generated" / "improved_relax"
    work_dir.mkdir(exist_ok=True)
    
    # Prepare tasks
    tasks = [(idx, texts[idx], work_dir, sk_path, dftb_binary, strategy) 
             for idx in failed_indices if idx < len(texts)]
    
    # Run
    improved_results = []
    start = time.time()
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_improved_relaxation, task): task[0] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            improved_results.append(result)
            n_done = len(improved_results)
            if n_done % 20 == 0 or n_done == len(tasks):
                n_passed = sum(1 for r in improved_results if r['status'] == 'passed')
                elapsed = time.time() - start
                print(f"  {n_done}/{len(tasks)} done | Newly passed: {n_passed} | {elapsed:.0f}s")
    
    improved_results.sort(key=lambda r: r['idx'])
    
    # Merge with original results
    orig_passed = [r for r in orig_results if r['status'] == 'passed']
    newly_passed = [r for r in improved_results if r['status'] == 'passed']
    still_failed = [r for r in improved_results if r['status'] != 'passed']
    
    print(f"\n{'='*60}")
    print(f"IMPROVED POST-PROCESSING RESULTS")
    print(f"{'='*60}")
    print(f"Originally passed: {len(orig_passed)}")
    print(f"Newly passed: {len(newly_passed)}")
    print(f"Still failed: {len(still_failed)}")
    
    total_passed = len(orig_passed) + len(newly_passed)
    total = len(orig_results)
    print(f"\nNew pipeline success: {total_passed}/{total} ({100*total_passed/total:.1f}%)")
    print(f"Previous:             {len(orig_passed)}/{total} ({100*len(orig_passed)/total:.1f}%)")
    print(f"Improvement:          +{len(newly_passed)} structures (+{len(newly_passed)/total*100:.1f}%)")
    
    # Physical validity of newly passed
    if newly_passed:
        new_phys = sum(1 for r in newly_passed if r.get('physically_valid', False))
        print(f"\nNewly passed physical validity: {new_phys}/{len(newly_passed)} ({100*new_phys/len(newly_passed):.1f}%)")
        
        nn_mins = [r['nn_min'] for r in newly_passed]
        nn_means = [r['nn_mean'] for r in newly_passed]
        print(f"  NN min:  {np.mean(nn_mins):.3f} ± {np.std(nn_mins):.3f} Å")
        print(f"  NN mean: {np.mean(nn_means):.3f} ± {np.std(nn_means):.3f} Å")
        
        # By class
        print(f"\n  Class breakdown (newly passed):")
        for cls in ['perovskite', 'heusler', 'hydride']:
            cls_new = [r for r in newly_passed if r.get('class') == cls]
            cls_phys = sum(1 for r in cls_new if r.get('physically_valid', False))
            if cls_new:
                print(f"    {cls}: {len(cls_new)} newly passed, {cls_phys} physically valid")
    
    # Strategy effectiveness
    still_timeout = sum(1 for r in still_failed if r['status'] == 'timeout')
    still_error = sum(1 for r in still_failed if r['status'] in ('error', 'no_convergence'))
    print(f"\n  Still timeout: {still_timeout}")
    print(f"  Still error: {still_error}")
    
    # Save
    # Convert numpy types
    for r in improved_results:
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                r[k] = float(v) if isinstance(v, np.floating) else int(v)
            elif isinstance(v, np.bool_):
                r[k] = bool(v)
    
    out_path = base / "generated" / "improved_results.json"
    with open(out_path, 'w') as f:
        json.dump({
            'original_passed': len(orig_passed),
            'newly_passed': len(newly_passed),
            'still_failed': len(still_failed),
            'total': total,
            'new_success_rate': total_passed / total,
            'strategy': strategy,
            'improved_results': improved_results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


# ================================================================
# STEP 3: COMPARE
# ================================================================
def compare(base):
    """Generate comparison figures and tables."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 8, 'axes.labelsize': 9,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'axes.linewidth': 0.6, 'figure.dpi': 1200, 'savefig.dpi': 1200,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    })
    
    DC = 180 / 25.4
    
    orig_results = load_results(base)
    
    imp_path = base / "generated" / "improved_results.json"
    if not imp_path.exists():
        print("ERROR: Run --mode rerun first")
        return
    with open(imp_path) as f:
        imp_data = json.load(f)
    
    imp_results = imp_data['improved_results']
    
    outdir = base / "figures-neurips"
    outdir.mkdir(exist_ok=True)
    
    # --- Figure: Before vs After improvement ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(DC, DC * 0.35))
    
    # (a) Convergence comparison
    orig_passed = sum(1 for r in orig_results if r['status'] == 'passed')
    new_passed = imp_data['original_passed'] + imp_data['newly_passed']
    total = imp_data['total']
    
    cats = ['Original\npipeline', 'Improved\npipeline']
    vals = [orig_passed / total * 100, new_passed / total * 100]
    colors = ['#C62828', '#2E7D32']
    bars = ax1.bar(cats, vals, color=colors, width=0.5, edgecolor='white', linewidth=0.3)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', fontsize=8, fontweight='600')
    ax1.set_ylabel('Convergence rate (%)')
    ax1.set_ylim(0, 100)
    ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, fontsize=10, fontweight='bold',
             va='top', ha='left')
    
    # (b) Physical validity comparison
    orig_phys = sum(1 for r in orig_results if r['status'] == 'passed' and r.get('physically_valid', False))
    new_phys = orig_phys + sum(1 for r in imp_results if r['status'] == 'passed' and r.get('physically_valid', False))
    
    vals2 = [orig_phys / total * 100, new_phys / total * 100]
    bars2 = ax2.bar(cats, vals2, color=colors, width=0.5, edgecolor='white', linewidth=0.3)
    for bar, val in zip(bars2, vals2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', fontsize=8, fontweight='600')
    ax2.set_ylabel('Net physical validity (%)')
    ax2.set_ylim(0, 100)
    ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes, fontsize=10, fontweight='bold',
             va='top', ha='left')
    
    # (c) Timeout breakdown by atom count
    orig_timeout = [r for r in orig_results if r['status'] == 'timeout']
    still_timeout = [r for r in imp_results if r['status'] == 'timeout']
    
    bins = [0, 30, 50, 80, 200]
    orig_counts = []
    new_counts = []
    labels = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        labels.append(f'{lo}-{hi}')
        orig_counts.append(sum(1 for r in orig_timeout if lo <= r['n_atoms'] < hi))
        new_counts.append(sum(1 for r in still_timeout if lo <= r['n_atoms'] < hi))
    
    x = np.arange(len(labels))
    w = 0.3
    ax3.bar(x - w/2, orig_counts, w, label='Original', color='#C62828', alpha=0.7, edgecolor='white')
    ax3.bar(x + w/2, new_counts, w, label='Improved', color='#2E7D32', alpha=0.7, edgecolor='white')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_xlabel('Atom count range')
    ax3.set_ylabel('Timeout count')
    ax3.legend(fontsize=6)
    ax3.text(0.05, 0.95, '(c)', transform=ax3.transAxes, fontsize=10, fontweight='bold',
             va='top', ha='left')
    
    fig.tight_layout(w_pad=2.5)
    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(outdir / f"fig7_improved_pipeline.{fmt}", format=fmt)
    plt.close(fig)
    print(f"Saved: fig7_improved_pipeline")
    
    # Print summary table for paper
    print(f"\n{'='*60}")
    print("TABLE FOR PAPER: Pipeline Improvement Summary")
    print(f"{'='*60}")
    print(f"{'Metric':<35s} {'Original':>12s} {'Improved':>12s}")
    print(f"{'─'*60}")
    print(f"{'Converged':<35s} {orig_passed:>10d}/500 {new_passed:>10d}/500")
    print(f"{'Convergence rate':<35s} {orig_passed/total*100:>11.1f}% {new_passed/total*100:>11.1f}%")
    print(f"{'Physically valid':<35s} {orig_phys:>10d}/500 {new_phys:>10d}/500")
    print(f"{'Net physical validity':<35s} {orig_phys/total*100:>11.1f}% {new_phys/total*100:>11.1f}%")


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=".")
    parser.add_argument("--mode", choices=['analyze', 'rerun', 'compare'], required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--strategy", default="full",
                        choices=["repair", "adaptive", "scc_tune", "full"],
                        help="Improvement strategy")
    args = parser.parse_args()
    
    base = Path(args.base)
    
    if args.mode == 'analyze':
        analyze(base)
    elif args.mode == 'rerun':
        rerun(base, workers=args.workers, strategy=args.strategy)
    elif args.mode == 'compare':
        compare(base)


if __name__ == "__main__":
    main()
