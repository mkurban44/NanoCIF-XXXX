#!/usr/bin/env python3
"""
NanoCIF Post-Processing: DFTB+ Relaxation of Generated Structures
Usage: python3 14_postprocess_relax.py --base . --workers 10

Pipeline: Generated NanoCIF → XYZ → DFTB+ gen → Relax → Evaluate
"""
import argparse
import json
import subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import time


def parse_nanocif(text):
    """Parse NanoCIF text → element list + coordinates."""
    lines = text.strip().split('\n')
    formula = None
    cls = None
    radius = None
    atoms = []
    coords = []
    in_loop = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('data_'):
            formula = line[5:]
        elif line.startswith('_class '):
            cls = line[7:]
        elif line.startswith('_radius '):
            try:
                radius = int(line[8:])
            except:
                pass
        elif line.startswith('loop_'):
            in_loop = True
        elif line.startswith('_atom_type'):
            continue
        elif in_loop:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    atoms.append(parts[0])
                    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except:
                    continue

    return {
        'formula': formula,
        'class': cls,
        'radius': radius,
        'atoms': atoms,
        'coords': np.array(coords) if coords else np.array([]),
    }


def write_gen_file(filepath, atoms, coords):
    """Write DFTB+ .gen format (cluster geometry)."""
    n = len(atoms)
    unique_elements = list(dict.fromkeys(atoms))  # preserve order
    type_map = {el: i + 1 for i, el in enumerate(unique_elements)}

    lines = [f"{n} C"]
    lines.append(" ".join(unique_elements))
    for i, (el, (x, y, z)) in enumerate(zip(atoms, coords)):
        lines.append(f"{i+1} {type_map[el]} {x:.6f} {y:.6f} {z:.6f}")

    Path(filepath).write_text("\n".join(lines) + "\n")
    return unique_elements


def write_dftb_input(workdir, unique_elements, sk_path, temperature=600):
    """Write dftb_in.hsd for nanoparticle relaxation."""
    # MaxAngularMomentum mapping
    ang_mom = {
        'H': 's', 'He': 's',
        'Li': 'p', 'Be': 'p', 'B': 'p', 'C': 'p', 'N': 'p', 'O': 'p', 'F': 'p', 'Ne': 'p',
        'Na': 'p', 'Mg': 'p', 'Al': 'p', 'Si': 'p', 'P': 'p', 'S': 'p', 'Cl': 'p', 'Ar': 'p',
        'K': 'p', 'Ca': 'd', 'Sc': 'd', 'Ti': 'd', 'V': 'd', 'Cr': 'd', 'Mn': 'd',
        'Fe': 'd', 'Co': 'd', 'Ni': 'd', 'Cu': 'd', 'Zn': 'd',
        'Ga': 'p', 'Ge': 'p', 'As': 'p', 'Se': 'p', 'Br': 'p', 'Kr': 'p',
        'Rb': 'p', 'Sr': 'd', 'Y': 'd', 'Zr': 'd', 'Nb': 'd', 'Mo': 'd',
        'Tc': 'd', 'Ru': 'd', 'Rh': 'd', 'Pd': 'd', 'Ag': 'd', 'Cd': 'd',
        'In': 'p', 'Sn': 'p', 'Sb': 'p', 'Te': 'p', 'I': 'p', 'Xe': 'p',
        'Cs': 'p', 'Ba': 'd', 'La': 'd', 'Lu': 'd',
        'Hf': 'd', 'Ta': 'd', 'W': 'd', 'Re': 'd', 'Os': 'd', 'Ir': 'd',
        'Pt': 'd', 'Au': 'd', 'Hg': 'd', 'Tl': 'p', 'Pb': 'p', 'Bi': 'p',
        'Po': 'p', 'At': 'p', 'Rn': 'p', 'Ra': 'd', 'Th': 'd',
    }

    # Check if metallic (has d-block transition metals)
    d_block = {'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
               'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
               'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg',
               'La','Lu'}
    is_metallic = any(el in d_block for el in unique_elements)
    temp = 1000 if is_metallic else temperature

    max_ang_lines = "\n".join(f'    {el} = "{ang_mom.get(el, "p")}"' for el in unique_elements)

    hsd = f"""Geometry = GenFormat {{
  <<< "input.gen"
}}

Hamiltonian = DFTB {{
  Scc = Yes
  MaxSccIterations = 1500
  SccTolerance = 1.0e-5
  Mixer = Broyden {{
    MixingParameter = 0.2
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


def run_relaxation(args_tuple):
    """Run DFTB+ relaxation for a single generated structure."""
    idx, nanocif_text, base_dir, sk_path, dftb_binary, timeout = args_tuple

    parsed = parse_nanocif(nanocif_text)
    if not parsed['atoms'] or len(parsed['atoms']) < 3:
        return {'idx': idx, 'status': 'parse_error', 'formula': None}

    formula = parsed['formula'] or f'unknown_{idx}'
    cls = parsed['class'] or 'unknown'
    radius = parsed['radius'] or 0

    workdir = base_dir / f"gen_{idx:04d}_{formula}_R{radius}"
    workdir.mkdir(parents=True, exist_ok=True)

    result = {
        'idx': idx,
        'formula': formula,
        'class': cls,
        'radius': radius,
        'n_atoms': len(parsed['atoms']),
        'status': 'failed',
    }

    try:
        # Write input files
        unique_elements = write_gen_file(workdir / "input.gen", parsed['atoms'], parsed['coords'])
        write_dftb_input(workdir, unique_elements, sk_path)

        # Run DFTB+
        proc = subprocess.run(
            [dftb_binary],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Check output
        geo_end = workdir / "geo_end.gen"
        detailed = workdir / "detailed.out"

        if geo_end.exists():
            # Parse energy from detailed.out
            energy = None
            if detailed.exists():
                for line in detailed.read_text().split('\n'):
                    if 'Total energy' in line and 'H' not in line.split(':')[-1]:
                        try:
                            energy = float(line.split()[-1])
                        except:
                            pass
                    if 'Total energy' in line and 'eV' in line:
                        try:
                            energy = float(line.split()[-2])
                        except:
                            pass

            # Read relaxed structure
            with open(geo_end) as f:
                gen_lines = f.readlines()
            n_relaxed = int(gen_lines[0].split()[0])
            relaxed_coords = []
            for i in range(2, 2 + n_relaxed):
                parts = gen_lines[i].split()
                relaxed_coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
            relaxed_coords = np.array(relaxed_coords)

            # Compute NN distances
            from scipy.spatial.distance import cdist
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

            # Compare with pre-relaxation
            if len(parsed['coords']) > 0:
                pre_dists = cdist(parsed['coords'], parsed['coords'])
                np.fill_diagonal(pre_dists, np.inf)
                pre_nn = pre_dists.min(axis=1)
                result['pre_nn_mean'] = float(pre_nn.mean())
                result['pre_nn_min'] = float(pre_nn.min())
        else:
            result['status'] = 'no_convergence'
            if proc.returncode != 0:
                result['error'] = proc.stderr[:200] if proc.stderr else 'unknown'

    except subprocess.TimeoutExpired:
        result['status'] = 'timeout'
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)[:200]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=".")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--dftb", default=None, help="Path to dftb+ binary")
    parser.add_argument("--sk_path", default=None, help="Path to SK files")
    args = parser.parse_args()

    base = Path(args.base)

    # Find DFTB+ binary
    dftb_binary = args.dftb
    if dftb_binary is None:
        for candidate in ["/home/mkurban/dftbplus-install/bin/dftb+",
                          "dftb+",
                          "/usr/local/bin/dftb+"]:
            if Path(candidate).exists() or subprocess.run(["which", candidate],
                                                           capture_output=True).returncode == 0:
                dftb_binary = candidate
                break

    if dftb_binary is None:
        print("ERROR: dftb+ binary not found. Use --dftb flag.")
        return

    # Find SK files
    sk_path = args.sk_path
    if sk_path is None:
        for candidate in ["/mnt/d/DFTB_shared/skfiles/ParameterSets/ptbp/complete_set",
                          base / "skfiles"]:
            if Path(candidate).exists():
                sk_path = str(candidate)
                break

    if sk_path is None:
        print("ERROR: SK files not found. Use --sk_path flag.")
        return

    print(f"DFTB+ binary: {dftb_binary}")
    print(f"SK files: {sk_path}")

    # Load generated structures
    gen_path = base / "generated" / "generated_nanocifs.txt"
    if not gen_path.exists():
        print(f"ERROR: {gen_path} not found. Run 13_generate_evaluate.py first.")
        return

    texts = [s.strip() for s in gen_path.read_text().split("\n\n") if s.strip()]
    print(f"Loaded {len(texts)} generated structures")

    # Work directory
    work_dir = base / "generated" / "relax_work"
    work_dir.mkdir(exist_ok=True)

    # Prepare tasks
    tasks = [(i, text, work_dir, sk_path, dftb_binary, args.timeout) for i, text in enumerate(texts)]

    # Run relaxations
    print(f"\nRelaxing {len(tasks)} structures with {args.workers} workers...")
    print(f"Timeout: {args.timeout}s per structure")
    print("=" * 60)

    results = []
    start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_relaxation, task): task[0] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            n_done = len(results)
            if n_done % 10 == 0 or n_done == len(tasks):
                n_passed = sum(1 for r in results if r['status'] == 'passed')
                elapsed = time.time() - start
                print(f"  {n_done}/{len(tasks)} done | Passed: {n_passed} | {elapsed:.0f}s")

    # Sort by index
    results.sort(key=lambda r: r['idx'])

    # ================================================================
    # SUMMARY
    # ================================================================
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"POST-PROCESSING RESULTS")
    print(f"{'='*60}")
    print(f"Total: {len(results)}")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    status_counts = Counter(r['status'] for r in results)
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count} ({100*count/len(results):.1f}%)")

    passed = [r for r in results if r['status'] == 'passed']
    if passed:
        print(f"\nPassed structures ({len(passed)}):")
        phys_valid = sum(1 for r in passed if r.get('physically_valid', False))
        print(f"  Physically valid (d_min > 0.8 Å): {phys_valid}/{len(passed)} ({100*phys_valid/len(passed):.1f}%)")

        nn_means = [r['nn_mean'] for r in passed]
        nn_mins = [r['nn_min'] for r in passed]
        energies = [r['energy_per_atom'] for r in passed if r.get('energy_per_atom')]

        print(f"  NN mean: {np.mean(nn_means):.3f} ± {np.std(nn_means):.3f} Å")
        print(f"  NN min:  {np.mean(nn_mins):.3f} ± {np.std(nn_mins):.3f} Å")
        if energies:
            print(f"  Energy/atom: {np.mean(energies):.2f} ± {np.std(energies):.2f} eV")

        # Improvement from relaxation
        pre_nn_mins = [r.get('pre_nn_min', 0) for r in passed if r.get('pre_nn_min')]
        post_nn_mins = [r['nn_min'] for r in passed if r.get('pre_nn_min')]
        if pre_nn_mins:
            print(f"\n  Before relaxation: NN min = {np.mean(pre_nn_mins):.3f} Å")
            print(f"  After relaxation:  NN min = {np.mean(post_nn_mins):.3f} Å")

        # Class breakdown
        print(f"\n  Class breakdown:")
        for cls in ['perovskite', 'heusler', 'hydride']:
            cls_passed = [r for r in passed if r.get('class') == cls]
            cls_phys = sum(1 for r in cls_passed if r.get('physically_valid', False))
            if cls_passed:
                print(f"    {cls}: {len(cls_passed)} passed, {cls_phys} physically valid")

    # Save results
    out_path = base / "generated" / "postprocess_results.json"
    # Convert numpy types for JSON
    for r in results:
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                r[k] = float(v) if isinstance(v, np.floating) else int(v)
            elif isinstance(v, np.bool_):
                r[k] = bool(v)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
