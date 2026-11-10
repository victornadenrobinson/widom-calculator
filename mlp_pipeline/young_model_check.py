"""One-frame GPU check of the transferred potential; no MD or files written."""
import argparse
import importlib
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', default='grace-ft:runs/grace/ft_mfinkc_cont/seed/1/final_model')
p.add_argument('--frames', default='runs/valid_m.extxyz')
a = p.parse_args()
if not Path(a.frames).is_file():
    p.error(f'frame file missing: {a.frames}; supply --frames PATH')
import tensorflow as tf
from ase.io import read
print('TensorFlow:', tf.__version__, flush=True)
gpus = tf.config.list_physical_devices('GPU')
print('Visible GPUs:', gpus, flush=True)
if not gpus:
    raise SystemExit('No TensorFlow GPU visible: run inside a Young GPU allocation.')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
atoms = read(a.frames, index=0)
print('Model:', a.model, '\nFrame:', a.frames, atoms.get_chemical_formula(), flush=True)
atoms.calc = importlib.import_module('04_benchmark').get_calc_spec(a.model, 'cuda')
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
if not all(np.isfinite(x).all() for x in (energy, forces, stress)):
    raise SystemExit('FAIL: non-finite model output')
print(f'Energy: {energy:.10f} eV; max |F|: {np.linalg.norm(forces, axis=1).max():.8f} eV/A')
print('Stress (eV/A^3):', stress)
print('PASS: model loaded and evaluated energy, forces and stress on a GPU.')
