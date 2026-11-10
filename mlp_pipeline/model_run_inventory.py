#!/usr/bin/env python3
"""Find evidence of a model in run names and text metadata; never alters runs.

A model path or directory name is evidence, not proof of identical weights.
Binary files, model weights, source snapshots and symlinks are not inspected.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys

TEXT_SUFFIXES = {'.json', '.jsonl', '.log', '.out', '.txt', '.csv', '.sh', '.sbatch', '.yaml', '.yml'}
SKIP = {'.git', '__pycache__', 'code', 'variables', 'assets', 'model_transfer_check'}


def model_fields(obj, needle, prefix=''):
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f'{prefix}.{key}' if prefix else str(key)
            if 'model' in key.lower() and isinstance(value, str) and needle in value.lower():
                yield name, value
            elif isinstance(value, (dict, list)):
                yield from model_fields(value, needle, name)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from model_fields(value, needle, f'{prefix}[{i}]')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', nargs='?', default='runs')
    p.add_argument('--model', default='mfinkc_cont', help='case-insensitive model substring')
    p.add_argument('--out', required=True, help='new CSV file for detailed evidence')
    p.add_argument('--max-mb', type=float, default=64, help='maximum bytes read per text file (default 64 MiB)')
    a = p.parse_args()
    root = Path(a.root)
    if not root.is_dir():
        p.error(f'not a directory: {root}')
    needle = a.model.lower()
    rows, issues = [], []
    limit = int(a.max_mb * 1024 * 1024)
    def add(kind, path, detail):
        rows.append((kind, str(path), detail.replace('\n', ' ')[:500]))
    def onerror(error):
        issues.append(str(error))
    for current, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
        path = Path(current)
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not (path / d).is_symlink())
        # We are inventorying runs, not the model checkpoint itself.
        if path == root / 'grace':
            dirs[:] = [d for d in dirs if needle not in d.lower()]
        if needle in path.name.lower():
            add('directory name', path, 'Name matches; inspect outputs to confirm model and completion')
        for name in sorted(files):
            f = path / name
            if f.is_symlink() or f.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                with f.open('rb') as inp:
                    raw = inp.read(limit + 1)
                if len(raw) > limit:
                    issues.append(f'{f}: scanned first {a.max_mb:g} MiB only')
                    raw = raw[:limit]
                if b'\0' in raw:
                    continue
                content = raw.decode('utf-8', errors='replace')
                if needle not in content.lower():
                    continue
                found = False
                if f.suffix.lower() == '.json':
                    try:
                        data = json.loads(content)
                        for key, value in model_fields(data, needle):
                            add('model field', f, f'{key} = {value}')
                            found = True
                    except (ValueError, RecursionError):
                        pass
                if not found:
                    lines = [line.strip() for line in content.splitlines() if needle in line.lower()]
                    # One row per file keeps large repeated MD logs manageable.
                    add('text mention', f, lines[0] + (f' ({len(lines)} matching lines)' if len(lines) > 1 else ''))
            except (OSError, ValueError) as exc:
                issues.append(f'{f}: {exc}')
    out = Path(a.out)
    try:
        with out.open('x', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['evidence', 'path', 'detail'])
            writer.writerows(rows)
    except OSError as exc:
        p.error(str(exc))
    print(f'Inventory of {root}: model substring {a.model!r}')
    for kind in ('model field', 'directory name', 'text mention'):
        subset = [r for r in rows if r[0] == kind]
        print(f'\n{kind.upper()}: {len(subset)}')
        for _, path, detail in subset:
            print(f'  {path}: {detail}')
    print(f'\nDetailed CSV: {out}')
    print('Model fields record a path/name, not a checksum; directory names and log mentions are weaker evidence.')
    print('Source snapshots, model weights, binary files and symlinks were not searched; a missing match does not prove a run used another model.')
    if issues:
        print(f'\nRead/scan limitations ({len(issues)}):', file=sys.stderr)
        for issue in issues:
            print(f'  {issue}', file=sys.stderr)


if __name__ == '__main__':
    main()
