"""Offline release checksum verification and lossless artifact reconstruction.

No network, provider SDK, model execution, training, or result generation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

def digest(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def safe_path(root: Path, name: str) -> Path:
    dest=(root/name).resolve()
    if not dest.is_relative_to(root.resolve()): raise ValueError('Path escapes destination: '+name)
    return dest

def verify_assets(folder: Path) -> None:
    failures=[]; count=0
    for line in (folder/'SHA256SUMS.txt').read_text(encoding='utf8').splitlines():
        if not line.strip(): continue
        expected,name=line.split(None,1); name=name.lstrip('* ')
        p=safe_path(folder,name)
        if not p.is_file() or digest(p)!=expected: failures.append(name)
        count+=1
    if failures: raise RuntimeError('Asset checksum failures: '+', '.join(failures))
    print(json.dumps({'asset_checksums':'PASS','assets':count}))

def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets-dir',type=Path,required=True)
    ap.add_argument('--git-root',type=Path)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--checksums-only',action='store_true')
    ap.add_argument('--verify-only',action='store_true',help='Check all sources without writing output files.')
    args=ap.parse_args(); assets=args.assets_dir.resolve()
    verify_assets(assets)
    if args.checksums_only: return 0
    catalog=json.loads((assets/'FILE_CATALOG.json').read_text(encoding='utf8'))
    if not args.verify_only and args.output is None: ap.error('--output is required unless --verify-only is used')
    gitroot=args.git_root.resolve() if args.git_root else None
    if gitroot:
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=gitroot,text=True).strip()
        if head!=catalog['baseline_commit']: raise RuntimeError('git-root must be checked out at '+catalog['baseline_commit'])
    verified_archives={}; errors=[]; verified=0; copied=0
    def archive(name: str, expected: str | None=None) -> Path:
        p=safe_path(assets,name)
        if not p.is_file(): raise FileNotFoundError('Required asset: '+name)
        if name not in verified_archives: verified_archives[name]=digest(p)
        if expected and verified_archives[name]!=expected: raise RuntimeError('Archive hash mismatch: '+name)
        return p
    out=args.output.resolve() if args.output else None
    for r in catalog['files']:
        try:
            if r['storage']=='new_alias': continue
            dest=safe_path(out,r['path']) if out else None
            if dest and dest.exists():
                if not dest.is_file() or digest(dest)!=r['sha256']: raise RuntimeError('Refusing to overwrite different bytes: '+r['path'])
                verified+=1; continue
            if r['storage'] in {'new_public','existing_public_release'}:
                dep=r.get('dependency',{})
                asset=r.get('asset') or dep['asset']; member=r.get('member') or dep['member']
                p=archive(asset,dep.get('asset_sha256'))
                with zipfile.ZipFile(p) as z:
                    with z.open(member) as f:
                        h=hashlib.sha256(); size=0
                        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b); size+=len(b)
                    if h.hexdigest()!=r['sha256'] or size!=r['bytes']: raise RuntimeError('ZIP member mismatch: '+r['path'])
                    if dest and not args.verify_only:
                        dest.parent.mkdir(parents=True,exist_ok=True)
                        with z.open(member) as src,dest.open('xb') as f: shutil.copyfileobj(src,f)
                        copied+=1
            elif r['storage']=='existing_public_git':
                if gitroot is None: raise FileNotFoundError('--git-root is required for '+r['dependency']['path'])
                p=safe_path(gitroot,r['dependency']['path'])
                if not p.is_file() or p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']: raise RuntimeError('Git/LFS source missing or incorrect: '+str(p))
                if dest and not args.verify_only:
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with p.open('rb') as src,dest.open('xb') as f: shutil.copyfileobj(src,f)
                    copied+=1
            else: raise RuntimeError('Unknown storage '+r['storage'])
            verified+=1
        except Exception as e: errors.append({'path':r['path'],'error':str(e)})
    bypath={r['path']:r for r in catalog['files']}
    for r in catalog['files']:
        if r['storage']!='new_alias': continue
        if args.verify_only:
            if r['canonical_path'] not in bypath or bypath[r['canonical_path']]['sha256']!=r['sha256']: errors.append({'path':r['path'],'error':'Alias target invalid'})
            else: verified+=1
        else:
            src=safe_path(out,r['canonical_path']); dest=safe_path(out,r['path'])
            try:
                if not src.is_file() or digest(src)!=r['sha256']: raise RuntimeError('Alias source not restored')
                if dest.exists():
                    if digest(dest)!=r['sha256']: raise RuntimeError('Refusing to overwrite different alias bytes')
                else:
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with src.open('rb') as s,dest.open('xb') as d: shutil.copyfileobj(s,d)
                    copied+=1
                verified+=1
            except Exception as e: errors.append({'path':r['path'],'error':str(e)})
    result={'status':'PASS' if not errors else 'INCOMPLETE','verified_paths':verified,'copied_paths':copied,'errors':errors,'network_calls':0,'model_calls':0,'training_runs':0}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if not errors else 1

if __name__=='__main__':
    sys.exit(main())
