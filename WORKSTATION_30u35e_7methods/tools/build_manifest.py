"""Hash portable source/docs/config and selected completed verification outputs."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from systemlab.common import sha,write_new

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
paths=[]
for folder in ('systemlab','tools','tests','configs','docs','verification'):
    paths.extend(path for path in (root/folder).rglob('*') if path.is_file() and
        '__pycache__' not in path.parts and path.suffix in ('.py','.md','.json','.txt','.jsonl'))
paths.extend(root/name for name in ('README.md','MANIFEST.md','requirements.txt','Dockerfile','compose.yaml'))
target=Path(a.output).resolve()
write_new(target,dict(schema='systemlab-delivery-manifest-v1',
    files={str(path.relative_to(root)):dict(sha256=sha(path),bytes=path.stat().st_size)
           for path in sorted(set(paths)) if path.resolve()!=target},
    scope='Code, documents, configuration and preserved author verification; no formal results',
    original_paper_rl_data_untouched=True))
