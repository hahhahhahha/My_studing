from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import shutil
import urllib.request
from pathlib import Path

from .common import local_windows_runtime, read, runtime, sha, write_new


def main():
    local_windows_runtime()
    parser=argparse.ArgumentParser(description='Real inference/queue/feedback prototype; new protocol')
    commands=parser.add_subparsers(dest='command',required=True)
    env=commands.add_parser('preflight')
    env.add_argument('--output',required=True)
    image=commands.add_parser('images')
    image.add_argument('--split-file',required=True)
    image.add_argument('--image-root',required=True)
    image.add_argument('--role',choices=['pilot','calibration','validation','test'],required=True)
    image.add_argument('--limit',type=int,default=100)
    image.add_argument('--output',required=True)
    event=commands.add_parser('schedule')
    event.add_argument('--manifest',required=True)
    event.add_argument('--users',type=int,default=16)
    event.add_argument('--duration',type=float,default=30)
    event.add_argument('--rate',type=float,default=20)
    event.add_argument('--seed',type=int,default=1)
    event.add_argument('--role',choices=['pilot','calibration','validation','test'],required=True)
    event.add_argument('--output',required=True)
    event.add_argument('--azure2021-csv')
    event.add_argument('--trace-start',type=float)
    event.add_argument('--trace-window',type=float,default=600)
    event.add_argument('--time-scale',type=float,default=1)
    event.add_argument('--time-unit',choices=['seconds','milliseconds'])
    minute=commands.add_parser('azure2019-schedule')
    minute.add_argument('--csv',required=True)
    minute.add_argument('--manifest',required=True)
    minute.add_argument('--columns',type=int,nargs='+',required=True)
    minute.add_argument('--users',type=int,default=16)
    minute.add_argument('--count-scale',type=float,required=True)
    minute.add_argument('--seed',type=int,default=1)
    minute.add_argument('--role',choices=['pilot','calibration','validation','test'],required=True)
    minute.add_argument('--output',required=True)
    for name in ('edge','controller','run','run-local'):
        command=commands.add_parser(name)
        command.add_argument('--config',required=True)
        command.add_argument('--output',required=True)
        if name=='edge':
            command.add_argument('--server',required=True)
        if name.startswith('run'):
            command.add_argument('--manifest',required=True)
            command.add_argument('--schedule',required=True)
            command.add_argument('--method',choices=['round_robin','score_softmax','stale_shortest','qla','qla_no_bias_probe','frozen_noql'],required=True)
            command.add_argument('--actor-bundle')
            command.add_argument('--entry-review',help='Required for test-role runs; complete independent entry JSON')
    actor=commands.add_parser('register-actor')
    actor.add_argument('--checkpoint',required=True)
    actor.add_argument('--source-root',required=True)
    actor.add_argument('--feature-scales',required=True)
    actor.add_argument('--edges',type=int,required=True)
    actor.add_argument('--allow-edge-count-transfer',action='store_true')
    actor.add_argument('--architecture',choices=['edgewise','shared_gaussian'],default='edgewise')
    actor.add_argument('--log-std-min',type=float,default=-5)
    actor.add_argument('--log-std-max',type=float,default=1)
    actor.add_argument('--output',required=True)
    analysis=commands.add_parser('analyze')
    analysis.add_argument('--client-dir',required=True)
    analysis.add_argument('--controller-dir')
    analysis.add_argument('--output',required=True)
    fit=commands.add_parser('calibrate')
    fit.add_argument('--client-dir',required=True)
    fit.add_argument('--edges-dir',required=True)
    fit.add_argument('--config',required=True)
    fit.add_argument('--manifest',required=True)
    fit.add_argument('--output',required=True)
    frozen=commands.add_parser('freeze')
    frozen.add_argument('--config',required=True)
    frozen.add_argument('--validation-client-dirs',nargs='+',required=True)
    frozen.add_argument('--output-dir',required=True)
    condition=commands.add_parser('register-condition')
    condition.add_argument('--base-config',required=True)
    condition.add_argument('--changes-json',required=True)
    condition.add_argument('--output-dir',required=True)
    paired=commands.add_parser('paired-compare')
    paired.add_argument('--analyses',nargs='+',required=True)
    paired.add_argument('--treatment',required=True)
    paired.add_argument('--baseline',required=True)
    paired.add_argument('--metric',default='cvar99_s')
    paired.add_argument('--output',required=True)
    fetch=commands.add_parser('fetch')
    fetch.add_argument('--source',choices=['imagenette','azure2021','resnet18'],required=True)
    fetch.add_argument('--expected-sha256')
    fetch.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='preflight':
        packages={}
        for package in ('numpy','torch','torchvision','Pillow'):
            try:
                packages[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package]=None
        write_new(args.output,dict(runtime=runtime(),installed_packages=packages,
            docker_executable=shutil.which('docker'),model_executed=False,network_services_started=False))
    elif args.command=='images':
        from .data import import_images
        import_images(args.split_file,args.image_root,args.role,args.output,args.limit)
    elif args.command=='schedule':
        from .data import schedule
        if args.azure2021_csv and not args.time_unit:
            parser.error('Azure2021 requires explicit --time-unit after schema verification')
        schedule(args.manifest,args.output,users=args.users,duration=args.duration,rate=args.rate,
                 seed=args.seed,role=args.role,azure=args.azure2021_csv,trace_start=args.trace_start,
                 trace_window=args.trace_window,time_scale=args.time_scale,time_unit=args.time_unit or 'seconds')
    elif args.command=='azure2019-schedule':
        from .data import minute_schedule
        minute_schedule(args.csv,args.manifest,args.output,columns=args.columns,users=args.users,
                        rate_scale=args.count_scale,seed=args.seed,role=args.role)
    elif args.command in ('edge','controller','run','run-local'):
        cfg=read(args.config)
        if args.command=='edge':
            from .edge import serve_edge
            serve_edge(cfg,args.server,args.output)
        elif args.command=='controller':
            from .feedback import serve_controller
            serve_controller(cfg,args.output)
        elif args.command=='run':
            from .client import replay
            replay(cfg,args.schedule,args.manifest,args.method,args.output,args.actor_bundle,args.entry_review)
        else:
            from .launcher import local_run
            local_run(args.config,cfg,args.schedule,args.manifest,args.method,args.output,args.actor_bundle,args.entry_review)
    elif args.command=='register-actor':
        from .policy import register_actor
        register_actor(args.checkpoint,args.source_root,args.output,args.edges,args.feature_scales,
            transfer=args.allow_edge_count_transfer,architecture=args.architecture,
            log_std_min=args.log_std_min,log_std_max=args.log_std_max)
    elif args.command=='analyze':
        from .analysis import analyze
        analyze(args.client_dir,args.output,args.controller_dir)
    elif args.command=='calibrate':
        from .analysis import calibrate
        calibrate(args.client_dir,args.edges_dir,read(args.config),args.manifest,args.output)
    elif args.command=='freeze':
        from .analysis import freeze
        freeze(read(args.config),args.validation_client_dirs,args.output_dir)
    elif args.command=='register-condition':
        from .analysis import register_condition
        register_condition(args.base_config,read(args.changes_json),args.output_dir)
    elif args.command=='paired-compare':
        from .analysis import paired_compare
        paired_compare(args.analyses,args.treatment,args.baseline,args.metric,args.output)
    elif args.command=='fetch':
        urls={'imagenette':'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz',
              'azure2021':'https://github.com/Azure/AzurePublicDataset/raw/master/data/AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar',
              'resnet18':'https://download.pytorch.org/models/resnet18-f37072fd.pth'}
        path=Path(args.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        part=path.with_suffix(path.suffix+'.part')
        if path.exists() or part.exists():
            raise FileExistsError('Existing/failed acquisition must be preserved; choose a new attempt')
        try:
            with urllib.request.urlopen(urls[args.source],timeout=60) as response, part.open('xb') as stream:
                shutil.copyfileobj(response,stream)
            fingerprint=sha(part)
            if args.expected_sha256 and fingerprint!=args.expected_sha256.lower():
                raise ValueError('Upstream/pinned SHA mismatch; partial archive preserved')
            with part.open('rb') as stream:
                if stream.read(128).startswith(b'version https://git-lfs.github.com'):
                    raise ValueError('Received an LFS pointer, not the requested dataset')
            part.rename(path)
            write_new(str(path)+'.receipt.json',dict(source=args.source,url=urls[args.source],
                sha256=fingerprint,bytes=path.stat().st_size,upstream_checksum_verified=bool(args.expected_sha256)))
        except Exception as error:
            write_new(str(part)+'.failure.json',dict(error=repr(error),source=args.source))
            raise


if __name__=='__main__':
    main()
