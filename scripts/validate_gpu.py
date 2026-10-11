"""Run one half of the fixed validation matrix on a selected physical GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--gpu",type=int,choices=(0,1),required=True)
    p.add_argument("--epochs",type=int,default=400)
    p.add_argument("--output-root",type=Path)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    cases={
        0:["mbb_3axis","mount_5axis","pedal_3axis","link_5axis","random_single","global_scale_coarse","no_design"],
        1:["mbb_5axis","mount_3axis","pedal_5axis","link_3axis","random_two_sided","local_z_scale","custom"],
    }[args.gpu]
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OPENBLAS_NUM_THREADS="4",OMP_NUM_THREADS="4")
    base=args.output_root or root/"results"/"validation"
    base.mkdir(parents=True,exist_ok=True)
    for case in cases:
        output=base/case
        command=[sys.executable,"-m","boxyto","optimize",str(root/"examples"/(case+".json")),
                 "--output",str(output),"--fe-backend","cupy","--epochs",str(args.epochs)]
        print(json.dumps({"gpu":args.gpu,"case":case,"state":"start"}),flush=True)
        with (base/(case+".log")).open("x") as log:
            subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        summary=json.loads((output/"summary.json").read_text())
        print(json.dumps({"gpu":args.gpu,"case":case,"state":"complete",**summary}),flush=True)

if __name__=="__main__":
    main()
