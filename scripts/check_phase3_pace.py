"""Read durable PACE job state over the user's existing SSH connection."""

from pathlib import Path
import argparse
import json
import os
import re
import subprocess


def check(connection, identity):
    config = json.loads(connection.read_text(encoding="utf-8"))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config["user"]) or not re.fullmatch(
        r"[A-Za-z0-9.-]+\.pace\.gatech\.edu", config["host"]
    ):
        raise ValueError("Expected a PACE username/hostname")
    script = r"""from pathlib import Path
import json,subprocess,socket
root=Path.home()/REMOTE_ROOT/'flight-delay'
output=root/'data/model_outputs/phase3_pace'
workflow=json.loads((output/'slurm/workflow.json').read_text())
ids=','.join(stage['job_id'] for stage in workflow['stages'])
result={'query_host':socket.gethostname(),'workflow':workflow}
for key,command in [
 ('queue',['squeue','-h','-j',ids,'-o','%i|%j|%T|%r|%P|%C']),
 ('accounting',['sacct','-X','-n','-P','-j',ids,'--format=JobID,JobName,State,ExitCode,Elapsed,AllocCPUS,NodeList'])]:
 p=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 result[key]=p.stdout.splitlines()
 result[key+'_returncode']=p.returncode
plan=output/'batch/B_tasks.json'
gates=sorted((output/'validation').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)
gate=json.loads(gates[0].read_text()) if gates else None
if gate:
 result['upstream_validation']={k:gate.get(k) for k in ['passed','cohort_count','input_digest','errors','node_local_staging_seconds','total_validation_seconds']}
run_id=None
if plan.exists():
 run_id=json.loads(plan.read_text())['run_id']
elif gate:
 for p in output.glob('phase3-*/context.json'):
  if json.loads(p.read_text()).get('config_hash')==gate['config_hash']:
   run_id=p.parent.name
   break
if run_id:
 names={p.name for p in (output/run_id/'development/units').glob('*.json')}
 for p in (output/'batch_workers').glob('*/task-*/attempt-*/'+run_id+'/development/units/*.json'):names.add(p.name)
 result['run_id']=run_id
 result['completed_fit_checkpoints']=len(names)
 for field,relative in [('final_fit_checkpoints','final/fits'),('final_month_checkpoints','final/months'),('final_pooled_checkpoints','final/pooled')]:
  result[field]=len(list((output/run_id/relative).glob('*.json')))
logs=sorted((output/'slurm').glob('*.log'),key=lambda p:p.stat().st_mtime,reverse=True)[:4]
result['recent_log_tails']={p.name:p.read_text(encoding='utf-8',errors='replace').splitlines()[-12:] for p in logs}
print(json.dumps(result))
""".replace("REMOTE_ROOT", repr(config["remote_relative_root"]))
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=12",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-i",
        str(identity),
        config["user"] + "@" + config["host"],
        "python3 -",
    ]
    completed = subprocess.run(
        command,
        input=script.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
        check=True,
    )
    result = json.loads(completed.stdout.decode("utf-8"))
    path = connection.parent / "status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connection",
        type=Path,
        default=Path("data/model_outputs/phase3/pace_transfer/connection.json"),
    )
    parser.add_argument(
        "--identity",
        type=Path,
        default=Path(os.environ.get("USERPROFILE", str(Path.home())))
        / ".ssh/id_ed25519_pace",
    )
    parser.add_argument("--logs", action="store_true")
    args = parser.parse_args()
    result = check(args.connection, args.identity)
    for line in result["queue"]:
        print(line)
    if result.get("upstream_validation"):
        print("Upstream validation:", json.dumps(result["upstream_validation"]))
    print("Completed fit checkpoints:", result.get("completed_fit_checkpoints", 0))
    print("Final fit checkpoints:", result.get("final_fit_checkpoints", 0))
    print("Final scored-month checkpoints:", result.get("final_month_checkpoints", 0))
    print("Final pooled checkpoints:", result.get("final_pooled_checkpoints", 0))
    if args.logs:
        print(json.dumps(result["recent_log_tails"], indent=2))
