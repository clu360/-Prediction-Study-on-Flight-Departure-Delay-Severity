import json
import subprocess
import sys
from flight_delay.config import Config
from flight_delay.pipeline import run
from test_discovery import make_year


def test_full_year_pipeline_and_reproducibility(tmp_path):
    raw=tmp_path/"raw"
    raw.mkdir()
    make_year(raw)
    out=tmp_path/"processed"
    config=Config(evaluation_airports=("DFW",))
    first=run(raw,out,2025,config)
    assert first["acceptance_passed"]
    assert first["counts"]["raw"]==24
    assert first["counts"]["modeling_cohort"]==12
    hashes=(out/"audit/logical_hashes.json").read_text()
    second=run(raw,out,2025,config)
    assert first["counts"]==second["counts"]
    assert hashes==(out/"audit/logical_hashes.json").read_text()
    # Exercise the actual documented module CLI in a separate interpreter.
    for command in [["audit","--processed-dir",str(out)],
                    ["discover","--raw-dir",str(raw),"--manifest",str(out/"audit/cli_manifest.json")]]:
        result=subprocess.run([sys.executable,"-m","flight_delay.cli",*command],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
