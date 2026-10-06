"""Generate three native text-to-video scenes with corrected SelfLift, then join."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
ROOT=Path(__file__).resolve().parent
REVISION="3ec17a324ced54151364f24f8b5fb6bf7e26414f"
TURBO="minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors"


def run():
    from check_correction import check
    from huggingface_hub import hf_hub_download
    from imageio_ffmpeg import get_ffmpeg_exe
    check()
    started=time.time()
    report={"status":"started","selflift":True,"correction":"paper_eq8_adaptive",
            "checkpoint":"hybrid", "turbo_revision":REVISION,"turbo_file":TURBO,"scenes":[]}
    receipt=ROOT/"story-job.json"
    def stage(name):
        report.update(stage=name,elapsed_seconds=time.time()-started)
        receipt.write_text(json.dumps(report,indent=2))
        print("STORY",name,flush=True)
    try:
        stage("download_8step_adapter")
        hf_hub_download("lightx2v/Minimax-h3-Turbo",TURBO,revision=REVISION,local_dir=ROOT/"models/turbo")
        scenes=json.loads((ROOT/"prompts/storm-guardian-scenes.json").read_text())
        output=ROOT/"output"
        output.mkdir(exist_ok=True)
        for scene in scenes:
            stage(f"generate_scene_{scene['scene']}")
            before=set(output.glob("selflift-quality-*.json"))
            command=[sys.executable,str(ROOT/"monitor_run.py"),"--mode","selflift-quality",
                     "--checkpoint","hybrid",
                     "--dtype","float32","--attention-fp16","--frames","124","--vram-limit","10",
                     "--seed",str(scene["seed"]),"--prompt-file",str(ROOT/"prompts"/scene["prompt_file"])]
            subprocess.run(command,check=True)
            new=list(set(output.glob("selflift-quality-*.json"))-before)
            if len(new)!=1:
                raise RuntimeError("Expected one new scene receipt")
            data=json.loads(new[0].read_text())
            if data["status"]!="success":
                raise RuntimeError("Scene did not export successfully")
            memory=output/(new[0].stem+"-memory.json")
            shutil.copyfile(ROOT/"memory-observation.json",memory)
            report["scenes"].append({"scene":scene["scene"],"receipt":new[0].name,
                "video_file":data["video_file"],"memory_file":memory.name})
        stage("assemble_15_seconds")
        listing=output/"storm-guardian-concat.txt"
        listing.write_text("".join("file '"+s["video_file"]+"'\n" for s in report["scenes"]))
        final=output/"storm-guardian-15s.mp4"
        subprocess.run([get_ffmpeg_exe(),"-y","-v","error","-f","concat","-safe","0","-i",str(listing),
                        "-t","15","-c:v","libx264","-crf","18","-preset","medium",
                        "-c:a","aac","-b:a","192k","-movflags","+faststart",str(final)],check=True)
        report.update(status="success",video_file=final.name)
        stage("complete")
    except Exception as error:
        report.update(status="failed",error=str(error))
        stage("failed")
        raise


if __name__=="__main__":
    run()
