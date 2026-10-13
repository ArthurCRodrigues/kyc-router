"""Small, bounded OCR systems lab. No server, container, training, or test data.

Run via 04e_ocr_resource_lab.ipynb. The worker runs in an ephemeral systemd user
cgroup; analysis is ordinary pandas/matplotlib. Notebook definitions are loaded
explicitly, hashed, and never include training cells. This is temporary reuse
before the jointly approved final source extraction, not the production API.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
from collections.abc import Mapping
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import queue
import subprocess
import threading
import time

import cv2
import nbformat
import numpy as np
import pandas as pd
import psutil
import torch
from transformers import GenerationConfig, TrOCRProcessor, VisionEncoderDecoderModel

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ocr_resource_profile_v1.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def cgroup_snapshot():
    membership = Path("/proc/self/cgroup").read_text().strip().split("0::", 1)[1]
    directory = Path("/sys/fs/cgroup") / membership.lstrip("/")
    def read(name):
        path = directory / name
        return path.read_text().strip() if path.exists() else None
    stats = {key: int(value) for key, value in
             (line.split() for line in (read("cpu.stat") or "").splitlines())}
    return dict(path=membership, cpu_max=read("cpu.max"), memory_max=read("memory.max"),
                memory_current=read("memory.current"), memory_peak=read("memory.peak"),
                memory_swap_max=read("memory.swap.max"), memory_events=read("memory.events"), cpu=stats)


class Sampler:
    """Own-process CPU/RSS and PyTorch memory; never claim device-wide GPU use."""
    def __init__(self):
        self.process = psutil.Process()
        self.start = time.perf_counter()
        self.phase = "startup"
        self.rows = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        previous_wall = time.perf_counter()
        previous_cpu = sum(self.process.cpu_times()[:2])
        while not self.stop_event.wait(0.1):
            now = time.perf_counter()
            cpu = sum(self.process.cpu_times()[:2])
            self.rows.append(dict(seconds=now-self.start, phase=self.phase,
                cpu_cores=(cpu-previous_cpu)/(now-previous_wall),
                rss_mib=self.process.memory_info().rss/2**20,
                gpu_allocated_mib=torch.cuda.memory_allocated()/2**20,
                gpu_reserved_mib=torch.cuda.memory_reserved()/2**20))
            previous_wall, previous_cpu = now, cpu

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()


def load_notebook_contract():
    """Execute only established imports/config/definitions, never fitting cells."""
    path = ROOT / "notebooks/04d_ocr_finetuning.ipynb"
    nb = nbformat.read(path, as_version=4)
    namespace = {"__name__": "ocr_notebook_contract", "Mapping": Mapping}
    provenance = {}
    for index in [1, *range(3, 25)]:
        cell = nb.cells[index]
        assert cell.cell_type == "code"
        digest = hashlib.sha256(cell.source.strip().encode()).hexdigest()
        if "source_sha256" in cell.metadata:
            assert digest == cell.metadata.source_sha256
        provenance[f"04d:{index}"] = digest
        exec(compile(cell.source, f"04d cell {index}", "exec"), namespace)
    return namespace, provenance


class ResourceLab:
    def __init__(self, output, profile):
        self.output, self.profile = Path(output), profile
        self.output.mkdir(parents=True, exist_ok=False)
        self.config = json.loads(CONFIG.read_text())
        self.amendment = None
        if profile=="small_tuned":
            amendment_path=ROOT/"configs/ocr_resource_profile_thread_control_v1.json"
            self.amendment=json.loads(amendment_path.read_text())
            self.config["torch_threads"]=self.amendment["torch_threads"]
        torch.set_num_threads(self.config["torch_threads"])
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)
        assert torch.cuda.is_available()
        torch.cuda.set_per_process_memory_fraction(
            self.config["pytorch_gpu_allocator_cap_gib"] * 2**30 /
            torch.cuda.get_device_properties(0).total_memory)
        self.device = torch.device("cuda:0")
        self.initial_cgroup = cgroup_snapshot()
        expected_quota = 4 if profile == "roomy" else 1
        quota, period = map(int, self.initial_cgroup["cpu_max"].split())
        assert quota / period == expected_quota
        assert int(self.initial_cgroup["memory_max"]) == 4*2**30
        assert self.initial_cgroup["memory_swap_max"] == "0"
        self.contract, self.provenance = load_notebook_contract()
        self.fields = self.contract["required_field_names"]
        manifest = json.loads((ROOT/"configs/ocr_finetuning_candidate_v1.json").read_text())
        checkpoint = ROOT / manifest["checkpoint_relative_path"]
        assert sha(checkpoint/"model.safetensors") == manifest["checkpoint_files_sha256"]["model.safetensors"]
        t = time.perf_counter()
        self.processor = TrOCRProcessor.from_pretrained(checkpoint, local_files_only=True)
        self.processor_load_ms = (time.perf_counter()-t)*1000
        t = time.perf_counter()
        self.model = VisionEncoderDecoderModel.from_pretrained(
            checkpoint, local_files_only=True, dtype=torch.float32, use_safetensors=True).eval()
        self.model_load_cpu_ms = (time.perf_counter()-t)*1000
        t = time.perf_counter()
        self.model.to(self.device)
        torch.cuda.synchronize()
        self.weights_to_gpu_ms = (time.perf_counter()-t)*1000
        policy = json.loads((ROOT/"configs/ocr_finetuning_v1.json").read_text())
        self.generation = GenerationConfig.from_pretrained(checkpoint, local_files_only=True)
        for key, value in policy["generation"].items():
            setattr(self.generation, key, value)
        self.model.config.decoder_start_token_id = self.generation.decoder_start_token_id
        self.model.config.pad_token_id = self.generation.pad_token_id
        self.model.config.eos_token_id = self.generation.eos_token_id
        self.observations = []
        ns = self.contract
        keys = sorted(ns["split_manifest"].loc[ns["split_manifest"].split=="validation", "identity_key"])
        for key in keys:
            assets = ns["load_alb_id_assets"](key, manifest_index=ns["manifest_by_identity"],
                dataset_root=ns["dataset_root"], template_record_index=ns["template_records_by_filename"],
                photo_record_index=ns["photo_records_by_filename"])
            assert assets["split"] == "validation"
            for source in ["template", "photo"]:
                self.observations.append(ns["make_canonical_observation"](
                    assets, source, ns["canonical_width"], ns["canonical_height"]))
        assert len(self.observations)==30
        prior = ROOT/"outputs/phase3_finetuning"/manifest["run_id"]/"selected_reload_fields.jsonl"
        prior_fields = pd.read_json(prior, lines=True)
        self.expected = {(key, source): dict(zip(group.field, group.raw_prediction))
            for (key,source),group in prior_fields.groupby(["identity_key","source"])}
        self.fingerprint = hashlib.sha256(json.dumps(dict(
            weights=manifest["checkpoint_files_sha256"], code=self.provenance,
            generation=policy["generation"]), sort_keys=True).encode()).hexdigest()
        self.rows, self.predictions, self.variant_stats = [], [], []
        self.prepared_cpu, self.prepared_gpu = {}, {}

    def crops(self, observation):
        return self.contract["extract_field_crops"](observation,
            self.contract["raw_to_canonical_field"], self.contract["field_crop_config"])

    def request(self, observation, variant="reference", trace=False):
        """Synchronized completed stages, not asynchronous enqueue-only timings."""
        torch.cuda.synchronize()
        begin, cpu_begin = time.perf_counter(), time.process_time()
        stages = dict(crop_ms=0., processor_ms=0., h2d_ms=0., generation_ms=0., decode_ms=0., postprocess_ms=0.)
        scope = torch.profiler.record_function if trace else lambda _: contextlib.nullcontext()
        key = (observation["identity_key"], observation["source"])
        t=time.perf_counter()
        with scope("crop_seven_fields"):
            crops=self.crops(observation)
        stages["crop_ms"]=(time.perf_counter()-t)*1000
        raw, tokens = {}, 0
        generation=copy.deepcopy(self.generation)
        generation.use_cache = variant != "reference"
        groups = [list(crops)] if variant=="batch7_kv" else [[field] for field in crops]
        with torch.inference_mode():
            for group in groups:
                t=time.perf_counter()
                with scope("cpu_image_processor"):
                    if variant=="prepared_cpu":
                        pixels=self.prepared_cpu[key][group[0]]
                    elif variant=="prepared_gpu":
                        pixels=self.prepared_gpu[key][group[0]]
                    else:
                        pixels=self.processor(images=[crops[f]["crop_rgb"] for f in group], return_tensors="pt").pixel_values
                stages["processor_ms"]+=(time.perf_counter()-t)*1000
                t=time.perf_counter()
                with scope("host_to_device_completed"):
                    pixels=pixels.to(self.device)
                    torch.cuda.synchronize()
                stages["h2d_ms"]+=(time.perf_counter()-t)*1000
                t=time.perf_counter()
                with scope("autoregressive_generation"):
                    generated=self.model.generate(pixel_values=pixels, generation_config=generation)
                    torch.cuda.synchronize()
                stages["generation_ms"]+=(time.perf_counter()-t)*1000
                t=time.perf_counter()
                with scope("device_to_host_and_token_decode"):
                    ids=generated.cpu().tolist()
                    text=self.processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    for field, string, sequence in zip(group,text,ids):
                        raw[field]=string
                        # Exclude decoder start and post-EOS batch padding, retain EOS.
                        tail=sequence[1:]
                        tokens += tail.index(generation.eos_token_id)+1 if generation.eos_token_id in tail else len(tail)
                stages["decode_ms"]+=(time.perf_counter()-t)*1000
        t=time.perf_counter()
        with scope("normalize_and_validate"):
            normalized=self.contract["normalize_document_values"](raw,self.fields)
            validation=self.contract["validate_document_values"](normalized, required_fields=self.fields,
                validation_config=self.contract["alb_id_validation_config"])
        stages["postprocess_ms"]=(time.perf_counter()-t)*1000
        total=(time.perf_counter()-begin)*1000
        return dict(raw=raw, fields=normalized, validation=validation), dict(
            **stages, other_ms=max(0.,total-sum(stages.values())), total_ms=total,
            cpu_ms=(time.process_time()-cpu_begin)*1000, generated_tokens=tokens)

    def key(self, observation):
        # Geometry belongs in the key: same pixels with another crop contract is not the same request.
        geometries=[self.contract["find_field_region"](observation["template_record"],field)["shape_attributes"]
                    for field in self.contract["raw_to_canonical_field"]]
        digest=hashlib.sha256(observation["canonical_rgb"].tobytes())
        digest.update(json.dumps([geometries, observation["template_source_width"]],sort_keys=True).encode())
        return self.fingerprint+digest.hexdigest()

    def measure_variants(self, sampler):
        result,cold=self.request(self.observations[0])
        assert result["raw"]==self.expected[(self.observations[0]["identity_key"],self.observations[0]["source"])]
        self.cold=cold
        for variant in ["reference", "kv", "batch7_kv"]:
            sampler.phase="latency_"+variant
            for obs in self.observations[:3]: self.request(obs,variant)
            torch.cuda.reset_peak_memory_stats()
            before=cgroup_snapshot()
            start=time.perf_counter()
            for repetition in range(self.config["repetitions"]):
                order=np.random.default_rng(self.config["seed"]+repetition).permutation(30)
                for index in order:
                    obs=self.observations[int(index)]
                    result, timings=self.request(obs,variant)
                    same=result["raw"]==self.expected[(obs["identity_key"],obs["source"])]
                    if variant=="reference": assert same
                    self.rows.append(dict(profile=self.profile,variant=variant,repetition=repetition,
                        identity_key=obs["identity_key"],source=obs["source"],exact_prediction_replay=same,**timings))
                    if repetition==0:
                        truth={field:item["ground_truth"] for field,item in self.crops(obs).items()}
                        scored,_=self.contract["score_document"](truth,result["raw"],required_fields=self.fields)
                        self.predictions.extend(dict(variant=variant,identity_key=obs["identity_key"],source=obs["source"],**row) for row in scored)
            after=cgroup_snapshot()
            self.variant_stats.append(dict(variant=variant,wall_seconds=time.perf_counter()-start,
                cpu_seconds=(after["cpu"]["usage_usec"]-before["cpu"]["usage_usec"])/1e6,
                throttled_seconds=(after["cpu"]["throttled_usec"]-before["cpu"]["throttled_usec"])/1e6,
                throttled_periods=after["cpu"]["nr_throttled"]-before["cpu"]["nr_throttled"],
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
            pd.DataFrame(self.rows).to_csv(self.output/"requests.csv",index=False)
            print(self.profile,variant,"median ms",pd.DataFrame(self.rows).query("variant==@variant").total_ms.median(),flush=True)

    def measure_caches(self, sampler):
        from collections import OrderedDict
        sampler.phase="prepared_input_caches"
        for obs in self.observations[:3]:
            key=(obs["identity_key"],obs["source"])
            self.prepared_cpu[key]={f:self.processor(images=c["crop_rgb"],return_tensors="pt").pixel_values for f,c in self.crops(obs).items()}
            self.prepared_gpu[key]={f:t.to(self.device) for f,t in self.prepared_cpu[key].items()}
        torch.cuda.synchronize()
        rows=[]
        for variant in ["kv","prepared_cpu","prepared_gpu"]:
            for obs in self.observations[:3]: self.request(obs,variant)
            for repetition in range(5):
                for obs in self.observations[:3]:
                    result,timings=self.request(obs,variant)
                    assert result["raw"]==self.expected[(obs["identity_key"],obs["source"])]
                    rows.append(dict(variant=variant,repetition=repetition,identity_key=obs["identity_key"],source=obs["source"],**timings))
        cache=OrderedDict()
        sampler.phase="result_cache"
        cache_rows=[]
        for obs in self.observations[:3]:
            t=time.perf_counter()
            key=self.key(obs)
            result,_=self.request(obs,"kv")
            cache[key]=json.dumps(result,ensure_ascii=False)
            if len(cache)>self.config["result_cache_capacity"]: cache.popitem(last=False)
            cache_rows.append(dict(kind="miss",total_ms=(time.perf_counter()-t)*1000))
        for i in range(self.config["result_cache_hit_repetitions"]):
            obs=self.observations[i%3]
            t=time.perf_counter()
            key=self.key(obs)
            serialized=cache.pop(key)
            cache[key]=serialized
            result=json.loads(serialized)
            elapsed=(time.perf_counter()-t)*1000
            assert result["raw"]==self.expected[(obs["identity_key"],obs["source"])]
            cache_rows.append(dict(kind="hit",total_ms=elapsed))
        pd.DataFrame(rows).to_csv(self.output/"prepared_cache.csv",index=False)
        pd.DataFrame(cache_rows).to_csv(self.output/"result_cache.csv",index=False)
        self.cache_bytes=dict(cpu=sum(t.numel()*t.element_size() for v in self.prepared_cpu.values() for t in v.values()),
            gpu=sum(t.numel()*t.element_size() for v in self.prepared_gpu.values() for t in v.values()),
            results=sum(len(v.encode()) for v in cache.values()))
        self.prepared_cpu.clear(); self.prepared_gpu.clear()
        gc.collect()

    def measure_transfers(self, sampler):
        sampler.phase="transfer_microbenchmark"
        rows=[]
        for fields in [1,7]:
            cpu=torch.randn(fields,3,384,384)
            pinned=cpu.pin_memory()
            gpu=torch.empty_like(cpu,device=self.device)
            for mode in ["pageable_blocking","pageable_nonblocking","pre_pinned_nonblocking","pin_inside_request"]:
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                for iteration in range(self.config["transfer_repetitions"]+5):
                    torch.cuda.synchronize()
                    t=time.perf_counter()
                    source=cpu.pin_memory() if mode=="pin_inside_request" else pinned if mode=="pre_pinned_nonblocking" else cpu
                    begin.record()
                    gpu.copy_(source,non_blocking=mode!="pageable_blocking")
                    end.record(); end.synchronize()
                    elapsed=(time.perf_counter()-t)*1000
                    if iteration>=5:
                        rows.append(dict(fields=fields,mode=mode,bytes=cpu.numel()*cpu.element_size(),
                            completed_wall_ms=elapsed,cuda_stream_ms=begin.elapsed_time(end)))
                assert torch.equal(gpu.cpu(),cpu)
            del cpu,pinned,gpu
        pd.DataFrame(rows).to_csv(self.output/"transfers.csv",index=False)

    def load_case(self, name, variant, rate, sampler, burst=False):
        """Actual paced arrivals and a bounded single GPU worker, not a queue formula."""
        sampler.phase="queue_"+name
        pending=queue.Queue(maxsize=self.config["queue_pending_capacity"])
        ready=threading.Event()
        results=[]
        origin=time.perf_counter()
        error=[]
        def worker():
            ready.set()
            while True:
                item=pending.get()
                try:
                    if item is None: return
                    started=time.perf_counter()
                    result,_=self.request(self.observations[item["request_id"]%30],variant)
                    finished=time.perf_counter()
                    obs=self.observations[item["request_id"]%30]
                    assert result["raw"]==self.expected[(obs["identity_key"],obs["source"])]
                    results.append(dict(**item,status="completed",start_seconds=started-origin,
                        finish_seconds=finished-origin,queue_ms=(started-origin-item["arrival_seconds"])*1000,
                        service_ms=(finished-started)*1000,response_ms=(finished-origin-item["arrival_seconds"])*1000))
                except BaseException as exc:
                    error.append(repr(exc))
                finally: pending.task_done()
        thread=threading.Thread(target=worker,daemon=True)
        thread.start(); ready.wait()
        origin=time.perf_counter()
        count=12 if burst else math.ceil(rate*self.config["load_duration_seconds"])
        for request_id in range(count):
            target=origin if burst else origin+request_id/rate
            delay=target-time.perf_counter()
            if delay>0: time.sleep(delay)
            arrival=time.perf_counter()-origin
            item=dict(case=name,variant=variant,request_id=request_id,arrival_seconds=arrival,
                planned_seconds=0. if burst else request_id/rate,queue_depth_on_arrival=pending.qsize())
            try: pending.put_nowait(item)
            except queue.Full:
                results.append(dict(**item,status="rejected",start_seconds=None,finish_seconds=None,
                    queue_ms=None,service_ms=None,response_ms=None))
        pending.join(); pending.put(None); thread.join()
        assert not error,error
        assert len(results)==count
        for row in results: row["profile"]=self.profile
        return results

    def traces(self, sampler):
        sampler.phase="profiler_trace"
        available=torch.profiler.supported_activities()
        activities=[torch.profiler.ProfilerActivity.CPU]
        if torch.profiler.ProfilerActivity.CUDA in available: activities.append(torch.profiler.ProfilerActivity.CUDA)
        summary=[]
        for variant in ["reference","kv"]:
            try:
                crop=self.crops(self.observations[0])["nationality"]["crop_rgb"]
                config=copy.deepcopy(self.generation); config.use_cache=variant!="reference"
                with torch.profiler.profile(activities=activities,record_shapes=False,with_stack=False) as prof:
                    with torch.inference_mode(),torch.profiler.record_function("one_nationality_crop"):
                        with torch.profiler.record_function("image_processor"):
                            pixels=self.processor(images=crop,return_tensors="pt").pixel_values
                        with torch.profiler.record_function("h2d"):
                            pixels=pixels.to(self.device); torch.cuda.synchronize()
                        with torch.profiler.record_function("generation"):
                            self.model.generate(pixel_values=pixels,generation_config=config)
                            torch.cuda.synchronize()
                prof.export_chrome_trace(str(self.output/f"trace_{variant}.json"))
                events=prof.key_averages()
                table=[dict(operator=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total,
                            self_device_us=e.self_device_time_total) for e in events]
                pd.DataFrame(table).to_csv(self.output/f"operators_{variant}.csv",index=False)
                summary.append(dict(variant=variant,status="captured",activities=[str(a) for a in activities],
                    summed_self_device_ms=sum(e.self_device_time_total for e in events)/1000))
            except Exception as exc:
                summary.append(dict(variant=variant,status="skipped",reason=repr(exc)))
        return summary

    def run(self, sampler):
        self.measure_variants(sampler)
        if self.profile=="roomy":
            self.measure_caches(sampler)
            self.measure_transfers(sampler)
        queue_rows=[]
        for name,variant,rate,burst in [
            ("quiet","reference",self.config["quiet_arrivals_per_second"],False),
            ("surge","reference",self.config["surge_arrivals_per_second"],False),
            ("surge_batch7","batch7_kv",self.config["surge_arrivals_per_second"],False),
            ("burst12","reference",0,True)]:
            queue_rows.extend(self.load_case(name,variant,rate,sampler,burst))
            pd.DataFrame(queue_rows).to_csv(self.output/"queue.csv",index=False)
        trace=self.traces(sampler) if self.profile=="roomy" else []
        pd.DataFrame(self.predictions).to_json(self.output/"predictions.jsonl",orient="records",lines=True,force_ascii=False)
        write_json(self.output/"variant_resources.json",self.variant_stats)
        metadata=dict(profile=self.profile,purpose=self.config["purpose"],config=self.config,
            config_sha256=sha(CONFIG),definition_provenance=self.provenance,
            thread_control_amendment=self.amendment,
            worker_source_sha256=sha(__file__),
            environment=dict(python=platform.python_version(),torch=torch.__version__,transformers=importlib.metadata.version("transformers"),
                gpu=torch.cuda.get_device_name(0),visible_gpu_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
                torch_threads=torch.get_num_threads(),cpu=platform.processor()),
            initial_cgroup=self.initial_cgroup,final_cgroup=cgroup_snapshot(),
            startup=dict(processor_load_ms=self.processor_load_ms,model_load_cpu_ms=self.model_load_cpu_ms,
                weights_to_gpu_ms=self.weights_to_gpu_ms,first_request=self.cold),
            cache_storage_bytes=getattr(self,"cache_bytes",None),profiler=trace,test_evaluated=False,phase3_closed=False)
        write_json(self.output/"metadata.json",metadata)


def worker(output,profile):
    with Sampler() as sampler:
        lab=ResourceLab(output,profile)
        lab.run(sampler)
    pd.DataFrame(sampler.rows).to_csv(Path(output)/"resource_samples.csv",index=False)
    print("Completed",profile,output,flush=True)


def run_experiment():
    config=json.loads(CONFIG.read_text())
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory=ROOT/"outputs/ocr_resource_profile"/run_id
    directory.mkdir(parents=True,exist_ok=False)
    write_json(directory/"config.json",config)
    for profile,quota in config["cpu_profiles"].items():
        command=["systemd-run","--user","--wait","--pipe","--collect",
            f"--unit=kyc-ocr-curiosity-{run_id.lower()}-{profile}",f"--working-directory={ROOT}",
            "-p",f"CPUQuota={quota}","-p",f"MemoryMax={config['memory_max']}",
            "-p",f"MemorySwapMax={config['memory_swap_max']}","-p","RuntimeMaxSec=600",
            "--setenv=TOKENIZERS_PARALLELISM=false","--setenv=OMP_NUM_THREADS=4",
            "--setenv=OPENBLAS_NUM_THREADS=4","--setenv=MKL_NUM_THREADS=4",
            str(ROOT/".venv/bin/python"),str(Path(__file__).resolve()),
            "--worker",str(directory/profile),"--profile",profile]
        print("Running",profile,"CPU quota",quota,"RAM",config["memory_max"],flush=True)
        completed=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=660)
        (directory/f"{profile}.log").write_text(completed.stdout+"\n"+completed.stderr)
        print(completed.stdout,flush=True)
        if completed.returncode:
            raise RuntimeError(f"{profile} failed ({completed.returncode}): {completed.stderr[-4000:]}")
    return directory


def run_thread_control(directory):
    directory=Path(directory).resolve()
    assert directory.parent==ROOT/"outputs/ocr_resource_profile"
    assert all((directory/p/"metadata.json").exists() for p in ["roomy","small"])
    amendment=json.loads((ROOT/"configs/ocr_resource_profile_thread_control_v1.json").read_text())
    write_json(directory/"thread_control_amendment.json",amendment)
    command=["systemd-run","--user","--wait","--pipe","--collect",
        f"--unit=kyc-ocr-thread-control-{directory.name.lower()}",f"--working-directory={ROOT}",
        "-p","CPUQuota=100%","-p","MemoryMax=4G","-p","MemorySwapMax=0","-p","RuntimeMaxSec=600",
        "--setenv=TOKENIZERS_PARALLELISM=false","--setenv=OMP_NUM_THREADS=1",
        "--setenv=OPENBLAS_NUM_THREADS=1","--setenv=MKL_NUM_THREADS=1",
        str(ROOT/".venv/bin/python"),str(Path(__file__).resolve()),
        "--worker",str(directory/"small_tuned"),"--profile","small_tuned"]
    result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=660)
    (directory/"small_tuned.log").write_text(result.stdout+"\n"+result.stderr)
    print(result.stdout,flush=True)
    assert result.returncode==0,result.stderr[-4000:]
    return directory


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--worker",type=Path,required=True)
    parser.add_argument("--profile",choices=["roomy","small","small_tuned"],required=True)
    arguments=parser.parse_args()
    worker(arguments.worker,arguments.profile)
