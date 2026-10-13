"""Read the systems-lab artifacts and draw compact, reproducible figures."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ["#3973ac", "#23a094", "#e39d32", "#9265ad", "#d76262", "#9ba8b6"]


def load_run(directory):
    directory=Path(directory)
    profiles=[p for p in ["roomy","small","small_tuned"] if (directory/p/"metadata.json").exists()]
    metadata={p:json.loads((directory/p/"metadata.json").read_text()) for p in profiles}
    data={name:pd.concat([pd.read_csv(directory/p/f"{name}.csv").assign(profile=p)
                         for p in metadata],ignore_index=True)
          for name in ["requests","queue","resource_samples"]}
    requests=data["requests"]
    assert len(requests)==180*len(profiles)
    assert requests.groupby(["profile","variant","identity_key","source"]).size().eq(2).all()
    assert requests.query("variant=='reference'").exact_prediction_replay.all()
    assert (requests.total_ms>0).all() and (requests.h2d_ms>=0).all()
    latency=requests.groupby(["profile","variant"],as_index=False).agg(
        requests=("total_ms","size"),median_ms=("total_ms","median"),
        p95_ms=("total_ms",lambda s:s.quantile(.95)),cpu_ms=("cpu_ms","mean"),
        prediction_agreement=("exact_prediction_replay","mean"))
    queue_rows=data["queue"]
    queue_summary=[]
    for (profile,case),group in queue_rows.groupby(["profile","case"],sort=False):
        done=group.query("status=='completed'")
        queue_summary.append(dict(profile=profile,case=case,offered=len(group),completed=len(done),
            rejected=int((group.status=="rejected").sum()),service_median_ms=done.service_ms.median(),
            response_p50_ms=done.response_ms.median(),response_p95_ms=done.response_ms.quantile(.95),
            queue_p95_ms=done.queue_ms.quantile(.95),
            max_arrival_lateness_ms=1000*(group.arrival_seconds-group.planned_seconds).max()))
    prepared=pd.read_csv(directory/"roomy/prepared_cache.csv")
    result_cache=pd.read_csv(directory/"roomy/result_cache.csv")
    transfers=pd.read_csv(directory/"roomy/transfers.csv")
    assert len(transfers)==480
    data.update(metadata=metadata,latency=latency,queue_summary=pd.DataFrame(queue_summary),
        prepared=prepared,result_cache=result_cache,transfers=transfers)
    return data


def style_axes(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x",color="#e6eaf0",linewidth=.7)
    ax.set_axisbelow(True)


def dashboard(data,destination):
    plt.rcParams.update({"font.size":10,"axes.titlesize":12,"axes.titleweight":"bold",
                         "figure.facecolor":"#f6f8fb","axes.facecolor":"#ffffff"})
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout="constrained")
    fig.suptitle("OCR, under the microscope",fontsize=22,fontweight="bold",color="#18334b")
    ax=axes[0,0]
    variants=["reference","kv","batch7_kv"]
    labels=["Official settings","Decoder KV cache","7 fields together + KV"]
    profiles=list(data["metadata"])
    for i,profile in enumerate(profiles):
        rows=data["latency"].query("profile==@profile").set_index("variant").loc[variants]
        bars=ax.barh(np.arange(3)+(i-(len(profiles)-1)/2)*.24,rows.median_ms,height=.22,
            color=COLORS[i],label={"roomy":"4 CPU / 4 threads","small":"1 CPU / 4 threads","small_tuned":"1 CPU / 1 thread"}[profile])
        ax.bar_label(bars,fmt="%.1f",padding=3,fontsize=9)
    ax.set(yticks=np.arange(3),yticklabels=labels,xlabel="Completed document latency (ms)",title="1  |  Less work, not just more CPUs")
    ax.invert_yaxis(); ax.legend(loc="lower right",fontsize=8); style_axes(ax)
    ax=axes[0,1]
    stages=["crop_ms","processor_ms","h2d_ms","generation_ms","decode_ms","postprocess_ms","other_ms"]
    names=["Crop","CPU pixels","H2D + sync","Generation","Decode + D2H","Validate","Other"]
    means=data["requests"].query("profile=='roomy'").groupby("variant")[stages].mean().loc[variants]
    left=np.zeros(3)
    for i,(stage,name) in enumerate(zip(stages,names)):
        ax.barh(np.arange(3),means[stage],left=left,color=(COLORS+["#657584"])[i],label=name)
        left+=means[stage].to_numpy()
    ax.set(yticks=np.arange(3),yticklabels=["Reference","KV","Batch 7 + KV"],xlabel="Mean synchronized stage time (ms)",
           title="2  |  Where a request spends its time")
    ax.invert_yaxis(); ax.legend(fontsize=8,loc="lower right",ncol=2); style_axes(ax)
    ax=axes[1,0]
    prepared=data["prepared"].groupby("variant").total_ms.median()
    hits=data["result_cache"].query("kind=='hit'").total_ms.median()
    values=[prepared["kv"],prepared["prepared_cpu"],prepared["prepared_gpu"],hits]
    bars=ax.barh(range(4),values,color=[COLORS[0],COLORS[1],COLORS[2],COLORS[4]])
    ax.set_xscale("log")
    ax.set(yticks=range(4),yticklabels=["KV: normal request","Prepared CPU tensors","Prepared GPU tensors","Exact result-cache HIT"],
           xlabel="Median ms, logarithmic scale",title="3  |  Four very different shortcuts")
    for bar,value in zip(bars,values):ax.text(value*1.10,bar.get_y()+bar.get_height()/2,f"{value:.3f} ms",va="center",fontsize=9)
    ax.set_xlim(min(values)*.5,max(values)*4); ax.invert_yaxis(); style_axes(ax)
    ax=axes[1,1]
    transfer=data["transfers"].groupby(["fields","mode"]).completed_wall_ms.median()
    modes=["pageable_blocking","pageable_nonblocking","pre_pinned_nonblocking","pin_inside_request"]
    for i,fields in enumerate([1,7]):
        vals=[transfer.loc[(fields,m)] for m in modes]
        bars=ax.barh(np.arange(4)+(.18 if i else -.18),vals,height=.32,color=COLORS[i],label=f"{fields} field(s)")
        ax.bar_label(bars,fmt="%.3f",padding=3,fontsize=8)
    ax.set(yticks=range(4),yticklabels=["Pageable / blocking","Pageable / non-blocking","Already pinned / non-blocking","Pin on request path"],
           xlabel="Completed H2D wall time (ms)",title="4  |  Faster transfer is not free")
    ax.invert_yaxis(); ax.legend(); style_axes(ax)
    fig.savefig(destination,dpi=140,bbox_inches="tight")
    plt.close(fig)


def load_figure(data,destination):
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout="constrained")
    fig.suptitle("When 8 requests/second meet one GPU worker",fontsize=20,fontweight="bold",color="#18334b")
    roomy_surges=data["queue"].query("profile=='roomy' and case in ['surge','surge_batch7']")
    common_end=max(roomy_surges.finish_seconds.max(),roomy_surges.arrival_seconds.max())+.15
    for ax,case,title in zip(axes[0],["surge","surge_batch7"],["Reference: wait or reject","Batch 7 + KV: same offered load"]):
        rows=data["queue"].query("profile=='roomy' and case==@case").sort_values("request_id")
        for row in rows.itertuples():
            if row.status=="rejected": ax.scatter(row.arrival_seconds,row.request_id,color=COLORS[4],marker="x",s=35)
            else:
                ax.barh(row.request_id,row.queue_ms/1000,left=row.arrival_seconds,height=.65,color="#bcc7d3")
                ax.barh(row.request_id,row.service_ms/1000,left=row.start_seconds,height=.65,color=COLORS[0] if case=="surge" else COLORS[1])
        ax.set(title=title,xlabel="Seconds since first arrival",ylabel="Request",ylim=(-1,41))
        ax.set_xlim(0,common_end)
        ax.invert_yaxis(); style_axes(ax)
        ax.text(.02,.02,"Gray = queue wait    Color = service    × = rejected",transform=ax.transAxes,fontsize=8)
    ax=axes[1,0]
    for profile,color in zip(data["metadata"],COLORS):
        rows=data["resource_samples"].query("profile==@profile and phase=='queue_surge'")
        ax.plot(rows.seconds-rows.seconds.min(),rows.cpu_cores,color=color,label={"roomy":"4 CPU / 4 threads","small":"1 CPU / 4 threads","small_tuned":"1 CPU / 1 thread"}[profile],linewidth=1.2)
    ax.axhline(1,color="#8994a0",linestyle="--",linewidth=1)
    ax.set(title="CPU work during reference overload",xlabel="Seconds in overload phase",ylabel="CPU cores busy (sampled)")
    ax.legend(); style_axes(ax)
    ax=axes[1,1]
    for profile,color in zip(data["metadata"],COLORS):
        rows=data["resource_samples"].query("profile==@profile and phase.str.startswith('latency_')")
        ax.plot(rows.seconds-rows.seconds.min(),rows.rss_mib,color=color,label=f"{profile}: process RAM")
        ax.plot(rows.seconds-rows.seconds.min(),rows.gpu_allocated_mib,color=color,linestyle="--",label=f"{profile}: torch GPU")
    ax.set(title="Resident cost during warm requests",xlabel="Seconds in latency experiments",ylabel="MiB")
    ax.legend(fontsize=7,ncol=2); style_axes(ax)
    fig.savefig(destination,dpi=140,bbox_inches="tight")
    plt.close(fig)


def tables(data):
    transfer=data["transfers"].groupby(["fields","mode"],as_index=False).agg(
        completed_wall_ms=("completed_wall_ms","median"),cuda_stream_ms=("cuda_stream_ms","median"))
    memory=[]
    for profile,meta in data["metadata"].items():
        samples=data["resource_samples"].query("profile==@profile")
        memory.append(dict(profile=profile,peak_sampled_rss_mib=samples.rss_mib.max(),
            peak_sampled_torch_gpu_mib=samples.gpu_allocated_mib.max(),
            cgroup_peak_mib=int(meta["final_cgroup"]["memory_peak"])/2**20,
            throttle_seconds=meta["final_cgroup"]["cpu"]["throttled_usec"]/1e6))
    return transfer,pd.DataFrame(memory)


def trace_figure(directory,destination):
    """Trace durations are diagnostic, not replacements for unprofiled timings."""
    fig,axes=plt.subplots(2,1,figsize=(15,6),layout="constrained")
    summaries=[]
    for ax,variant in zip(axes,["reference","kv"]):
        events=json.loads((Path(directory)/"roomy"/f"trace_{variant}.json").read_text())["traceEvents"]
        enclosing=next(e for e in events if e.get("name")=="one_nationality_crop" and e.get("cat")=="user_annotation")
        origin=enclosing["ts"]
        annotations=[e for e in events if e.get("cat")=="user_annotation" and e["name"]!="one_nationality_crop"]
        for e in annotations:
            color=COLORS[{"image_processor":2,"h2d":1,"generation":0}[e["name"]]]
            ax.broken_barh([((e["ts"]-origin)/1000,e["dur"]/1000)],(2.1,.55),facecolors=color)
        for cat,y,color in [("kernel",1.1,COLORS[0]),("gpu_memcpy",.1,COLORS[1])]:
            intervals=[((e["ts"]-origin)/1000,e["dur"]/1000) for e in events if e.get("cat")==cat and e.get("ph")=="X"]
            ax.broken_barh(intervals,(y,.55),facecolors=color)
        intervals=sorted((e["ts"],e["ts"]+e["dur"]) for e in events if e.get("cat")=="kernel" and e.get("ph")=="X")
        union=0.; end=-float("inf")
        for start,finish in intervals:
            union+=max(0.,finish-max(start,end)); end=max(end,finish)
        summaries.append(dict(variant=variant,traced_request_ms=enclosing["dur"]/1000,
            gpu_kernel_union_ms=union/1000,gpu_kernel_count=len(intervals),
            gpu_copy_count=sum(e.get("cat")=="gpu_memcpy" for e in events)))
        ax.set(title=f"{variant}: one nationality crop (instrumented)",xlabel="Milliseconds from trace start",
            yticks=[.375,1.375,2.375],yticklabels=["GPU copies","GPU kernels","CPU scopes"],ylim=(-.1,2.9))
        style_axes(ax)
    fig.suptitle("The GPU works in bursts, not one continuous block",fontsize=18,fontweight="bold")
    common_end=max(row["traced_request_ms"] for row in summaries)*1.03
    for ax in axes: ax.set_xlim(0,common_end)
    fig.savefig(destination,dpi=140,bbox_inches="tight"); plt.close(fig)
    return pd.DataFrame(summaries)
