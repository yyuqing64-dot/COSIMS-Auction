"""Observation and paired-evaluation utilities for threshold calibration."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,os,platform,sys,time,gc
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from policies.cosims_auction_B import COSIMSAuctionB
from sim.env import Env
from eval.metrics import compute_metrics

LOAD=1.2
Q=0.95


def plain(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,dict): return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    return x

def dump(path,value):
    p=Path(path); t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(plain(value),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    t.replace(p)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(json.dumps(plain(x),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def task_digest(tasks,inputs=False):
    if inputs:
        a=np.array([[t.tid,t.src,t.arrival_t,t.cpu_demand,t.data_mb] for t in tasks],dtype='<f8')
    else:
        a=np.array([[t.tid,t.src,t.arrival_t,t.cpu_demand,t.data_mb,-1 if t.dest is None else t.dest,
                    int(t.is_cloud),t.start_t,t.finish_t,t.backlog_time_total,t.backlog_work_total] for t in tasks],dtype='<f8')
    return hashlib.sha256(a.tobytes()).hexdigest()

def thresholds(samples):
    a=np.asarray(samples,float)
    assert a.ndim==2 and a.shape[1]==4 and len(a)>0 and np.isfinite(a).all() and (a>=0).all()
    return {'tau_W':float(np.quantile(a[:,0],Q,method='linear')),
            'tau_S':float(np.quantile(a[:,1],Q,method='linear')),
            'service_reference':float(np.median(a[:,2])),
            'interval_reference':float(np.median(a[:,3])),
            'n_calibration_observations':len(a),
            'empirical_wait_exceed_pct':float(np.mean(a[:,0]>np.quantile(a[:,0],Q))*100),
            'empirical_growth_exceed_pct':float(np.mean(a[:,1]>np.quantile(a[:,1],Q))*100)}

def self_tests():
    t=thresholds([[i,max(0,i-2),2.,.2] for i in range(1,11)])
    assert np.isclose(t['tau_W'],9.55) and np.isclose(t['tau_S'],7.55)
    assert thresholds([[1.,0.,1.,.2],[1.,0.,1.,.2]])['tau_S']==0
    return {'known_quantile':True,'zero_growth_handling':True}

class CalibrationObserver(COSIMSAuctionB):
    def __init__(self,cfg,rng,directory):
        super().__init__(cfg,rng)
        self.observations=[]
        self.warmup=cfg['warmup_time']
        self.f=gzip.open(directory/'calibration_signals.csv.gz','wt',encoding='utf-8',newline='',compresslevel=1)
        self.writer=csv.writer(self.f)
        self.writer.writerow(['task_id','source','arrival_t','ot_preferred_edge','waiting_signal',
                             'max_positive_growth_signal','reference_service_time','effective_interval','included'])

    def select_dest(self,task,env):
        prev=np.zeros(env.n) if not self.initialized else self.prev_backlog.copy()
        last=0.0 if not self.initialized else self.last_backlog_t
        destination=super().select_dest(task,env)
        # Observation is outside all native scheduling timers and before assignment.
        now=float(task.arrival_t)
        backlog=np.maximum(env.available_time-now,0)
        dt=max(now-last,1e-9)
        dt=float(np.clip(dt,self.slope_dt_min if self.slope_dt_min>0 else dt,
                            self.slope_dt_max if self.slope_dt_max>0 else dt))
        growth=float(max(0,np.max((backlog-prev)/dt)))
        top1=int(np.argmax(self.P[task.src]))
        wait=float(backlog[top1]); service=float(task.cpu_demand/env.xi[top1])
        included=now>=self.warmup
        if included: self.observations.append([wait,growth,service,dt])
        self.writer.writerow([task.tid,task.src,now,top1,wait,growth,service,dt,int(included)])
        return destination

class RateObserver(COSIMSAuctionB):
    def __init__(self,cfg,rng):
        super().__init__(cfg,rng)
        self.warmup=cfg['warmup_time']
        self.count=0; self.triggered=0; self.dec_seconds=0.
        self.wait_exceeded=0; self.growth_exceeded=0
        self.trigger_records=[]

    def select_dest(self,task,env):
        prev=np.zeros(env.n) if not self.initialized else self.prev_backlog.copy()
        last=0.0 if not self.initialized else self.last_backlog_t
        before=self._trig_fired_calls; timer=self._decision_time_sum
        destination=super().select_dest(task,env)
        decision=self._decision_time_sum-timer
        now=float(task.arrival_t)
        # All observations remain outside the original decision timer.
        if now>=self.warmup:
            B=np.maximum(env.available_time-now,0.)
            dt=max(now-last,1e-9)
            dt=float(np.clip(dt,self.slope_dt_min if self.slope_dt_min>0 else dt,
                                self.slope_dt_max if self.slope_dt_max>0 else dt))
            G=float(max(0,np.max((B-prev)/dt)))
            top1=int(np.argmax(self.P[task.src])); W=float(B[top1])
            fired=int(self._trig_fired_calls!=before)
            self.count+=1; self.triggered+=fired; self.dec_seconds+=decision
            self.wait_exceeded+=int(W>self.wait_tau); self.growth_exceeded+=int(G>self.slope_th)
            self.trigger_records.append([task.tid,W,G,fired,int(W>self.wait_tau),int(G>self.slope_th)])
        return destination

def execute(cfg,seed,kind='plain',directory=None):
    rng=np.random.default_rng(seed)
    if kind=='calibration': policy=CalibrationObserver(cfg,rng,directory)
    elif kind=='rates': policy=RateObserver(cfg,rng)
    else: policy=COSIMSAuctionB(cfg,rng)
    env=Env(cfg,rng,policy); tasks=env.run(LOAD)
    return tasks,env,policy

def state(tasks,env):
    return {'input_sha256':task_digest(tasks,True),'trajectory_sha256':task_digest(tasks),
            'rng_sha256':digest(env.rng.bit_generator.state),
            'queue_sha256':digest([env.available_time,env.cloud_available_time])}

def save_tasks(directory,tasks,rates):
    lookup={int(a[0]):a[1:] for a in rates}
    with gzip.open(directory/'tasks_and_triggers.csv.gz','wt',encoding='utf-8',newline='',compresslevel=1) as f:
        w=csv.writer(f)
        w.writerow(['task_id','source','arrival_t','cpu_demand','data_mb','destination','start_t','finish_t',
                    'waiting_signal','growth_signal','triggered','wait_exceeded','growth_exceeded'])
        for t in tasks:
            w.writerow([t.tid,t.src,t.arrival_t,t.cpu_demand,t.data_mb,'cloud' if t.dest is None else f'edge:{t.dest}',
                        t.start_t,t.finish_t,*lookup.get(t.tid,['','','','',''])])

def run_evaluation(cfg,seed,variant,directory):
    directory.mkdir(parents=True,exist_ok=True)
    # Plain run provides uncontaminated timing, followed by read-only rate recording.
    tasks,env,policy=execute(cfg,seed)
    base_state=state(tasks,env)
    native_costs=policy.get_cost_stats()
    metrics=compute_metrics(tasks,cfg['warmup_time'],cfg['sim_time'])
    del tasks,env,policy; gc.collect()
    tasks,env,policy=execute(cfg,seed,'rates')
    observed_state=state(tasks,env)
    validation={k:base_state[k]==observed_state[k] for k in base_state}
    assert all(validation.values()),validation
    row={'seed':seed,'variant':variant,'load_factor':LOAD,'tau_W':cfg['cosims']['event']['wait_tau'],
         'tau_S':cfg['cosims']['event']['slope_th'],'evaluation_tasks':policy.count,**metrics,
         'trigger_rate_pct':100*policy.triggered/policy.count,
         'waiting_activation_pct':100*policy.wait_exceeded/policy.count,
         'growth_activation_pct':100*policy.growth_exceeded/policy.count,
         'native_full_run_decision_ms':native_costs['decision_time_ms_mean'],
         'observed_post_warmup_decision_ms':1000*policy.dec_seconds/policy.count}
    save_tasks(directory,tasks,policy.trigger_records)
    dump(directory/'metrics.json',row)
    dump(directory/'state.json',base_state)
    dump(directory/'read_only_validation.json',{'passed':True,'equalities':validation})
    return row,base_state
