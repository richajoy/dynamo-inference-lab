#!/usr/bin/env python3
"""Router/queue metrics for one run window from Prometheus. Runs ON the node (stdlib only):
  python3 bench/prom-window.py <t0_ms> <t1_ms>          (window_ms from a run's summary.json)
Prints max / avg over the window of the router queue and per-worker load gauges, plus router decision counters:
  dynamo_frontend_router_queue_pending_requests   requests held in the router queue (only with --router-queue-threshold)
  dynamo_frontend_router_queue_pending_isl_tokens prompt tokens waiting in that queue
  dynamo_frontend_worker_active_prefill_tokens    router's view of in-flight prefill per worker (kv modes)
  dynamo_frontend_worker_active_decode_blocks     router's view of resident decode blocks per worker
  dynamo_component_router_non_max_overlap_selections_total  times the router picked a worker that did NOT have
                                                  the best prefix overlap (load won over locality)
  dynamo_router_overhead_total_ms                 routing decision cost
"""
import json, subprocess, sys, urllib.parse, urllib.request

t0, t1 = int(sys.argv[1]) // 1000, int(sys.argv[2]) // 1000
ip = subprocess.check_output(["kubectl", "-n", "monitoring", "get", "svc", "prometheus-kube-prometheus-prometheus",
                              "-o", "jsonpath={.spec.clusterIP}"], text=True)
rng = f"{max(t1 - t0, 60)}s"
Q = {
    "queue_pending_requests max": f"max(max_over_time(dynamo_frontend_router_queue_pending_requests[{rng}]))",
    "queue_pending_requests avg": f"max(avg_over_time(dynamo_frontend_router_queue_pending_requests[{rng}]))",
    "queue_pending_isl_tokens max": f"max(max_over_time(dynamo_frontend_router_queue_pending_isl_tokens[{rng}]))",
    "active_prefill_tokens max per worker": f"max(max_over_time(dynamo_frontend_worker_active_prefill_tokens[{rng}]))",
    "active_decode_blocks max per worker": f"max(max_over_time(dynamo_frontend_worker_active_decode_blocks[{rng}]))",
    "non_max_overlap_selections": f"sum(increase(dynamo_component_router_non_max_overlap_selections_total[{rng}]))",
    "router_requests": f"sum(increase(dynamo_component_router_requests_total[{rng}]))",
    "routing_overhead_ms avg": f"sum(increase(dynamo_router_overhead_total_ms_sum[{rng}])) / clamp_min(sum(increase(dynamo_router_overhead_total_ms_count[{rng}])),1)",
}
out = {}
for k, q in Q.items():
    url = f"http://{ip}:9090/api/v1/query?" + urllib.parse.urlencode({"query": q, "time": t1 + 15})
    r = json.load(urllib.request.urlopen(url, timeout=10))["data"]["result"]
    out[k] = round(float(r[0]["value"][1]), 3) if r else None
print(json.dumps(out, indent=2))
