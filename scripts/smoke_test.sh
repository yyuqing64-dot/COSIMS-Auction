#!/usr/bin/env bash
set -euo pipefail

python - <<'PY2'
from sim.network import Network
from sim.env import Env
from policies.rr import RoundRobin
from policies.greedy_latency import GreedyLatency
from policies.sqlf import ShortestQueueFirst
from policies.cap_greedy import CapAwareGreedy
from policies.cosims_sinkhorn import COSIMSSinkhorn
from policies.cosims_auction_A import COSIMSAuctionA
from policies.cosims_auction_B import COSIMSAuctionB
print('Import smoke test passed.')
PY2
