#!/usr/bin/env python3
"""Score a stored plan's trajectory quality and write its quality log.

``save_plan`` writes the log for every NEW plan. This exists for the plans already
on disk — the ones whose inefficiency was found by hand, and which are the only
baseline available for judging whether a change helped.

    # one plan, or the newest if --tree-id is omitted
    python scripts/trajectory_quality_report.py --cell welding_cell_2_small
    python scripts/trajectory_quality_report.py --cell welding_cell_2_small --tree-id bt_1788022470_welding_cell_2_small

    # every stored plan
    python scripts/trajectory_quality_report.py --cell welding_cell_2_small --all

Prints the headline numbers and the findings; the full report goes to
``<cell>/agentic_planner/logs/<tree_id>.json``. ``--dry-run`` prints without
writing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.trajectory.tools import plan_store, trajectory_quality  # noqa: E402
from utils import cell_layout  # noqa: E402


def _report(cell_id: str, tree_id: str, cell_base, dry_run: bool) -> int:
    path = cell_layout.plan_file(cell_id, tree_id, cell_base)
    if not path.is_file():
        print(f"no such plan: {path}")
        return 1
    plan = json.load(open(path))
    if (plan.get("status") or plan_store.STATUS_OK) == plan_store.STATUS_FAILED:
        # A failed run's partial trajectories are worth reading but scoring them as
        # a plan would compare a fragment against a whole program.
        print(f"{tree_id}: skipped — plan is marked failed")
        return 0

    robots = trajectory_quality._load_robots(cell_id, cell_base)
    report = trajectory_quality.score_plan(plan, robots=robots)
    s = report["summary"]

    print(f"\n{tree_id}")
    print(f"  score      {report['efficiency_score']}  ({report['verdict']})"
          f"   components {report['score_components']}")
    print(f"  makespan   {s['makespan_s']} s over {s['motion_legs']} legs")
    print(f"  parallel   {s['parallel_efficiency']}"
          f"   serial {s['serial_fraction']}   balance {s['load_balance']}")
    print(f"  travel     {s['joint_travel_rad']} rad for "
          f"{s['direct_distance_rad']} rad of moves"
          f"  — {s['total_excess_travel_rad']} rad wasted")
    print(f"  worst leg  {s['worst_excess_travel_rad']} rad excess"
          f"   ratios: worst {s['worst_detour_ratio']}"
          f" mean {s['mean_detour_ratio']}")
    print(f"  dwell      {s['total_interior_dwell_s']} s interior")
    sched = report.get("schedule") or {}
    if sched.get("available"):
        print(f"  per arm    busy {sched['busy_s']}  idle {sched['idle_s']}"
              f"  util {sched['utilisation']}")
    if report["findings"]:
        print("  findings:")
        for f in report["findings"]:
            where = f.get("node") or f.get("robot_ns") or "plan"
            print(f"    [{f['severity']:<6}] {f['kind']:<22} {where}: {f['detail']}")
    else:
        print("  findings:  none")

    if dry_run:
        print("  (dry run — nothing written)")
        return 0
    out = trajectory_quality.write_quality_log(cell_id, plan, cell_base,
                                               robots=robots)
    print(f"  written    {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True, help="cell id, e.g. welding_cell_2_small")
    ap.add_argument("--tree-id", help="plan to score (default: the newest)")
    ap.add_argument("--all", action="store_true", help="score every stored plan")
    ap.add_argument("--cell-base", help="override the cells root")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the report without writing the log")
    args = ap.parse_args()

    if args.all:
        ids = [p["tree_id"] for p in plan_store.list_plans(args.cell)
               if p.get("status") != plan_store.STATUS_FAILED]
    elif args.tree_id:
        ids = [args.tree_id]
    else:
        plans = [p for p in plan_store.list_plans(args.cell)
                 if p.get("status") != plan_store.STATUS_FAILED]
        if not plans:
            print(f"no usable plans stored for {args.cell}")
            return 1
        ids = [plans[0]["tree_id"]]

    rc = 0
    for tree_id in ids:
        rc |= _report(args.cell, tree_id, args.cell_base, args.dry_run)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
