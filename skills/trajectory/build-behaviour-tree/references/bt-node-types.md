# BT node types — BehaviorTree.CPP v4

The tree format is **BehaviorTree.CPP v4** ([behaviortree.dev](https://www.behaviortree.dev/docs/intro),
[BehaviorTree/BehaviorTree.CPP](https://github.com/BehaviorTree/BehaviorTree.CPP)), so a
plan can be opened in Groot2, ticked by a C++ `BehaviorTreeFactory`, or diffed as
XML. It is not a repo-local dialect.

Two representations carry the same tree:

- **XML** — the interchange artifact. `<root BTCPP_format="4"
  main_tree_to_execute="MainTree">` containing one `<BehaviorTree ID="MainTree">`
  with exactly one root child, plus a `<TreeNodesModel>` describing the custom
  nodes' ports (Groot2 needs it to draw them; a C++ factory ignores it).
- **JSON mirror** — what travels over MCP/DDS and what the Python executor walks.
  Each node is `{"type": <registration ID>, "name": <instance label>,
  "ports": {...}, "children": [...]}`. `type` is the XML element name exactly as
  BT.CPP registers it, `name` is the optional instance label, and `ports` are the
  XML attributes.

`agents/trajectory/tools/btcpp.py` is the whole format layer: `to_btcpp` (convert
+ canonicalise), `to_xml` / `from_xml` (serialise / parse), and the dialect-agnostic
accessors `node_kind` / `port` / `children` / `resolve` / `trajectory_of` that every
tree walker in the repo goes through.

## The blackboard: why waypoints are not attributes

A welding trajectory is hundreds of waypoints. Inlining it as an XML attribute
would make the file unreadable, defeating the point of adopting the format. BT.CPP's
own answer is the blackboard: a port whose value is the literal string `{key}` reads
blackboard entry `key` at tick time.

So bulky or structured port values — `trajectory`, `interlock_target`,
`depends_on_robots`, `depends_on_joints`, `depends_on_positions` — live in a side-car
`blackboard` dict on the tree, and the XML carries only the `{key}` reference. Keys
are readable and stable, e.g. `traj_robot1_W1`, `interlock_robot2_W2`.

**The blackboard must be loaded before the tree is ticked.** `assemble_behaviour_tree`
returns it in the tree's `blackboard` field; `export_behaviour_tree_xml` and the
`get_plan_xml` MCP tool return it alongside the XML.

## Built-in control nodes

| Node | Ports | Meaning here |
|---|---|---|
| `Sequence` | — | Children in order; stops on the first failure. |
| `SequenceWithMemory` | — | As `Sequence`, but does not re-tick children that already succeeded. |
| `Parallel` | `success_count` (default `-1`), `failure_count` (default `1`) | Ticks all children concurrently. `success_count="-1"` means *all children must succeed* — that is what the assembler emits for a parallel step. |
| `Delay` | `delay_msec` | Waits, then ticks its single child. Replaces the old per-node `start_delay_s` field. |

Other built-ins BT.CPP provides and `btcpp.py` recognises structurally —
`Fallback`, `ReactiveSequence`, `ReactiveFallback`, `Timeout` (`msec`),
`RetryUntilSuccessful` (`num_attempts`), `Inverter`, `ForceSuccess`,
`ForceFailure` — are valid in the format, but the Python executor implements only
the four above and the four leaves below. An unimplemented node fails loudly with
"unregistered node" rather than being silently skipped.

## Custom leaves

### `ExecuteTrajectory` (Action)

Dispatches one precomputed joint trajectory to one robot and waits for completion.

| Port | Default | Meaning |
|---|---|---|
| `robot_ns` | *required* | Robot to move, e.g. `robot1`. |
| `operation_id` | *required* | Unique id of the operation, e.g. `R1_W1`. |
| `operation_type` | `weld` | `weld` \| `rivet` \| `glue` \| `move` … |
| `trajectory` | *required* | `{key}` into the blackboard → `[{"positions": [j1..j6], "time_s": float}, …]`. |
| `hold_duration_s` | `0.5` | Dwell at the working point. |
| `velocity_scale` | `1.0` | Playback scaling. |
| `has_interlock` | `false` | Whether the gating ports below are enforced. |
| `interlock_target` | `""` | `{key}` → staging pose the other arm must reach. |
| `depends_on_robots` | `""` | `{key}` → list of robot ids to wait on. |
| `depends_on_positions` | `""` | `{key}` → `[{"robot", "position", "tolerance_m"}, …]`. |

### `MoveHome` (Action)

Returns one robot to its HOME configuration.

| Port | Default | Meaning |
|---|---|---|
| `robot_ns` | *required* | Robot to home. |
| `trajectory` | `""` | `{key}` → planned path home. Empty means a blind home move. |

### `WaitForRobot` (Action)

Blocks until other arms have cleared. Succeeds when every dependency is satisfied.

| Port | Default | Meaning |
|---|---|---|
| `robot_ns` | *required* | The waiting robot. |
| `depends_on_robots` | `""` | `{key}` → robot ids to wait on. |
| `depends_on_joints` | `""` | `{key}` → required joint configurations. |
| `depends_on_positions` | `""` | `{key}` → required Cartesian clearances. |

### `CheckSafety` (Condition)

| Port | Default | Meaning |
|---|---|---|
| `condition` | `estop_clear` | Named safety predicate; failure aborts the branch. |

## Example

```xml
<root BTCPP_format="4" main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <Sequence>
      <Parallel success_count="-1" failure_count="1">
        <ExecuteTrajectory name="weld_robot1_W1" robot_ns="robot1" operation_id="W1"
                           operation_type="weld" trajectory="{traj_robot1_W1}"
                           hold_duration_s="0.5" velocity_scale="1.0"
                           has_interlock="false"/>
        <Delay name="delay_weld_robot2_W2" delay_msec="1500">
          <ExecuteTrajectory name="weld_robot2_W2" robot_ns="robot2" operation_id="W2"
                             operation_type="weld" trajectory="{traj_robot2_W2}"
                             has_interlock="true"
                             interlock_target="{interlock_robot2_W2}"/>
        </Delay>
      </Parallel>
      <WaitForRobot name="wait_robot1" robot_ns="robot1"
                    depends_on_robots="{deps_robot1_n5}"
                    depends_on_positions="{waitpos_robot1_n5}"/>
      <CheckSafety condition="estop_clear"/>
      <MoveHome name="home_robot1" robot_ns="robot1" trajectory="{traj_robot1_n7}"/>
    </Sequence>
  </BehaviorTree>
  <TreeNodesModel>
    <!-- port declarations for ExecuteTrajectory / MoveHome / WaitForRobot / CheckSafety -->
  </TreeNodesModel>
</root>
```

## Legacy dialect

Trees recorded before the format change used `SEQUENCE` / `PARALLEL` / `EXECUTE` /
`HOME` / `INTERLOCK_WAIT` / `SAFETY_GATE` with top-level fields instead of `ports`.
Because every walker reads through `node_kind` / `port`, those trees still replay,
and `to_btcpp` converts one on demand (it is idempotent, so re-converting a v4 tree
is a no-op). Do not author new trees in the legacy dialect.
