## BehaviourNode.msg — single node in a BehaviourTree

# Node types
string TYPE_PARALLEL      = "PARALLEL"
string TYPE_SEQUENCE      = "SEQUENCE"
string TYPE_EXECUTE       = "EXECUTE"
string TYPE_INTERLOCK_WAIT = "INTERLOCK_WAIT"
string TYPE_SAFETY_GATE   = "SAFETY_GATE"
string TYPE_HOME          = "HOME"

string node_id
string node_type          # one of TYPE_* constants above
string robot_ns           # for EXECUTE / INTERLOCK_WAIT / HOME
string operation_id       # e.g. "W001"
string operation_type     # "weld" | "rivet" | "glue" | "laser_cut" | "mark" | "move"

# Trajectory — JSON-encoded list of {positions, time_s}
string precomputed_trajectory_json

# Interlock
bool has_interlock
string interlock_target_json  # {"ee_position": [...], "joint_positions": [...]}

# Gates
string[] depends_on_robots
string depends_on_positions_json  # JSON list of position gate dicts

# Timing
float32 start_delay_s
float32 hold_duration_s
float32 velocity_scale

# Gate condition (SAFETY_GATE)
string condition

# Children (PARALLEL / SEQUENCE) — node IDs, resolved by cell agent
string[] children_ids
