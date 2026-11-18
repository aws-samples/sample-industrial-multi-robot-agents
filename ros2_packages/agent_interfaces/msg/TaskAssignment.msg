## TaskAssignment.msg — dispatched from cell agent to robot agent

std_msgs/Header header

string cell_id
string robot_ns
string operation_id
string operation_type     # weld | rivet | glue | laser_cut | mark | move

# Trajectory — JSON-encoded for flexibility
string trajectory_json    # [{"positions": [j1..j6], "time_s": float}]

float32 hold_duration_s   # hold at target before completing
float32 velocity_scale    # 0.1 to 1.0
