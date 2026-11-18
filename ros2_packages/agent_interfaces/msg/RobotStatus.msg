## RobotStatus.msg — published by robot agent at 10 Hz

std_msgs/Header header

string cell_id
string robot_ns
string status             # idle | busy | error | estop

float64[] joint_positions    # 6 joints
float64[3] ee_position       # world-frame EE XYZ
float64[4] ee_quaternion     # world-frame EE orientation

string current_task          # operation_id being executed, empty if idle
float32 task_progress        # 0.0–1.0
string error_message         # non-empty if status == error
