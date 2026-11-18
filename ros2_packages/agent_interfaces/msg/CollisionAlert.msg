## CollisionAlert.msg — published by cell agent on collision detection

std_msgs/Header header

string cell_id
string severity           # warning | critical
float32 min_clearance_m
string robot1_ns
string robot2_ns          # empty if obstacle collision
float64[3] collision_point
string action_taken       # estop | warning | none
