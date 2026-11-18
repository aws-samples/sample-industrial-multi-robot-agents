# Operation Parameters Reference

Default parameters per operation type used by the MRTA pipeline and trajectory planner.

| operation    | approach_offset_m | hold_s | thermal_gap | tcp_link        |
|-------------|------------------|--------|-------------|-----------------|
| weld        | 0.05             | 2.0    | 2           | welding_tcp     |
| rivet       | 0.03             | 0.5    | 1           | rivet_tcp       |
| glue        | 0.02             | 0.0*   | 1           | glue_tcp        |
| laser_cut   | 0.00             | 0.0*   | 3           | laser_tcp       |
| laser_weld  | 0.00             | 0.0*   | 3           | laser_tcp       |
| mark        | 0.00             | 0.0    | 1           | tool0           |
| move        | 0.00             | 0.0    | 0           | tool0           |

`*` path-following operation — continuous motion along a path, no dwell at target.

## Thermal gap explanation

Thermal gap = minimum number of execution steps between consecutive welds on
the same face/panel. Prevents thermal deformation from heat accumulation.
- gap=2: standard spot weld (0.3s dwell, low heat per weld)
- gap=3: laser operations (3 kW beam, higher heat concentration)
- gap=1: rivet / mark (minimal heat)

## Approach vector convention

The approach vector points AWAY from the surface toward the robot:
- Top face (robot approaches from above): [0, 0, -1]
- Front face (+X): [-1, 0, 0]
- Back face (-X): [+1, 0, 0]
- Right face (+Y): [0, -1, 0]
- Left face (-Y): [0, +1, 0]

The planner places the TCP at:
`target + approach_vector * approach_offset_m`
before moving to the final `target` position.
