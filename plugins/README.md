# plugins/ — trajectory backends from other repositories

Clone (or copy) an external backend here, e.g. `plugins/robot_ballet/`, and
register it when starting the stack:

```bash
make sim SIM=<cell> TRAJECTORY_BACKEND_PLUGINS="robot-ballet=plugins.robot_ballet.umarell:SPEC"
```

The repository is mounted live into the trajectory agent, so code here is
importable there as `plugins.<name>…` without rebuilding the image. Everything
in this directory except this README is ignored by git: a plugin keeps its own
repository. See [`docs/creating-a-backend.md`](../docs/creating-a-backend.md).
