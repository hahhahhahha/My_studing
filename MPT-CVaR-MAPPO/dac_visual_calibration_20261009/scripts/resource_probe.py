"""Resource skill fallback: retain detector errors rather than inventing fields."""
import importlib.util
import shutil
from datetime import datetime
from pathlib import Path

from common import BASE, save

path = Path("C:/Users/CKX/.codex/skills/get-available-resources/scripts/detect_resources.py")
spec = importlib.util.spec_from_file_location("resource_skill", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
value = {"datetime": datetime.now().isoformat(), "source": str(path),
         "note": "Original aggregate detector failed on psutil disk response; individual probes preserve unknowns"}
for key, probe in (("os", module.get_os_info), ("cpu", module.get_cpu_info),
                   ("memory", module.get_memory_info), ("gpu", module.get_gpu_info)):
    try:
        value[key] = probe()
    except Exception as error:
        value[key] = {"error": repr(error)}
disk = shutil.disk_usage(BASE)
value["disk"] = {"path": str(BASE), "total_bytes": disk.total, "free_bytes": disk.free,
                 "source": "stdlib shutil.disk_usage"}
value["gpu_interpretation"] = "No detected GPU in this probe is not proof of absence; CPU-only measurement does not require GPU"
save(BASE / "protocol/RESOURCES.json", value)
print(value)
