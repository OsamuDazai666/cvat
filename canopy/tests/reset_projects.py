"""Delete the canopy test projects (and their tasks) so init-projects can recreate them. Test helper only."""
import os
from cvat_sdk import make_client
NAMES = {"Canopy Vehicles", "Canopy Plates", "Canopy Text OCR"}
with make_client(os.environ["CVAT_URL"], credentials=(os.environ["CVAT_USER"], os.environ["CVAT_PASSWORD"])) as c:
    for p in c.projects.list():
        if p.name in NAMES:
            p.remove()
            print("deleted", p.name)
