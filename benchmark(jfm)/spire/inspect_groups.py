import zarr
from arraylake import Client

client = Client()
session = client.get_repo("artamta/s2s-research").readonly_session("main")

# Inspect groups in the Zarr store
root = zarr.open(session.store, mode='r')
groups = list(root.group_keys())
print(f"Total groups found: {len(groups)}")
for g in groups:
    print(f"- {g}")
