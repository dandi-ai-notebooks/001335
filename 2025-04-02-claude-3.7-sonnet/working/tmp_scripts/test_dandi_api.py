"""
This script tests the DANDI API to understand how to correctly access metadata.
"""

from dandi.dandiapi import DandiAPIClient

# Initialize the DANDI API client
client = DandiAPIClient()
dandiset = client.get_dandiset("001335")

# Print the object type
print(f"Type of dandiset object: {type(dandiset)}")

# Print available attributes and methods
print("\nDandiset attributes and methods:")
for item in dir(dandiset):
    if not item.startswith('__'):  # Skip internal methods/attributes
        print(f"- {item}")

# Try to access metadata via different methods
print("\nTrying to access metadata:")

try:
    metadata = dandiset.get_metadata()
    print("\nSuccessfully accessed metadata via get_metadata()")
    print(f"Metadata type: {type(metadata)}")
    
    # Print keys if it's a dict-like object
    if hasattr(metadata, 'keys'):
        print("Metadata keys:", list(metadata.keys()))
    
    # Try to access specific fields that we need
    if hasattr(metadata, 'get'):
        print(f"Name: {metadata.get('name', 'Not found')}")
        print(f"Description: {metadata.get('description', 'Not found')}")
except Exception as e:
    print(f"Error accessing via get_metadata(): {e}")

# Try to get metadata using get_raw_metadata()
try:
    if hasattr(dandiset, 'get_raw_metadata'):
        raw_meta = dandiset.get_raw_metadata()
        print("\nSuccessfully accessed raw_metadata")
        print(f"Raw metadata type: {type(raw_meta)}")
        
        if hasattr(raw_meta, 'keys'):
            print("Raw metadata keys:", list(raw_meta.keys()))
except Exception as e:
    print(f"Error accessing via get_raw_metadata(): {e}")

# Try accessing individual properties through direct attributes
print("\nTrying direct attribute access:")
try:
    print(f"dandiset.identifier: {dandiset.identifier}")
except Exception as e:
    print(f"Error accessing identifier: {e}")

try:
    print(f"dandiset.version_id: {dandiset.version_id}")
except Exception as e:
    print(f"Error accessing version_id: {e}")

# Try to get the version object and access its attributes
try:
    version = dandiset.version
    print("\nAccessed version object:")
    print(f"Type of version: {type(version)}")
    print("Version attributes:", [attr for attr in dir(version) if not attr.startswith('__')])
    
    if hasattr(version, 'get_metadata'):
        ver_meta = version.get_metadata()
        print("\nVersion metadata:")
        print(f"Name: {ver_meta.get('name', 'Not found')}")
        print(f"Description: {ver_meta.get('description', 'Not found')}")
except Exception as e:
    print(f"Error accessing version: {e}")