import hashlib

def compute_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# Compute the hash of the client.py script
clientScriptHash = compute_file_hash("client.py")

print("SCRIPT_HASH IS: ", clientScriptHash)