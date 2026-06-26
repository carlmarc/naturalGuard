import subprocess

BUCKET_PREFIX = "gs://bucket-amplify/ai_detector/data/train"

def list_objects(prefix):
    cmd = ["gcloud", "storage", "ls", f"{prefix}**"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]

def download_object(obj, destination):
    cmd = ["gcloud", "storage", "cp", obj, destination]
    subprocess.run(cmd, check=True)

objects = list_objects(BUCKET_PREFIX)

print("Found objects:")
for i, obj in enumerate(objects, 1):
    print(f"{i}: {obj}")

choices = input("Enter file numbers to download, comma-separated: ")
selected = [objects[int(x.strip()) - 1] for x in choices.split(",")]

for obj in selected:
    print(f"Downloading {obj} ...")
    download_object(obj, ".")

print("Done.")
