import os

root = "."

for dirpath, dirnames, filenames in os.walk(root):
    total = len(filenames)
    wav = sum(1 for f in filenames if f.lower().endswith(".wav"))
    print(f"{dirpath} | total: {total} | wav: {wav}")
