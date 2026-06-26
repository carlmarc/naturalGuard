# 1. Show what Python is actually importing as src.dataset.metadata
python -c "from src.dataset import metadata; print(metadata.__file__)"

# 2. Show what's defined in that file
python -c "from src.dataset import metadata; print([x for x in dir(metadata) if not x.startswith('_')])"

# 3. Show the file size and first few lines
ls -la src/dataset/metadata.py
head -5 src/dataset/metadata.py