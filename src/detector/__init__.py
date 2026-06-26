"""
src/detector — Core model package for Audio AI Detector.

Modules (build order matches CLAUDE.md):
    features   — MultiScaleSTFT, PerFrequencyInstanceNorm, SpectrogramAugmenter
    model      — BandCNN, WindowEncoder, DistanceHead, MultilabelHead   (TODO)
    dataset    — PyTorch Datasets per training phase                    (TODO)
    lightning  — DetectorLightning with phase dispatch                  (TODO)
    metrics    — AP, AUC, per-class metrics                             (TODO)
"""
