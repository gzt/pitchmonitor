# Vocal Pitch Monitor

Trying to one-shot making a vocal pitch monitor using Claude in Python.

```
# Install deps (sounddevice needs portaudio)
sudo dnf install portaudio portaudio-devel
pip install sounddevice matplotlib numpy scipy --break-system-packages

python3 vocal_pitch_monitor.py
```

It took two shots: one to build the app and then one to tweak the UI a bit.
