# Vocal Pitch Monitor

Trying to one-shot making a vocal pitch monitor using Claude in Python.

```
# Install deps (sounddevice needs portaudio)
sudo dnf install portaudio portaudio-devel
pip install sounddevice matplotlib numpy scipy --break-system-packages

python3 vocal_pitch_monitor.py
```

It took two shots: one to build the app and then one to tweak the UI a bit.

I then tested it on implementing this in C. This took two tries as my system
uses PipeWire pretending to be PulseAudio, so we need to rewrite as PortAudio 
needs to use its PulseAudio backend instead, which will route through PipeWire 
correctly. Then I ran out of free Claude usage before it fixed it completely. 
So... ? I guess I'm on Python. I'm not committing those files until they work.
I'll give it one more pass.

```
# Fedora deps:
#   sudo dnf install gtk4-devel portaudio-devel fftw-devel
#
# Ubuntu/Debian deps:
#   sudo apt install libgtk-4-dev libportaudio-dev libfftw3-dev

make all
./pitchmonitor
```

I can obviously do the math part and have no real professional interest in the UI or figuring out
audio streams and device management in Linux. I do not care about PulseAudio, PipeWire, etc.
The interesting part is that there are things I
should know more about and explore here, so what I ought to do is perhaps take these apps, 
gut that functionality, and rewrite it myself as an exercise.

Namely:

1. I don't really have much experience with live (as opposed to static) data streams.
2. I'm a brutish Python programmer, still more comfortable in R, and would have had to
think for a bit about the best way to put some of these things into Pythonic data 
structures when it should really be second nature for these simple things.
3. I could spend some time remembering C and how to work with FFTW objects, but 
it really is a pain to deal with pointers, structs, and all that jazz. And the 
obvious problems dealing with audio.
