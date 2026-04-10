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

This is impressive: it made a functioning application in Python that did what I want 
in pretty much one shot after being told what I wanted and that I was on fedora 43 and was making
something for personal use and therefore didn't have to worry about portability. It isn't 
production level as you would need to fiddle with it to get it to work on your own system,
but that's one well-formed prompt away for somebody who can use Python.

Update (11:00 a.m.): after being allowed by Claude to run through this again, it took 
a couple rounds, we debugged the problems. There were two issues:

1. It couldn't compile because of a name mismatch between functions in the latest version
of the audio tools vs what is in Fedora (0.1 behind). This means this program will be obsolete
fairly soon, actually, as Fedora is going to update it in the near future.
2. The first try got the normalization constant for FFTW wrong, so the "confidence" was 
never correct. I'll have to look back at the math.
