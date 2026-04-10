# Makefile — Vocal Pitch Monitor
#
# Fedora deps:
#   sudo dnf install gtk4-devel portaudio-devel fftw-devel
#
# Ubuntu/Debian deps:
#   sudo apt install libgtk-4-dev libportaudio-dev libfftw3-dev

CC      := gcc
TARGET  := pitchmonitor
SRC     := pitchmonitor.c

# pkg-config pulls in all the right flags for GTK4, PortAudio, FFTW3
PKG_LIBS := gtk4 portaudio-2.0 fftw3

CFLAGS  := -O2 -Wall -Wextra -std=c11 \
           $(shell pkg-config --cflags $(PKG_LIBS))
LIBS    := $(shell pkg-config --libs $(PKG_LIBS)) \
           -lpthread -lm

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

# Debug build — adds sanitisers and disables optimisation
debug: $(SRC)
	$(CC) -O0 -g -fsanitize=address,undefined -Wall -Wextra -std=c11 \
	      $(shell pkg-config --cflags $(PKG_LIBS)) \
	      -o $(TARGET)-debug $< $(LIBS)

install: $(TARGET)
	install -Dm755 $(TARGET) $(DESTDIR)$(PREFIX)/bin/$(TARGET)

PREFIX  ?= /usr/local

clean:
	rm -f $(TARGET) $(TARGET)-debug
