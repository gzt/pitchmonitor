/*
 * pitchmonitor.c  —  Vocal Pitch Monitor
 *
 * Real-time pitch detection and scrolling display.
 * GTK4 + Cairo rendering, PipeWire/ALSA audio via PortAudio or ALSA,
 * FFTW3 for autocorrelation-based pitch detection.
 *
 * Build:  see Makefile
 * Deps (Fedora):
 *   sudo dnf install gtk4-devel portaudio-devel fftw-devel
 */

#include <gtk/gtk.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <portaudio.h>
#include <fftw3.h>

/* ── Constants ────────────────────────────────────────────────────────────── */

#define SAMPLE_RATE      44100
#define CHUNK_FRAMES     4096
#define MIN_FREQ         60.0
#define MAX_FREQ         1200.0
#define CONF_THRESHOLD   0.30
#define HISTORY_MAX_SEC  120
#define DEFAULT_HIST_SEC 20
#define DEFAULT_OCT_LO   3
#define DEFAULT_OCT_HI   5
#define DRAW_FPS         20        /* ms between redraws = 1000/FPS */
#define DRAW_INTERVAL_MS (1000 / DRAW_FPS)

/* Pitch history ring buffer — sized for max history at one entry/chunk */
#define HISTORY_MAXLEN   ((HISTORY_MAX_SEC * SAMPLE_RATE) / CHUNK_FRAMES + 2)

/* ── Music theory ─────────────────────────────────────────────────────────── */

static const char *NOTE_NAMES[12] = {
    "C","C#/Db","D","D#/Eb","E","F","F#/Gb","G","G#/Ab","A","A#/Bb","B"
};
static const char *KEYS[12] = {
    "C","C#","D","D#","E","F","F#","G","G#","A","A#","B"
};

/* Scale interval sets (semitones from root) */
static const int MAJOR_INTERVALS[]    = {0,2,4,5,7,9,11};
static const int MINOR_INTERVALS[]    = {0,2,3,5,7,8,10};
static const int CHROMATIC_INTERVALS[]= {0,1,2,3,4,5,6,7,8,9,10,11};
#define N_MAJOR    7
#define N_MINOR    7
#define N_CHROMATIC 12

typedef enum { SCALE_CHROMATIC, SCALE_MAJOR, SCALE_MINOR } ScaleType;

static double midi_to_freq(double midi, double a4) {
    return a4 * pow(2.0, (midi - 69.0) / 12.0);
}
static double freq_to_midi(double freq, double a4) {
    return 69.0 + 12.0 * log2(freq / a4);
}

typedef struct {
    const char *name;
    int         octave;
    double      nearest_freq;
    double      cents;
} NoteInfo;

static bool freq_to_note_info(double freq, double a4, NoteInfo *out) {
    if (freq <= 0.0) return false;
    double midi     = freq_to_midi(freq, a4);
    int    near     = (int)round(midi);
    out->cents      = (midi - near) * 100.0;
    out->octave     = (near / 12) - 1;
    out->name       = NOTE_NAMES[((near % 12) + 12) % 12];
    out->nearest_freq = midi_to_freq((double)near, a4);
    return true;
}

/* ── FFTW pitch detection ─────────────────────────────────────────────────── */

typedef struct {
    int      n;            /* CHUNK_FRAMES */
    double  *in_buf;       /* input window */
    fftw_complex *fft_out; /* FFT of zero-padded input */
    double  *acf_buf;      /* inverse FFT → autocorrelation */
    fftw_plan plan_fwd;
    fftw_plan plan_inv;
} PitchDetector;

static PitchDetector *pitch_detector_new(int n) {
    PitchDetector *pd = calloc(1, sizeof(*pd));
    pd->n       = n;
    pd->in_buf  = fftw_malloc(sizeof(double) * n * 2);
    pd->fft_out = fftw_malloc(sizeof(fftw_complex) * (n + 1));
    pd->acf_buf = fftw_malloc(sizeof(double) * n * 2);
    /* Forward: real n*2 → complex n+1 */
    pd->plan_fwd = fftw_plan_dft_r2c_1d(n * 2, pd->in_buf,
                                          pd->fft_out, FFTW_ESTIMATE);
    /* Inverse: complex n+1 → real n*2 */
    pd->plan_inv = fftw_plan_dft_c2r_1d(n * 2, pd->fft_out,
                                          pd->acf_buf, FFTW_ESTIMATE);
    return pd;
}

static void pitch_detector_free(PitchDetector *pd) {
    fftw_destroy_plan(pd->plan_fwd);
    fftw_destroy_plan(pd->plan_inv);
    fftw_free(pd->in_buf);
    fftw_free(pd->fft_out);
    fftw_free(pd->acf_buf);
    free(pd);
}

/*
 * detect_pitch()
 * Returns frequency in Hz and confidence [0,1] via out params.
 * Uses FFT autocorrelation + parabolic interpolation.
 */
static void detect_pitch(PitchDetector *pd, const float *audio, int sr,
                          double *freq_out, double *conf_out)
{
    int N = pd->n;

    /* Compute RMS and mean for normalisation */
    double mean = 0.0, peak = 0.0;
    for (int i = 0; i < N; i++) mean += audio[i];
    mean /= N;

    for (int i = 0; i < N; i++) {
        double v = audio[i] - mean;
        pd->in_buf[i] = v;
        if (fabs(v) > peak) peak = fabs(v);
    }
    if (peak < 1e-6) { *freq_out = 0.0; *conf_out = 0.0; return; }

    /* Normalise and zero-pad second half */
    for (int i = 0; i < N; i++) pd->in_buf[i] /= peak;
    memset(pd->in_buf + N, 0, sizeof(double) * N);

    /* FFT autocorrelation: X·X* → IFFT */
    fftw_execute(pd->plan_fwd);
    int M = N + 1;
    for (int i = 0; i < M; i++) {
        double re = pd->fft_out[i][0], im = pd->fft_out[i][1];
        pd->fft_out[i][0] = re*re + im*im;
        pd->fft_out[i][1] = 0.0;
    }
    fftw_execute(pd->plan_inv);

    /* Normalise IFFT output.
     * FFTW does not normalise inverse transforms — the raw output is scaled
     * by the transform size (N*2).  acf_buf[0] is the zero-lag energy,
     * already scaled by N*2, so dividing every element by acf_buf[0]
     * gives a normalised ACF in [-1, 1] without any extra factor. */
    double norm = pd->acf_buf[0];
    if (norm < 1e-10) { *freq_out = 0.0; *conf_out = 0.0; return; }
    double inv_norm = 1.0 / norm;

    /* Search for peak in lag range corresponding to [MIN_FREQ, MAX_FREQ] */
    int lag_min = (int)(sr / MAX_FREQ);
    int lag_max = (int)(sr / MIN_FREQ);
    if (lag_min < 1) lag_min = 1;
    if (lag_max >= N) lag_max = N - 1;
    if (lag_min >= lag_max) { *freq_out = 0.0; *conf_out = 0.0; return; }

    int    best_lag  = lag_min;
    double best_val  = pd->acf_buf[lag_min] * inv_norm;
    for (int lag = lag_min + 1; lag <= lag_max; lag++) {
        double v = pd->acf_buf[lag] * inv_norm;
        if (v > best_val) { best_val = v; best_lag = lag; }
    }

    /* Diagnostic: print ACF peak info periodically */
    static int acf_diag = 0;
    if (++acf_diag >= 20) {
        fprintf(stderr, "  acf_peak=%.3f at lag=%d (freq=%.1f) norm=%.1f\n",
                best_val, best_lag, best_lag > 0 ? (double)sr/best_lag : 0.0, norm);
        acf_diag = 0;
    }

    double confidence = best_val < 0.0 ? 0.0 : (best_val > 1.0 ? 1.0 : best_val);
    if (confidence < 0.2) { *freq_out = 0.0; *conf_out = confidence; return; }

    /* Parabolic interpolation for sub-sample lag accuracy */
    double lag_precise = best_lag;
    if (best_lag > 0 && best_lag < N - 1) {
        double y0 = pd->acf_buf[best_lag - 1] * inv_norm;
        double y1 = pd->acf_buf[best_lag    ] * inv_norm;
        double y2 = pd->acf_buf[best_lag + 1] * inv_norm;
        double denom = 2.0 * (2.0*y1 - y0 - y2);
        if (fabs(denom) > 1e-10)
            lag_precise += (y2 - y0) / denom;
    }

    *freq_out = (lag_precise > 0.0) ? (double)sr / lag_precise : 0.0;
    *conf_out = confidence;
}

/* ── Pitch smoother ───────────────────────────────────────────────────────── */

typedef struct {
    double alpha;
    int    hold_frames;
    double midi_smooth;
    bool   has_value;
    int    silent_count;
} PitchSmoother;

static void smoother_init(PitchSmoother *s, double alpha, int hold) {
    s->alpha       = alpha;
    s->hold_frames = hold;
    s->has_value   = false;
    s->silent_count= 0;
    s->midi_smooth = 0.0;
}

static double smoother_update(PitchSmoother *s, double freq, double conf,
                               double threshold, double *conf_out)
{
    bool active = (freq > 0.0 && conf >= threshold);
    *conf_out = conf;
    if (active) {
        s->silent_count = 0;
        double midi_raw = freq_to_midi(freq, 440.0); /* a4 fixed here; adjusted later */
        if (!s->has_value) {
            s->midi_smooth = midi_raw;
            s->has_value   = true;
        } else {
            s->midi_smooth += s->alpha * (midi_raw - s->midi_smooth);
        }
        return midi_to_freq(s->midi_smooth, 440.0);
    } else {
        s->silent_count++;
        if (s->silent_count >= s->hold_frames) s->has_value = false;
        return 0.0;
    }
}

/* ── Ring buffer ──────────────────────────────────────────────────────────── */

typedef struct {
    double freq[HISTORY_MAXLEN];
    double conf[HISTORY_MAXLEN];
    int    head;   /* next write position */
    int    len;    /* current fill level   */
    int    cap;    /* max capacity         */
} RingBuf;

static void ringbuf_init(RingBuf *r, int cap) {
    memset(r, 0, sizeof(*r));
    r->cap = cap < HISTORY_MAXLEN ? cap : HISTORY_MAXLEN;
}

static void ringbuf_push(RingBuf *r, double freq, double conf) {
    r->freq[r->head] = freq;
    r->conf[r->head] = conf;
    r->head = (r->head + 1) % r->cap;
    if (r->len < r->cap) r->len++;
}

/* Fill arrays oldest→newest; returns count written */
static int ringbuf_snapshot(const RingBuf *r, double *fout, double *cout, int maxn) {
    int n = r->len < maxn ? r->len : maxn;
    int start = (r->head - n + r->cap * 2) % r->cap;
    for (int i = 0; i < n; i++) {
        int idx = (start + i) % r->cap;
        fout[i] = r->freq[idx];
        cout[i] = r->conf[idx];
    }
    return n;
}

/* ── Audio capture (PortAudio) ────────────────────────────────────────────── */

#define AUDIO_RINGBUF_CHUNKS 32
typedef struct {
    float   buf[AUDIO_RINGBUF_CHUNKS][CHUNK_FRAMES];
    int     write_pos;
    int     read_pos;
    int     count;
    pthread_mutex_t lock;
    /* Accumulator for variable-size callbacks */
    float   accum[CHUNK_FRAMES * 4];
    int     accum_fill;
} AudioQueue;

static void audioqueue_init(AudioQueue *q) {
    memset(q, 0, sizeof(*q));
    pthread_mutex_init(&q->lock, NULL);
}

static int pa_callback(const void *input, void *output,
                        unsigned long frames,
                        const PaStreamCallbackTimeInfo *ti,
                        PaStreamCallbackFlags flags, void *userdata)
{
    (void)output; (void)ti; (void)flags;
    AudioQueue *q = userdata;
    const float *in = input;
    if (!in) return paContinue;

    pthread_mutex_lock(&q->lock);

    unsigned long remaining = frames;
    const float *src = in;

    while (remaining > 0) {
        int space = CHUNK_FRAMES - q->accum_fill;
        int take  = (int)remaining < space ? (int)remaining : space;
        memcpy(q->accum + q->accum_fill, src, take * sizeof(float));
        q->accum_fill += take;
        src           += take;
        remaining     -= take;

        if (q->accum_fill == CHUNK_FRAMES) {
            if (q->count < AUDIO_RINGBUF_CHUNKS) {
                memcpy(q->buf[q->write_pos], q->accum, CHUNK_FRAMES * sizeof(float));
                q->write_pos = (q->write_pos + 1) % AUDIO_RINGBUF_CHUNKS;
                q->count++;
            }
            q->accum_fill = 0;
        }
    }

    pthread_mutex_unlock(&q->lock);
    return paContinue;
}

static bool audioqueue_pop(AudioQueue *q, float *out) {
    pthread_mutex_lock(&q->lock);
    if (q->count == 0) { pthread_mutex_unlock(&q->lock); return false; }
    memcpy(out, q->buf[q->read_pos], CHUNK_FRAMES * sizeof(float));
    q->read_pos = (q->read_pos + 1) % AUDIO_RINGBUF_CHUNKS;
    q->count--;
    pthread_mutex_unlock(&q->lock);
    return true;
}

/* ── Application state ────────────────────────────────────────────────────── */

typedef struct AppState {
    /* GTK widgets */
    GtkWidget   *window;
    GtkWidget   *draw_area;
    GtkWidget   *lbl_freq;
    GtkWidget   *lbl_note;
    GtkWidget   *lbl_octave;
    GtkWidget   *lbl_nearest;
    GtkWidget   *lbl_cents;
    GtkWidget   *lbl_conf_pct;
    GtkWidget   *cents_canvas;
    GtkWidget   *conf_canvas;
    GtkWidget   *device_combo;
    GtkWidget   *key_combo;
    GtkWidget   *scale_combo;
    GtkWidget   *spin_a4;
    GtkWidget   *spin_oct_lo;
    GtkWidget   *spin_oct_hi;
    GtkWidget   *smooth_combo;
    GtkWidget   *hist_combo;
    GtkWidget   *btn_startstop;

    /* Audio */
    PaStream    *pa_stream;
    AudioQueue   audio_queue;
    bool         audio_running;
    int          actual_sample_rate;   /* may be 48000 on PipeWire */
    int          n_devices;
    int          device_ids[64];
    char         device_names[64][64];
    int          selected_device_idx;

    /* Pitch detection */
    PitchDetector *detector;
    PitchSmoother  smoother;
    pthread_t      worker_thread;
    bool           worker_running;

    /* Shared state (worker writes, GTK timer reads) */
    pthread_mutex_t state_lock;
    RingBuf         history;
    double          current_freq;
    double          current_conf;

    /* Display settings */
    double      a4;
    int         key_idx;        /* 0-11 */
    ScaleType   scale_type;
    double      smooth_alpha;
    int         history_sec;
    int         oct_lo, oct_hi;
    double      conf_threshold;

    /* Cached display freq (smoothed, for label updates) */
    double      display_freq;
    double      display_conf;
} AppState;

/* ── Pitch worker thread ──────────────────────────────────────────────────── */

static void *pitch_worker(void *arg) {
    AppState *app = arg;
    float chunk[CHUNK_FRAMES];

    while (app->worker_running) {
        if (!audioqueue_pop(&app->audio_queue, chunk)) {
            struct timespec ts = {0, 5000000}; /* 5 ms */
            nanosleep(&ts, NULL);
            continue;
        }

        /* RMS gate */
        double rms = 0.0;
        for (int i = 0; i < CHUNK_FRAMES; i++) rms += chunk[i]*chunk[i];
        rms = sqrt(rms / CHUNK_FRAMES);

        double freq = 0.0, conf = 0.0;
        if (rms >= 0.001) {
            int sr = app->actual_sample_rate > 0 ? app->actual_sample_rate : SAMPLE_RATE;
            detect_pitch(app->detector, chunk, sr, &freq, &conf);
            if (freq < MIN_FREQ || freq > MAX_FREQ) { freq = 0.0; conf = 0.0; }
        }

        /* Diagnostic: print every ~2 seconds (every ~20 chunks at 93ms/chunk) */
        static int diag_counter = 0;
        if (++diag_counter >= 20) {
            fprintf(stderr, "rms=%.4f freq=%.1f conf=%.2f\n", rms, freq, conf);
            diag_counter = 0;
        }

        /* Smooth on MIDI scale; we fix a4=440 in smoother, adjust display later */
        double conf_s;
        double freq_s = smoother_update(&app->smoother, freq, conf,
                                         app->conf_threshold, &conf_s);

        pthread_mutex_lock(&app->state_lock);
        ringbuf_push(&app->history, freq_s, conf);
        app->current_freq = freq_s;
        app->current_conf = conf;
        pthread_mutex_unlock(&app->state_lock);
    }
    return NULL;
}

/* ── PortAudio helpers ────────────────────────────────────────────────────── */

/*
 * Find the PulseAudio host API index, or -1 if not available.
 * On PipeWire systems PortAudio's ALSA backend fails; the PulseAudio
 * backend routes through PipeWire's PulseAudio compatibility layer fine.
 */
static PaHostApiIndex find_pulse_api(void) {
    int n = Pa_GetHostApiCount();
    fprintf(stderr, "Available PortAudio host APIs:\n");
    for (int i = 0; i < n; i++) {
        const PaHostApiInfo *info = Pa_GetHostApiInfo(i);
        if (!info) continue;
        fprintf(stderr, "  [%d] type=%d name=%s\n", i, info->type, info->name ? info->name : "(null)");
        if (info->name && (
                strstr(info->name, "PulseAudio") ||
                strstr(info->name, "pulse")      ||
                strstr(info->name, "PipeWire")   ||
                strstr(info->name, "pipewire")))
            return i;
    }
    return -1;
}

static void enumerate_devices(AppState *app) {
    app->n_devices = 0;

    /*
     * Prefer PulseAudio host API devices on PipeWire systems.
     * Fall back to all-API enumeration if PulseAudio API not found.
     */
    PaHostApiIndex pulse_api = find_pulse_api();

    int n = Pa_GetDeviceCount();
    /* First pass: PulseAudio devices only (if available) */
    if (pulse_api >= 0) {
        const PaHostApiInfo *hinfo = Pa_GetHostApiInfo(pulse_api);
        for (int ii = 0; ii < hinfo->deviceCount && app->n_devices < 64; ii++) {
            int pa_idx = Pa_HostApiDeviceIndexToDeviceIndex(pulse_api, ii);
            const PaDeviceInfo *info = Pa_GetDeviceInfo(pa_idx);
            if (info && info->maxInputChannels > 0) {
                app->device_ids[app->n_devices] = pa_idx;
                snprintf(app->device_names[app->n_devices],
                         sizeof(app->device_names[0]),
                         "%d: %.55s", pa_idx, info->name);
                app->n_devices++;
            }
        }
        fprintf(stderr, "Using PulseAudio host API (%d input devices)\n",
                app->n_devices);
    }

    /* Fallback: enumerate all APIs */
    if (app->n_devices == 0) {
        fprintf(stderr, "PulseAudio API not found, enumerating all devices\n");
        for (int i = 0; i < n && app->n_devices < 64; i++) {
            const PaDeviceInfo *info = Pa_GetDeviceInfo(i);
            if (info && info->maxInputChannels > 0) {
                app->device_ids[app->n_devices] = i;
                snprintf(app->device_names[app->n_devices],
                         sizeof(app->device_names[0]),
                         "%d: %.55s", i, info->name);
                app->n_devices++;
            }
        }
    }
}

static bool start_audio(AppState *app, int device_list_idx) {
    if (app->pa_stream) {
        Pa_StopStream(app->pa_stream);
        Pa_CloseStream(app->pa_stream);
        app->pa_stream = NULL;
    }
    if (device_list_idx < 0 || device_list_idx >= app->n_devices) return false;

    int pa_dev = app->device_ids[device_list_idx];
    const PaDeviceInfo *devinfo = Pa_GetDeviceInfo(pa_dev);
    if (!devinfo) return false;

    /*
     * Use the device's native sample rate if it differs from our preferred
     * rate — PipeWire's default is 48000, not 44100.
     * We store the actual rate so the pitch detector uses the right value.
     */
    double sr = devinfo->defaultSampleRate;
    if (sr <= 0.0) sr = SAMPLE_RATE;
    app->actual_sample_rate = (int)sr;
    fprintf(stderr, "Opening device %d (%s) at %.0f Hz\n",
            pa_dev, devinfo->name, sr);

    PaStreamParameters params = {
        .device                    = pa_dev,
        .channelCount              = 1,
        .sampleFormat              = paFloat32,
        .suggestedLatency          = devinfo->defaultLowInputLatency,
        .hostApiSpecificStreamInfo = NULL
    };

    PaError err = Pa_OpenStream(&app->pa_stream, &params, NULL,
                                 sr, CHUNK_FRAMES, paClipOff,
                                 pa_callback, &app->audio_queue);
    if (err != paNoError) {
        fprintf(stderr, "PortAudio open error: %s\n", Pa_GetErrorText(err));
        /* Retry with paFramesPerBufferUnspecified — some PulseAudio devices
           refuse a fixed buffer size */
        err = Pa_OpenStream(&app->pa_stream, &params, NULL,
                             sr, paFramesPerBufferUnspecified, paClipOff,
                             pa_callback, &app->audio_queue);
        if (err != paNoError) {
            fprintf(stderr, "PortAudio retry error: %s\n", Pa_GetErrorText(err));
            return false;
        }
        fprintf(stderr, "Opened with variable buffer size\n");
    }
    err = Pa_StartStream(app->pa_stream);
    if (err != paNoError) {
        fprintf(stderr, "PortAudio start error: %s\n", Pa_GetErrorText(err));
        Pa_CloseStream(app->pa_stream);
        app->pa_stream = NULL;
        return false;
    }
    app->audio_running = true;
    fprintf(stderr, "Audio started on device %d at %d Hz\n",
            pa_dev, app->actual_sample_rate);
    return true;
}

static void stop_audio(AppState *app) {
    app->audio_running = false;
    if (app->pa_stream) {
        Pa_StopStream(app->pa_stream);
        Pa_CloseStream(app->pa_stream);
        app->pa_stream = NULL;
    }
}

/* ── Cairo drawing helpers ────────────────────────────────────────────────── */

#define COL_BG      0.051, 0.059, 0.078   /* #0d0f14 */
#define COL_PANEL   0.075, 0.086, 0.118   /* #13161e */
#define COL_ACCENT  0.000, 0.898, 0.627   /* #00e5a0 */
#define COL_DIM     0.165, 0.184, 0.239   /* #2a2f3d */
#define COL_MUTED   0.420, 0.447, 0.502   /* #6b7280 */
#define COL_TEXT    0.910, 0.918, 0.941   /* #e8eaf0 */
#define COL_YELLOW  0.941, 0.753, 0.251   /* #f0c040 */
#define COL_RED     1.000, 0.310, 0.431   /* #ff4f6e */
#define COL_DARK    0.227, 0.204, 0.173   /* dim note bg */

static void set_col(cairo_t *cr, double r, double g, double b) {
    cairo_set_source_rgb(cr, r, g, b);
}
static void set_cola(cairo_t *cr, double r, double g, double b, double a) {
    cairo_set_source_rgba(cr, r, g, b, a);
}

/* ── Cents bar drawing ────────────────────────────────────────────────────── */

static void draw_cents_bar(GtkDrawingArea *area, cairo_t *cr,
                            int w, int h, gpointer data)
{
    AppState *app = data;
    double cents = 0.0;
    NoteInfo ni;
    if (app->display_conf >= app->conf_threshold &&
        freq_to_note_info(app->display_freq, app->a4, &ni))
        cents = ni.cents;

    set_col(cr, COL_BG); cairo_paint(cr);

    double mid = w / 2.0;
    /* Track */
    set_col(cr, COL_DIM);
    cairo_rectangle(cr, 10, h/2.0 - 3, w - 20, 6);
    cairo_fill(cr);
    /* Centre tick */
    set_cola(cr, 1,1,1, 0.25);
    cairo_rectangle(cr, mid - 1, h/2.0 - 8, 2, 16);
    cairo_fill(cr);

    double clamped = cents < -50 ? -50 : cents > 50 ? 50 : cents;
    double px      = mid + (clamped / 50.0) * (mid - 12);
    double abs_c   = fabs(clamped);

    if (abs_c < 8)       set_col(cr, COL_ACCENT);
    else if (abs_c < 20) set_col(cr, COL_YELLOW);
    else                 set_col(cr, COL_RED);

    /* Bar */
    double bx = px < mid ? px : mid;
    cairo_rectangle(cr, bx, h/2.0 - 4, fabs(px - mid), 8);
    cairo_fill(cr);

    /* Marker dot */
    if (abs_c < 8)       set_col(cr, COL_ACCENT);
    else if (abs_c < 20) set_col(cr, COL_YELLOW);
    else                 set_col(cr, COL_RED);
    cairo_arc(cr, px, h/2.0, 7, 0, 2*G_PI);
    cairo_fill(cr);

    /* Labels */
    set_col(cr, COL_MUTED);
    cairo_set_font_size(cr, 8);
    cairo_move_to(cr, 10, h - 4); cairo_show_text(cr, "-50c");
    cairo_move_to(cr, mid - 4, h - 4); cairo_show_text(cr, "0");
    cairo_move_to(cr, w - 26, h - 4); cairo_show_text(cr, "+50c");
}

/* ── Confidence bar drawing ───────────────────────────────────────────────── */

static void draw_conf_bar(GtkDrawingArea *area, cairo_t *cr,
                           int w, int h, gpointer data)
{
    AppState *app = data;
    set_col(cr, COL_DIM); cairo_paint(cr);
    double pct = app->display_conf;
    if (pct > 0.0) {
        double fw = pct * w;
        if (fw > w) fw = w;
        if (pct > 0.75)      set_col(cr, COL_ACCENT);
        else if (pct > 0.5)  set_col(cr, COL_YELLOW);
        else                 set_col(cr, COL_RED);
        cairo_rectangle(cr, 0, 0, fw, h);
        cairo_fill(cr);
    }
}

/* ── Main plot drawing ────────────────────────────────────────────────────── */

/* Log-scale Y position mapping */
static double freq_to_y(double freq, double y_min_f, double y_max_f, double h) {
    if (freq <= 0.0) return -1.0;
    double log_min = log2(y_min_f);
    double log_max = log2(y_max_f);
    double log_f   = log2(freq);
    double t = (log_f - log_min) / (log_max - log_min);
    return h * (1.0 - t);
}

static void draw_plot(GtkDrawingArea *area, cairo_t *cr,
                       int w, int h, gpointer data)
{
    AppState *app = data;

    /* Background */
    set_col(cr, COL_BG); cairo_paint(cr);

    double a4    = app->a4;
    int    lo    = app->oct_lo;
    int    hi    = app->oct_hi;
    if (lo > hi) { int t = lo; lo = hi; hi = t; }

    double y_min_f = midi_to_freq((double)(lo + 1) * 12, a4) * 0.92;
    double y_max_f = midi_to_freq((double)(hi + 1) * 12 + 11, a4) * 1.08;

    const int *intervals;
    int n_intervals;
    switch (app->scale_type) {
        case SCALE_MAJOR:    intervals = MAJOR_INTERVALS;    n_intervals = N_MAJOR;    break;
        case SCALE_MINOR:    intervals = MINOR_INTERVALS;    n_intervals = N_MINOR;    break;
        default:             intervals = CHROMATIC_INTERVALS; n_intervals = N_CHROMATIC; break;
    }

    bool is_chromatic = (app->scale_type == SCALE_CHROMATIC);
    int  root_semi    = app->key_idx;

    /* ── Scale guide lines ── */
    cairo_set_font_size(cr, 8.0);
    for (int oct = lo - 1; oct <= hi + 1; oct++) {
        for (int ii = 0; ii < n_intervals; ii++) {
            int semi  = root_semi + intervals[ii];
            int midi  = (oct + 1) * 12 + semi;
            double f  = midi_to_freq((double)midi, a4);
            double y  = freq_to_y(f, y_min_f, y_max_f, (double)h);
            if (y < 0 || y > h) continue;

            bool is_root = (((midi % 12) + 12) % 12 == root_semi);
            double alpha, lw;
            if (is_root)          { alpha = 0.90; lw = 1.0; }
            else if (is_chromatic){ alpha = 0.22; lw = 0.3; }
            else                  { alpha = 0.45; lw = 0.5; }

            if (is_root) set_cola(cr, COL_ACCENT, alpha);
            else         set_cola(cr, COL_DIM, alpha);

            cairo_set_line_width(cr, lw);
            if (is_root) {
                /* Dashed for root */
                double dash[] = {4, 3};
                cairo_set_dash(cr, dash, 2, 0);
            } else {
                cairo_set_dash(cr, NULL, 0, 0);
            }
            cairo_move_to(cr, 0,   y);
            cairo_line_to(cr, w - 28, y);
            cairo_stroke(cr);
            cairo_set_dash(cr, NULL, 0, 0);

            /* Note label on right edge (non-chromatic, or roots) */
            if (!is_chromatic || is_root) {
                int   note_idx = ((midi % 12) + 12) % 12;
                int   note_oct = midi / 12 - 1;
                char  label[16];
                snprintf(label, sizeof(label), "%s%d",
                         NOTE_NAMES[note_idx], note_oct);
                if (is_root) set_cola(cr, COL_ACCENT, 1.0);
                else         set_cola(cr, COL_MUTED,  0.7);
                cairo_move_to(cr, w - 26, y + 4);
                cairo_show_text(cr, label);
            }
        }
    }

    /* ── Pitch history trace ── */
    double snap_freq[HISTORY_MAXLEN];
    double snap_conf[HISTORY_MAXLEN];
    int n_snap;
    pthread_mutex_lock(&app->state_lock);
    n_snap = ringbuf_snapshot(&app->history, snap_freq, snap_conf, HISTORY_MAXLEN);
    pthread_mutex_unlock(&app->state_lock);

    if (n_snap > 1) {
        cairo_set_line_width(cr, 2.0);
        set_col(cr, COL_ACCENT);
        cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
        cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);

        bool in_segment = false;
        for (int i = 0; i < n_snap; i++) {
            double f = snap_freq[i];
            double t = (double)(i - n_snap + 1) / (double)(SAMPLE_RATE / CHUNK_FRAMES);
            /* t goes from -history_sec to ~0 */
            double x = ((t + app->history_sec) / (double)app->history_sec) * w;
            double y = freq_to_y(f, y_min_f, y_max_f, (double)h);

            if (f > 0.0 && y >= 0.0 && y <= h) {
                if (!in_segment) {
                    cairo_move_to(cr, x, y);
                    in_segment = true;
                } else {
                    cairo_line_to(cr, x, y);
                }
            } else {
                if (in_segment) { cairo_stroke(cr); in_segment = false; }
            }
        }
        if (in_segment) cairo_stroke(cr);
    }

    /* ── Current dot ── */
    double cf = app->display_freq;
    if (cf > 0.0 && app->display_conf >= app->conf_threshold) {
        double y = freq_to_y(cf, y_min_f, y_max_f, (double)h);
        if (y >= 0.0 && y <= h) {
            set_col(cr, COL_ACCENT);
            cairo_arc(cr, w - 1.0, y, 6.0, 0, 2*G_PI);
            cairo_fill(cr);
            set_cola(cr, 1,1,1, 0.12);
            cairo_arc(cr, w - 1.0, y, 8.0, 0, 2*G_PI);
            cairo_fill(cr);
        }
    }

    /* ── Time axis ticks ── */
    set_col(cr, COL_MUTED);
    cairo_set_font_size(cr, 8.0);
    cairo_set_line_width(cr, 0.5);
    for (int s = 0; s <= app->history_sec; s += 5) {
        double x = ((double)s / app->history_sec) * w;
        set_cola(cr, COL_DIM, 0.5);
        cairo_move_to(cr, x, 0); cairo_line_to(cr, x, h);
        cairo_stroke(cr);
        char tbuf[8];
        snprintf(tbuf, sizeof(tbuf), "-%ds", app->history_sec - s);
        set_col(cr, COL_MUTED);
        cairo_move_to(cr, x + 2, h - 4);
        cairo_show_text(cr, tbuf);
    }
}

/* ── GTK timer: update display ────────────────────────────────────────────── */

static gboolean update_display(gpointer data) {
    AppState *app = data;

    pthread_mutex_lock(&app->state_lock);
    double freq = app->current_freq;
    double conf = app->current_conf;
    pthread_mutex_unlock(&app->state_lock);

    app->display_freq = freq;
    app->display_conf = conf;

    bool active = (freq > 0.0 && conf >= app->conf_threshold);
    char buf[64];

    if (active) {
        snprintf(buf, sizeof(buf), "%.1f Hz", freq);
        gtk_label_set_text(GTK_LABEL(app->lbl_freq), buf);

        NoteInfo ni;
        if (freq_to_note_info(freq, app->a4, &ni)) {
            gtk_label_set_text(GTK_LABEL(app->lbl_note), ni.name);
            snprintf(buf, sizeof(buf), "octave %d", ni.octave);
            gtk_label_set_text(GTK_LABEL(app->lbl_octave), buf);
            snprintf(buf, sizeof(buf), "nearest: %.2f Hz  (%s%d)",
                     ni.nearest_freq, ni.name, ni.octave);
            gtk_label_set_text(GTK_LABEL(app->lbl_nearest), buf);
            snprintf(buf, sizeof(buf), "cents: %+.1f", ni.cents);
            gtk_label_set_text(GTK_LABEL(app->lbl_cents), buf);
        }
    } else {
        gtk_label_set_text(GTK_LABEL(app->lbl_freq),    "--- Hz");
        gtk_label_set_text(GTK_LABEL(app->lbl_note),    "--");
        gtk_label_set_text(GTK_LABEL(app->lbl_octave),  "octave --");
        gtk_label_set_text(GTK_LABEL(app->lbl_nearest), "nearest: --- Hz");
        gtk_label_set_text(GTK_LABEL(app->lbl_cents),   "cents: +/-0");
    }

    snprintf(buf, sizeof(buf), "%d%%", (int)(conf * 100));
    gtk_label_set_text(GTK_LABEL(app->lbl_conf_pct), buf);

    gtk_widget_queue_draw(app->draw_area);
    gtk_widget_queue_draw(app->cents_canvas);
    gtk_widget_queue_draw(app->conf_canvas);

    return G_SOURCE_CONTINUE;
}

/* ── UI callbacks ─────────────────────────────────────────────────────────── */

static void on_device_changed(GtkComboBoxText *cb, gpointer data) {
    AppState *app = data;
    int idx = gtk_combo_box_get_active(GTK_COMBO_BOX(cb));
    if (idx < 0 || idx >= app->n_devices) return;
    app->selected_device_idx = idx;
    bool was_running = app->audio_running;
    stop_audio(app);
    if (was_running) start_audio(app, idx);
}

static void on_key_changed(GtkComboBoxText *cb, gpointer data) {
    AppState *app = data;
    app->key_idx = gtk_combo_box_get_active(GTK_COMBO_BOX(cb));
    gtk_widget_queue_draw(app->draw_area);
}

static void on_scale_changed(GtkComboBoxText *cb, gpointer data) {
    AppState *app = data;
    app->scale_type = (ScaleType)gtk_combo_box_get_active(GTK_COMBO_BOX(cb));
    gtk_widget_queue_draw(app->draw_area);
}

static void on_a4_changed(GtkSpinButton *sb, gpointer data) {
    AppState *app = data;
    app->a4 = gtk_spin_button_get_value(sb);
    gtk_widget_queue_draw(app->draw_area);
}

static void on_oct_lo_changed(GtkSpinButton *sb, gpointer data) {
    AppState *app = data;
    app->oct_lo = (int)gtk_spin_button_get_value(sb);
    gtk_widget_queue_draw(app->draw_area);
}

static void on_oct_hi_changed(GtkSpinButton *sb, gpointer data) {
    AppState *app = data;
    app->oct_hi = (int)gtk_spin_button_get_value(sb);
    gtk_widget_queue_draw(app->draw_area);
}

static void on_smooth_changed(GtkComboBoxText *cb, gpointer data) {
    AppState *app = data;
    const char *s = gtk_combo_box_text_get_active_text(cb);
    if (s) {
        double a = atof(s);
        if (a >= 0.01 && a <= 1.0) app->smoother.alpha = a;
    }
}

static void on_hist_changed(GtkComboBoxText *cb, gpointer data) {
    AppState *app = data;
    const char *s = gtk_combo_box_text_get_active_text(cb);
    if (s) {
        int secs = atoi(s);
        if (secs >= 5 && secs <= HISTORY_MAX_SEC) {
            app->history_sec = secs;
            pthread_mutex_lock(&app->state_lock);
            int cap = (secs * SAMPLE_RATE) / CHUNK_FRAMES + 2;
            ringbuf_init(&app->history, cap);
            pthread_mutex_unlock(&app->state_lock);
        }
    }
    gtk_widget_queue_draw(app->draw_area);
}

static void on_startstop_clicked(GtkButton *btn, gpointer data) {
    AppState *app = data;
    if (app->audio_running) {
        stop_audio(app);
        gtk_button_set_label(btn, "START");
    } else {
        start_audio(app, app->selected_device_idx);
        gtk_button_set_label(btn, "STOP");
    }
}

static void on_window_destroy(GtkWidget *w, gpointer data) {
    AppState *app = data;
    app->worker_running = false;
    stop_audio(app);
    pthread_join(app->worker_thread, NULL);
    pitch_detector_free(app->detector);
    Pa_Terminate();
}

/* ── CSS styling ──────────────────────────────────────────────────────────── */

static const char *APP_CSS =
    "window, .view { background-color: #0d0f14; color: #e8eaf0; }"
    ".panel { background-color: #13161e; padding: 8px; }"
    ".freq-label { font-family: Courier New, monospace; font-size: 34px;"
    "  font-weight: bold; color: #00e5a0; }"
    ".note-label { font-family: Courier New, monospace; font-size: 26px;"
    "  font-weight: bold; color: #e8eaf0; }"
    ".small-label { font-family: Courier New, monospace; font-size: 10px;"
    "  color: #6b7280; }"
    ".detail-label { font-family: Courier New, monospace; font-size: 12px;"
    "  color: #6b7280; }"
    ".ctrl-label { font-family: Courier New, monospace; font-size: 9px;"
    "  color: #6b7280; margin-right: 2px; }"
    "combobox, spinbutton, entry {"
    "  background-color: #13161e; color: #e8eaf0;"
    "  border-color: #2a2f3d; font-family: Courier New, monospace; font-size: 10px; }"
    "button { background-color: #13161e; color: #ff4f6e;"
    "  font-family: Courier New, monospace; font-weight: bold;"
    "  border-color: #2a2f3d; }";

/* ── Build UI ─────────────────────────────────────────────────────────────── */

static void build_ui(AppState *app) {
    GtkCssProvider *css = gtk_css_provider_new();
    gtk_css_provider_load_from_string(css, APP_CSS);
    gtk_style_context_add_provider_for_display(
        gdk_display_get_default(),
        GTK_STYLE_PROVIDER(css),
        GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);

    app->window = gtk_window_new();
    gtk_window_set_title(GTK_WINDOW(app->window), "Vocal Pitch Monitor");
    gtk_window_set_default_size(GTK_WINDOW(app->window), 1000, 680);
    g_signal_connect(app->window, "destroy",
                     G_CALLBACK(on_window_destroy), app);

    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_window_set_child(GTK_WINDOW(app->window), vbox);

    /* ── Info bar ── */
    GtkWidget *info_bar = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    gtk_widget_add_css_class(info_bar, "panel");
    gtk_box_append(GTK_BOX(vbox), info_bar);

    app->lbl_freq = gtk_label_new("--- Hz");
    gtk_widget_add_css_class(app->lbl_freq, "freq-label");
    gtk_box_append(GTK_BOX(info_bar), app->lbl_freq);

    GtkWidget *note_col = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
    gtk_box_append(GTK_BOX(info_bar), note_col);
    app->lbl_note = gtk_label_new("--");
    gtk_widget_add_css_class(app->lbl_note, "note-label");
    gtk_box_append(GTK_BOX(note_col), app->lbl_note);
    app->lbl_octave = gtk_label_new("octave --");
    gtk_widget_add_css_class(app->lbl_octave, "small-label");
    gtk_box_append(GTK_BOX(note_col), app->lbl_octave);

    /* Separator */
    GtkWidget *sep = gtk_separator_new(GTK_ORIENTATION_VERTICAL);
    gtk_box_append(GTK_BOX(info_bar), sep);

    GtkWidget *detail_col = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
    gtk_box_append(GTK_BOX(info_bar), detail_col);
    app->lbl_nearest = gtk_label_new("nearest: --- Hz");
    gtk_widget_add_css_class(app->lbl_nearest, "detail-label");
    gtk_box_append(GTK_BOX(detail_col), app->lbl_nearest);
    app->lbl_cents = gtk_label_new("cents: +/-0");
    gtk_widget_add_css_class(app->lbl_cents, "detail-label");
    gtk_box_append(GTK_BOX(detail_col), app->lbl_cents);

    /* Cents bar */
    app->cents_canvas = gtk_drawing_area_new();
    gtk_widget_set_size_request(app->cents_canvas, 220, 48);
    gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(app->cents_canvas),
                                    draw_cents_bar, app, NULL);
    gtk_box_append(GTK_BOX(info_bar), app->cents_canvas);

    /* Conf block (right-aligned) */
    GtkWidget *conf_col = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
    gtk_widget_set_hexpand(conf_col, TRUE);
    gtk_widget_set_halign(conf_col, GTK_ALIGN_END);
    gtk_box_append(GTK_BOX(info_bar), conf_col);
    GtkWidget *lbl_conf_hdr = gtk_label_new("CONFIDENCE");
    gtk_widget_add_css_class(lbl_conf_hdr, "small-label");
    gtk_box_append(GTK_BOX(conf_col), lbl_conf_hdr);
    app->conf_canvas = gtk_drawing_area_new();
    gtk_widget_set_size_request(app->conf_canvas, 80, 14);
    gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(app->conf_canvas),
                                    draw_conf_bar, app, NULL);
    gtk_box_append(GTK_BOX(conf_col), app->conf_canvas);
    app->lbl_conf_pct = gtk_label_new("0%");
    gtk_widget_add_css_class(app->lbl_conf_pct, "small-label");
    gtk_box_append(GTK_BOX(conf_col), app->lbl_conf_pct);

    /* ── Controls bar ── */
    GtkWidget *ctrl = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    gtk_widget_set_margin_start(ctrl, 8);
    gtk_widget_set_margin_end(ctrl,   8);
    gtk_widget_set_margin_top(ctrl,   4);
    gtk_widget_set_margin_bottom(ctrl,4);
    gtk_box_append(GTK_BOX(vbox), ctrl);

#define CLBL(text) do { \
    GtkWidget *_l = gtk_label_new(text); \
    gtk_widget_add_css_class(_l, "ctrl-label"); \
    gtk_box_append(GTK_BOX(ctrl), _l); } while(0)

    /* Key */
    CLBL("KEY");
    app->key_combo = gtk_combo_box_text_new();
    for (int i = 0; i < 12; i++)
        gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(app->key_combo), KEYS[i]);
    gtk_combo_box_set_active(GTK_COMBO_BOX(app->key_combo), app->key_idx);
    gtk_box_append(GTK_BOX(ctrl), app->key_combo);
    g_signal_connect(app->key_combo, "changed", G_CALLBACK(on_key_changed), app);

    /* Scale */
    CLBL("SCALE");
    app->scale_combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(app->scale_combo), "Chromatic");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(app->scale_combo), "Major");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(app->scale_combo), "Minor");
    gtk_combo_box_set_active(GTK_COMBO_BOX(app->scale_combo), (int)app->scale_type);
    gtk_box_append(GTK_BOX(ctrl), app->scale_combo);
    g_signal_connect(app->scale_combo, "changed", G_CALLBACK(on_scale_changed), app);

    /* A4 */
    CLBL("A4=");
    app->spin_a4 = gtk_spin_button_new_with_range(400, 480, 1);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(app->spin_a4), app->a4);
    gtk_widget_set_size_request(app->spin_a4, 70, -1);
    gtk_box_append(GTK_BOX(ctrl), app->spin_a4);
    CLBL("Hz");
    g_signal_connect(app->spin_a4, "value-changed", G_CALLBACK(on_a4_changed), app);

    /* Smooth */
    CLBL("SMOOTH");
    app->smooth_combo = gtk_combo_box_text_new();
    const char *smooth_vals[] = {"0.08","0.15","0.25","0.40","0.65","1.0", NULL};
    for (int i = 0; smooth_vals[i]; i++)
        gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(app->smooth_combo), smooth_vals[i]);
    gtk_combo_box_set_active(GTK_COMBO_BOX(app->smooth_combo), 2); /* 0.25 */
    gtk_widget_set_size_request(app->smooth_combo, 70, -1);
    gtk_box_append(GTK_BOX(ctrl), app->smooth_combo);
    g_signal_connect(app->smooth_combo, "changed", G_CALLBACK(on_smooth_changed), app);

    /* History */
    CLBL("HISTORY");
    app->hist_combo = gtk_combo_box_text_new();
    const char *hist_vals[] = {"10","15","20","30","60", NULL};
    for (int i = 0; hist_vals[i]; i++)
        gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(app->hist_combo), hist_vals[i]);
    gtk_combo_box_set_active(GTK_COMBO_BOX(app->hist_combo), 2); /* 20s */
    gtk_widget_set_size_request(app->hist_combo, 60, -1);
    gtk_box_append(GTK_BOX(ctrl), app->hist_combo);
    CLBL("s");
    g_signal_connect(app->hist_combo, "changed", G_CALLBACK(on_hist_changed), app);

    /* Octave range */
    CLBL("OCT");
    app->spin_oct_lo = gtk_spin_button_new_with_range(1, 7, 1);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(app->spin_oct_lo), app->oct_lo);
    gtk_widget_set_size_request(app->spin_oct_lo, 52, -1);
    gtk_box_append(GTK_BOX(ctrl), app->spin_oct_lo);
    CLBL("-");
    app->spin_oct_hi = gtk_spin_button_new_with_range(1, 7, 1);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(app->spin_oct_hi), app->oct_hi);
    gtk_widget_set_size_request(app->spin_oct_hi, 52, -1);
    gtk_box_append(GTK_BOX(ctrl), app->spin_oct_hi);
    g_signal_connect(app->spin_oct_lo, "value-changed", G_CALLBACK(on_oct_lo_changed), app);
    g_signal_connect(app->spin_oct_hi, "value-changed", G_CALLBACK(on_oct_hi_changed), app);

    /* Device */
    CLBL("DEVICE");
    app->device_combo = gtk_combo_box_text_new();
    int default_sel = 0;
    /* Prefer PulseAudio API default; fall back to global default */
    PaHostApiIndex pulse_api = find_pulse_api();
    int pa_def = (pulse_api >= 0)
        ? Pa_HostApiDeviceIndexToDeviceIndex(
              pulse_api,
              Pa_GetHostApiInfo(pulse_api)->defaultInputDevice)
        : Pa_GetDefaultInputDevice();
    for (int i = 0; i < app->n_devices; i++) {
        gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(app->device_combo),
                                        app->device_names[i]);
        if (app->device_ids[i] == pa_def) default_sel = i;
    }
    gtk_combo_box_set_active(GTK_COMBO_BOX(app->device_combo), default_sel);
    app->selected_device_idx = default_sel;
    gtk_widget_set_size_request(app->device_combo, 220, -1);
    gtk_widget_set_hexpand(app->device_combo, TRUE);
    gtk_box_append(GTK_BOX(ctrl), app->device_combo);
    g_signal_connect(app->device_combo, "changed", G_CALLBACK(on_device_changed), app);

    /* Start/stop */
    app->btn_startstop = gtk_button_new_with_label("STOP");
    gtk_box_append(GTK_BOX(ctrl), app->btn_startstop);
    g_signal_connect(app->btn_startstop, "clicked",
                     G_CALLBACK(on_startstop_clicked), app);

    /* ── Main plot ── */
    app->draw_area = gtk_drawing_area_new();
    gtk_widget_set_vexpand(app->draw_area, TRUE);
    gtk_widget_set_hexpand(app->draw_area, TRUE);
    gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(app->draw_area),
                                    draw_plot, app, NULL);
    gtk_box_append(GTK_BOX(vbox), app->draw_area);
}

/* ── Application activate ────────────────────────────────────────────────── */

static void on_activate(GtkApplication *gapp, gpointer data) {
    AppState *app = data;

    /* Initialise PortAudio */
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        fprintf(stderr, "PortAudio init error: %s\n", Pa_GetErrorText(err));
        return;
    }
    enumerate_devices(app);

    /* Initialise pitch detection */
    app->detector = pitch_detector_new(CHUNK_FRAMES);
    smoother_init(&app->smoother, 0.25, 3);

    /* Shared state */
    pthread_mutex_init(&app->state_lock, NULL);
    int hist_cap = (app->history_sec * SAMPLE_RATE) / CHUNK_FRAMES + 2;
    ringbuf_init(&app->history, hist_cap);
    app->current_freq = 0.0;
    app->current_conf = 0.0;

    /* Build UI */
    build_ui(app);
    gtk_application_add_window(gapp, GTK_WINDOW(app->window));
    gtk_window_present(GTK_WINDOW(app->window));

    /* Start worker thread */
    app->worker_running = true;
    pthread_create(&app->worker_thread, NULL, pitch_worker, app);

    /* Start audio on default device */
    start_audio(app, app->selected_device_idx);

    /* GTK timer for UI updates */
    g_timeout_add(DRAW_INTERVAL_MS, update_display, app);
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    AppState app = {
        .a4                 = 440.0,
        .key_idx            = 0,
        .scale_type         = SCALE_CHROMATIC,
        .smooth_alpha       = 0.25,
        .history_sec        = DEFAULT_HIST_SEC,
        .oct_lo             = DEFAULT_OCT_LO,
        .oct_hi             = DEFAULT_OCT_HI,
        .conf_threshold     = CONF_THRESHOLD,
        .actual_sample_rate = SAMPLE_RATE,
    };
    audioqueue_init(&app.audio_queue);

    GtkApplication *gapp = gtk_application_new(
        "org.pitchmonitor", G_APPLICATION_DEFAULT_FLAGS);
    g_signal_connect(gapp, "activate", G_CALLBACK(on_activate), &app);
    int status = g_application_run(G_APPLICATION(gapp), argc, argv);
    g_object_unref(gapp);
    return status;
}
