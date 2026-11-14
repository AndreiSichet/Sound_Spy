# Sound Spy — Project Insights

**An app that lets you choose what to hear.** It listens to the room through a
microphone, works out which kinds of sound are present (speech, music, dishes,
footsteps…), and makes the sounds you picked clearer while turning everything
else down.

Written: 14 November 2026, after step 01 · Last updated: 14 November 2026, after step 02
(audio capture).

> **Read this with its date in mind.** Sound Spy is built one step at a time.
> At the moment the *skeleton* of the app is finished and running, and it can
> record from a real microphone, but it does not yet process sound live or
> remove anything. Every section below says whether it describes
> something that **exists today** or something that is **planned**, and in
> which step. When a later step changes something written here, the old
> sentence should be corrected where it stands rather than quietly deleted.

---

## Who this document is for

This document is for anyone who wants to understand the project without
having followed its development. You don't need to be a programmer. It
explains every technical term the first time it is used, and Chapter 2 is a
glossary.

After reading it you should be able to explain:

- what Sound Spy is supposed to do, and how much of it exists today;
- what the separate parts are, and why they are separate;
- how sound is represented inside a computer;
- how a microphone is chosen, how recordings are stored, and how to tell whether the microphone
  gives honest sound (Chapter 8);
- how the website and the Python program talk to each other;
- how to start everything, and how to check that it works.

It goes alongside [README.md](README.md), which is the short version. Two
more technical documents go deeper: [docs/architecture.md](docs/architecture.md)
(the exact rules and message formats) and [docs/decisions.md](docs/decisions.md)
(a dated list of every important choice).

---

## Table of contents

1. [The project in one page](#1-the-project-in-one-page)
2. [Glossary](#2-glossary)
3. [How sound lives inside a computer](#3-how-sound-lives-inside-a-computer)
4. [How the whole app fits together](#4-how-the-whole-app-fits-together)
5. [The core building blocks](#5-the-core-building-blocks)
6. [The processing engine](#6-the-processing-engine)
7. [Configuration — settings live in files, not in code](#7-configuration--settings-live-in-files-not-in-code)
8. [Audio capture — getting sound in from a microphone](#8-audio-capture--getting-sound-in-from-a-microphone)
9. [The backend's web interface — REST and WebSocket](#9-the-backends-web-interface--rest-and-websocket)
10. [The database](#10-the-database)
11. [The website](#11-the-website)
12. [The command line tool](#12-the-command-line-tool)
13. [Following a request from start to finish](#13-following-a-request-from-start-to-finish)
14. [What comes next — the plan, step by step](#14-what-comes-next--the-plan-step-by-step)
15. [Running everything](#15-running-everything)
16. [Testing everything](#16-testing-everything)
17. [Problems found and fixed](#17-problems-found-and-fixed)
18. [Design decisions and why](#18-design-decisions-and-why)
19. [Known limitations](#19-known-limitations)
20. [Third-party code](#20-third-party-code)
21. [Quick reference](#21-quick-reference)

---

## 1. The project in one page

### What it will do

Imagine sitting in a crowded university cafeteria. Someone is talking to you,
but there is chatter from every table, music from a speaker, plates clattering
and chairs scraping. Sound Spy is meant to let you say *"I want to hear
speech"*. It then makes speech clearer and turns the dishes, music and chairs
down.

The app has **three tabs**. Each one is a different way of telling it what you
want to hear:

| Tab | You choose… | The app… | Needs |
|---|---|---|---|
| **1. Semantic** | one or more *types* of sound, e.g. speech | finds those types in the mix and keeps them, turning the rest down | any microphone |
| **2. Spatial** | sound types **and a direction**, e.g. "speech, from my left" | listens toward that direction, then keeps the chosen types | a microphone *array* |
| **3. Sources** | one **specific source** on a map, e.g. "that person" | follows that source and keeps only it | a microphone array |

"Semantic" means *by meaning*: you choose by what kind of sound it is. Tab 1 is
built first and works with the laptop's built-in microphone. Tabs 2 and 3 need
a **microphone array**, a small board with several microphones at known
positions (explained in Chapter 2). The project will use a ReSpeaker XVF3800
array, which has not arrived yet.

The cleaned-up sound plays live through the laptop's headphones or speakers.

### What it is made of

Two programs that run side by side on the same laptop:

1. **The backend** (Python). This program does all the real work. It records
   from the microphone, processes the sound, plays the result, stores
   recordings, and answers questions from the website. It lives in
   `backend/`.
2. **The frontend** (a website built with React). This is what you see and
   click: the tabs, the buttons, and later the live sound visualisations. It
   lives in `frontend/` and runs in your web browser.

"Front" and "back" refer to what the user sees (front) versus what happens
behind the scenes (back). Keeping both in one repository is called a
**monorepo**.

### Where it stands today

| | Status |
|---|---|
| The folder structure, rules and tools | **done** |
| The data types every part of the code shares | **done** (Chapter 5) |
| The skeleton of the processing engine | **done** (Chapter 6) |
| The backend's web interface: health, configuration, live channel | **done** (Chapter 9) |
| The database tables | **done** (Chapter 10) |
| The website: three tabs, connection indicator | **done** (Chapter 11) |
| Choosing a microphone, recording it, checking its levels | **done**, step 02 (Chapter 8) |
| Live monitoring and playback | planned, step 03 |
| Recognising sound types | planned, step 05 |
| Keeping the chosen sounds, removing the rest | planned, steps 06–07 |
| Direction and source tracking (tabs 2–3) | planned, modules 3–5 |

So today you can start both programs and open the website. You see the three
tabs and a green "connected" light, and the Semantic tab lists the seven sound
types. The tabs do not process any sound yet. That is deliberate: step 01 was
about building a frame solid enough that every later step only fills it in.

*Since step 02:* from the command line you can also list the microphones
(`soundspy devices`), test one (`soundspy check-mic`) and make recordings
(`soundspy record`), and the backend lists those recordings at
`GET /api/recordings`. The website doesn't show them yet; that comes with the
live session in step 03.

### Why it exists

It is a university project (Telecommunications, Technical University of
Cluj-Napoca). The ideas behind the sound-separation part come from published
research, mainly a project called **SemanticHearing** (Veluri et al., 2023;
see Chapter 14). Sound Spy borrows its ideas but, so far, none of its code.

---

## 2. Glossary

The terms are grouped by topic rather than listed alphabetically. Everything
later in the document assumes them.

### Sound and signal processing

**Sound wave.** Sound is air pressure going rapidly up and down. A microphone
turns that pressure into an electrical voltage that goes up and down the same
way.

**Sample.** A single measurement of that voltage at one instant, stored as a
number. A recording is just a long list of samples.

**Sample rate.** How many samples are taken per second, measured in hertz
(Hz). 48,000 Hz (48 kHz) means 48,000 numbers per second. A higher rate can
capture higher-pitched sounds. Speech models often work at 16 kHz, laptop
microphones usually record at 48 kHz.

**Channel.** One microphone's stream of samples. A normal laptop microphone
gives one or two channels. A 4-microphone array gives four, one per
microphone.

**Frame.** In this project, one moment in time across all channels: one sample
from each microphone, taken at the same instant. A 4-channel recording has
four numbers per frame.

**Mono / stereo / multichannel.** One channel, two channels, or more than two.

**Chunk.** A short slice of audio, for example 1,024 frames, which is about
21 milliseconds at 48 kHz. The app processes sound one chunk at a time,
because live sound can't wait for a whole recording to finish.

**float32.** The way each sample is stored: a decimal number between −1.0 and
+1.0, using 32 bits of memory.

**Decibel (dB).** A logarithmic scale for loudness, where each 20 dB step means
a sound wave ten times bigger. Turning something down by **30 dB** shrinks it
to about 3% of its size. To the ear it sounds very quiet, though not silent.

**dBFS (decibels relative to full scale).** Loudness measured against the
loudest value a sample can hold (1.0), which is 0 dBFS. Everything quieter is
negative, so −20 dBFS is quiet and −1 dBFS is very loud.

**RMS and peak.** Two ways to describe how loud a chunk is. **Peak** is the
single biggest sample. **RMS** ("root mean square") is a kind of average that
matches perceived loudness much better.

**Clipping.** When a sound is too loud to fit in the allowed range, its peaks
are cut flat. You hear this as harsh distortion, and it can't be undone, so
the app will warn about it.

**Frequency.** How fast a sound wave vibrates: low hum versus high whistle.

**Spectrum / spectrogram.** A spectrum shows how much of each frequency is in
a sound at one moment. A **spectrogram** puts many spectra side by side: time
runs left to right, frequency bottom to top, and brightness shows loudness.
It is the standard "picture" of a sound, and the app will draw one live.

**STFT (short-time Fourier transform).** The mathematical operation that
turns a stream of samples into a spectrogram. It cuts the sound into short,
overlapping pieces and finds the frequencies in each piece. The **inverse
STFT** turns a spectrogram back into sound.

**Mel scale / log-mel spectrogram.** A spectrogram rearranged to match human
hearing, which separates low pitches finely and high pitches coarsely. This
is the usual input to sound-recognition models.

**Mask.** A grid of numbers between 0 and 1, one for each time-and-frequency
cell of a spectrogram. Multiplying by the mask keeps the cells near 1 and
silences the cells near 0. This is the classic way to "remove" sounds.

**Wiener filter.** A well-known recipe for choosing mask values from an
estimate of how much of each cell is wanted sound and how much is noise.

**Attenuation.** Turning a sound down. "Attenuate the rest by 30 dB" means
reduce everything you didn't choose to about 3% of its size.

**SNR (signal-to-noise ratio).** How much louder the wanted sound is than
everything else, in dB.

**SI-SDR (scale-invariant signal-to-distortion ratio).** A score, in dB, for
how close a cleaned-up sound is to the true clean sound. "Scale-invariant"
means it ignores overall volume, so a cleaned sound that is just quieter
isn't punished. **SI-SDRi** (the *improvement*) is the score after cleaning
minus the score before. It is the main way extraction quality will be
measured.

**Microphone array.** Several microphones at known, fixed positions on one
board. A sound from the left reaches the left microphone a fraction of a
millisecond before the right one, and that tiny delay reveals the direction.

**Mic geometry.** The positions of the microphones in an array, in metres.
Direction-finding needs them.

**DOA (direction of arrival).** Working out which direction a sound is coming
from, using those tiny delays between microphones.

**Beamforming.** Combining an array's microphones so that sound from one
chosen direction adds up while sound from other directions partly cancels.
It works like pointing an invisible directional microphone.

### Machine learning

**Model.** A program whose behaviour was learned from examples instead of
being written by hand. You give it a piece of audio, and it gives back, say,
the chance that speech is present.

**Training.** Showing a model many examples whose answers are known, until it
learns to produce the answers itself.

**Inference.** Using an already-trained model on new input. This is the
opposite of training, and it is what happens live in the app.

**Pretrained model / fine-tuning.** A pretrained model was already trained by
someone else on a huge dataset. Fine-tuning means training it a little more
on our own data so it learns *our* sound types. This is much cheaper than
starting from nothing.

**Label / class.** The name of a category, such as "speech" or "dishes". Sound
Spy has seven classes (Chapter 5).

**Multi-label.** More than one class can be true at once: a cafeteria has
speech *and* dishes *and* music at the same time. The detector (step 05)
therefore answers "how likely is each class?" separately for every class,
rather than picking one.

**Multi-hot vector.** A list with one slot per class, holding 1 for "yes" and
0 for "no". With the classes in the order speech, music, footsteps, dishes,
chair scraping, alarm, traffic, the vector `[1, 0, 0, 1, 0, 0, 0]` means
"speech and dishes". Models read and write class choices in this form.

**AudioSet / ontology.** AudioSet is a large public collection of labelled
sound clips from Google, organised as a tree of over 500 sound categories
("Speech", "Dishes, pots, and pans", "Vehicle"…). That tree is called an
*ontology*. Sound Spy maps each of its seven classes onto names in that
tree, so that clips from different public datasets can be pooled together.

**Dataset.** A collection of labelled examples. The project plans to use
public ones (ESC-50, FSD50K) plus its own recordings.

**Training / validation / test split.** The examples are divided into three
piles. The model *learns* from the training pile. Its settings are *chosen*
using the validation pile. The test pile is used *once, at the end*, to
report an honest score. A model scored on examples it has already seen looks
much better than it really is.

**Leakage.** Accidentally letting information about the answer slip into the
input. One example: the same original recording ends up in both the training
and test piles. Leakage makes a model look excellent on paper and useless in
real life.

**Synthetic mixture.** An artificial training example made by adding a clean
"target" sound (say, speech) to "background" sounds at a chosen loudness.
Because we made the mix, we know exactly what the clean answer should be.

**Target sound extraction (TSE).** Pulling one chosen kind of sound out of a
mixture. A *label-conditioned* TSE model receives the mixture plus a
multi-hot vector saying what to keep.

**Causal / streaming.** A causal model only uses sound that has already
happened, never the future. That is what lets it run live, chunk by chunk.

**Latency.** The delay between a sound happening and the processed version
reaching your ears. For a hearing app it has to stay small.

### Web and network

**Server / client.** A server is a program that waits for requests and
answers them. A client is the program that asks. Here, the backend is the
server and the website is the client.

**localhost / 127.0.0.1.** The computer's own address for talking to itself.
`127.0.0.1` is the numeric version (IPv4). `localhost` is a name that may
point to `127.0.0.1` or to `::1`, the newer IPv6 version of the same idea.
This difference caused a real problem (Chapter 17).

**Port.** A number that tells programs on one computer apart, like the room
number within a building's address. The backend uses port **8765**, and the
website's development server uses **5173**.

**API (application programming interface).** A set of addresses that another
program can call to get data or make something happen. The answer is data,
not a web page.

**REST / endpoint.** REST is a common style of API built on ordinary web
requests. An endpoint is one address in it, for example `GET /api/health`.

**HTTP methods.** **GET** means "give me something, change nothing". **POST**
means "create or start something". **PUT** means "replace something".

**HTTP status code.** A three-digit number in every reply that says how the
request went:

| Code | Meaning |
|---|---|
| 200 | OK |
| 404 | not found |
| 422 | the request was understood but its contents were invalid |
| 501 | **not implemented** — the address exists, the feature comes in a later step |
| 503 | a service we depend on is unavailable |

**JSON.** A plain-text format for structured data that almost every
programming language understands, for example
`{"status": "ok", "version": "0.1.0"}`.

**WebSocket.** A connection that stays open so *both* sides can send
messages whenever they like. A normal request is a single question and a
single answer. The app uses a WebSocket for live data, because the server
needs to push many updates per second without being asked each time.

**Heartbeat.** A small "I'm still here" message sent at regular intervals.
The backend sends one every 2 seconds on the WebSocket.

**CORS (cross-origin resource sharing).** A browser safety rule. By default, a
web page from one address may not read data from another address. The
backend explicitly allows the development website's address and no other.

**Proxy.** A go-between that forwards requests. In development, the website's
server forwards anything starting with `/api` or `/ws` to the backend. To the
browser, everything then appears to come from one address.

### Python and the backend

**Package.** A folder of Python code that can be installed and imported. The
backend's package is called `soundspy`.

**Virtual environment (venv).** A private, self-contained Python installation
for one project. Its libraries can't clash with other projects. Here it lives
in `backend/.venv`.

**Editable install.** `pip install -e .` installs the package so that Python
reads the files straight from the folder, and edits take effect without
reinstalling.

**Dataclass.** A compact way to define a Python class that mainly holds data,
for example an `AudioBuffer`. *Frozen* means it can't be changed after it is
created.

**Abstract base class (ABC) / interface.** A class that describes *what*
methods something must have, without saying *how* they work. Real classes then
fill in the "how". For example, every processing stage must have a
`process(chunk)` method.

**FastAPI.** The Python library used to build the backend's web interface.

**uvicorn.** The program that actually runs a FastAPI app and listens on a
port.

**Pydantic.** A library that checks data has the right shape. FastAPI uses it
to validate incoming requests and to describe responses.

**SQLite.** A complete database stored in a single file
(`data/soundspy.db`). There is no separate database server to install.

**SQLAlchemy / ORM.** SQLAlchemy is the library that connects Python to the
database. An ORM ("object-relational mapper") lets you work with Python
objects instead of writing database commands by hand: one object is one row
in a table.

**Thread.** One line of work inside a running program. A program can have
several threads running at the same time, for example one capturing sound
while another processes it.

**Queue.** A waiting line used to pass data safely from one thread to another.

**asyncio / event loop.** Python's system for juggling many slow tasks, such
as network connections, without waiting on each one in turn. The web server
runs on an event loop.

### The website

**React.** A library for building web interfaces out of reusable pieces called
**components**, such as a tab, a button or the connection indicator.

**TypeScript.** JavaScript with types added. You declare, for example, that a
value must be a number, and mistakes are caught before the code runs.

**Hook.** A React function, named `useSomething`, that lets a component
remember things or react to events. One example is `useSessionSocket`, which
opens the WebSocket.

**Store (Zustand).** One shared place where the website keeps its current
state: is the backend reachable, which tab is open, what the server last
said. Any component can read it, and every component that uses a value updates
when that value changes.

**Vite.** The tool that runs the website during development (with instant
reload on save) and packs it into small files for production.

**Tailwind CSS.** A styling system where you describe the look directly on
each element using short class names, such as `bg-slate-950` for a very dark
background.

### Audio capture (added in step 02)

**PortAudio / sounddevice.** PortAudio is a widely used library for talking to sound cards on
any operating system. `sounddevice` is the Python package that wraps it. Only Sound Spy's `io/`
package uses it.

**Host API.** One of the routes by which Windows lets a program reach the sound hardware:
**MME** (oldest; resamples to 44.1 kHz and truncates names to 31 characters), **DirectSound**,
**WASAPI** (modern, native rate, low delay) and **WDM-KS** (almost direct to the driver). The same
microphone is listed once per host API.

**Twin.** The same physical device seen through different host APIs.

**Device profile.** A small settings file saying which microphone to use and how (Chapter 8.2).

**`raw_channels`.** The list of a device's channels that are real microphones and should be kept
(Chapter 8.3).

**Callback.** A function the sound card calls on its own thread each time a new block of samples
is ready. It must be quick.

**Overflow.** The sound card had new samples but the program hadn't collected the previous ones
in time, so some were lost.

**Stall.** The stream is open but no sound arrives at all, usually because another program holds
the microphone or Windows blocks access to it.

**Sidecar.** A small JSON file stored next to a recording, with the same name, describing it:
device, sample rate, label, notes and so on.

**Noise floor.** The level of the background when nothing is happening: room tone, fans, the
microphone's own electronic hiss. Every real microphone has one.

**Noise suppression / noise gate / AGC.** Processing that "tidies up" sound before an app sees
it. *Noise suppression* removes steady background noise, a *noise gate* silences everything
below a threshold, and *AGC* (automatic gain control) keeps changing the volume. Windows "audio
enhancements" and "Voice Clarity" do this. Fine for video calls; harmful for Sound Spy, which
needs the honest sound.

**Cross-correlation / lag.** A way of comparing two signals by sliding one against the other and
finding where they match best. The amount of sliding is the *lag*. Between two microphones, the
lag shows which one heard a sound first.

**Duplicated mono.** One signal copied into several channels, so they are identical. It looks
like stereo but contains no spatial information.

### Testing and tooling

**Test.** A small program that checks a piece of the real program behaves as
expected. They run automatically and quickly.

**pytest / Vitest.** The test runners for Python and for the website.

**Fake (test double).** A stand-in for something tests must not touch for
real. Here, a fake `sounddevice` pretends to be the sound card, so no test
ever needs a microphone.

**Fixture.** A ready-made thing a test needs, such as a temporary database,
prepared automatically before the test runs.

**Lint / ruff / ESLint.** Tools that read code without running it and flag
mistakes and style problems.

**Type check.** Running the TypeScript compiler purely to check the types
match.

**Stub.** A placeholder: a file that contains only a description of its
future job and a `TODO(step-XX)` note saying when it will be written.

---

## 3. How sound lives inside a computer

This chapter is short but important, because the whole app is built around
one decision it describes.

### From air to numbers

A microphone measures air pressure tens of thousands of times per second.
Each measurement becomes one number between −1.0 and +1.0, where 0 is
silence. At 48 kHz, one second of sound from one microphone is a list of
48,000 numbers.

With several microphones, every instant produces one number *per microphone*.
The natural way to store that is a table:

```
              mic 1     mic 2     mic 3     mic 4
  frame 0     0.012    0.010     0.013     0.011
  frame 1     0.019    0.017     0.021     0.018
  frame 2     0.025    0.024     0.026     0.025
  ...
```

Each **row is one moment** (a *frame*) and each **column is one microphone**
(a *channel*). The four columns look almost identical, but the tiny
differences between them are exactly what will later reveal *where* a sound
came from.

### The one audio format: `AudioBuffer`

Every piece of audio in Sound Spy travels as an `AudioBuffer`, which is that
table plus two labels:

| Part | What it holds |
|---|---|
| `samples` | the table: rows = frames, columns = channels, each value a float32 |
| `sample_rate` | how many rows make one second, e.g. 48,000 |
| `metadata` | free notes: which device, which profile, when it was recorded… |

**The table always has two dimensions, even for a single microphone.** A
mono recording is a table with one column, not a flat list. That sounds
pedantic, but it is the reason the laptop microphone today and the
4-microphone array later can go through the *same* code. No function is
allowed to assume how many columns there are.

### Why "never throw channels away" is a rule

The simplest way to handle multichannel sound is to average the channels
into one, or keep only the first (`audio[:, 0]` in Python). Doing that early
would quietly destroy the direction information Tabs 2 and 3 depend on.

So the project has a rule: channels may be reduced in **only two places**.
One is a component called the `ChannelReducer`, which averages or selects
channels when a later stage genuinely needs a single channel. The other is
the beamformer, which reduces channels *on purpose*, by aiming at a
direction. Anywhere else, it counts as a bug. (The `ChannelReducer` exists since
step 02, Chapter 8.7, and an automatic test now scans the code for `[:, …]`-style
channel indexing outside the few files allowed to use it.)

---

## 4. How the whole app fits together

### The big picture

```
   ┌────────────────────────── Backend (Python, 127.0.0.1:8765) ──────────────────────────┐
   │                                                                                     │
   │  Microphone ─► AudioSource ─► Engine (processes the sound chunk by chunk)            │
   │  (or a file)                    │                                                    │
   │                                 ├─► SpatialFilter  points toward a direction (tab 2/3)│
   │                                 ├─► Extractor      keeps chosen sounds ──► AudioSink ─┼─► headphones
   │                                 │                                                    │
   │                                 ├─► Detector  (side)  which sounds are present ─┐    │
   │                                 ├─► Localizer (side)  where they are             ├─► live messages
   │                                 └─► Analyzer  (side)  loudness, spectrogram    ─┘    │
   │                                                                                     │
   │  Web interface: health, settings, start/stop, recordings ◄──► SQLite database        │
   └─────────────────────────────────────────────────────────────────────────────────────┘
                              ▲   web requests + one WebSocket
                              │
   ┌──────────────────────────┴────────── Website (React, localhost:5173) ────────────────┐
   │  tabs, buttons, connection light, and later live pictures of the sound              │
   └─────────────────────────────────────────────────────────────────────────────────────┘
```

Read it from the microphone downward:

1. Sound comes in through an **AudioSource**: the microphone, or a recorded
   file played back as if it were live.
2. The **engine** cuts it into chunks and passes each chunk along a line of
   stages.
3. Stages in the main line **change** the sound. The spatial filter (later)
   focuses on a direction, and the extractor keeps the chosen sound types.
4. Side stages only **look** at the sound and report what they notice: which
   sounds are present, where they are, how loud it is.
5. The changed sound leaves through an **AudioSink**, which plays it on the
   headphones or saves it to a file.
6. The side stages' reports travel to the website over the WebSocket, so it
   can draw them live.

Today, the parts that exist are the web interface, the database, the website
and the empty frame of the engine. The sound path itself starts in step 02.

*Update after step 02:* the first piece of the sound path now exists. The
microphone and file **AudioSources** work, as does a file **AudioSink**
(Chapter 8). What doesn't exist yet is the engine running live between them
(step 03) and the stages that change the sound (steps 05–07).

### Why two programs instead of one

**The backend owns all the sound.** The browser never records from the
microphone and never plays live sound. Two reasons:

- **Browsers "improve" microphone audio on their own.** They apply echo
  cancellation, noise suppression and automatic volume control. For a
  project whose whole purpose is to control what you hear, an invisible
  filter applied before our code ever sees the sound is unacceptable.
- **Browsers handle multi-microphone devices poorly.** The 4-microphone
  array needs all its channels, unmodified.

So the website is only a **remote control and a display**. It sends choices
to the backend and shows what the backend reports. The one exception
(planned) is playing back *saved* files, to compare "before" and "after".

### The rule that keeps the core independent

Inside the backend there is a further split:

- **The library**: everything that knows about sound (data types, engine,
  processing, models). It doesn't know a website exists.
- **The API layer** (`soundspy/api/`): a thin shell that turns web requests
  into library calls.

The API may use the library, but the library must never use the API or its
web libraries (FastAPI, Starlette, SQLAlchemy). A test enforces this
(Chapter 16). The benefit is that the sound-processing code can be used,
tested and studied on its own: from the command line, in a notebook, or in a
university report, with no server running.

### The two speeds

The project runs on two very different clocks. Mixing them up is the most
common source of confusion.

**The slow clock: preparing data and training models.** This means
downloading datasets, building training mixtures and fine-tuning models. It
takes hours and happens rarely. Its result is a set of model files on disk.

**The fast clock: live listening.** A chunk arrives every ~20 ms and must be
processed before the next one arrives. It uses the model files the slow clock
produced and never changes them.

The only thing that passes between the two clocks is files.

---

## 5. The core building blocks

**Exists today.** Location: `backend/src/soundspy/core/`.

These are the shared "nouns" of the whole project. They contain no audio
processing at all. They are agreements about what the data looks like, which
every other part relies on. Getting them right first is what lets the later
steps be written independently.

### 5.1 `AudioBuffer` — a piece of sound

Described in Chapter 3. In code it is a *frozen dataclass*: once created, it
can't be changed. It also checks itself when it is created and refuses
anything malformed:

- `samples` must be a NumPy array (a fast table of numbers) of float32 values;
- it must have exactly two dimensions and at least one channel;
- `sample_rate` must be a positive whole number.

It answers three questions: `num_frames`, `num_channels`, and `duration_s`
(the length in seconds). It deliberately has **no** "convert to mono" method,
because that would invite exactly the channel-dropping the rules forbid.

### 5.2 `Direction` — where to listen

Two numbers:

- **`azimuth_deg`**: the compass direction, from 0 up to (but not including)
  360 degrees. Values outside that range are wrapped around automatically,
  so 370° becomes 10° and −90° becomes 270°.
- **`width_deg`**: how wide the listening "beam" is, more than 0 and at most
  360 degrees. The default is 30°.

### 5.3 `TargetSpec` — what the user wants

This is the single object that says what the user wants to hear, and it
serves all three tabs. Each tab fills in different fields, and each
processing stage reads only the fields it cares about:

| Field | Set by | Meaning |
|---|---|---|
| `labels` | Tab 1 (and 2) | which sound types to keep. **Empty means "let everything through"** |
| `direction` | Tab 2 | where to listen |
| `source_id` | Tab 3 | which tracked source to follow |
| `attenuation_db` | all | how strongly to turn the rest down; default **30 dB**, never negative |

A `TargetSpec` can check itself against the list of known classes and
refuses a name it doesn't recognise, such as `"laughter"`. A typo therefore
fails loudly instead of silently matching nothing.

Having **one** such object, rather than a separate one per tab, means a
processing stage never needs to know which tab is open.

### 5.4 Events — what the side stages report

Every report carries `t`: the number of seconds since the listening session
started. There are five kinds:

| Event | What it says | Used for |
|---|---|---|
| `Detection` | "class X is present with probability p between time a and b" | which sounds are present (step 05) |
| `SoundSource` | "a source of type X is at direction D with this activity" | the source map (module 5) |
| `LevelFrame` | loudness per channel: RMS, peak, clipping yes/no | level meters (step 03) |
| `SpectrogramFrame` | one vertical slice of the spectrogram | the live spectrogram (step 03) |
| `EngineStatus` | idle / running / error, plus device, sample rate, channels | the status shown on the website |

Only `EngineStatus` is actually sent today: the engine always reports
"idle", because there is no live session yet.

### 5.5 The seven sound classes — `LabelSet` and `classes.yaml`

The list of sound types the user can choose lives in one settings file,
`backend/configs/classes.yaml`:

| ID (internal) | Shown to the user as | Tooltip | AudioSet names (data only) |
|---|---|---|---|
| `speech` | Speech | People talking: conversations, voices, announcements | Speech |
| `music` | Music | Songs and instrumental music, live or from speakers | Music |
| `footsteps` | Footsteps | People walking: steps on hard floors | Walk, footsteps |
| `dishes` | Dishes & cutlery | Plates, cups, pots and cutlery clinking | Dishes, pots, and pans; Cutlery, silverware |
| `chair_scrape` | Chair scraping | Chairs and furniture dragged or squeaking on the floor | Scrape; Squeak |
| `alarm` | Alarms | Alarms, beeps and ringing signals | Alarm |
| `traffic` | Traffic | Road traffic and passing vehicles | Traffic noise, roadway noise; Vehicle |

Each class has **four separate pieces of information**, and keeping them
apart is a deliberate design choice:

- **The ID** (`dishes`) is what the code, the models and the database use. It
  **never changes**, because trained models and stored records refer to it.
- **The display name** ("Dishes & cutlery") is what the website shows. It can
  be reworded freely.
- **The description** is the tooltip text. It is also handy for the
  university report.
- **The AudioSet names** are used **only** when building training data from
  public datasets. Users never see them: "Dishes, pots, and pans" is a
  dataset category, not a friendly label.

Two of the AudioSet mappings (`chair_scrape` and `traffic`) are marked "to be
verified in step 04", because the best match in the AudioSet tree is not
obvious.

**The order of the list matters.** A class's position in the file is its slot
in every multi-hot vector. If the file were reordered, a model trained when
"dishes" was slot 4 would suddenly read slot 4 as something else. New classes
are therefore always added at the end, and the list is never shuffled.

`LabelSet` is the Python object built from that file. It can convert a class
name to its position and back, turn a set of names into a multi-hot vector
and back (with a threshold, so that a probability of 0.5 or more counts as
"yes"), and reject unknown names.

---

## 6. The processing engine

The *contracts* below **exist today**. The machinery that runs them live is
planned for step 03. Location: `backend/src/soundspy/engine/`.

### 6.1 One idea: everything is a stage that handles one chunk

Every step of the processing is a small object with a simple job: *take one
chunk, return one chunk*. There are two kinds:

- **`Processor`**, in the main line. It **changes** the sound, as a
  beamformer or an extractor does. It can even change the channel count:
  a beamformer takes four channels in and gives one out.
- **`SideProcessor`**, an observer. It **looks** at the sound without
  changing it and returns a list of events, as the detector, analyser and
  localiser do.

Both kinds can be told about the current `TargetSpec`, and both can be
**reset** to forget what they remember. "Remember" matters here: these stages
are *stateful*. A filter, for example, has to remember the end of the
previous chunk to process the next one smoothly.

**Why chunk by chunk, even for a recording?** Because then there is only one
way of doing things. Processing a saved file simply means feeding it through
the same stages chunk by chunk, as if it were coming from a microphone. When
real-time listening arrives, nothing has to be redesigned. It's the same
code, just fed faster. The rule is strict: **no stage may require the whole
file.**

A trivial stage called `Passthrough` already exists. It returns the chunk
unchanged and stands in for stages that haven't been written yet.

### 6.2 The `ProcessorChain` — stages in a line

The chain holds the main stages in order, plus any side stages. For each
chunk it:

1. shows the **incoming** chunk to every side stage (so the detector hears
   what the microphone heard, not the cleaned-up version) and collects their
   reports;
2. passes the chunk through the main stages in order;
3. returns the processed chunk **and** all collected reports.

Before it can run, the chain must be **configured** with the sample rate and
channel count. Each main stage is told what it will receive and answers how
many channels it will produce, and that answer is passed to the next stage.
The chain then knows the final output format before any sound flows.

The chain is careful by design. It refuses to run if it hasn't been
configured, and it refuses any chunk whose sample rate or channel count
doesn't match the configuration. This guard has already caught a real mistake
(Chapter 17).

### 6.3 Modes — which stages each tab uses

| Mode (tab) | Main stages, in order | Side stages | Needs an array? |
|---|---|---|---|
| **Semantic** | extractor | detector, analyser | no |
| **Spatial** | spatial filter → extractor | detector, analyser, localiser | **yes** |
| **Sources** | spatial filter → extractor | detector, analyser, localiser, tracker | **yes** |

Today only this table exists. Building real chains from it comes later. The
website already uses it to decide whether a tab is available (Chapter 11).

### 6.4 How live processing will run (planned, step 03)

Live audio needs several things to happen at once, so the work will be split
across threads:

```
 sound card ──► input queue ──► engine worker thread ──► output queue ──► headphones
                                         │
                                         └─► reports ──► web server's event loop ──► WebSocket ──► website
```

- The **sound card** calls our code every few milliseconds with fresh
  samples. That code only drops the samples into a queue. Anything slower
  could make the sound stutter.
- The **engine worker** takes chunks from the queue, runs the chain, and puts
  the result in the output queue for playback.
- The side stages' **reports** are handed over to the web server's event loop
  (using a Python function made for passing work safely between threads,
  `call_soon_threadsafe`), which sends them to every connected browser.
- Only **one session** runs at a time. When the user changes their choice,
  the new `TargetSpec` takes effect at the start of the next chunk, never
  halfway through one.

---

## 7. Configuration — settings live in files, not in code

**Exists today.** Location: `backend/configs/`, read by
`backend/src/soundspy/config.py`.

### 7.1 The rule and the reason

Sample rates, chunk sizes, file paths, ports, the class list and the model
settings all live in settings files written in **YAML** (a simple
human-readable text format), not inside the Python code. Changing a value
means editing a text file, not code, and the same code can run with different
settings. The `TODO(step-XX)` markers in these files show which values are
only placeholders for now.

### 7.2 The files

| File | What it holds |
|---|---|
| `default.yaml` | folder paths, which microphone profile is active, engine chunk size (1,024 frames), backend address and port, allowed website address, heartbeat interval (2 s), database location |
| `classes.yaml` | the seven classes (Chapter 5.5) |
| `devices/laptop.yaml` | the laptop's own microphone: system default device, preferred host API, native sample rate, all channels, no geometry |
| `devices/respeaker_xvf3800.yaml` | the future 4-mic array: a device name to look for, channels 2–5 (the raw microphones; channels 0–1 are the board's own processed output), geometry to be filled in |

Since step 02 a profile has five fields (`device`, `hostapi`, `sample_rate`,
`raw_channels`, `geometry`) and is checked when it is loaded. Chapter 8.2 explains
each field.
| `detection/default.yaml` | the future sound detector (model not yet chosen) |
| `extraction/mask_baseline.yaml` | the future classical extractor (spectrogram settings, mask type) |
| `extraction/neural_tse.yaml` | the future neural extractor |

### 7.3 Device profiles — swapping microphones without touching code

A **device profile** describes one microphone setup: which device to open,
which of its channels are real microphones, and where those microphones sit.
Switching from the laptop microphone to the array will mean changing one line
in `default.yaml` (`active_device_profile: respeaker_xvf3800`), not editing
the program. For a single run, `--profile respeaker_xvf3800` on any audio command
does the same.

The profile also decides what the website offers. The laptop profile has no
geometry, so the backend reports Tabs 2 and 3 as unavailable and the website
says *"Requires a microphone array"*.

### 7.4 Where the "project root" is

Paths in the settings, such as `data/raw`, are measured from the **project
root**: the folder that contains both `backend/` and `frontend/`. So
recordings, models and outputs live at the top level, shared by everything,
instead of being buried inside `backend/`.

```
Sound_Spy/                 ← project root
├── backend/               the Python program
├── frontend/              the website
├── data/                  recordings, datasets, the database file   (not in git)
├── models/                downloaded model weights                  (not in git)
├── outputs/               processed audio, plots                    (not in git)
├── experiments/           one folder per training run
├── docs/                  architecture, decisions, literature
└── notebooks/             exploration only
```

`data/`, `models/` and `outputs/` are kept out of version control. They hold
large files that can be rebuilt or downloaded again, and recordings that
shouldn't be shared.

---

## 8. Audio capture — getting sound in from a microphone

**Exists today** (step 02). Location: `backend/src/soundspy/io/`, `backend/src/soundspy/dsp/`,
and the `devices`, `check-mic` and `record` commands.

This is the first part of Sound Spy that touches real sound. It answers four questions: *which
microphone?*, *how does its sound reach our code?*, *how is a recording stored?* and *is the
microphone actually giving us honest sound?* The last one turned out to matter more than
expected (8.8).

### 8.1 The short version

```
 device profile (laptop.yaml)                       which mic, which channels, which rate
        │
        ▼
 resolve_input()  ──►  "Microphone Array (Realtek(R) Audio), WASAPI, index 9, 48 kHz, channels 0–1"
        │
        ▼
 MicSource  ──►  read(5 s)  ──►  one AudioBuffer (240,000 frames × 2 channels)
        │                              │
        └─►  chunks(…)  ──►  a stream of small AudioBuffers (for live use, step 03)
                                       │
                                       ▼
                          save_recording()  ──►  test_20261114-160527-863.wav
                                                  test_20261114-160527-863.json  (the sidecar)
                                       │
                                       ▼
                          the API indexes the sidecar  ──►  GET /api/recordings
```

### 8.2 Choosing a microphone

**Host APIs, and why every microphone appears several times.** Windows offers several *host
APIs*: different routes by which a program can talk to the sound hardware. **MME** is the oldest,
**DirectSound** came next, **WASAPI** is the modern one, and **WDM-KS** talks to the driver
almost directly. The library underneath (PortAudio, used through `sounddevice`) lists every
device once *per host API*. On the development laptop the built-in microphone appears four
times:

```
  idx  ch   rate  host API              name
    9   2  48000  Windows WASAPI        Microphone Array (Realtek(R) Audio)
    5   2  44100  Windows DirectSound   Microphone Array (Realtek(R) Audio)
 *  1   2  44100  MME                   Microphone Array (Realtek(R) Au
   15   2  44100  Windows WDM-KS        Microphone Array (Realtek HD Audio Mic input)
```

(`*` marks the system default.) These are **twins**: one physical microphone reached through
four different doors. Notice two things:

- The **system default is the MME twin**, because PortAudio takes its defaults from MME on
  Windows. MME resamples everything to 44.1 kHz and cuts names to **31 characters**
  ("…Realtek(R) Au").
- The **WASAPI twin** reports the microphone's real rate, **48 kHz**, and has lower delay.

So Sound Spy uses a **preference order**: WASAPI, then DirectSound, then MME, then WDM-KS (Core
Audio and ALSA on Mac and Linux). When the profile says "the system default", the app finds the
default's twins by comparing names case-insensitively on their first 31 characters, the part
MME keeps. It then picks the most preferred twin. On this laptop that turns MME device 1 into
WASAPI device 9.

**WASAPI's one fussy rule.** In normal ("shared") mode WASAPI only accepts the device's own
sample rate. Sound Spy therefore always opens WASAPI streams with *auto-conversion* switched on,
so a profile asking for, say, 16 kHz still works. Windows then converts the rate.

**The device profile** (`backend/configs/devices/<name>.yaml`) says which microphone to use:

| Field | Meaning | `laptop` |
|---|---|---|
| `device` | `null` = system default; a number = that device index; text = part of the name (any case) | `null` |
| `hostapi` | `null` = the preference order above; or a host API name, e.g. `"MME"` | `null` |
| `sample_rate` | `null` = the device's own rate; or a number in Hz | `null` |
| `raw_channels` | `null` = every channel; or a list of channel numbers to keep (8.3) | `null` |
| `geometry` | microphone positions in metres, one `[x, y, z]` per kept channel (module 3) | `null` |

Profiles are **checked when loaded**. An unknown field, a negative channel number, a sample rate
that isn't a positive whole number, or a geometry list whose length doesn't match the channels
is rejected with a message naming the profile, the file and the field. A name that matches two
different microphones is refused too ("'mic' matches several input devices: …"), rather than
quietly picking one.

Every audio command also takes `--profile <name>` and `--device <index or name>`, which override
the active profile for one run.

### 8.3 `raw_channels`, using the ReSpeaker as the example

The ReSpeaker XVF3800 array (arriving for module 3), with its 6-channel firmware, delivers six
channels:

| Device channel | What it carries |
|---|---|
| 0, 1 | the board's own *processed* output (its built-in beamforming and echo cancelling) |
| 2, 3, 4, 5 | the four **raw microphones** |

Sound Spy wants the raw microphones, because doing its own processing is the whole point. So the
profile says `raw_channels: [2, 3, 4, 5]`. The app then:

1. **opens** `max(raw_channels) + 1` = 6 channels, since you can't ask a sound card for
   "channels 2 to 5" only;
2. **keeps** channels 2, 3, 4, 5, in that order, the moment each block of sound arrives.

From then on the rest of the app sees a clean 4-channel `AudioBuffer`, and its channels 0–3 are
the array's microphones 1–4. Which device channels they came from is recorded in the metadata
(`channel_indices: [2, 3, 4, 5]`). If the device has fewer channels than the profile needs (for
example the wrong firmware), opening it fails with a message saying exactly that.

This is a *hardware selection*, deciding which wires are microphones. It is not a downmix, so it
is allowed in `io/` (Chapter 3's rule is about not *throwing away* microphones).

### 8.4 How sound arrives: `MicSource`

The sound card doesn't wait for us. Every few milliseconds it hands over a fresh block of samples
by calling a function of ours, the **callback**, on its own thread. The callback keeps the
profile's channels and hands the block on. It does nothing slow, because a slow callback makes
the sound card lose audio (an **overflow**, which is counted).

There are two ways to take sound out:

- **`read(seconds)`**: "give me exactly 5 seconds". It returns exactly `round(5 × 48,000)` =
  240,000 frames, no more and no less. Recording and `check-mic` use this.
- **`chunks(n)`**: "keep giving me pieces of `n` frames". These go through a small waiting line
  (a queue) that holds up to 32 pieces. If whoever is reading falls behind and the line is full,
  the **oldest** piece is thrown away and counted. For live listening, fresh sound is worth more
  than complete sound: a growing backlog would mean hearing things later and later. The live
  meter uses this, and so will the live session in step 03.

Two safety nets:

- **Stall detection.** If no sound arrives for 2 seconds, the app stops with a clear message
  ("is another app using the microphone, or is microphone access disabled in Windows privacy
  settings?") instead of waiting forever.
- **Always closing.** Leaving `chunks()` early, or calling `close()`, always stops the stream. No
  background thread is left holding the microphone.

**The start-up silence (found on real hardware).** On the development laptop, a freshly opened
WASAPI stream delivers **exactly 500 ms of digital zeros** before the real sound starts:
measured, 24,000 frames at 48 kHz, every time. An already-running stream never does this. Left
alone, every recording would begin with half a second of fake silence, and the level report
would claim a noise floor of −120 dBFS. A real microphone **never** outputs exact zeros, because
there is always a little electronic hiss. So right after opening, `MicSource` throws away leading
frames that are exactly zero. It does this for at most 1 second, so a genuinely muted microphone
still delivers its silence and gets reported as muted. The number of discarded frames is shown in
the `check-mic` report.

*Retest, same day:* with Windows audio enhancements switched off, the start-up silence
**disappeared** (`start-up silence discarded: 0 frames` on every run). The 500 ms of zeros were
the enhancement processing warming up, not the microphone. The discard stays in place as a
harmless safety net for other machines.

**Files work the same way.** `FileSource` plays a WAV file through the same two methods, so later
stages can't tell a file from a microphone. Its last piece is padded with silence to full length,
with a note saying how much of it is real (`valid_frames`). It can also loop, and can pace itself
at real-time speed, which step 03 will use to fake a live microphone from a recording.
`FileSink` does the opposite: it writes pieces to a WAV file one by one, so a long session never
has to fit in memory.

### 8.5 How a recording is stored

`soundspy record --seconds 5 --label test` produces two files in `data/raw/` with the same name:

```
test_20261114-160527-863.wav     the sound
test_20261114-160527-863.json    the "sidecar": everything known about the sound
```

The name is the label (or `rec`), then the date and time down to the millisecond. The **WAV** is
32-bit floating point, at the microphone's own rate, with every kept channel: nothing is
resampled or mixed. The **sidecar** is a small text file (this is the real one):

```json
{
  "file": "test_20261114-160527-863.wav",
  "sample_rate": 48000,
  "channels": 2,
  "duration_s": 5.0,
  "created": "2026-11-14T16:05:27.863+02:00",
  "device": "Microphone Array (Realtek(R) Audio)",
  "hostapi": "Windows WASAPI",
  "profile": "laptop",
  "channel_indices": [0, 1],
  "label": "test",
  "notes": "",
  "overflows": 0,
  "app_version": "0.1.0"
}
```

**Recordings are never overwritten.** Both files are created in "must not exist yet" mode, so a
second recording can never replace the first, even by accident.

### 8.6 How the database knows about recordings

The **sidecar is the source of truth**, and the database is only an **index** of the sidecars,
like a library catalogue. The catalogue can always be rebuilt by walking the shelves.

- When the backend starts, and every time the website asks for `GET /api/recordings`, it walks
  through `data/raw/`. It adds a database row for every sidecar it doesn't know yet, and
  **deletes** rows whose WAV file has disappeared. If the file comes back, the next walk adds it
  again.
- `soundspy record` only writes files. It never touches the database, so the sound library
  doesn't need any database code (Chapter 4's rule), and you can record with the backend
  stopped. The recording shows up the next time the API looks.
- Paths are stored relative to the project root (`data/raw/test_….wav`), so the database still
  works if the project folder moves. Times are stored in UTC, because SQLite doesn't remember
  time zones and mixed offsets would break the "newest first" order.

### 8.7 Measuring levels

`dsp/levels.py` measures, per channel:

| Measure | What it tells you |
|---|---|
| **RMS dBFS** | average loudness; a full-scale sine wave reads −3.01 dBFS |
| **peak dBFS** | the loudest single sample; 0 dBFS is the maximum |
| **noise floor** | the background level: the quietest 10 % of 50 ms windows (room tone, fan, mic hiss) |
| **clipping** | the share of samples at or above 0.999, which are too loud and distorted |
| **digital silence** | the share of the recording made of *stretches* (≥ 5 ms) of exact 0.0, which only a gate, suppression or a muted input produces. Isolated zeros are normal (Chapter 17.12) |

Total silence is reported as −120 dBFS rather than "minus infinity".

**Comparing two channels.** To tell whether two channels come from two *different* microphones,
the app slides one channel against the other by up to ±1 ms and finds where they match best (a
*cross-correlation*). The shift is the **lag**: a **positive lag** means the second channel heard
the sound *later*. A sound from the left reaches the left microphone first. Microphones a few
centimetres apart differ by a fraction of a millisecond, which is a few samples at 48 kHz.

**`ChannelReducer`** (`dsp/channels.py`) also exists now. It turns any number of channels into
one, either by averaging (`mean`) or by picking one (`select`), and always returns the 2-D
`(frames, 1)` shape. It is the one sanctioned place to do this (Chapter 3), and a test scans the
code to make sure nobody does it anywhere else.

### 8.8 Reading the `check-mic` output

This is the real output from the development laptop, in a quiet room:

```
Device : Microphone Array (Realtek(R) Audio)  [Windows WASAPI, index 9]
Profile: laptop   channels kept: [0, 1] of 2 opened   sample rate: 48000 Hz
Recording 5 s in 2... 1... now.
done.

 ch   RMS dBFS   peak dBFS   noise floor   clipping
  0     -116.0       -90.2        -120.0     0.000 %
  1     -116.0       -90.2        -120.0     0.000 %
duration: 5.00 s   overflows: 0   start-up silence discarded: 24000 frames

Channel comparison (against channel 0, lags within +/-1 ms):
  ch 1: zero-lag corr +1.0000, peak corr +1.0000 at lag +0 samples (+0.000 ms)
        -> duplicated mono: no spatial information

Warnings:
  ! all channels below -55 dBFS RMS: muted or very quiet input
  ! channel 0: noise floor -120.0 dBFS is cleaner than any real microphone (their own hiss
    sits well above -100 dBFS): noise suppression is probably active - turn off Windows
    audio enhancements / Voice Clarity for this microphone
  ! channel 1: (the same)
```

How to read it, line by line:

- **Device / Profile**: which microphone was used and how. Check it is the one you expect.
- **The table**: one row per channel. Normal speech close to a laptop is roughly −30 to −20 dBFS
  RMS. A quiet room with an honest microphone shows a noise floor somewhere around −70 to
  −50 dBFS.
- **overflows**: should be 0. Anything else means the computer was too busy to collect sound.
- **start-up silence discarded**: the 500 ms of WASAPI start-up zeros (8.4).
- **Channel comparison**: only shown for 2+ channels. The verdict is one of:

| Verdict | Meaning |
|---|---|
| **duplicated mono: no spatial information** | the channels are identical sample for sample: one signal copied twice |
| **distinct microphones** | the best match is at a non-zero lag: sound reaches them at different times, as with real separate mics |
| **very similar, possibly merged by the driver** | lag 0 and correlation above 0.995: do the clap test to be sure |
| **distinct signals** | different, but no clear time shift |

- **Warnings:**

| Warning | Likely cause |
|---|---|
| clipping above 0.1 % | too loud: lower the input gain or move away |
| all channels below −55 dBFS | muted, or very quiet |
| noise floor above −45 dBFS | a noisy room or fan, or automatic gain control pumping the hiss up |
| noise floor below −100 dBFS | **too clean to be real**: software noise suppression is removing the room sound |
| more than 1 % of the recording in stretches of exact digital silence | a noise gate, noise suppression, or a muted input |

**What the laptop's output says.** Two separate problems show up:

1. **The two channels are bit-identical.** The laptop's "microphone array" reaches Sound Spy as
   one signal copied into two channels. For Tab 1 (sound *types*) that doesn't matter. It does
   mean the built-in microphone can never be used for direction finding: that needs the
   ReSpeaker.
2. **Noise suppression is active.** Over a 5-second take *every* 50 ms window was below
   −100 dBFS, the loudest at −108.9. Even the best microphone and converter hiss far louder than
   that, so something between the microphone and Sound Spy is wiping out the quiet room sound.
   That something is almost certainly Windows' audio enhancements ("Voice Clarity") or the Realtek
   driver's own effects. It must be switched off before recording training data, or Sound Spy
   would learn from sound that a different program has already "cleaned".

**After switching the enhancements off (retest, same day).** Both problems were caused by the
enhancements, and both are gone. The same laptop, recording speech:

```
 ch   RMS dBFS   peak dBFS   noise floor   clipping
  0      -41.8       -19.3         -64.1     0.000 %
  1      -42.1       -19.6         -65.7     0.000 %
duration: 5.00 s   overflows: 0   start-up silence discarded: 0 frames

Channel comparison (against channel 0, lags within +/-1 ms):
  ch 1: zero-lag corr +0.8854, peak corr +0.8854 at lag +0 samples (+0.000 ms) -> distinct signals

No level warnings.
```

- The **noise floor is now −64 to −66 dBFS**: the honest hiss of a real microphone and room.
- The **two channels are no longer identical**. The laptop really has **two microphones**; the
  "duplicated mono" was the enhancement processing merging them into one signal. Speech from in
  front reaches both at the same moment, so the lag is 0 and the correlation high, but not 1.0.
  The clap test (8.9) proves they are separate.
- **The converter is 16-bit, and Windows adds +7.44 dB of digital gain.** Every sample is a whole
  multiple of 2.354/32768, i.e. a 16-bit value multiplied by 2.354. Two consequences:
  - samples can exceed 1.0, up to +7.44 dBFS, which is the converter's real ceiling. Your loud
    claps flattened at exactly +7.4 dBFS on both channels: that is true clipping, so clap a bit
    softer or further away;
  - quiet sound regularly lands on the converter's "0" step, which produces isolated exact
    zeros. That is normal (Chapter 17.12).

### 8.9 The clap test

The channel comparison can only judge what it hears. To check whether two channels really are two
microphones:

1. Run `soundspy check-mic` and, during the recording, **clap once near the left edge** of the
   laptop.
2. Run it again and **clap near the right edge**.
3. Compare the **lag** in the two reports. With two real microphones the sign **flips**: positive
   one time, negative the other, because the clap reaches the nearer microphone first. If it
   stays at `+0` with "duplicated mono" both times, there is only one signal.

**Real result on the development laptop** (enhancements off):

```
clap 1: ch 1: ... peak corr +0.2217 at lag +6 samples (+0.125 ms) -> distinct microphones
clap 2: ch 1: ... peak corr +0.2865 at lag -5 samples (-0.104 ms) -> distinct microphones
```

The sign flips, so these are **two separate microphones**. A lag of 6 samples at 48 kHz is
0.125 ms, during which sound travels about 4.3 cm, so the microphones are at least that far apart.
With lag +6 the sound reached channel 0 first, so if that was the left-side clap, channel 0 is the
left microphone. (The correlation is low, 0.2–0.3, because a 5-second take is mostly room sound
and only a short moment is the clap. The lag still comes out clearly.)

### 8.10 Switching off Windows audio enhancements

Settings → System → Sound → choose the microphone under *Input* → **Properties**. Turn **Audio
enhancements** off; on newer Windows versions also turn off **Voice Clarity**, if present.
Some Realtek drivers have their own effects (noise suppression, beamforming) in the Realtek
Audio Console, so check there too. The development laptop also has **Nahimic** installed (its
virtual "Nahimic" speakers appear in `soundspy devices`). Nahimic has its own microphone effects,
such as noise suppression, which must be switched off in the Nahimic app as well. Then run `soundspy check-mic` again: an honest microphone shows
a noise floor in the −70 to −50 dBFS range and no "noise suppression" warning.

---

## 9. The backend's web interface — REST and WebSocket

**Exists today**, partly. Location: `backend/src/soundspy/api/`.

### 9.1 The addresses

| Address | What it does | Status |
|---|---|---|
| `GET /api/health` | "Are you alive?" Answers `{"status": "ok", "version": "0.1.0"}` | **works** |
| `GET /api/config` | the class list (ID, display name, description), which tabs are available and why not, the active device profile | **works** |
| `GET /api/devices` | list microphones and speakers, and which one the active profile resolves to | **works** (since step 02) |
| `POST /api/session/start` / `stop` | start or stop live listening | 501, step 03 |
| `PUT /api/session/target` | change what to listen to (a `TargetSpec`) | 501, step 03 |
| `GET /api/recordings` | list recordings, newest first (re-reads `data/raw` first, Chapter 8.6) | **works** (since step 02) |
| `GET /api/recordings/{id}`, `/{id}/audio` | view one recording, download its sound | 501, step 03 |
| `POST /api/recordings/{id}/process` | clean up a saved recording | 501, step 06 |
| `WS /ws/session` | the live channel (below) | **works**: status and error messages |

**Why have addresses that only say "not implemented"?** Because the complete
shape of the interface is then visible from day one. The automatic
documentation page at http://127.0.0.1:8765/docs lists every address, the
website can be written against the real shape, and each later step fills in
an address that already exists. The 501 reply says *which* step will
implement it: `{"detail": "not implemented (step-03)"}`.

**`GET /api/devices` in detail.** It answers with the active profile's name, every input and
output device (index, name, host API, channel count, native rate, whether it is the system
default) and `resolved_input`: the device, channels and sample rate the profile turns into. If the
profile can't be matched, for example because the named microphone is unplugged, the answer is
still **200**, with `resolved_input: null` and an `error` sentence saying why, so a page can show
the reason instead of a generic failure. Only when the sound library itself can't be loaded does
it answer **503**.

### 9.2 The live channel: `/ws/session`

When the website connects:

1. The backend immediately sends the current **status**: "idle", since no
   session exists yet.
2. Every **2 seconds** after that, it sends the status again as a
   heartbeat.
3. If the website sends something the backend doesn't understand, the
   backend answers with an **error** message instead of crashing or staying
   silent. That covers broken JSON, an unknown message type, or a
   `set_target` request (which is recognised but not implemented until
   step 03).

### 9.3 One envelope for every message

Every message, in both directions, has the same three parts:

```json
{ "type": "status",
  "t": 0.0,
  "payload": { "state": "idle", "mode": null, "device": null,
               "sample_rate": null, "channels": null,
               "message": "no active session" } }
```

- **`type`** says what kind of message it is: `status`, `levels`,
  `spectrogram`, `detections`, `sources` or `error` from the backend, and
  `set_target` from the website.
- **`t`** is the number of seconds since the session started (0 when idle).
- **`payload`** holds the actual contents. Its shape depends on the type.

With a single envelope, the website can sort every incoming message by its
`type`, and a new kind of message never breaks the old ones. The website
simply ignores types it doesn't know yet, with a warning in the browser's
console.

### 9.4 Who may talk to it

- The backend listens on **127.0.0.1**, so only programs on this same laptop
  can reach it. It is a desktop app, not a public website.
- **CORS** allows exactly one other web address, the development website at
  `http://localhost:5173`. Tests check that a different address is refused.
- There are no passwords and no rate limiting yet (see Chapter 19).

---

## 10. The database

**Exists today** (the tables are created, nothing writes to them yet).
Location: `backend/src/soundspy/api/db/`. File: `data/soundspy.db`.

*Corrected after step 02:* the `recordings` table is now filled. It is an
**index** of the sidecar files in `data/raw/`, rebuilt from them on startup and on
every `GET /api/recordings` (Chapter 8.6). `processing_runs` stays empty until
step 06.

### 10.1 What it stores, and what it doesn't

The database stores **information about** audio, not the audio itself. Sound
files stay on disk as WAV files. Each one gets a matching **JSON "sidecar"**
file with its details, plus a database row pointing to its path. Databases are
good at searching and linking records, while sound files are large and are
better left as plain files.

### 10.2 The two tables

**`recordings`**: one row per raw recording.

| Column | Meaning |
|---|---|
| `id` | a number the database assigns: 1, 2, 3… |
| `created_at` | when it was recorded |
| `path` | where the WAV file is; no two rows can share one |
| `duration_s`, `sample_rate`, `channels` | the basic facts about the sound |
| `device`, `profile` | which microphone, under which device profile |
| `label` | optional: a sound class, for dataset recordings |
| `notes` | free text |

**`processing_runs`**: one row each time a recording is cleaned up offline.

| Column | Meaning |
|---|---|
| `recording_id` | which recording; must point to a real row |
| `mode` | semantic / spatial / sources |
| `target_json` | the `TargetSpec` used, stored as JSON |
| `method` | `mask_baseline` or `neural_tse`; nothing else is accepted |
| `output_path` | where the cleaned file was written |
| `metrics_json` | quality scores, e.g. SI-SDR improvement |
| `status` | `pending`, `done` or `failed`; nothing else is accepted |
| `error` | what went wrong, if it failed |

The database itself enforces several rules, so mistakes can't slip in:
duplicate paths are refused, a processing run must belong to an existing
recording, unknown methods and statuses are refused, and deleting a recording
also deletes its runs.

When a WAV file disappears from disk, its `recordings` row is **deleted** at the
next sync (decided in step 02; the decision log explains why, and when to revisit
it). Paths are stored relative to the project root, and times in UTC.

### 10.3 Raw recordings are never changed

A recording is saved at its original sample rate with **all** its channels,
and it is never edited afterwards. Converted copies (resampled, or reduced to
fewer channels) are made fresh when needed, or stored separately in
`data/processed/`. That way any recording can always be processed again later
with a better method.

### 10.4 How the database starts up

When the backend starts, it creates any missing tables. There is no
"migration" tool yet, meaning a tool that records step by step how a
database's structure changes over time. The decision log notes that one
(Alembic) should be added as soon as the table structure starts changing.

---

## 11. The website

**Exists today** as a shell. Location: `frontend/`.

### 11.1 What you see

- A header with the app name and a **connection light**.
- Three tabs: **Semantic**, **Spatial**, **Sources**.
- **Semantic** lists the seven sound types as buttons, using their display
  names, with the description as a tooltip. The buttons are greyed out and
  marked *"Selection available in step 03"*.
- **Spatial** and **Sources** say *"Requires a microphone array"*, because
  the backend reports them unavailable with the laptop profile.
- A dark colour theme. The final visual design comes later.

### 11.2 The connection light

It combines two separate checks:

1. Every **5 seconds** the website calls `/api/health`. Does the backend
   answer?
2. Is the **WebSocket** open?

| Light | Label | Meaning |
|---|---|---|
| green | connected | the backend answers and the live channel is open |
| amber | reconnecting | one of the two is up, the other is still coming back |
| red | disconnected / disconnected · reconnecting | the backend is not answering; the website keeps retrying |
| grey | connecting | the very first moments after the page opens |

If the WebSocket drops, the website tries again after 0.5 seconds, then 1, 2,
4… doubling each time up to a limit of 10 seconds, and goes back to quick
retries as soon as a connection succeeds. This is called **exponential
backoff**. It recovers quickly from a short blip without bombarding a backend
that is down for longer.

### 11.3 How the code is organised

| Folder / file | Job |
|---|---|
| `src/api/types.ts` | TypeScript descriptions of what the backend sends, mirroring the backend's own definitions |
| `src/api/client.ts` | a small helper for calling `/api/...` addresses |
| `src/api/ws.ts` | the self-reconnecting WebSocket, and the code that checks every incoming message |
| `src/store/appStore.ts` | the shared state: backend up or down, WebSocket state, open tab, configuration, last status |
| `src/hooks/useSessionSocket.ts` | opens the WebSocket when the app starts and puts what arrives into the store |
| `src/components/ConnectionIndicator.tsx` | the connection light |
| `src/components/viz/` | empty for now; live waveform, spectrogram and source-map components go here |
| `src/tabs/` | one component per tab |
| `src/App.tsx` | puts the header, tabs and panels together |

The backend's descriptions of its messages and the website's are currently
two hand-written copies that must agree. The plan is to generate the
website's copy automatically from the backend's published description
(`npm run gen:api` is a placeholder for that).

### 11.4 The development proxy

During development, the website is served at `localhost:5173` and the backend
runs at `127.0.0.1:8765`. The website's server **forwards** every `/api/...`
and `/ws/...` request to the backend. To the browser everything then comes
from one address, which keeps things simple and avoids CORS problems. The
forwarding goes to `127.0.0.1` rather than `localhost`, on purpose
(Chapter 17).

---

## 12. The command line tool

**Exists today.** The backend also installs a command called `soundspy`, used
for developer jobs that don't need the website: running the server, recording
test clips, preparing data and training models.

| Command | What it will do | Status |
|---|---|---|
| `soundspy serve` | start the backend (address and port from settings; `--reload` restarts it when code changes) | **works** |
| `soundspy devices` | list microphones and speakers, marking the system default (`*`) and the one the profile uses (`>>`); `--json` for machine-readable output | **works** (step 02) |
| `soundspy check-mic` | record a few seconds and report the levels, channel comparison and warnings (Chapter 8.8); `--meter` for a live level meter | **works** (step 02) |
| `soundspy record` | record a clip into `data/raw` as WAV + sidecar (`--seconds`, `--label`, `--notes`, `--out`) | **works** (step 02) |
| `soundspy record-dataset` | guided recording of labelled clips | step 04 |
| `soundspy data` | download datasets, build mixtures | step 04 |
| `soundspy train` | train a model | step 05 |
| `soundspy detect` | find sound types in a recording | step 05 |
| `soundspy extract` | clean up a recording | step 06 |

Placeholder commands print `not implemented yet (step-XX)` and exit normally.

All audio commands accept `--profile <name>` and `--device <index or name>`. A
problem with the setup (no such device, invalid profile, a stalled microphone)
prints one line starting with `error:` and exits with code 1, rather than a long
Python error.

---

## 13. Following a request from start to finish

Here is what actually happens today when you open the website with both
programs running.

**Step 1: the page loads.** The browser asks `localhost:5173` for the page,
and Vite sends the React app.

**Step 2: two things start at once.**
- The connection light calls `GET /api/health`.
- `useSessionSocket` opens the WebSocket to `/ws/session`.

**Step 3: the proxy forwards both** to the backend at `127.0.0.1:8765`.

**Step 4: the backend answers the health call.** FastAPI sends the request
to `routes/system.py`, which replies `{"status": "ok", "version": "0.1.0"}`.
The store records "backend: up".

**Step 5: the backend accepts the WebSocket.** `api/ws.py` registers the
connection and immediately sends a `status` message ("idle"). It then starts
the 2-second heartbeat. The store records "connected" and the status.

**Step 6: the light turns green**, because both checks are now positive.

**Step 7: the website asks for the configuration.** Once the backend is
known to be up, the app calls `GET /api/config` once. The backend reads the
class list and the active device profile from its settings. It works out
that Tabs 2 and 3 need geometry the laptop profile doesn't have, and sends
everything back.

**Step 8: the tabs fill in.** The Semantic tab shows the seven display
names. The Spatial and Sources tabs show the "requires a microphone array"
message.

**If you now stop the backend:** the next health check fails and the
WebSocket closes. The light goes red ("disconnected · reconnecting") and the
website keeps retrying. **When you start the backend again**, the next retry
succeeds and the light turns green on its own, with no page reload needed.

---

## 14. What comes next — the plan, step by step

Everything in this chapter is **planned**, not built. Concepts are explained
here so the later chapters will make sense when they arrive.

| Step | What gets built | Tab |
|---|---|---|
| 02 | microphone capture, device listing, recording to disk, level measurement, `ChannelReducer` — **done** (Chapter 8) | – |
| 03 | the live engine (threads, queues, playback), live level meters and spectrogram on the website, a record button | 1 |
| 04 | datasets: AudioSet mapping, downloading ESC-50 and FSD50K, own recordings, splits, synthetic mixtures | – |
| 05 | sound detection: fine-tuning a pretrained model, live detections on the website | 1 |
| 06 | first extractor (classical masking) + quality scores; clean up saved recordings and compare before/after | 1 |
| 07 | neural extractor that runs live | 1 |
| 08 | full evaluation, robustness, documentation for the university report | 1 |
| modules 3–4 | simulated array, direction finding, beamforming, Spatial tab | 2 |
| module 5 | source tracking, Sources tab | 3 |

### 14.1 Recognising sounds (step 05)

A **pretrained** audio model already knows a lot about sound in general. It
will be **fine-tuned** to answer seven yes/no questions at once, one per
class. Its input is a log-mel spectrogram of a short window of sound, and its
output is seven probabilities. Run continuously on the live stream, it
produces `Detection` events: *"speech, 92%, from 12.0 s to 13.0 s"*. Which
pretrained model to use has not been decided yet.

### 14.2 Keeping the chosen sounds — two ways (steps 06 and 07)

Both methods sit behind the same `Extractor` interface, so either can be
plugged into the engine.

**The classical baseline (step 06): detector-guided masking.** Convert the
sound to a spectrogram. Use the detector to estimate which time-and-frequency
cells belong to the chosen sounds, and build a mask: close to 1 for cells to
keep, small for the rest. The value is never below the attenuation floor, so
nothing drops to unnatural total silence. Multiply, then convert back to
sound. It is transparent and easy to reason about, and it gives a reference
score that the neural method has to beat.

**The neural method (step 07): label-conditioned target sound extraction.**
A neural network receives two things: the mixed sound, and a multi-hot vector
saying which classes to keep. It outputs only those classes. The design
follows **SemanticHearing**'s *Waveformer* model, which is built to work in
small chunks with a memory of what came before, so it can run live with low
latency.

### 14.3 Training data from mixtures (step 04)

There is no recording of "cafeteria speech, with the exact speech-only
version alongside". So training examples are **made**: take a clean clip of
a chosen class (the *foreground*), add clips of other sounds (the
*background*) at a controlled loudness, and keep the clean foreground as the
known answer. Datasets with different label vocabularies can all contribute,
because every clip's label is translated through the AudioSet ontology into
the project's seven classes. The data is split carefully, so that no original
recording ends up in both training and test, which would be leakage.

### 14.4 Measuring quality

- **Detection:** how well the probabilities rank "present" above "absent" for
  each class (mean average precision), and how often the yes/no decisions are
  right (F1 score).
- **Extraction:** **SI-SDR improvement**, measured on synthetic mixtures where
  the clean answer is known. A positive improvement means the output is
  closer to the clean target than the raw mixture was.

### 14.5 Space (modules 3–5)

With the 4-microphone array, the delays between microphones reveal where each
sound comes from (direction of arrival). **Beamforming** then focuses
listening in a chosen direction before the extractor runs. In Tab 3, a
**tracker** combines "what" (detections) with "where" (directions) into
persistent sources the user can tap on a map. Until the hardware arrives,
**simulated** array recordings (made with a room-acoustics library called
pyroomacoustics) will let this work start without it.

### 14.6 A note on reused code

If any code from SemanticHearing (MIT licence) is adapted in future, the
adapted file keeps its original copyright notice and the reuse gets recorded
in a third-party notices file. So far none has been adapted.

*Updated in step 02:* there is no separate notices file any more. Adapted code is
recorded in Chapter 20, "Third-party code".

---

## 15. Running everything

All commands are for **PowerShell** on Windows 11.

### 15.1 What you need

- **Python 3.11.** On this machine, plain `python` is version 3.14, which is
  too new for the project, so the backend is always created with the `py
  -3.11` launcher.
- **Node.js** (version 22 or newer; this machine has 24) for the website.
- Ports **8765** and **5173** free.

### 15.2 First-time setup

```powershell
# backend
cd backend
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e ".[api,dev]"

# website (in a second terminal)
cd frontend
npm install
```

`[api,dev]` installs two optional groups of libraries: the web server pieces,
and the testing tools. Two more groups, `ml` (PyTorch and friends) and
`spatial` (pyroomacoustics), are not needed until steps 05 and module 3. When
the time comes, PyTorch must be installed **first**, from PyTorch's own
download site with the matching graphics-card (CUDA) version. Otherwise pip
installs a version that ignores the graphics card. The README has the exact
command.

### 15.3 Starting it

```powershell
# terminal 1 — the backend
cd backend
.venv\Scripts\activate
soundspy serve            # API docs: http://127.0.0.1:8765/docs

# terminal 2 — the website
cd frontend
npm run dev               # open http://localhost:5173
```

Stop either one with **Ctrl+C**.

### 15.4 What you should see

- At http://127.0.0.1:8765/docs, an automatically generated page listing
  every backend address. You can try them out from there.
- At http://localhost:5173, the three tabs, a **green "connected"** light
  showing the version, and the seven sound types on the Semantic tab.
- Stop the backend and the light turns red within about 5 seconds. Start it
  again and it turns green by itself.

### 15.5 Using the microphone

```powershell
# in backend/, with the venv active
soundspy devices                              # which microphone the profile uses
soundspy check-mic                            # 5 s test + report (Chapter 8.8)
soundspy check-mic --meter                    # live level meter, Ctrl+C to stop
soundspy record --seconds 10 --label speech   # saved in data/raw/
```

Before recording anything worth keeping, switch off Windows audio enhancements /
Voice Clarity for the microphone (Chapter 8.10) and check that `check-mic` no
longer warns about noise suppression.

---

## 16. Testing everything

### 16.1 Running the checks

```powershell
# backend (in backend/, with the venv active)
pytest              # 185 tests (after step 02), about 10 seconds
ruff check .        # code-quality check

# website (in frontend/)
npm run test        # 5 tests
npm run lint        # code-quality check
npm run typecheck   # TypeScript type check
npm run build       # proves the production build works
```

### 16.2 Tests never need hardware

No test opens a real microphone. Before any test runs, a **fake
`sounddevice`** replaces the real sound library. It pretends there is one
microphone and refuses any attempt to actually record. Tests that would need
a graphics card or downloaded datasets are marked and skipped unless asked for
(`pytest --run-gpu`, `pytest --run-data`). The whole suite therefore runs on
any machine, in seconds.

*Since step 02 the fake is much richer.* It imitates a Windows laptop: the same
built-in microphone under MME (name cut to 31 characters, 44.1 kHz) and under
WASAPI (full name, 48 kHz), a 6-channel reSpeaker array, and several speakers.
Its microphone stream runs on a background thread, like a real sound card, and
produces predictable sound: by default, noise where each channel is the previous
one delayed by 3 samples. Tests can switch it to other signals, make it stall,
report overflows, or start with a burst of digital silence the way the laptop's
WASAPI stream does.

### 16.3 What the backend tests cover

| Area | Examples of what is checked |
|---|---|
| `AudioBuffer` | refuses 1-D arrays, the wrong number type, zero channels, invalid sample rates; reports length and channels correctly |
| `Direction`, `TargetSpec` | 370° becomes 10°; invalid widths are refused; unknown class names are refused; attenuation can't be negative |
| `LabelSet` and classes | the seven classes load in order with names and descriptions; class ↔ position and multi-hot round trips; old or incomplete file formats are refused |
| settings | paths resolve to the project root; the laptop profile has no geometry; the backend port is 8765 |
| engine | a chain of passthrough stages returns the input unchanged; side stages see the *input*; channel counts flow correctly through the chain; target changes reach every stage |
| web interface | health and config answers; every unfinished address returns 501 with the right step; CORS refuses strangers; the WebSocket sends status on connect, a heartbeat, and error replies |
| database | tables are created; a recording can be stored and read back; duplicate paths, unknown methods and links to missing recordings are refused |
| architecture | **the library never imports FastAPI, Starlette, SQLAlchemy or the API layer**, checked twice: by reading every source file, and by loading the whole library in a fresh Python and confirming none of those got loaded |
| command line | `--help` lists every command; each placeholder prints its step and exits normally |
| device choice (step 02) | the WASAPI twin of the MME default is chosen; names match in any case; an ambiguous name is refused; the ReSpeaker profile opens 6 channels and keeps 2–5 in order; too few channels, unknown devices and unknown host APIs give clear errors |
| device profiles (step 02) | every invalid field is rejected with a message naming the profile and file |
| microphone source (step 02) | `read()` returns exactly the requested length as float32; chunks arrive in order; leaving early or closing from another thread stops the stream; a slow reader gets the newest chunks and the drops are counted; a stall is reported within the timeout; start-up silence is discarded but a muted mic still shows as silent |
| files (step 02) | exact chunks with a padded last chunk; looping; real-time pacing; `raw_channels` from a profile; writing and reading back gives identical samples; nothing is overwritten |
| recordings (step 02) | file name and every sidecar field; no overwrite; the API indexes new files, lists newest first and forgets deleted ones |
| levels (step 02) | a full-scale sine reads −3.01 dBFS RMS and 0 dBFS peak; clipping, digital silence and noise floor on known signals; a 7-sample delay is found with the right sign; identical channels are recognised |
| channel rule (step 02) | a scan of the source code fails if channels are indexed (`[:, …]`) or averaged anywhere except the four allowed files; proven by planting a violation and watching it fail |
| audio commands (step 02) | `devices --json` and the table markers; `check-mic` report, meter and warnings; `record` writes WAV + sidecar |

### 16.4 What the website tests cover

- The three tabs and the connection light appear.
- Class buttons show display names, not internal IDs, and are disabled with
  the description as a tooltip.
- The Spatial tab shows the array message when the backend says it is
  unavailable.
- The light turns **green** when the backend answers and the WebSocket opens,
  and shows **"disconnected · reconnecting"** when both fail.
- Unknown or broken WebSocket messages are ignored with a warning.

Tests run against a *simulated* browser, which can check what the page says
but not how it looks. Layout is checked by opening the page in a real
browser.

### 16.5 A manual check worth repeating

With both programs running, stop the backend, wait for the red light, then
start it again and watch it turn green without reloading the page. This was
done for step 01 at the network level: the health check, the configuration,
and the WebSocket heartbeat and error replies were all confirmed through the
website's proxy, with the backend up, stopped, and restarted.

**After any change to audio capture**, run `soundspy devices`, `soundspy check-mic`
and a short `soundspy record` on the real laptop. Then start `soundspy serve` and
confirm the recording appears at `GET /api/recordings`. Tests can only prove the
code behaves correctly with the *fake* sound card. The two biggest surprises of
step 02, start-up silence and noise suppression (Chapter 17.8–17.9), only showed
up on real hardware.

---

## 17. Problems found and fixed

Each of these was real. Most were silent: nothing crashed, and something was
simply wrong. They are the most instructive part of this document.

### 17.1 Port 8000 was secretly taken

**Symptom:** with our backend stopped, the website should have shown
"connection refused". Instead, something on port 8000 answered *"Not
Found"*.

**Cause:** Docker Desktop and WSL (the Linux layer built into Windows) were
both listening on port 8000, for an unrelated container. While our backend
ran, it won on `127.0.0.1`. When it stopped, the other program answered
instead. Worse, `localhost` can resolve to the IPv6 address `::1`, where the
WSL relay was listening, so a request to `localhost:8000` could reach the
wrong program even while ours was running.

**Fix:** the backend moved to port **8765**, still bound to `127.0.0.1`, and
the website's proxy points at `127.0.0.1` explicitly rather than at
`localhost`.

**Lesson:** "the port is free" is an assumption to check, not a fact.

### 17.2 Windows paths came out scrambled

**Symptom:** a test checking where the database file goes failed: the path
came back as `C%3A%5CUsers%5C...`.

**Cause:** the database library (SQLAlchemy 2.1), when turning a database
address back into text, **percent-encodes** the colon and backslashes of a
Windows path. Those are the codes web addresses use for special characters.

**Fix:** pass the address around as the library's own `URL` object, never as
text. The database file now lands exactly where intended:
`data/soundspy.db` at the project root.

### 17.3 The newest TypeScript was too new

TypeScript 7 (a rewrite of the TypeScript compiler) was the latest release,
but the code-checking tool `typescript-eslint` only supports versions below
6.1. The project therefore uses **TypeScript 6.0**, a deliberate choice
recorded in the decision log. Upgrade when the checker catches up.

### 17.4 The installer refused a file outside its folder

The backend's package description pointed at `../README.md`, one folder up.
The Python packaging tool refuses to include files from outside the package's
own folder, so the first install failed. The reference was removed.

### 17.5 The engine's own guard caught a mistake in a test

A test set up a chain for 48,000 Hz audio, then fed it a 16,000 Hz chunk by
accident. The chain refused the chunk, as designed (Chapter 6.2). The test
was wrong, not the code, and the guard proved its worth on day one.

### 17.6 Dataset names were about to reach the user

The first version of the website showed AudioSet category names such as
"Dishes, pots, and pans" as class descriptions. Those are dataset labels, not
friendly text. Each class now has a separate display name and description,
and the backend **no longer sends** the AudioSet names to the website at all.
A test makes sure they stay out.

### 17.7 Known and harmless

- **One test warning:** the web library (Starlette 1.10) suggests a newer
  testing helper (`httpx2`) instead of the one used (`httpx`). Nothing is
  broken. It is noted for a later tidy-up.
- **"Extraneous" packages in `npm ls`:** these are optional helper packages
  that the styling and build tools use on Windows. They are harmless.
- **A leftover `.venv` folder at the project root** belongs to the project's
  old name (SoundFocus, step 00). It is no longer used and can be deleted.

*The entries below were found in step 02, mostly on real hardware.*

### 17.8 Every new recording started with half a second of fake silence

**Symptom:** the first real `check-mic` reported a noise floor of −120 dBFS,
the lowest value the app can display.

**Cause:** measured directly, the first **500 ms** of a freshly opened WASAPI
stream on the laptop are **exact zeros**: 24,000 frames at 48 kHz, every time.
After that, real sound flows. The quietest 10 % of the recording was simply that
silent start.

**Fix:** after opening a microphone, leading frames that are exactly zero are
discarded, for at most 1 second (Chapter 8.4). A real microphone never produces
exact zeros, and the 1-second cap means a truly muted microphone still reaches
the report and gets flagged.

**Lesson:** a measurement that looks impossibly good deserves suspicion, not
celebration.

### 17.9 The laptop's sound is being "cleaned" before Sound Spy sees it

**Symptom:** with the start-up silence removed, the noise floor still read
−120 dBFS. Over a whole 5-second take every 50 ms window stayed below −100 dBFS,
and some samples were as small as 0.0000004, far below what the microphone's
converter can even represent.

**Cause:** no real microphone is that quiet; its own electronic hiss is far
louder. Something between the microphone and Sound Spy is removing the room
sound. That is almost certainly Windows audio enhancements / Voice Clarity or the
Realtek driver's noise suppression.

**What changed:** `check-mic` now says so in plain words ("noise floor … is
cleaner than any real microphone: noise suppression is probably active"), and
also warns about stretches of exact digital silence. Switching the enhancements
off is a setting on the laptop, not code (Chapter 8.10). Until it's done, any
recording is of pre-cleaned sound.

**Also found:** the laptop's two channels are **bit-identical**: one signal copied
twice. Its built-in "microphone array" can therefore never tell directions, which
confirms the plan to use the ReSpeaker for Tabs 2–3.

**Outcome (retest):** with the enhancements switched off, the noise floor became an
honest −65 dBFS, the 500 ms start-up silence vanished, and the two channels turned
out to be **two real microphones** (the clap test flipped the lag from +6 to −5
samples). So the "duplicated mono" above was the enhancement processing, not the
hardware. Two laptop microphones a few centimetres apart can't replace the
ReSpeaker for real direction finding, but they are no longer "mono".

### 17.10 The same microphone wasn't recognised as itself

**Symptom:** in a test, the app chose the MME microphone instead of its WASAPI
twin.

**Cause:** MME cuts names to 31 characters, and on this name the cut fell just
after a space: "Microphone Array (Intel® Smart ". The matching code removed
surrounding spaces *before* cutting the other name to 31 characters, so one side
was 30 characters long and the other 31, and they never matched.

**Fix:** cut first, then remove spaces. The test fake reproduces exactly this
name, so the mistake can't come back unnoticed.

### 17.12 A warning that cried wolf: isolated zeros are normal

**Symptom:** after the enhancements were switched off, every `check-mic` still
warned that 1.4–5 % of samples were "exact digital silence", although the noise
floor now looked perfectly honest.

**Investigation, on the real recording:**
- the exact zeros came in about 3,200 tiny stretches, none longer than
  5 samples (0.1 ms). A noise gate would silence whole milliseconds;
- every sample was a whole multiple of 2.354/32768: a 16-bit converter plus
  +7.44 dB of Windows gain (Chapter 8.8);
- the value 0 occurred exactly as often as its neighbours ±1 (1.40 % vs 1.34 % /
  1.42 %). That is ordinary rounding of a quiet signal, not something forcing
  values to zero.

**Fix:** "digital silence" now counts only **stretches of at least 5 ms**. The
same recording now reports no warnings, and a test reproduces a quiet 16-bit
signal with isolated zeros to make sure it stays that way.

**Lesson:** a check written from one bad recording needs a second, good recording
before it can be trusted not to fire on healthy input.

### 17.11 Times without a time zone

SQLite stores dates as plain text without a time zone. A recording made at
16:05 in Romania and one made at 14:30 UTC would sort wrongly if each were stored
in its own local time. All recording times are therefore stored in UTC, and the
API marks them as UTC when it sends them.

---

## 18. Design decisions and why

The full dated list is in [docs/decisions.md](docs/decisions.md). The
reasoning behind the most important ones:

**Why does the backend handle all audio, not the browser?** Browsers apply
their own echo cancellation, noise suppression and volume control, and they
handle multi-microphone devices poorly. A project about controlling what you
hear can't have an invisible filter in front of it.

**Why chunk-by-chunk processing even for files?** So there is only one way of
processing. Offline and live use identical code, and real time needs no
redesign later.

**Why one `TargetSpec` for all three tabs?** Stages only read the fields they
need, so they never need to know which tab is open, and adding a tab doesn't
change them.

**Why keep the library free of web code?** So the sound processing can be
tested, used from the command line, and explained in a report without a
server, and so the web layer stays thin and replaceable.

**Why is the microphone setup a settings file?** So switching from the laptop
microphone to the array is a one-line change, not a code change.

**Why are class IDs separate from display names?** Trained models and stored
records refer to the ID, so it must never change. The display name can be
reworded at any time without touching a model.

**Why SQLite?** It needs no setup, it is a single file, and it is plenty for a
desktop app. If a bigger database were ever needed, only the address in the
settings would change.

**Why no Docker?** Getting at a microphone and a graphics card from inside a
container on Windows is impractical, so both programs run directly on the
machine.

**Why build a classical extractor before the neural one?** It is
transparent, quick to build, and gives an honest reference score. Without a
baseline, there is no way to tell whether the neural model is actually good.

**Why do unfinished addresses return 501 instead of not existing?** So the
full shape of the interface is visible and documented from the start, and
each step fills in a slot that already exists.

**Why are hardware tests faked?** So the test suite runs anywhere, in
seconds, and a missing microphone never makes a test fail for the wrong
reason.

**Why prefer WASAPI over the system default?** (step 02) The system default
on Windows is the MME version of the microphone, which resamples to 44.1 kHz and
cuts names short. WASAPI gives the microphone's real rate (48 kHz here) with less
delay.

**Why throw away the *oldest* chunk when the reader falls behind?** (step 02)
For live listening, hearing *now* matters more than hearing *everything*. Keeping
a backlog would make the delay grow without limit. The drops are counted, so they
are never hidden.

**Why is the sidecar the source of truth, not the database?** (step 02)
Recordings are plain files that can be copied, backed up or moved to another
computer, and the database can always be rebuilt from them. It also means the
recording command needs no database code at all.

**Why delete the index row when a file disappears, instead of marking it
missing?** (step 02) Marking would need a new database column, and the project
has no migration tool yet. If the file comes back, the next sync re-adds it.
This will be revisited once offline processing runs exist (step 06), because
deleting a recording's row also deletes its runs.

---

## 19. Known limitations

These are understood and accepted for now, not oversights.

1. **No audio is processed yet.** Step 01 built the frame. The sound path
   starts in step 02. *(After step 02: sound can be captured and recorded, but
   nothing processes or plays it yet. The live session comes in step 03.)*
2. **Tabs 2 and 3 need hardware that hasn't arrived.** The ReSpeaker array's
   microphone positions are still to be filled in from its datasheet.
3. **The website's message types are typed by hand.** They must be kept in
   step with the backend's manually until automatic generation is set up.
4. **No security.** No login, no rate limiting. That is acceptable only
   because the backend listens on `127.0.0.1` and so can't be reached from
   other computers.
5. **No database migrations yet.** Fine while the tables are new. To be added
   when their structure starts changing.
6. **Two of the AudioSet mappings are provisional** (chair scraping and
   traffic) until they are checked against the real ontology in step 04.
7. **Nothing on the website has been checked for appearance in a real
   browser** beyond a quick look. Automated tests can't see layout.
8. **Laptop microphones on Windows may have "audio enhancements" switched
   on** (noise suppression, automatic volume). Step 02 needs to detect or
   avoid these, for the same reason the browser is not trusted with audio.
   *(Step 02: confirmed on the development laptop. `check-mic` now detects it
   (Chapter 8.8), but switching it off is a Windows setting the user has to
   change (Chapter 8.10). The app can't do it for them.)*
9. **The laptop's built-in microphone is effectively mono** (step 02): both
   channels are identical, so it carries no direction information.
   *(Corrected after the retest: that was the Windows enhancements. With them off
   the laptop delivers two real microphones, about 4+ cm apart. That is still too
   small and fixed an array for the Spatial/Sources tabs, which need the
   ReSpeaker.)*
10. **Microphones plugged in while the backend runs aren't seen** (step 02).
    PortAudio reads the device list once, at start-up, so restart the backend
    (or re-run the command) after plugging in a USB microphone.
11. **The website doesn't show devices or recordings yet** (step 02). They exist
    in the API (`/api/devices`, `/api/recordings`). The page that uses them comes
    with the live session in step 03.

---

## 20. Third-party code

Sound Spy is original code. It uses ordinary libraries installed with `pip` and `npm` (NumPy,
SciPy, sounddevice, soundfile, FastAPI, React and so on), each under its own licence; those are
dependencies, not adapted code, and are not listed here.

The rule for **adapted** code, meaning code copied from another project and modified: the
adapted file keeps its original copyright and licence header, and the component is listed in the
table below with where it came from, its licence, and which of our files contain it.

| Component | Source | Licence | Adapted files |
|---|---|---|---|
| *(none so far)* | | | |

**Ideas, not code.** The extraction design (Chapter 14) follows the **SemanticHearing** project
(Veluri et al., UIST 2023, https://github.com/vb000/SemanticHearing, MIT licence): the
label-conditioned Waveformer model, chunked streaming inference, the AudioSet class mapping and
synthetic training mixtures. No code from it has been adapted. If any is in future, it gets a row
above.

---

## 21. Quick reference

### Start

```powershell
cd backend;  .venv\Scripts\activate;  soundspy serve     # http://127.0.0.1:8765/docs
cd frontend; npm run dev                                 # http://localhost:5173
```

### Check

```powershell
cd backend;  pytest;  ruff check .
cd frontend; npm run test;  npm run lint;  npm run typecheck;  npm run build
```

### Microphone

```powershell
soundspy devices [--json]                      # devices; >> = the one the profile uses
soundspy check-mic [--seconds 5] [--meter]     # level report / live meter
soundspy record --seconds 10 --label speech    # data/raw/speech_<time>.wav + .json
#   every audio command: --profile <name>  --device <index or name>
```

### Addresses

```
GET  /api/health                        is the backend alive, which version
GET  /api/config                        classes, which tabs are available, device profile
GET  /api/devices                       devices + what the profile resolves to (503 = no PortAudio)
GET  /api/recordings                    recordings, newest first (re-indexes data/raw)
POST /api/session/start | stop          501 until step 03
PUT  /api/session/target                501 until step 03
GET  /api/recordings/{id}[/audio]       501 until step 03
POST /api/recordings/{id}/process       501 until step 06
WS   /ws/session                        live channel: status every 2 s, error replies
```

### The numbers worth remembering

| | |
|---|---|
| Backend address | `127.0.0.1:8765` |
| Website (development) | `localhost:5173` |
| Database | `data/soundspy.db` |
| Sound classes | 7 |
| Default suppression of unchosen sounds | 30 dB (to about 3% of its size) |
| Engine chunk | 1,024 frames (~21 ms at 48 kHz) |
| Heartbeat / health check | every 2 s / every 5 s |
| Python / Node | 3.11 / 24 |
| Laptop microphone | WASAPI index 9, 48 kHz, 2 real mics (≥ ~4.3 cm apart), 16-bit + 7.44 dB gain |
| Honest noise floor (enhancements off) | about −65 dBFS |
| WASAPI start-up silence (discarded) | 500 ms with enhancements on, none with them off |
| Mic stall timeout / live chunk queue | 2 s / 32 chunks, oldest dropped first |
| Backend tests / website tests | 185 / 5 (after step 02) |

### Files that matter most

| File | Why |
|---|---|
| `backend/configs/classes.yaml` | the sound classes — IDs, display names, descriptions, AudioSet mapping |
| `backend/configs/default.yaml` | paths, active microphone profile, port, database location |
| `backend/src/soundspy/core/` | the shared data types every part relies on |
| `backend/src/soundspy/engine/` | the chunk-by-chunk processing frame |
| `backend/configs/devices/*.yaml` | the microphone profiles |
| `backend/src/soundspy/io/devices.py` | choosing a microphone (host-API twins, preference order) |
| `backend/src/soundspy/io/mic_source.py` | the live microphone: read, chunks, stall and start-up handling |
| `backend/src/soundspy/io/storage.py` | WAV + sidecar recordings |
| `backend/src/soundspy/dsp/levels.py` | RMS, peak, noise floor, clipping, channel comparison |
| `backend/src/soundspy/cli/common.py` | the `check-mic` report and its warnings |
| `backend/src/soundspy/api/app.py` | where the backend's web interface is assembled |
| `backend/src/soundspy/api/ws.py` | the live channel |
| `frontend/src/App.tsx` | the website's layout and tabs |
| `frontend/src/api/ws.ts` | the self-reconnecting live connection |
| `frontend/vite.config.ts` | the development proxy (must point at the backend's port) |
| `docs/architecture.md` | the exact technical rules and message formats |
| `docs/decisions.md` | every important decision, dated, with its reason |
