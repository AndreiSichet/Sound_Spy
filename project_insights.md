# Sound Spy — Project Insights

**An app that lets you choose what to hear.** It listens to the room through a
microphone, works out which kinds of sound are present (speech, music, dishes,
footsteps…), and makes the sounds you picked clearer while turning everything
else down.

Written: 14 November 2026, after step 01 · Last updated: 15 November 2026, after step 06
(extraction baseline).

> **Read this with its date in mind.** Sound Spy is built one step at a time.
> *Corrected after step 07:* the Semantic tab works end to end. It listens through the laptop
> microphone, recognises seven sound types live, keeps the ones you select and turns the rest
> down, in your headphones and for saved recordings. Two methods do the keeping: a simple first
> version (the "baseline", step 06) and a neural network (step 07) that also works while sounds
> overlap. Both are measured, and both have clear limits. The Spatial and Sources tabs still
> wait for the microphone array.
> Every section below says whether it describes
> something that **exists today** or something that is **planned**, and in
> which step. When a later step changes something written here, the old
> sentence should be corrected where it stands rather than quietly deleted.

---

## Who this document is for

This document is for anyone who wants to understand the project without
having followed its development. You don't need to be a programmer. It
explains every technical term the first time it is used, and Chapter 3 is a
glossary.

**If you only want to use the app, read Chapter 2**: it is the user's manual, and needs none of
the rest.

After reading it you should be able to explain:

- what Sound Spy is supposed to do, and how much of it exists today;
- how to use it: every button, what you should see and hear (Chapter 2);
- what the separate parts are, and why they are separate;
- how sound is represented inside a computer;
- how a microphone is chosen, how recordings are stored, and how to tell whether the microphone
  gives honest sound (Chapter 9);
- how the website and the Python program talk to each other;
- how to start everything, and how to check that it works.

It goes alongside [README.md](README.md), which is the short version. Two
more technical documents go deeper: [docs/architecture.md](docs/architecture.md)
(the exact rules and message formats) and [docs/decisions.md](docs/decisions.md)
(a dated list of every important choice).

---

## Table of contents

1. [The project in one page](#1-the-project-in-one-page)
2. [Using Sound Spy today — a guided tour](#2-using-sound-spy-today--a-guided-tour)
3. [Glossary](#3-glossary)
4. [How sound lives inside a computer](#4-how-sound-lives-inside-a-computer)
5. [How the whole app fits together](#5-how-the-whole-app-fits-together)
6. [The core building blocks](#6-the-core-building-blocks)
7. [The processing engine](#7-the-processing-engine)
8. [Configuration — settings live in files, not in code](#8-configuration--settings-live-in-files-not-in-code)
9. [Audio capture — getting sound in from a microphone](#9-audio-capture--getting-sound-in-from-a-microphone)
10. [Live session — from microphone to headphones and screen](#10-live-session--from-microphone-to-headphones-and-screen)
11. [Training data — from public datasets to mixture recipes](#11-training-data--from-public-datasets-to-mixture-recipes)
12. [Detection — which sounds are present right now](#12-detection--which-sounds-are-present-right-now)
13. [Extraction (baseline) — keeping the selected sounds](#13-extraction-baseline--keeping-the-selected-sounds)
14. [Neural extraction — a network that pulls the chosen sounds out](#14-neural-extraction--a-network-that-pulls-the-chosen-sounds-out)
15. [The backend's web interface — REST and WebSocket](#15-the-backends-web-interface--rest-and-websocket)
16. [The database](#16-the-database)
17. [The website](#17-the-website)
18. [The command line tool](#18-the-command-line-tool)
19. [Following a request from start to finish](#19-following-a-request-from-start-to-finish)
20. [What comes next — the plan, step by step](#20-what-comes-next--the-plan-step-by-step)
21. [Running everything](#21-running-everything)
22. [Testing everything](#22-testing-everything)
23. [Problems found and fixed](#23-problems-found-and-fixed)
24. [Design decisions and why](#24-design-decisions-and-why)
25. [Known limitations](#25-known-limitations)
26. [Third-party code](#26-third-party-code)
27. [Quick reference](#27-quick-reference)

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
positions (explained in Chapter 3). The project will use a ReSpeaker XVF3800
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
| The data types every part of the code shares | **done** (Chapter 6) |
| The skeleton of the processing engine | **done** (Chapter 7) |
| The backend's web interface: health, configuration, live channel | **done** (Chapter 15) |
| The database tables | **done** (Chapter 16) |
| The website: three tabs, connection indicator | **done** (Chapter 17) |
| Choosing a microphone, recording it, checking its levels | **done**, step 02 (Chapter 9) |
| Live session: listening in headphones, live pictures, recording from the page | **done**, step 03 (Chapter 10) |
| Recognising sound types | **done**, step 05 (Chapter 12) |
| Keeping the chosen sounds, removing the rest | **done**: baseline, step 06 (Chapter 13) and neural extractor, step 07 (Chapter 14) |
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

*Since step 03:* the Semantic tab runs a **live session**. Press Start and the
level meters, spectrogram and waveform follow the microphone. Tick Listen (with
headphones) to hear it, and press Record to save what it hears. Saved recordings
are listed on the page and play in the browser. Nothing is removed from the sound
yet (Chapter 10).

*Since step 05:* the page shows, four times a second, which of the seven sound types it hears
(Chapter 12).

*Since step 06:* selecting sound types changes what you hear: they are kept and everything else
is turned down by 30 dB (adjustable), live in the headphones and, with the Process button, for
saved recordings, where an A/B player compares before and after. The method is a simple baseline
with clear limits; they are measured in Chapter 13.7.

*Since step 07:* a second method, **Neural**, is the default: a small neural network
(Waveformer, fine-tuned on the laptop for Sound Spy's seven types) that pulls the selected
sounds out of the mix sample by sample, also while other sounds overlap them, with half the
baseline's delay. On the test mixtures it is 8 dB better than the baseline as you hear it live
(+3.3 against −4.9 dB). It is no miracle: in about one mixture in nine it keeps the wrong sound,
and a perfect gate would still do better (Chapter 14.9). A **Method** selector switches between
the two, the default attenuation is now 15 dB, and an output level meter sits next to the input
meters.

### The story so far, step by step

Each step was specified by the project owner, built, tested on the laptop and written up here.

| Step | Date | What was built, in plain words | Read more |
|---|---|---|---|
| 00–01 | 13–14 Nov | **The frame.** The two programs (backend in Python, website in React), the shared data types, the rules every later step follows, a settings system, the database, the website with its three tabs and connection light. Nothing touched sound yet. | Chapters 5–8, 14–16 |
| 02 | 14 Nov | **Hearing.** Finding and choosing the microphone through *device profiles*, recording to WAV files with a description file next to each (the *sidecar*), measuring levels, and `soundspy check-mic`. Discovered that Windows' "audio enhancements" faked a single microphone and hid quiet sound; with them off, the laptop has two real microphones. | Chapter 9 |
| 03 | 14 Nov | **The live session.** Sound flows continuously from the microphone, in 20 ms pieces, through a processing chain to your headphones, with level meters, a spectrogram and a waveform on the page, and a Record button. About 94 ms of delay, no glitches in 60 s. | Chapter 10 |
| 04 | 15 Nov | **Training material.** Two public sound collections (ESC-50 and FSD50K, 53,197 clips) downloaded to drive D:, sorted into the seven sound types with the help of Google's AudioSet ontology, split fairly into training / validation / test, and turned into 24,000 recipes for synthetic 6-second mixtures. | Chapter 11 |
| 05 | 15 Nov | **Recognising.** A small pretrained network (EfficientAT) fine-tuned on the mixtures for 3.7 hours on the laptop's graphics card. It recognises the seven types far better than before fine-tuning (test score 0.56 vs 0.41) and runs four times a second live, plus `soundspy detect` for recordings. The owner's room test: every type lit up correctly, with two understandable confusions (classical music → alarm, singing → speech). | Chapter 12 |
| 06 | 15 Nov | **Keeping what you choose.** The detector now drives a *gate* and a noise reducer: selected sounds are kept, the rest is turned down. Bypass for A/B listening, input/output spectrograms, offline Process with an A/B player, a measuring framework that compares five versions of the method, and a database that can always be rebuilt from the files. The owner's listening test went smoothly. | Chapters 2 and 13 |
| 07 | 16 Nov | **Pulling sounds apart.** A published network for real-time sound extraction (Waveformer, MIT licence) was adapted, checked against its original code, and fine-tuned for the seven sound types: 25 GB of training mixtures rendered once to drive D:, half of them in simulated rooms, and 7.9 hours of training in two runs (the first one taught a lesson about the loss). It runs live at 44.1 kHz inside the 48 kHz session through a streaming resampler, adds 10.6 ms of delay, and is mixed back with a little of the original. Method selector, output level meter, 15 dB default. Test score +3.3 dB (baseline: −4.9 live, +2.2 offline) | Chapter 14 |

Along the way, problems were found and fixed, from a training run that silently ran out of memory
to Python's memory clean-up causing clicks; each one is written up in Chapter 23, and every
important choice with its reason in Chapter 24.

### Why it exists

It is a university project (Telecommunications, Technical University of
Cluj-Napoca). The ideas behind the sound-separation part come from published
research, mainly a project called **SemanticHearing** (Veluri et al., 2023;
see Chapter 20). Sound Spy borrows its ideas and, since step 07, the code of the network at its heart (Waveformer, from the same research group; Chapter 26).

---

## 2. Using Sound Spy today — a guided tour

**Exists today** (rewritten after step 07). This chapter is the user's manual. It needs no
technical background: it says what every button does, what you should see and hear, and what to
do when something looks wrong. The chapters after it explain *how* each part works inside.

### 2.1 What you need

- **The laptop with its built-in microphone.** Windows' audio "enhancements" for the microphone
  must be switched **off**; with them on, the two microphones sound identical and quiet sounds
  vanish (Chapter 9.10).
- **Wired headphones.** You will listen to the microphone live. Through the laptop's own
  speakers the sound would go straight back into the microphone and howl. Bluetooth headphones
  work but add 100–200 ms of extra delay.
- **The trained detection model** (`experiments/det_mn10_v1/checkpoints/best.pt`, made in step
  05). Without it the app still runs, but the buttons do not show which sounds are present, and
  the Baseline method only reduces steady background noise.
- **The trained extraction model** (`experiments/tse_waveformer_v1/checkpoints/best.pt`, made
  in step 07) for the **Neural** method. Without it the app uses the Baseline method and says so.
- **A web browser** (Chrome, Edge or Firefox).

### 2.2 Starting and stopping

Open two PowerShell windows:

```powershell
# window 1: the backend (does all the sound work)
cd C:\Users\User\Desktop\Sound_Spy\backend
.venv\Scripts\activate
soundspy serve

# window 2: the website
cd C:\Users\User\Desktop\Sound_Spy\frontend
npm run dev
```

Then open **http://localhost:5173**. In the header, the **connection light** should be green
and read *connected*. Red (*disconnected · reconnecting*) means the website cannot reach the
backend: check that window 1 is still running. The page reconnects by itself as soon as the
backend is back; there is no need to reload.

To stop, press **Ctrl+C** in each window. Closing the backend ends a running session cleanly;
an offline job that was still waiting or running is marked *failed (interrupted)* at the next
start.

### 2.3 The screen at a glance

The **Semantic** tab is the working tab. **Spatial** and **Sources** only say *"Requires a
microphone array"*: they need hardware that has not arrived yet.

```
┌─ Choose what to hear ───────────────────────────────────────────────────────────┐
│ [Start] Source: Microphone ▾   running                                          │  ← session bar
│ Input · Format · Latency · RTF p95 · Overflow/underrun/dropped ·                │
│ Detector · Inference · Skipped windows · Extraction · Model time                │
├──────────────────────────────────────────────────────────────────────────────────┤
│ ☐ Listen   Volume ───●── 0 dB   [Mute]                                          │  ← listening
│ Method: Neural ▾   Attenuation ───●──── 15 dB   [Bypass (A/B)]                  │  ← extraction
│ [● Record]  label (optional)  notes (optional)                                  │  ← recording
├──────────────────────────────────────────────────────────────────────────────────┤
│ (Speech 92%) (Music 4%) (Footsteps 1%) (Dishes & cutlery 0%) …                  │  ← sound types
│ Keeping 1 sound type(s); everything else is turned down by 15 dB.               │
│ ▒▒▒▒▒▒▓▓▓▓▓▓▒▒ detection timeline (last 30 s, one lane per type) ▒▒▒▒▒▒▒▒▒▒▒▒▒▒ │
├────────┬─────────────────────────────────────────────────────────────────────────┤
│ level  │ spectrogram: input  (above)                                             │
│ meters │ spectrogram: output (below)        [Input | Output | Both]             │
│ in|out │ waveform (input)                                                        │
├────────┴─────────────────────────────────────────────────────────────────────────┤
│ Recordings   Process with: Neural ▾                                             │
│ [Play] [Detect] [Process] name  label  duration  date                           │
│   └─ run: done  Speech  rest −15 dB  Neural … [Original|Processed] ──●── Download│
└──────────────────────────────────────────────────────────────────────────────────┘
```

| Part | What it is for |
|---|---|
| **Start / Stop** | Switches Sound Spy on: it starts listening, recognising and processing. Starting takes about 3–5 seconds, mostly loading the two models onto the graphics card. |
| **Source** | *Microphone*, or *File: …*, any saved recording played in a loop at normal speed. A file gives the same sound every time, which makes it ideal for comparing settings. Can only be changed while stopped. |
| **Session figures** (only while running) | **Input** and **Format**: which microphone, its sample rate and channel count (*48000 Hz · 2 ch* on the laptop). **Latency**: how long until a sound reaches your ears, about 95–115 ms with Neural (105–125 ms with Baseline). **RTF p95**: how hard the computer works for the sound (about 0.63 with Neural on the graphics card, 0.17 with Baseline; above 0.8 it turns red and a warning appears). **Overflow / underrun / dropped**: three counters of tiny audio glitches, which should stay at 0. **Detector**: the model and where it runs (`det_mn10_v1 · cuda` = the graphics card). **Inference**: how long one recognition takes (about 28 ms). **Skipped windows**: recognitions skipped because the previous one was not finished (should be 0). **Extraction**: the sound-keeping method, where its model runs and the delay it adds (`Neural · cuda · +10.6 ms`, or `Baseline · +21.33 ms`); for Baseline it says *no gating* in amber when there is no detection model; *bypassed* while Bypass is on. **Model time** (Neural only): how long the network needs for one call, about 7.5 ms on the graphics card; it has to stay well under 20 ms. |
| **Listen** | Sends the processed sound to your headphones. The first time, it asks you to confirm you are wearing headphones (the browser remembers your answer). It is greyed out, with a note, if the session could not open an output device. |
| **Volume, Mute** | Loudness in your headphones, −40 to +12 dB. A safety limiter always stops the sound before it gets dangerously loud. |
| **Method** | How the selected sounds are kept. **Neural** (the default once it is trained): a network that pulls the selected sounds out of the mix, also while other sounds overlap them (Chapter 14). **Baseline**: the detector-driven gate of step 06 (Chapter 13). Can only be changed while stopped. A method that cannot run is greyed out; hover it to read why. |
| **Attenuation** | How far everything you did **not** select is turned down: 0 dB = not at all, 15 dB (default) = to about 18 % of its strength, 30 dB = to about 3 %, 40 dB = the most. It only matters while something is selected. |
| **Bypass (A/B)** | Lets you hear the unprocessed sound instead, to compare (section 2.6). |
| **Record** | Saves what the microphone hears, with an optional label and notes. The button shows the running time while recording; press it again to stop. The recording is always the **raw, unprocessed** sound, so it can be processed again later with better methods. |
| **Sound-type buttons** ("chips") | One per sound type. The bar and percentage show how sure the app is that the sound is present right now (updated four times a second). The button turns **green** while the sound counts as present. **Click** to select it (blue border); click again to deselect. Several can be selected at once. The line below says what is selected and how strongly the rest is turned down. |
| **Detection timeline** | The last 30 seconds, one lane per sound type, coloured where it was detected (brighter = more certain). "Now" is at the right edge. |
| **Level meters** | **Input:** one bar per microphone: loudness, a white peak line, a held marker of the loudest moment in the last second, and a red **clip light** when the sound hits the ceiling. Normal speech close by reads about −30 to −20 dBFS; a quiet room about −60. **Output** (the narrow meter next to them, labelled *out*): the same for the sound after extraction. With something selected it sits lower than the input whenever other sounds are being removed. |
| **Spectrogram** | A picture of the sound by pitch over the last 10 seconds (section 2.7). The switch under it picks **Input**, **Output** or **Both**. |
| **Waveform** | The last 5 seconds of the input: tall = loud. |
| **Recordings** | Every saved recording, newest first, with **Play**, **Detect** and **Process**, and a **Process with** choice of method (section 2.10). |
| **Notices** (bottom right) | Warnings and errors, e.g. *"Sound detection is off for this session"*. Click **×** to dismiss. The same message is never shown twice at once. |

### 2.4 Your first session, step by step

1. Put on your headphones. Leave *Source* on **Microphone** and press **Start**.
2. Wait ~3–5 s until the state reads *running*. The meters, spectrogram and waveform start
   to move. Check the figures: *Detector* says `cuda`, *Skipped windows* is 0, *Extraction* says
   `Neural · cuda`.
3. Tick **Listen** and confirm the headphones. You now hear the room, about a tenth of a second
   late. With nothing selected, nothing is changed.
4. Talk. The **Speech** button turns green within about half a second, and a lane fills in the
   detection timeline.
5. **Click Speech** to select it. Stop talking: the room becomes quieter, and the output level
   meter drops below the input meters. Talk again: your voice comes through at once.
6. Press **Bypass (A/B)** and talk: you hear the room as it really is. Press it again to return to
   the processed sound.
7. Move **Attenuation** to 30 dB: everything that is not speech nearly disappears, and any
   mistake of the network becomes easier to hear. At 5–10 dB the effect is gentle.
8. Press **Stop**, switch **Method** to *Baseline*, press **Start** and repeat steps 4–6 to hear
   the difference between the two methods (section 2.8).
9. Press **Stop** when you are done. Your selection, the method, the attenuation, Listen, volume
   and Bypass are kept for the next session while the backend runs.

### 2.5 Choosing what to hear

- **Nothing selected:** the sound passes unchanged (the line under the buttons reads *"Select one
  or more sounds to enhance them"*).
- Every change takes effect within one 20 ms piece of sound and fades in or out over 10–20 ms, so
  it never clicks.

What a selection does depends on the **Method**:

- **Neural.** The network rebuilds the sound with only the selected types in it, continuously;
  everything it removed is added back *attenuation* dB quieter. Selecting **several types** asks
  for all of them at once. It does not wait for the detector, and it can keep a voice while
  removing music that plays at the same moment. Its typical mistake is to let a piece of the
  wrong sound through, or to dull the right one (section 2.8).
- **Baseline.** The selected type is kept whenever the detector believes it is present (with
  several types: when **any** of them is present); while it is absent, everything is turned down
  by the attenuation. While the chosen sound is present, it lets the sound through (the *gate*
  is open) and reduces whatever steady background it learnt during the pauses (a fan, a hum,
  hiss). It does **not** remove other sounds that happen at the same time.

### 2.6 Bypass (A/B): what it is and how to use it

**A** and **B** are the two versions of the same moment: the **processed** sound (what Sound Spy
makes of it) and the **unprocessed** one (what the microphone hears, mixed to one channel). With
**Bypass on**, your headphones get the unprocessed version; with it off, the processed one.

Two details make it a fair comparison:

- **Same delay.** The processing makes the sound a little late (10.6 ms with Neural, 21 ms with
  Baseline). The unprocessed version is delayed by exactly the same amount, so pressing Bypass
  changes only the processing, never the moment you hear. You can switch in the middle of a word
  without a jump.
- **Processing keeps running.** Bypass only changes which version goes to the headphones, so
  switching back is instant.

Bypass changes **only what you hear**. The pictures do not change: the output spectrogram keeps
showing the processed sound, so with Bypass on you can *hear* the original while *seeing* what
the processing would have done. The session bar shows *bypassed* while it is on.

How to use it: select a sound type, listen for a few seconds, then press Bypass and listen to the
same situation. If the processed version is not clearly better, try a lower attenuation, or the
sound is one the method handles poorly (section 2.8).

### 2.7 Reading the pictures

**The spectrogram** shows sound by pitch: time runs left to right (now = right edge), low pitches
at the bottom, high ones at the top, and colour shows loudness (dark = quiet, yellow = loud). A
voice is a stack of bright horizontal stripes (its harmonics) that wiggle; a clink is a thin
vertical line; music is many steady stripes; a fan or hum is a constant band; footsteps are short
vertical smudges low down.

With **Both** chosen, the **upper** picture is the input (the microphone) and the **lower** one is
the output (after processing). What you should see:

| Situation | Upper picture (input) | Lower picture (output), **Neural** | Lower picture (output), **Baseline** |
|---|---|---|---|
| Nothing selected | the room | the same as above | the same as above |
| A type selected, not present | the room | somewhat darker everywhere (not always the full 15 dB) | darker everywhere (15 dB = a few colour steps) |
| A type selected and present | the room with the sound | the sound's own shapes stay bright (a voice's stripes, a clink's line), from its very start; what lies between and around them is darker | everything is bright while the sound lasts, starting about 0.3 s after it begins and staying for about a second after it ends |
| A type selected, sound present, and other sounds at the same time | everything | the chosen sound's shapes stay bright, the other sounds' shapes are dimmer (not gone: 15 dB) | everything that happens *during* the chosen sound comes through |
| Attenuation at 0 dB | the room | the same as above | the same as above |
| Bypass on | the room | unchanged by Bypass: still the processed sound | the same |

**The input level meters and the waveform** show the **input**: they tell you whether the
microphone hears properly. **The output level meter** shows what the processing lets through:
with a type selected and absent it sits below the input: with Baseline by about what the
attenuation says, with Neural usually by less.

**The detection timeline and the buttons** show what the app recognises. Green means "counts as
present". The detector decides from the last full second of sound and smooths its answers, so a
type turns green about 0.3–0.5 s after the sound starts and stays green 0.6–1 s after it stops.
This is deliberate: without smoothing the buttons would flicker.

### 2.8 What you should hear

The table below describes the **Baseline** method (as measured and listened to in step 06,
at 30 dB). With the default 15 dB the "almost silence" in it becomes "a clearly quieter room".
What changes with **Neural** follows after it.

| Situation (headphones, Listen on, Method = Baseline) | What you hear |
|---|---|
| Nothing selected | everything, about 0.1 s late |
| Speech selected, nobody talks | almost silence (the room 30 dB down) |
| Speech selected, you talk | your voice; the very first syllable may be quiet or cut, and the room stays audible for about a second after you stop |
| Speech selected, steady fan or hum in the room, you talk | your voice, with the hum somewhat reduced even while you talk |
| Speech selected, music plays, you are silent | the music turned down, **unless the song has singing** (vocals count as speech, Chapter 25 item 21) |
| Speech selected, music plays, you talk | your voice **and** the music. This is the method's main limit: while a chosen sound is present, whatever else sounds at the same time comes through with it |
| Dishes selected, a single clink | often nothing, or the second after the clink. Short sounds are the hardest case live |
| Dishes selected, cutlery clinking for several seconds | the clinking (after the first moment), the room in between it |
| Footsteps selected, someone walks | the steps, once the walking has gone on for a moment |
| Music selected, classical music | the music; "Alarm" may light up too, which only matters if Alarm is selected |
| Several types selected | the sound whenever any of them is present |
| A sudden loud noise (a door) with Speech selected | faintly, 30 dB down, unless the detector mistakes it for speech |
| A chosen sound the app fails to recognise | turned down like everything else. A lower attenuation makes this less drastic |
| Bypass on | everything, unprocessed, equally late |

**Why it behaves like this, in plain words.** The method has no idea *which part* of the sound is
speech; it only knows, from the detector, *whether* speech is present right now. It works like a
volume fader: up while speech is present, down otherwise. Two consequences:

1. **It reacts late.** The detector needs a moment to be sure, and waits a moment before it
   lets go:

```
 what happens:      ........|==== you talk ====|...................
 button green:      ...........|==================|=====|..........   on ~0.3 s late, off ~1 s late
 fader up (gate):   ...........|========================|==|.......   + 0.3 s hold, 0.2 s fade
 what you hear:     quiet......|your voice........|room...|quiet...
```

   Saved recordings processed with **Process** do not have this problem: the whole file is
   analysed first, so the fader moves in time with the sound.
2. **It cannot pull one sound out from under another.** When the fader is up, everything comes
   through; only steady background is reduced. Separating sounds that overlap needs a model that
   has learnt what each type sounds like: that is the Neural method.

**With Method = Neural.** These are expectations from the measurements on test mixtures
(Chapter 14.9), not yet from long listening in a real room, so treat them as things to check:

| Situation (headphones, Listen on, Method = Neural, 15 dB) | What to expect |
|---|---|
| Speech selected, nobody talks | the room, somewhat quieter. Measured: about 8 dB, less than the slider's 15, because the network does not go fully quiet when the selected sound is absent |
| Speech selected, you talk | your voice **from the first syllable** (no detector to wait for), and the room goes quiet again as soon as you stop |
| Speech selected, music plays, you talk | your voice in front, the music behind it. This is the case the Baseline cannot handle at all |
| Speech selected, music with singing | the singing counts as speech here too |
| Dishes selected, a single clink | the clink itself, at the moment it happens (short sounds are no longer missed because of a slow detector) |
| Footsteps, dishes, alarm or speech selected | the classes with the largest measured gain (3.6–4.4 dB on the test mixtures) |
| Music or traffic selected | a smaller gain (about 2.5 dB), but still where the Baseline achieves nothing |
| Chair scraping selected | hardly any effect: the weakest class for both methods |
| Several types selected | all of them together; the measured gain is smaller than with one type |
| The network picks the wrong sound | you hear a piece of something you did not select at nearly full level, or the selected sound turns dull. At 15 dB this is mild; at 30 dB it is obvious. Compare with Bypass |
| A reverberant room | only slightly worse than a dry one in the measurements (0.4 dB; the training included simulated rooms) |

The two methods fail differently. Baseline is **late and coarse** (it opens for everything, a
moment too late). Neural is **immediate and selective, but can be wrong about the content**.
Comparing them on the same sound, with Bypass as the reference, is the best way to learn both.

### 2.9 Recording

Type an optional **label** (e.g. `cafeteria-test`) and **notes**, press **Record**, and press it
again (it then reads *Stop 0:12* or similar) to finish. The recording appears at the top of the
list. It contains every microphone channel, unprocessed, at the microphone's own sample rate, as a
WAV file plus a small JSON file with its details in `data\raw\`. Recordings are never changed
afterwards.

### 2.10 Working with saved recordings

Each recording in the list has three buttons:

- **Play** plays the original in the browser (press *Close* to hide the player).
- **Detect** shows, under the recording, one lane per sound type with the stretches where it was
  found (hover a stretch for its exact times and certainty). Recordings longer than 2 minutes are
  refused here; use `soundspy detect` for those.
- **Process** keeps the sound types **currently selected** above, turning the rest down by the
  **current attenuation**, with the method chosen in **Process with** at the top of the list
  (it starts on the session's method). It is greyed out until something is selected (the tooltip
  says so). Recordings longer than 5 minutes are refused here; use `soundspy extract`.

After Process, a **run** appears under the recording:

| What the row says | Meaning |
|---|---|
| *waiting…* | queued: only one recording is processed at a time, the others wait their turn |
| *processing…* | being processed (for a one-minute recording about 7 s with Neural and about 6 s with Baseline, plus a few seconds the first time while the model loads) |
| *done* | finished: the A/B player appears |
| *failed* | it did not work; the reason is written underneath |

The row also shows the selected types, *rest −15 dB* (the attenuation used), the method
(*Neural* or *Baseline*) and how long processing took. Every run is kept, so you can process the
same recording with both methods, or with different selections, and compare them in their A/B
players.

**The A/B player** of a finished run has one **Play/Pause** button, one **position** slider and
an **Original / Processed** switch. Switching keeps the position: you hear the same moment with
and without processing. It starts on *Processed*. **Download** saves the processed WAV.

With **Baseline**, offline processing is **better than live**: the whole file is analysed
first, with each moment judged from the sound around it, so the fader moves in time with the
sound. On the test mixtures it scored about 7 dB better than live (Chapter 13.7). With
**Neural** there is no such difference: the network does the same thing offline and live
(Chapter 14.9).

A recording whose file was deleted or moved stays in the list, greyed out with **file missing**:
it cannot be played or processed, but its runs and their processed sound remain available.

### 2.11 The same from the command line

Run these in window 1 (stop the server first with Ctrl+C, or use a third window with
`.venv\Scripts\activate`):

| Command | What it does |
|---|---|
| `soundspy devices` | lists microphones and speakers |
| `soundspy check-mic` | records a few seconds and reports levels and problems (Chapter 9.8) |
| `soundspy record --seconds 10 --label test` | records into `data\raw` |
| `soundspy detect data\raw\<file>.wav` | which sound types are in a recording, and when |
| `soundspy extract data\raw\<file>.wav --labels speech` | keeps speech, turns the rest down by 15 dB (`--attenuation-db`, `--method neural_tse` or `mask_baseline`, `--out copy.wav`); the result appears on the page as a run |
| `soundspy evaluate extraction --method neural_tse` | measures both extraction methods on 500 test mixtures, dry and in simulated rooms (about an hour; `--method mask_baseline` alone: about 4 minutes) |
| `soundspy db rebuild` | rebuilds the app's database from the files on disk (never loses anything; Chapter 16) |

The data, training and detection-evaluation commands are described in Chapters 11 and 12, and training the neural extractor in Chapter 14.

### 2.12 When something looks wrong

| You notice | Likely cause and what to do |
|---|---|
| Red connection light | the backend is not running: start `soundspy serve` in window 1 |
| *"could not open the audio device"* after Start | a device was just plugged in or out, or another app holds the microphone: wait a moment and press Start again |
| **Listen** greyed out, *"Listening unavailable"* | the output device changed: Stop and Start again |
| Listen is on but you hear nothing | is **Mute** on, or the volume low? Is a type selected that is not present (silence is then correct)? Press Bypass, or deselect everything, to check |
| Notice *"Sound detection is off for this session"*, *Extraction … no gating* in amber | the detection model file is missing: the buttons show nothing, and Baseline then only reduces steady noise (Neural still works). Check that `experiments\det_mn10_v1\checkpoints\best.pt` exists |
| Notice *"Neural extraction is not available … Using the baseline"*, *Neural (not available)* in the Method list | the trained extraction model is missing: check that `experiments\tse_waveformer_v1\checkpoints\best.pt` exists (it is made by `soundspy train extraction`, Chapter 14.6) |
| The Method list is greyed out | a session is running: press Stop first |
| *RTF p95* is red, crackles, with Method = Neural | the network is not keeping up: is *Extraction* showing `cpu` instead of `cuda`, or is something else using the graphics card (a training run)? See Chapter 14.7 |
| Clicks or crackles, the underrun counter rises | the computer is busy with something else: close heavy programs; check that RTF stays low |
| *Skipped windows* keeps rising | the detector cannot keep up: is it running on `cpu` instead of `cuda`? |
| A button lights up for the wrong sound | a detector mistake. Known ones: classical music → Alarm, songs with singing → Speech (Chapter 25) |
| **Process** greyed out | nothing is selected above, or the recording's file is missing |
| Notice *"Processing failed: recording is … s long; processing on the page is limited to 300 s"* | use `soundspy extract` for long recordings (the same for *Detect* above 120 s: use `soundspy detect`) |
| A run says *failed* | the reason is written under it, e.g. the recording file is gone |
| After reloading the page, no buttons are selected but the sound is still processed | the backend remembers the last selection, but a freshly loaded page does not know it (a known gap). Click a type twice (select, deselect), or select what you want again, to bring them back in step |
| Both microphone channels identical, the room at −120 dBFS | Windows' microphone enhancements are back on: switch them off (Chapter 9.10) |

### 2.13 Good ways to try it

- **Use a file as the source** for comparisons: record 20–30 s once (e.g. talking over music),
  then choose *File: …* as the source. The same sound plays in a loop, so you can try settings and
  Bypass on identical material.
- **Compare the two methods on one recording:** select a type, choose *Neural* in **Process with**
  and press **Process**, then choose *Baseline* and press it again. The two runs appear under the
  recording, each with its own A/B player.
- **Compare live with offline (Baseline):** record a test, select a type, listen live, then press
  **Process** on the recording and use the A/B player. The difference is the detector's lag.
- **Try different attenuations:** 30 dB gives the strongest effect; 10–15 dB sounds more natural
  and hides the moments when the app is late or wrong.
- **Watch Both spectrograms** while you talk over music: with Baseline you can see the music come
  through whenever your voice opens the gate; with Neural the music's stripes stay dimmer.
- **Listen to the evaluation examples** in `outputs\eval\ext_neural_tse\` (described in its
  `index.md`): for each test mixture there is the input, the correct answer and the result of
  every system, Baseline and Neural side by side.

### 2.14 What it cannot do yet

- Separate overlapping sounds *reliably*. The Neural method does separate them, but its gain is
  modest and uneven across the sound types, and it sometimes keeps the wrong sound
  (Chapter 14.9–14.10).
- Tell two sounds of the **same type** apart (your voice from another voice): both are "speech".
- Use the direction of a sound (Spatial tab, modules 3–4) or follow one source (Sources tab,
  module 5); both need the microphone array.

---

## 3. Glossary

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
Spy has seven classes (Chapter 6).

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

*Added in step 04:*

**Mid.** The permanent ID of an AudioSet category, such as `/m/09x0r` for "Speech". Names can be
spelled differently from dataset to dataset; mids can't, so all translation goes through them.

**Foreground / background / excluded.** The role of a dataset clip (Chapter 11.6): *foreground*
clips contain exactly one of our classes and can be asked for; *background* clips contain none of
them; *excluded* clips mix one of our classes with something else and are not used.

**Recipe.** A few lines of JSON describing one synthetic mixture exactly: which clips, which part
of each, where, how loud, and which classes form the answer. Audio is made from it on demand.

**Activity map.** For each clip, its loudness every 0.1 s, used to place events where something
actually sounds instead of on silence.

*Added in step 05:*

**Fine-tuning.** Continuing the training of a model that was already trained on a large dataset
(here AudioSet), on our own task, with small learning rates, so it keeps what it knew.

**Zero-shot.** Using a pretrained model for our task *without* any training on it (here: reading
our classes off its 527 AudioSet outputs). The baseline a fine-tuned model must beat.

**AP / mAP.** Average precision: how well a detector ranks the windows that contain a class above
those that don't (1.0 = perfect, ~the share of positive windows = random). mAP = mean over classes.

**Threshold / hysteresis.** The probability above which a class counts as "present". With
hysteresis it switches on at the threshold but off only somewhat below it, so it does not flicker.

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
This difference caused a real problem (Chapter 23).

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

**Device profile.** A small settings file saying which microphone to use and how (Chapter 9.2).

**`raw_channels`.** The list of a device's channels that are real microphones and should be kept
(Chapter 9.3).

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

### Live session (added in step 03)

**Session.** Sound Spy switched on: continuously reading the microphone, processing, showing and
optionally playing and recording the sound (Chapter 10).

**Worker thread.** The one background thread that processes the chunks of a session.

**Underrun.** The headphones needed sound but none was ready, so a short silence (a click) was
played instead.

**Prebuffer / jitter buffer.** A small amount of sound collected before playback starts, so small
timing irregularities don't cause underruns. It costs a little delay.

**Real-time factor (RTF).** Processing time divided by the length of the sound processed. Below 1
the computer keeps up; above 1 it falls behind.

**p95 (95th percentile).** The value that 95 % of measurements stay below: a "nearly worst case"
that ignores the rare extreme.

**Limiter.** An automatic volume control that only acts on peaks, turning the sound down just
enough to keep it below a ceiling.

**Acoustic feedback.** Sound from a speaker going back into the microphone, being amplified and
played again, until it howls. The reason Listen needs headphones.

**Canvas / requestAnimationFrame.** A `<canvas>` is a drawing surface on a web page. The browser
calls `requestAnimationFrame` once per screen refresh, which is the right moment to redraw it.

### Extraction (added in step 06)

**Extraction.** Keeping the selected sounds and turning everything else down. The stage that
does it is the *extractor* (Chapter 13).

**Gate.** A volume control that a program moves by itself: up while a wanted sound is present,
down otherwise. Sound Spy's gate opens in 10 ms, stays open 300 ms after the sound stops (the
*hold*) and closes over 200 ms, so nothing clicks and word endings survive.

**Attenuation / floor.** How far "down" is: 30 dB by default, i.e. the unwanted sound keeps
about 3 % of its amplitude. Nothing is ever muted completely.

**Noise estimate.** The method's idea of what "everything else" sounds like, per frequency:
learnt while the wanted sound is absent, and subtracted while it is present.

**Minimum statistics.** A way to estimate steady background noise without knowing when the
wanted sound is present: the quietest level each frequency reached in the last 1.5 s.

**Activity source.** Where the extractor learns *whether* the wanted sound is present: the live
detector, a timeline computed for a whole file, or (in tests) the truth.

**Oracle.** In an evaluation, a part of the system replaced by the *true answer*, to see how good
the rest could be at best. *Oracle activity* = the true moments the target sounds; *IRM oracle*
(ideal ratio mask) = the best possible gain for every time-frequency cell.

**SI-SDRi, SNRi.** The improvement in SI-SDR (or SNR) over the unprocessed mixture. 0 dB = no
better than doing nothing; negative = worse.

**Suppression.** How much quieter the output is than the input, in dB, when the wanted sound is
not there at all (the right output is then silence).

**Bypass (A/B).** Hearing the unprocessed sound instead of the processed one, to compare. A and B
are the two versions; a fair comparison needs both to be equally delayed.

**Processing run.** One offline processing of a recording: a processed WAV plus a JSON sidecar
in `outputs/processed/`.

**Garbage collection.** Python's periodic clean-up of memory that is no longer used. A full
clean-up stops every thread of the program for its duration.

### Neural extraction (added in step 07)

**Target sound extraction (TSE).** Giving a network a mixture and the names of the wanted sound
types, and getting back only those sounds. Unlike a gate, it works while sounds overlap
(Chapter 14).

**Waveformer.** The published network Sound Spy's neural extractor is built on: small, causal,
made for real-time extraction (Chapter 14.2).

**Causal.** Using only the past and the present, never the future. A live system has no choice.

**Lookahead.** The small piece of future a system waits for before it answers. Waveformer waits
0.7 ms (32 samples); it becomes part of the delay.

**Encoder, decoder, mask.** The three parts of the network: the encoder turns sound into a
compact description, the decoder decides which parts of it belong to the wanted sounds (the
mask, numbers between "drop" and "keep"), and a last layer turns the kept parts back into sound.

**Label embedding / conditioning.** How the network is told what you want: the on/off switches
of the seven classes are turned into numbers that steer the rest of the network.

**Parameters.** The adjustable numbers inside a network (3.86 million here). Training sets them.

**Loss.** The single number that says how wrong the network's output is for one example.
Training is the search for parameters that make it small (Chapter 14.5).

**SNR and SI-SNR.** Two "how close is the output to the target" scores in dB. SNR also demands
the right loudness; SI-SNR ignores loudness. SI-SNR is the same thing as SI-SDR.

**Residual mixing.** Adding back, at reduced level, what the network removed:
`output = estimate + g × (input − estimate)`. It is how the attenuation slider works for the
neural method (Chapter 14.8).

**Streaming state.** The small memories a network keeps between two pieces of sound, so that
processing piece by piece gives the same result as processing everything at once.

**Resampling.** Converting sound from one sample rate to another (48 kHz ↔ 44.1 kHz here). A
*streaming* resampler does it chunk by chunk without seams.

**Reverberation, impulse response (RIR), RT60.** Reverberation is the sound that reaches you a
moment later from the walls. An impulse response is the room's answer to a single click from one
position; passing a dry sound through it puts the sound "into" that room. RT60 is how long the
echo takes to fade by 60 dB (0.2 s = a small furnished room, 0.8 s = a large bare one).

**Cache, shard, memory mapping.** The cache holds the training mixtures already rendered, in
large files (*shards*) of 256 examples each. Memory mapping lets the program treat such a file
as if it were in memory while the operating system loads only the parts that are read.

**Crop.** A randomly placed 3-second piece of a 6-second training example.

**Gradient accumulation.** Adding up the corrections from several small batches before applying
them, when one large batch does not fit the graphics card's memory.

**Early stopping, patience.** Ending the training when the validation score has not improved for
a set number of rounds (the patience), and keeping the best round.

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

## 4. How sound lives inside a computer

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
step 02, Chapter 9.7, and an automatic test now scans the code for `[:, …]`-style
channel indexing outside the few files allowed to use it.)

---

## 5. How the whole app fits together

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
(Chapter 9). What doesn't exist yet is the engine running live between them
(step 03) and the stages that change the sound (steps 05–07).

*Update after step 03:* the engine now runs live. Microphone or file → chunks →
chain → speaker output, with the analyzer's live pictures and recording
(Chapter 10). The extractor in the chain is still a stand-in that changes
nothing; the detector and localizer don't exist yet.

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
(Chapter 22). The benefit is that the sound-processing code can be used,
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

## 6. The core building blocks

**Exists today.** Location: `backend/src/soundspy/core/`.

These are the shared "nouns" of the whole project. They contain no audio
processing at all. They are agreements about what the data looks like, which
every other part relies on. Getting them right first is what lets the later
steps be written independently.

### 6.1 `AudioBuffer` — a piece of sound

Described in Chapter 4. In code it is a *frozen dataclass*: once created, it
can't be changed. It also checks itself when it is created and refuses
anything malformed:

- `samples` must be a NumPy array (a fast table of numbers) of float32 values;
- it must have exactly two dimensions and at least one channel;
- `sample_rate` must be a positive whole number.

It answers three questions: `num_frames`, `num_channels`, and `duration_s`
(the length in seconds). It deliberately has **no** "convert to mono" method,
because that would invite exactly the channel-dropping the rules forbid.

### 6.2 `Direction` — where to listen

Two numbers:

- **`azimuth_deg`**: the compass direction, from 0 up to (but not including)
  360 degrees. Values outside that range are wrapped around automatically,
  so 370° becomes 10° and −90° becomes 270°.
- **`width_deg`**: how wide the listening "beam" is, more than 0 and at most
  360 degrees. The default is 30°.

### 6.3 `TargetSpec` — what the user wants

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

### 6.4 Events — what the side stages report

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

### 6.5 The seven sound classes — `LabelSet` and `classes.yaml`

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
*(Step 04: checked against the real ontology. Every name exists, so nothing had to change
(Chapter 11.5). What remains is a question of meaning, not spelling: AudioSet's "Vehicle" also
contains aircraft, boats and trains (Chapter 25).)*

**The order of the list matters.** A class's position in the file is its slot
in every multi-hot vector. If the file were reordered, a model trained when
"dishes" was slot 4 would suddenly read slot 4 as something else. New classes
are therefore always added at the end, and the list is never shuffled.

`LabelSet` is the Python object built from that file. It can convert a class
name to its position and back, turn a set of names into a multi-hot vector
and back (with a threshold, so that a probability of 0.5 or more counts as
"yes"), and reject unknown names.

---

## 7. The processing engine

The *contracts* below **exist today**. The machinery that runs them live is
planned for step 03. Location: `backend/src/soundspy/engine/`.

### 7.1 One idea: everything is a stage that handles one chunk

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

### 7.2 The `ProcessorChain` — stages in a line

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
(Chapter 23).

### 7.3 Modes — which stages each tab uses

| Mode (tab) | Main stages, in order | Side stages | Needs an array? |
|---|---|---|---|
| **Semantic** | extractor | detector, analyser | no |
| **Spatial** | spatial filter → extractor | detector, analyser, localiser | **yes** |
| **Sources** | spatial filter → extractor | detector, analyser, localiser, tracker | **yes** |

Today only this table exists. Building real chains from it comes later. The
website already uses it to decide whether a tab is available (Chapter 17).

### 7.4 How live processing will run (planned, step 03)

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

*Implemented in step 03, exactly as planned here; Chapter 10 describes how it
works. The chain can now also name its stages (`reducer`, `extractor`,
`analyzer`), so a later step can swap the stand-in `extractor` for the real one.*

---

## 8. Configuration — settings live in files, not in code

**Exists today.** Location: `backend/configs/`, read by
`backend/src/soundspy/config.py`.

### 8.1 The rule and the reason

Sample rates, chunk sizes, file paths, ports, the class list and the model
settings all live in settings files written in **YAML** (a simple
human-readable text format), not inside the Python code. Changing a value
means editing a text file, not code, and the same code can run with different
settings. The `TODO(step-XX)` markers in these files show which values are
only placeholders for now.

### 8.2 The files

| File | What it holds |
|---|---|
| `default.yaml` | folder paths, which microphone profile is active, engine chunk size (1,024 frames), backend address and port, allowed website address, heartbeat interval (2 s), database location. *Since step 03:* the chunk is `engine.chunk_ms` = 20 ms, and four new sections hold every live-session setting: `capture` (stall timeout, input queue, start-up silence), `engine` (chunk, RTF warning, status rate), `output` (device, prebuffer, queue, limiter, gain) and `analysis` (rates and sizes of the live pictures). These have **no fallback values in code**: a missing or invalid key stops the app at start-up with a message naming it |
| `classes.yaml` | the seven classes (Chapter 6.5) |
| `default.yaml` → `paths` | *Step 04:* the large folders (`external`, `processed`, `mixtures`, `model_cache`) now point to drive D:; absolute paths are allowed (Chapter 11.2) |
| `devices/laptop.yaml` | the laptop's own microphone: system default device, preferred host API, native sample rate, all channels, no geometry |
| `devices/respeaker_xvf3800.yaml` | the future 4-mic array: a device name to look for, channels 2–5 (the raw microphones; channels 0–1 are the board's own processed output), geometry to be filled in |

Since step 02 a profile has five fields (`device`, `hostapi`, `sample_rate`,
`raw_channels`, `geometry`) and is checked when it is loaded. Chapter 9.2 explains
each field.
| `detection/default.yaml` | the future sound detector (model not yet chosen). *Step 05:* the real detector: backbone and pretrained weights (URL + pinned SHA-256), window (1 s), input context (10 s), labelling, the checkpoint used live, live settings (hop, smoothing, device), training, augmentation and evaluation settings; every key required and checked (Chapter 12) |
| `detection/audioset_labels.csv` | *Step 05:* the 527 AudioSet outputs of the pretrained model (index, mid, name), from AudioSet via EfficientAT |
| `extraction/mask_baseline.yaml` | the future classical extractor (spectrogram settings, mask type) |
| `extraction/neural_tse.yaml` | the neural extractor (step 07): the network's shape, where its published weights come from (with a checksum), how the 41 published classes map to ours, the trained file to use, live settings, the cache and the simulated rooms, and every training setting (Chapter 14) |
| `data/default.yaml` | *Step 04:* everything about training data: which datasets are on, the role warning threshold, activity settings, split folds and fractions, and the mixture rules (counts, seeds, ranges, peak limit). Every key is required and checked at start-up |
| `data/esc50_to_audioset.yaml` | *Step 04:* ESC-50's 50 labels mapped by hand to AudioSet mids (Chapter 11.5) |

### 8.3 Device profiles — swapping microphones without touching code

A **device profile** describes one microphone setup: which device to open,
which of its channels are real microphones, and where those microphones sit.
Switching from the laptop microphone to the array will mean changing one line
in `default.yaml` (`active_device_profile: respeaker_xvf3800`), not editing
the program. For a single run, `--profile respeaker_xvf3800` on any audio command
does the same.

The profile also decides what the website offers. The laptop profile has no
geometry, so the backend reports Tabs 2 and 3 as unavailable and the website
says *"Requires a microphone array"*.

### 8.4 Where the "project root" is

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

*Step 04:* downloaded datasets, the clip index and the mixture recipes do **not** go into
`data/`: they live on drive D: (`D:/SoundSpyData/`, Chapter 11.2). `data/` keeps the database,
the own recordings (`raw/`) and the own labelled clips (`custom/`).

`data/`, `models/` and `outputs/` are kept out of version control. They hold
large files that can be rebuilt or downloaded again, and recordings that
shouldn't be shared.

---

## 9. Audio capture — getting sound in from a microphone

**Exists today** (step 02). Location: `backend/src/soundspy/io/`, `backend/src/soundspy/dsp/`,
and the `devices`, `check-mic` and `record` commands.

This is the first part of Sound Spy that touches real sound. It answers four questions: *which
microphone?*, *how does its sound reach our code?*, *how is a recording stored?* and *is the
microphone actually giving us honest sound?* The last one turned out to matter more than
expected (8.8).

### 9.1 The short version

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

### 9.2 Choosing a microphone

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

### 9.3 `raw_channels`, using the ReSpeaker as the example

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
is allowed in `io/` (Chapter 4's rule is about not *throwing away* microphones).

### 9.4 How sound arrives: `MicSource`

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

### 9.5 How a recording is stored

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

### 9.6 How the database knows about recordings

The **sidecar is the source of truth**, and the database is only an **index** of the sidecars,
like a library catalogue. The catalogue can always be rebuilt by walking the shelves.

- When the backend starts, and every time the website asks for `GET /api/recordings`, it walks
  through `data/raw/`. It adds a database row for every sidecar it doesn't know yet, and
  **deletes** rows whose WAV file has disappeared. If the file comes back, the next walk adds it
  again.
- `soundspy record` only writes files. It never touches the database, so the sound library
  doesn't need any database code (Chapter 5's rule), and you can record with the backend
  stopped. The recording shows up the next time the API looks.
- Paths are stored relative to the project root (`data/raw/test_….wav`), so the database still
  works if the project folder moves. Times are stored in UTC, because SQLite doesn't remember
  time zones and mixed offsets would break the "newest first" order.

### 9.7 Measuring levels

`dsp/levels.py` measures, per channel:

| Measure | What it tells you |
|---|---|
| **RMS dBFS** | average loudness; a full-scale sine wave reads −3.01 dBFS |
| **peak dBFS** | the loudest single sample; 0 dBFS is the maximum |
| **noise floor** | the background level: the quietest 10 % of 50 ms windows (room tone, fan, mic hiss) |
| **clipping** | the share of samples at or above 0.999, which are too loud and distorted |
| **digital silence** | the share of the recording made of *stretches* (≥ 5 ms) of exact 0.0, which only a gate, suppression or a muted input produces. Isolated zeros are normal (Chapter 23.12) |

Total silence is reported as −120 dBFS rather than "minus infinity".

**Comparing two channels.** To tell whether two channels come from two *different* microphones,
the app slides one channel against the other by up to ±1 ms and finds where they match best (a
*cross-correlation*). The shift is the **lag**: a **positive lag** means the second channel heard
the sound *later*. A sound from the left reaches the left microphone first. Microphones a few
centimetres apart differ by a fraction of a millisecond, which is a few samples at 48 kHz.

**`ChannelReducer`** (`dsp/channels.py`) also exists now. It turns any number of channels into
one, either by averaging (`mean`) or by picking one (`select`), and always returns the 2-D
`(frames, 1)` shape. It is the one sanctioned place to do this (Chapter 4), and a test scans the
code to make sure nobody does it anywhere else.

### 9.8 Reading the `check-mic` output

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
    zeros. That is normal (Chapter 23.12).

### 9.9 The clap test

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

### 9.10 Switching off Windows audio enhancements

Settings → System → Sound → choose the microphone under *Input* → **Properties**. Turn **Audio
enhancements** off; on newer Windows versions also turn off **Voice Clarity**, if present.
Some Realtek drivers have their own effects (noise suppression, beamforming) in the Realtek
Audio Console, so check there too. The development laptop also has **Nahimic** installed (its
virtual "Nahimic" speakers appear in `soundspy devices`). Nahimic has its own microphone effects,
such as noise suppression, which must be switched off in the Nahimic app as well. Then run `soundspy check-mic` again: an honest microphone shows
a noise floor in the −70 to −50 dBFS range and no "noise suppression" warning.

---

## 10. Live session — from microphone to headphones and screen

**Exists today** (step 03). Location: `backend/src/soundspy/engine/session.py`,
`engine/factory.py`, `engine/buffers.py`, `io/speaker_sink.py`, `analysis/analyzer.py`,
`api/services/session_manager.py`, and the Semantic tab of the website.

A **session** is Sound Spy "switched on": it listens continuously, passes the sound through the
processing chain, can play the result into your headphones, draws it live on the page, and can
record it. Nothing is removed from the sound yet: the stage that will keep only the chosen sounds
(the *extractor*) is still a stand-in that changes nothing. Step 05 onwards replaces it.

*Corrected after step 06:* the extractor is real now. Selecting sound types on the page keeps
them and turns everything else down; Chapter 13 describes how. What changed in the session itself
is noted in the sections below.

### 10.1 What you see and do

On the Semantic tab:

1. Choose a **source**: the *Microphone*, or a *File* (any saved recording, played on a loop at
   normal speed, which is handy for demos and testing without talking).
2. Press **Start**. The bar shows the input device, sample rate and channel count, the
   **latency**, the **RTF** (how hard the computer is working) and three problem counters
   (overflow / underrun / dropped).
3. The **level meters**, **spectrogram** and **waveform** start moving.
4. Optionally tick **Listen** to hear the sound in your headphones, with a volume slider and a
   mute button.
5. Optionally press **Record**, with a label and notes. The recording appears in the list below,
   with a **Play** button that plays it in the browser.
6. Press **Stop**.

### 10.2 How the sound flows

```
 microphone ──► MicSource ──► 20 ms chunks ──► ProcessorChain ──────────────► SpeakerSink ──► headphones
 (or a file)   (queue, drop-oldest)              │  main: ChannelReducer(mean)   (queue, limiter)
                                                 │        → extractor (stand-in)
                                                 │  side: Analyzer (sees all input channels)
                                                 │           └─► levels / waveform / spectrogram
                                                 │
                                                 └─► RecordingWriter ◄── the RAW input chunk, all channels
```

- **Chunks of 20 ms.** The engine works on pieces of `engine.chunk_ms` = 20 ms: 960 frames at
  48 kHz. Small chunks mean low delay. Each chunk goes through the **chain** described in
  Chapter 7.
- **The chain for the Semantic tab** (`build_chain`): first a `ChannelReducer` mixes the
  microphones to one channel (that's what goes to your headphones, and what extraction models
  will take), then the **extractor**, a stand-in that passes sound through unchanged for now. On
  the side, the **analyzer** looks at the *original* chunk with all its channels and produces the
  live pictures.
- **What gets recorded is the raw input**, every microphone channel, untouched: exactly what
  `soundspy record` saves. A recording of the processed sound would be worthless once the
  processing improves; the raw sound can always be processed again.

*Since step 06* the chain has three more parts (Chapter 13.5):

```
 main:       ChannelReducer(mean) ──► extractor (detector-gated Wiener filter) ──► output
                     │                                                               │
                     └─► the "dry" signal, delayed like the output (for Bypass)      │
 side:       Analyzer (input) + Detector ──► DetectionState ──► read by the extractor
 post_side:  a second Analyzer on the OUTPUT ──► levels / waveform / spectrogram, source "output"
```

- The **extractor** is no longer a stand-in. The detector tells it, through a shared
  `DetectionState`, whether the selected sounds are present.
- **Observers on the output** (`post_side`): the second analyser sees what you hear, so the page
  can show the output spectrogram next to the input one. Every live picture now says which side
  it shows (`source: "input"` or `"output"`).
- **Bypass**: the chain also keeps the unprocessed mono sound, delayed exactly as much as the
  processing delays the output. With Bypass on, that is what goes to the headphones.

### 10.3 Threads: who does what

Several things must happen at once, so the work is split:

| Thread | Job | Must never |
|---|---|---|
| **Sound card input** (PortAudio's) | hands over fresh samples every few ms; `MicSource` cuts them into chunks and puts them in a queue | wait for anything |
| **Session worker** (`soundspy-session`) | takes each chunk, runs the chain, feeds the speaker queue and the recording, reports events | block the sound card |
| **Sound card output** (PortAudio's) | takes audio from the speaker queue whenever the headphones need more | wait for anything |
| **Web server event loop** | sends events to every open browser over the WebSocket | slow down the audio |

The worker reports events by calling `on_event`. The `SessionManager` only **schedules** them onto
the web server's event loop (`loop.call_soon_threadsafe`) and returns at once, so a slow network
connection can never make the audio stutter.

**Changes happen between chunks.** When you move the volume slider, start recording or (from step
05) choose a different sound, the request is stored and the worker applies it just before the
next chunk. A chunk is never processed half with the old setting and half with the new one.

### 10.4 Listening: the speaker output

The microphone and the headphones are two separate devices, each with its own clock. They never
run at *exactly* the same speed. Between them sits a small queue (`SpeakerSink`):

- **Prebuffer.** Playback starts only once 2 chunks (40 ms) are waiting. This small cushion
  absorbs timing jitter.
- **Underrun.** If the queue runs dry, the missing moment is played as silence (a click) and
  counted. Playback then waits for the cushion again rather than clicking repeatedly.
- **Overflow.** If more than 8 chunks pile up (the input clock running fast), the **oldest** is
  thrown away and counted. That keeps the delay from creeping up over a long session.
- **Mono to stereo.** The single mixed channel is copied to both ears.
- **Volume and mute.** −40 to +12 dB. Mute plays silence but keeps everything running.
- **Safety limiter — always on.** The laptop's microphone can deliver sound *above* full scale
  (Windows adds +7.44 dB, Chapter 9.8), and the volume can add another +12 dB. Before anything
  reaches your ears, a limiter reacts instantly to any peak above −1 dBFS and turns the sound down
  just enough, then lets go over about 50 ms. A final hard clip catches rounding errors. Nothing
  louder than −1 dBFS can ever be sent to the headphones.

**Why headphones are required.** With the laptop's own speakers, the sound comes out, goes back
into the microphone, comes out again a little louder, and within a second it howls (*acoustic
feedback*). So **Listen is off by default**, and the first time it is switched on the page asks
you to confirm you are using headphones. The browser remembers your answer.

### 10.5 Latency: how long until you hear it

**Latency** is the delay between a sound happening and you hearing it in the headphones. Sound
Spy estimates it by adding up the stages:

| Part | On the development laptop |
|---|---|
| input stream (PortAudio / WASAPI) | 22 ms |
| one chunk (the engine waits for a full chunk) | 20 ms |
| speaker queue (prebuffer of 2 chunks, varying) | about 20–40 ms |
| output stream | 22 ms |
| **total shown** | **about 84–104 ms, typically 94 ms** |

For comparison, a delay under about 30 ms feels instant. Around 100 ms you notice a slight echo of
your own voice, but other people's voices sound natural. The cushion is the part that can be
tuned: with `prebuffer_chunks: 1` the typical delay drops to about 84 ms, but in a 30-second
test that caused **1 underrun** (a click), while 2 caused none. So 2 is the default.

The estimate leaves out the converters inside the sound hardware and the Bluetooth link if the
headphones are wireless (often 100–200 ms more). Wired headphones are best.

*Since step 06* one more part is added, the extractor's STFT (Chapter 13.3):

| Part | On the development laptop |
|---|---|
| input stream + one chunk + speaker queue + output stream (as above) | about 84–104 ms |
| **extractor (STFT, 1,024 samples)** | **21.3 ms** |
| **total shown** | **about 105–125 ms** (measured over 60 s: between 105 and 125, median 125) |

The 21 ms is there whether or not anything is selected (the sound always passes through the
STFT, so that selecting a class does not make it jump). The session status reports the
extractor's share separately, and the session bar shows it as `mask_baseline · +21.33 ms`.

### 10.6 Is the computer keeping up? The real-time factor

For every chunk the worker measures how long processing took. The **real-time factor (RTF)** is
that time divided by the chunk's length:

- RTF 0.06 means 20 ms of sound took 1.2 ms to process, so there's plenty of room;
- RTF 1.0 means it only *just* keeps up;
- above 1.0, sound arrives faster than it's processed and chunks start being dropped.

The status shows the 95th percentile over the last 2 seconds, so occasional slow moments count.
If it goes above **0.8**, a warning appears on the page. It's not a problem today; it's there
because the neural models of steps 05–07 will be heavy, and it is better to see that coming.

**Measured on the development laptop** (60 s live microphone session, listening on, muted):

| | |
|---|---|
| chunks processed | 3,031 (50 per second, as expected) |
| processing time per chunk | 0.56 ms mean, 1.21 ms p95 |
| RTF p95 | **0.06** |
| input overflows / output underruns / dropped chunks | **0 / 0 / 0** |
| backend CPU | about 12 % of one core |

### 10.7 The live pictures

All three come from the **analyzer**, which runs on the backend at fixed rates that don't depend
on the chunk size, and arrive over the WebSocket:

| Picture | Rate | What it shows | How to read it |
|---|---|---|---|
| **Level meters** | 20 per second | one bar per microphone: average loudness (RMS), the peak as a white line, the highest peak of the last second as a held marker, and a red **clip light** | normal speech close up: around −30 to −20 dBFS. The scale goes to +6 because this microphone can exceed full scale. The red light means some samples hit the ceiling |
| **Waveform** | 25 per second | the last 5 s of the mixed sound, as the highest and lowest point of each tiny time slice | tall = loud. Speech looks like bursts with gaps; steady noise like a uniform band |
| **Spectrogram** | 25 per second | the last 10 s: time left→right, pitch bottom→top, brightness = loudness (dark = quiet, yellow = loud) | see below |

**How to read a spectrogram.** Each vertical slice is one moment, split into 128 pitch bands.
Low sounds (a hum, a fridge) are near the bottom; high ones (cutlery, hissing, the "s" in speech)
near the top. The bands follow the **mel scale**, like hearing, so the lower part gets more room:
100 Hz to 1 kHz takes about as much height as 1 to 5 kHz. What different sounds look like:

- **speech**: stacked horizontal stripes (the harmonics of the voice) that wobble with
  intonation, in syllable-sized bursts, mostly below 4 kHz;
- **music**: steady, long horizontal lines (held notes);
- **a clap or a dropped plate**: a bright *vertical* line across almost all heights (a short
  burst with energy everywhere);
- **footsteps**: a regular series of short low-to-mid bursts;
- **steady noise** (fan, traffic): a smooth, even haze.

Learning to see these patterns is exactly what the detection model of step 05 will learn to do.

**Why the page stays smooth.** About 70 messages a second arrive. If each updated the page the
normal way, the browser would rebuild the page 70 times a second. Instead they go into a small
memory buffer (`liveFrames`), and the pictures are drawn on `<canvas>` elements in step with the
screen's refresh (`requestAnimationFrame`), only when something changed. If the browser tab falls
behind, the server throws away old *picture* messages for that tab (never status or warnings), so
a slow tab can't build up a backlog.

*Since step 06* there are two sets of pictures: the **input** ones described above, and the same
three for the chain's **output**. The page uses the output spectrogram: a switch under the
spectrogram chooses **Input**, **Output** or **Both** (stacked, the default). With a sound type
selected, what was turned down is bright in the upper picture and dark in the lower one. The
level meters and the waveform still show the input.

**Smooth audio and Python's memory clean-up (step 06).** While a session runs, the long-lived
objects of the program are "frozen": kept out of Python's garbage collector. A full clean-up
otherwise took 45–75 ms, during which every thread stands still, the audio thread included, and
that was enough for an occasional click (Chapter 23.31).

### 10.8 Starting from a file

A session can also play a saved WAV file, looped, at normal speed, through exactly the same chain.
It lets the app be shown without talking into the microphone, and lets tests run without any
hardware. File paths must be inside the project folder.

### 10.9 When something goes wrong

If the worker hits a problem (the microphone is unplugged or stalls, or a stage crashes), the
session switches to **error** with a message, sends that state to the page (which shows it as a
notice), finishes any recording and frees the devices. The web server keeps running, and you can
press Start again. Stopping the backend while a session runs also stops the session cleanly; a
test checks that no audio thread is left behind.

**When devices change (headphones plugged in or out).** The sound library reads the list of
devices once and keeps using what it found. Plugging headphones into the laptop's jack makes
Windows reconfigure its audio devices, and afterwards opening the microphone or speakers through
the old list can fail. Since the owner's first test of step 03 found this (Chapter 23.18), every
**Start** first re-reads the device list. If opening still fails, it re-reads once more and
retries, and only then reports an error that says what probably happened. If a session starts but
the output can't be opened, a notice gives the reason and the **Listen** box is disabled with an
explanation, instead of failing when clicked.

---

## 11. Training data — from public datasets to mixture recipes

**Exists today** (step 04). Location: `backend/src/soundspy/data/` (`download.py`,
`ontology.py`, `roles.py`, `splits.py`, `activity.py`, `collectors/`, `prepare.py`, `stats.py`,
`mixtures.py`, `aliases.py`), `dsp/resample.py`, `cli/data_cmd.py`, `cli/record_dataset_cmd.py`,
`backend/configs/data/`.

The models of steps 05–07 learn from examples. This chapter is about where those examples come
from. No model is trained yet; this step only gets the material ready.

### 11.1 Why the examples are *made*, not found

Two models will be trained:

1. a **detector**: "which of the seven classes can be heard in this second of sound?";
2. an **extractor**: "here is a mixture and a set of classes; give back only those".

The extractor needs pairs that don't exist in nature: a noisy mixture **and** the exact clean
version of the wanted part. So, as in SemanticHearing (Veluri et al., UIST 2023), the pairs are
**built**: take clean clips of the chosen classes (the **foreground**), add other sounds (the
**background**), and keep the clean foreground as the known answer. Because the computer did the
mixing, the right answer is known exactly, sample by sample.

The clips come from two public datasets (ESC-50 and FSD50K) and, later, from the owner's own
recordings.

### 11.2 Where the data lives: C: and D:

The laptop's SSD has two partitions: C: (about 31 GB free) and D: (about 240 GB free). The
datasets need roughly 60 GB while they are being unpacked, so **everything large goes to D:**.
The code, the database and the owner's own recordings stay in the project on C:.

| `paths` key (`default.yaml`) | Where | What |
|---|---|---|
| `raw` | `data/raw` (C:) | own free recordings (step 02) |
| `custom` | `data/custom` (C:) | own **labelled** clips (`soundspy record-dataset`) |
| `external` | `D:/SoundSpyData/external` | downloaded datasets and their archives |
| `processed` | `D:/SoundSpyData/processed` | clip index, activity maps |
| `mixtures` | `D:/SoundSpyData/mixtures` | mixture recipes |
| `model_cache` | `D:/SoundSpyData/models` | pretrained weights (from step 05) |

A path in `default.yaml` can be absolute (`D:/...`) or relative (then it is measured from the
project root, as before). `soundspy data paths` prints where everything resolves to, with the
free space of each drive.

**Model caches.** Libraries like PyTorch and Hugging Face download pretrained weights into the
user profile on C: by default. At start-up (`soundspy ...` and the API) Sound Spy sets
`TORCH_HOME`, `HF_HOME` and `XDG_CACHE_HOME` to sub-folders of `model_cache`, **but only if they
are not already set**, so a choice the user made in Windows always wins.

**Aliases instead of full paths.** The clip index never stores `D:\SoundSpyData\...`. It stores
**aliases** like `fsd50k://FSD50K.dev_audio/64760.wav` or `custom://speech/speech_0001.wav`,
which are turned into real paths through `paths` when a clip is loaded (`data/aliases.py`).
Moving the whole data folder to another drive therefore only needs a config change.

### 11.3 The datasets and their licences

| Dataset | What | Size | Licence |
|---|---|---|---|
| **AudioSet ontology** | the tree of 632 sound categories (`ontology.json`) | 0.3 MB | CC BY-SA 4.0 |
| **ESC-50** | 2,000 clips of 5 s, 44.1 kHz, 50 classes (dog, rain, footsteps…), 5 official folds | 0.6 GB | **CC BY-NC 3.0** |
| **FSD50K** | 51,197 clips (0.3–30 s), 44.1 kHz mono, 200 AudioSet classes, Freesound | 25 GB download, ~34 GB unpacked | per clip: CC0, CC-BY, **CC-BY-NC**, Sampling+ |

"NC" means *non-commercial*. That is fine for a university project, but **it must be looked at
again before any public or commercial use** of the data or of models trained on it. FSD50K's
per-clip licence is stored in the clip index (`license` column), so the non-commercial clips can
be filtered out later if needed.

### 11.4 Downloading (`soundspy data download`)

```powershell
soundspy data download ontology esc50     # small, a minute
soundspy data download fsd50k             # 25 GB; asks before starting (--yes to skip)
soundspy data cleanup-archives            # later, to delete the downloaded zips
```

What it does, and why:

- **Free space first.** Before anything is fetched it checks the drive (FSD50K: remaining
  download + ~36 GB unpacked + 2 GB margin) and stops with a clear message if it doesn't fit.
- **Resumable.** Bytes go to `<name>.part`. If the connection drops, or the command is stopped
  with Ctrl+C, running the **same command** again asks the server for "the rest from byte N"
  (an HTTP *Range* request) instead of starting over. Network errors are retried up to 8 times
  with growing pauses (2, 4, 8 … s).
- **Checked.** FSD50K's file list, sizes and MD5 checksums are read from the Zenodo record's API
  (never typed into the code). A file is renamed from `.part` to its real name only after its
  checksum matched, so a file with its final name is a verified file. For the GitHub downloads
  (ontology, ESC-50) there is no published checksum; the SHA-256 is computed and recorded.
- **Unpacked and counted.** ESC-50 and FSD50K's small zips are unpacked by Python (entries that
  would land outside the target folder are refused). FSD50K's audio is a **split zip**
  (`.zip` + `.z01` … `.z05`), which only **7-Zip** can unpack; if `7z.exe` is not found (on PATH
  or in `C:\Program Files\7-Zip`), the command stops with *"Install 7-Zip: `winget install
  7zip.7zip`"*, after downloading everything. Afterwards the number of WAV files is compared with
  the CSVs (ESC-50: 2,000; FSD50K: 40,966 dev + 10,231 eval).
- **Done once.** A `.done` file is written last; with it present, the command does nothing.
- **Recorded.** `SOURCE.json` in each dataset folder: URL, date, version (Zenodo version or the
  GitHub commit), licence and every file's size, MD5 and SHA-256.
- **Archives are kept** until `soundspy data cleanup-archives`, which lists them with sizes and
  asks before deleting (`--yes` skips the question). Only datasets that are completely unpacked
  are offered.

Layout on D: afterwards:

```
D:/SoundSpyData/external/
├── ontology/   ontology.json, SOURCE.json, .done
├── esc50/      ESC-50-master/ (audio/, meta/esc50.csv), archives/, SOURCE.json, .done
└── fsd50k/     FSD50K.dev_audio/, FSD50K.eval_audio/, FSD50K.ground_truth/,
                FSD50K.metadata/, FSD50K.doc/, archives/, SOURCE.json, .done
```

### 11.5 From dataset labels to our seven classes

Every dataset names its sounds differently. They are translated through the **AudioSet
ontology**, a tree in which every category has a permanent ID called a **mid**
(`/m/09x0r` = "Speech"). Two details matter:

- **A category can have several parents.** "Vehicle horn, car horn, honking" sits under *Car*
  (→ *Motor vehicle* → *Vehicle*) **and** under *Alarm*. So it's not really a tree but a web
  without loops, and `ontology.py` walks it with a "visited" list so it can never go round in
  circles. `descendants(mid)` = the category and everything below it; `ancestors(mid)` =
  everything above it.
- **FSD50K writes names differently** (`Walk_and_footsteps` for "Walk, footsteps"). Names are
  therefore never matched by spelling: FSD50K ships the mid of every label, and the mapping is
  built from those mids.

**The classes** (`configs/classes.yaml`) each list one or more AudioSet names. All of them were
checked against the real ontology in this step and **all exist** (`validate_classes` would list
the closest valid names for any that didn't). So no name had to be changed.
`soundspy data prepare` refuses to run if a name is ever wrong.

**ESC-50** has its own 50 labels. `configs/data/esc50_to_audioset.yaml` maps each one by hand
to an AudioSet mid (the AudioSet name is written next to it as a comment), or to `null` when
nothing fits (drinking/sipping, can opening, washing machine). Every mid is checked when
`prepare` runs. Mappings marked `UNCERTAIN` in the file are for the owner to review.
*(Step 05: `null` is gone; every class without a match now says explicitly `background` or
`excluded`, Chapter 11.14.)*

### 11.6 Foreground, background or excluded

Each clip gets a **role**, which decides how it may be used in a mixture:

| Role | Meaning | Used as |
|---|---|---|
| `fg:<class>` | contains one of our classes and nothing else | a sound the user could ask for |
| `bg` | contains none of our classes | background noise |
| `excluded` | contains one of our classes **plus** something else | not used |

The rule, for a class *c*: **F(c)** is the class's AudioSet categories and everything below them;
**A(c)** is everything above them. A clip is foreground for *c* if all its labels are in F(c) or
A(c) and at least one is in F(c). Labels from A(c) are tolerated because FSD50K **smears**
labels upwards: a speech clip is labelled `Speech, Human_voice`, and the `Human_voice` part adds
nothing.

Examples:

| FSD50K labels | Role | Why |
|---|---|---|
| `Speech, Human_voice` | `fg:speech` | Human_voice is just the smeared parent |
| `Guitar, Plucked_string_instrument, Musical_instrument, Music` | `fg:music` | all under Music |
| `Dog, Domestic_animals_and_pets, Animal` | `bg` | none of our classes |
| `Speech, Human_voice, Music` | `excluded` | speech **and** music: as a "speech" example it would teach the extractor to keep music too |
| `Vehicle_horn…, Car, Vehicle, Alarm` | `excluded` | it is both "alarm" and "traffic" (overlapping classes); counted and reported |

**Why "excluded" matters.** If a clip with speech and music were used as the "speech" target, the
extractor would be taught that music belongs in a speech answer. Throwing such clips away costs
quantity but protects the meaning of every class.

**One refinement** (deviation from the spec's literal rule): smearing follows *every* parent, so
the car horn above also carries `Alarm`, which is not an ancestor of the *traffic* class. The
literal rule would call that clip "traffic mixed with something else" instead of noticing the
overlap. So before testing, labels that are ancestors of **another label of the same clip** are
dropped (they were added by smearing). Everywhere else this gives exactly the spec's result.

**Overlapping classes.** "Alarm" includes car horns, sirens of emergency vehicles and bicycle
bells; "traffic" (AudioSet *Vehicle*) includes the same horns and sirens, and "Music" includes
bicycle bells (under *Bell*). *(Step 05: traffic is now "Motor vehicle (road)"; horns, emergency
sirens and car alarms are still alarm + traffic overlaps, bicycle bells now only music + alarm.)* Clips of those sounds are foreground for two classes at once and are
excluded; `prepare` prints how many.

### 11.7 The clip index and activity maps (`soundspy data prepare`)

`prepare` reads every dataset's own metadata (the **collectors**: `collectors/esc50.py`,
`fsd50k.py`, `custom.py`), opens every audio file once and writes one table,
`D:/SoundSpyData/processed/clip_index.parquet` (plus `clip_index.csv` to look at in a
spreadsheet):

| Column | Example |
|---|---|
| `clip_id` | `fsd50k:64760`, `esc50:1-100032-A-0`, `custom:speech_0003` |
| `dataset` | `fsd50k` |
| `path` | `fsd50k://FSD50K.dev_audio/64760.wav` (alias) |
| `duration_s`, `sample_rate`, `channels` | from the file itself |
| `mids`, `raw_labels` | AudioSet mids and the dataset's own labels |
| `role` | `fg:speech` / `bg` / `excluded` |
| `split` | `train` / `val` / `test` |
| `group` | `uploader:<name>` (FSD50K), `1-100032` (ESC-50 source recording), `session:<id>` (own) |
| `license` | `CC-BY 3.0`, `CC BY-NC 3.0`, `own recording` |

**Activity maps.** Many clips are mostly silence (a dog barks once in 5 s). For every clip,
`prepare` measures the loudness every 0.1 s (an **RMS envelope**, in dB). A moment counts as
**active** if it is within 30 dB of the clip's loudest moment (and above −90 dBFS). Mixtures use
this to put events where something actually sounds. The envelopes (not the yes/no decisions) are
stored in `processed/activity/<dataset>.npz`, so the 30 dB threshold can change without
re-reading the audio.

Reading 51,000 files takes a while, so it runs on **all CPU cores in parallel**, shows a progress
bar, and **remembers** what it already measured: a second `prepare` reads only new clips and
takes seconds.

`soundspy data stats` prints the **coverage table**: for each class, dataset and split, how many
clips and how many minutes there are, and warns about classes with fewer than
`min_fg_clips_warning` (100) training clips.

### 11.8 Splits without leakage

Clips that come from the same recording, or the same person's microphone and room, sound alike. If
one ended up in training and its sibling in testing, the test would measure memory, not skill.
That is **leakage**. Every clip therefore has a `group`, and a group always stays in one split:

- **FSD50K:** eval → test, exactly as published (dev and eval share no uploader). For dev,
  FSD50K only *proposes* a train/val split, and on the real data **926 uploaders appear in both**
  (57 % of the val clips have a "sibling" in train). So every uploader is moved wholly to the
  side that already holds most of its dev clips (ties → train). 2,010 of the 40,966 dev clips
  change side; val keeps 2,660 clips instead of 4,170. FSD50K's README allows a different split
  as long as it is stated, which is what this paragraph does (deviation from the spec, Chapter
  20.23). *(Step 05: then whole training uploaders move to val until every class has at least
  40 val clips, Chapter 11.14.)*
- **ESC-50:** official folds 1–3 → train, 4 → val, 5 → test (configurable). A fold holds all the
  pieces (`-A`, `-B`, …) of one original recording.
- **Own clips:** whole **recording sessions** are shuffled (fixed seed) and assigned to a split,
  15 % val and 15 % test.

`prepare` refuses to write the index if any group appears in two splits, and a test checks it.
Mixtures only combine clips of **their own split**, foreground and background alike.

### 11.9 Mixture recipes (`soundspy data mix`)

Instead of writing tens of thousands of mixed WAV files, Sound Spy writes **recipes**: a few
lines that say exactly how to build each mixture. One recipe per line in
`D:/SoundSpyData/mixtures/<split>.jsonl`:

```json
{"id": "train-000001", "duration_s": 6.0,
 "events": [
   {"clip_id": "fsd50k:64760", "role": "fg", "class": "speech",
    "clip_offset_s": 1.3, "onset_s": 0.4, "dur_s": 3.2, "gain_db": 1.5},
   {"clip_id": "esc50:2-102414-A-17", "role": "bg", "class": null,
    "clip_offset_s": 0.0, "onset_s": 0.0, "dur_s": 5.0, "gain_db": -9.0}],
 "present": ["speech"], "target": ["speech"], "target_to_rest_db": 3.2}
```

Read it as: *"take 3.2 s of clip 64760 starting at 1.3 s, place it 0.4 s into a 6-second
mixture, 1.5 dB louder than the reference; add 5 s of the ESC-50 clip as background, 9 dB quieter;
the user asked for speech; make the speech 3.2 dB louder than everything else together."*

How recipes are drawn (all numbers in `configs/data/default.yaml`, section `mixtures`):

- 1–3 **different** foreground classes and 1–3 background clips per mixture.
- Classes are drawn **uniformly**, not in proportion to how many clips they have. FSD50K has far
  more music than footsteps; uniform drawing makes sure footsteps are seen just as often.
- Every event starts inside an **active** part of its clip and lasts at least 0.5 s; clips shorter
  than that are used whole. Background events are made as long as possible.
- The **target** is 1–2 of the classes present. In 5 % of recipes it is a class that is **not**
  present, so the correct answer is silence. This teaches the extractor not to invent sounds.
- The whole file is **deterministic**: the same seed (`seed: {train: 1, val: 2, test: 3}`) gives
  the same recipes on any computer.

Why recipes and not audio files? 24,000 six-second mixtures would be ~25 GB at 44.1 kHz, fixed
at one sample rate. The recipes are a few MB, can be rendered at whatever sample rate step 05
chooses, and can be regenerated in seconds.

### 11.10 Rendering a recipe

`render(recipe, sample_rate, …)` in `mixtures.py` turns a recipe into sound:

1. Each event's piece of audio is read from its file (only that piece), mixed down to mono with
   the `ChannelReducer`, and **resampled** to the requested rate with `dsp/resample.py`
   (polyphase filtering, `scipy.signal.resample_poly`; 44.1 → 16 kHz is the exact ratio 160/441,
   so 6 s stay exactly 96,000 samples).
2. It is brought to a **reference loudness** (`reference_rms_dbfs: -25`, measured only over its
   active moments, so silence doesn't make it look quiet) and then gets its `gain_db`.
3. It is added to the **target** sum if its class is in the target, otherwise to the **rest**.
4. The target is scaled so that target vs. rest is exactly `target_to_rest_db`.
5. **mixture = target + rest**, exactly, sample for sample.
6. If the mixture would exceed the **peak limit** (0.99), target and rest are scaled by **the same
   factor**, so the relationship between input and answer is never broken.

It returns the mixture and target (mono `AudioBuffer`s), the rest, the **present** and
**target** multi-hot vectors, and a **class-activity** table: for each time frame (10 per second
by default, configurable), which classes actually sound. The detector of step 05 needs that to
learn "which classes are active in each second". Rendering the same recipe twice gives identical
samples.

Spatial rendering (each sound arriving at the 4 microphones from a direction, in a simulated
room) comes with module 3; the place where it plugs in is marked `TODO(module-3)`.

### 11.11 Listening to previews (`soundspy data preview`)

```powershell
soundspy data preview --split val --n 5
```

renders the first five validation recipes at 44.1 kHz into `outputs/previews/`:
`<id>_mix.wav` (what the microphone would hear), `<id>_target.wav` (what the extractor should
output) and `<id>.json` (the recipe plus the applied scaling). Listening to a few is the quickest
way to judge whether the mixtures sound believable.

### 11.12 Recording your own clips (`soundspy record-dataset`)

```powershell
soundspy record-dataset --classes speech dishes background --clips 10 --seconds 5 --notes "kitchen, laptop mic"
```

Public datasets are recorded with other microphones in other rooms. A few of the owner's own clips,
made with the real microphone in real places, help the models later. The command walks through
the classes one by one:

- `Enter` records one clip after a short countdown; `r` redoes the last take; `d` discards it;
  `s` skips to the next class; `q` quits.
- After each take it prints the level report (Chapter 9.8) and warns about clipping or silence.
- Clips go to `data/custom/<class>/<class>_0001.wav` (+ sidecar), on C:. Numbering continues
  across runs.
- `background` is a class too: room tone or cafeteria noise **without** any of the seven classes.
- Every run is one **session**; its id goes into each sidecar (`session_id`, `class`, `notes`).
  The split is decided per session, so takes from one sitting never land in both train and test.

The next `soundspy data prepare` picks them up (`custom` collector, role from the chosen class).
A take is only ever deleted by `r` or `d` right after it was recorded.

### 11.13 The real run on the laptop

15 November 2026, on the development laptop (12 CPU threads, NVMe SSD, home internet).

| Stage | Result | Time |
|---|---|---|
| `data download ontology esc50` | 632 categories; ESC-50 2,000 clips verified | 1.2 min |
| `data download fsd50k` | 11 files, 24.7 GB, every MD5 matched | 36 min (~11 MB/s) |
| (7-Zip installed, same command again) | dev 40,966 + eval 10,231 WAVs = the CSVs | 7.8 min (incl. re-hashing 25 GB) |
| `data prepare` (first time) | 53,197 clips indexed, activity maps for all | ~2.2 min |
| `data prepare` (again) | nothing re-read | 5 s |
| `data mix --split all` | 20,000 / 2,000 / 2,000 recipes, 15 MB | 3.2 s |
| `data preview --split val --n 5` | 5 mixtures + targets at 44.1 kHz | 3 s |

**Roles:** 28,351 background, 21,175 foreground, 3,671 excluded (3,296 because a class was
mixed with other sounds, 146 as alarm + traffic overlaps, 109 as music + alarm + traffic: bicycle
bells). Among the background clips, 3,146 carry `Human_voice` without `Speech` (shouts,
whispering, screams…) and 5,183 carry `Domestic_sounds_and_home_sounds` without dishes or
cutlery (doors, drawers, keys…).

**Coverage** (`soundspy data stats`, clips and minutes per split, all datasets):

| Class | train | val | test |
|---|---|---|---|
| speech | 1,268 (79 min) | 45 (3 min) | 218 (20 min) |
| music | 12,552 (1,071 min) | 638 (77 min) | 1,857 (218 min) |
| footsteps | 305 (29 min) | 30 (3 min) | 54 (6 min) |
| dishes | 445 (19 min) | 17 (1 min) | 95 (8 min) |
| chair_scrape | 254 (15 min) | 19 (2 min) | 57 (6 min) |
| alarm | 932 (125 min) | 46 (6 min) | 208 (31 min) |
| traffic | 1,830 (387 min) | 137 (31 min) | 168 (29 min) |
| background | 20,576 (2,607 min) | 1,847 (279 min) | 5,928 (945 min) |
| excluded | 1,344 (192 min) | 281 (53 min) | 2,046 (442 min) |

Every class has at least 100 training clips, so no warning fired. The thin spots are in
**validation**: dishes (17 clips), chair scraping (19) and footsteps (30) have very few distinct
clips there, and speech only 45. Recipes still draw every class equally often (train: 5,605–5,822
mixtures per class), but validation will reuse those few clips many times. Own recordings would
help most here.

**Disk:** C: 30.6 GB free before → 32.7 GB after (nothing large was written to C:; the change is
other activity). D: 240.1 GB free before → 196.9 GB after. On D:, `fsd50k/` takes 60 GB (25 GB archives,
removable with `cleanup-archives`, + 35 GB unpacked), `esc50/` 1.6 GB, `processed/` 33 MB
(index + activity maps) and `mixtures/` 16 MB. The previews in `outputs/previews` are 11 MB.
*(After step 05: the owner had the archives deleted with `cleanup-archives` once training had
shown the data is good: 25.3 GB freed (FSD50K 24.68 GB + ESC-50 0.65 GB), D: 222 GB free. The
unpacked audio stays; the archives can be downloaded again at any time.)*


### 11.14 Fixes after the review (step 05)

After looking at the step-04 numbers, the owner approved three changes, applied in this order:

1. **Traffic = road traffic.** `classes.yaml` now maps *traffic* to "Traffic noise, roadway
   noise" and **"Motor vehicle (road)"** instead of "Vehicle". Aircraft, trains and boats are now
   background (still useful: realistic sounds the extractor has to suppress). Key and order did not
   change, so nothing that depends on the label order broke. Traffic foreground clips went from
   2,135 to 848.
2. **No more `null` in the ESC-50 mapping.** Each ESC-50 class without an AudioSet match is
   decided explicitly: `background` if it cannot contain any of our classes, `excluded` if it
   could. *drinking_sipping*, *can_opening* and *washing_machine* became **background** (they are
   exactly the clinks and hums of a cafeteria); **engine** became **excluded** (AudioSet's "Engine"
   is not under "Motor vehicle (road)", but most ESC-50 engine clips are cars or motorbikes, i.e.
   road traffic). All other ESC-50 classes keep their mids.
3. **A fuller validation set.** FSD50K's uploader-disjoint split left only 17 dishes, 19 chair
   scraping and 30 footsteps clips in val, too few for reliable scores. `prepare` now moves
   **whole training uploaders** into val, greedily (the uploader with the largest share of
   still-needed clips first, ties by size then name; deterministic), until every class has at
   least `splits.fsd50k.min_val_clips_per_class` = **40** val clips. Test and the ESC-50 folds
   never change. The moved uploaders are logged in `D:/SoundSpyData/processed/val_rebalance.json`.
   Result: **28 uploaders** moved; the training set lost 21 chair-scraping, 23 dishes and 10
   footsteps clips (−8.3 %, −5.2 %, −3.4 %), nothing else.

Coverage after the fixes (clips, all datasets):

| Class | train | val | test |
|---|---|---|---|
| speech | 1,268 | 45 | 218 |
| music | 12,552 | 638 | 1,857 |
| footsteps | 295 | 40 | 54 |
| dishes | 422 | 40 | 95 |
| chair_scrape | 233 | 40 | 57 |
| alarm | 932 | 46 | 208 |
| traffic | 781 | 41 | **26** |
| background | 21,780 | 1,969 | 6,232 |
| excluded | 1,185 | 259 | 1,884 |

Every class has ≥ 40 validation clips. New thin spot: **traffic test has only 26 clips**: ESC-50's
"traffic" clips were all airplanes, helicopters and trains, and the test split is never changed.
Recipes were regenerated (`mix --split all`, 3 s).
---

## 12. Detection — which sounds are present right now

**Exists today** (step 05). Location: `backend/src/soundspy/detection/` (`model.py`,
`backbones/`, `inference.py`, `metrics.py`, `windows.py`, `weights.py`), `dsp/features.py`,
`training/` (`datasets.py`, `train.py`, `evaluate.py`, `experiment.py`), the CLI commands
`train`, `evaluate`, `detect`, and the class chips and timeline on the Semantic tab.

The detector answers one question, several times a second: *for each of the seven sound types,
how likely is it that it can be heard in the last second?* Nothing is removed from the sound yet
(that is steps 06–07), but the page now shows what the microphone hears.

### 12.1 The model and why this one

The model is **EfficientAT `mn10_as`** (Schmid, Koutini & Widmer, ICASSP 2023): a MobileNetV3, the
kind of compact network designed for phones, re-trained by its authors on **AudioSet** (2 million
YouTube clips, 527 sound categories). It has **4.88 million parameters** (a 20 MB file) and reaches
an AudioSet mAP of 0.471, close to models fifty times its size.

Why it fits Sound Spy:

- **Small and fast.** One window takes about 28 ms on the laptop's 4 GB GPU and about 70 ms on a
  single CPU thread, far less than the 250 ms between windows.
- **It already knows sounds.** Its 527 categories include all of ours (Speech, Music, Walk/footsteps,
  Dishes, Cutlery, Scrape, Squeak, Alarm, Motor vehicle …). Fine-tuning only has to sharpen that
  knowledge for our seven classes and our mixtures, not teach hearing from scratch.
- **Permissive licence (MIT)**, so its code could be adapted (Chapter 26).

### 12.2 From sound to probabilities

```
1 s of sound (mono, 32 kHz)
  → padded with silence to 10 s           (the model was trained on 10 s clips, see below)
  → log-mel spectrogram, 128 bands        (EfficientAT's exact front-end)
  → MobileNetV3 body → 1,280 numbers      (the pretrained "embedding")
  → new head: 7 outputs → sigmoid         (one independent probability per class)
```

- **The front-end must match exactly.** A pretrained network only understands the precise
  picture of sound it was trained on: 32 kHz, pre-emphasis, an FFT of 1,024 with an 800-sample
  window every 10 ms, 128 Kaldi mel bands up to 15 kHz, log and a fixed normalisation.
  `dsp/features.py` re-implements it, and a test compares our output with the original code's on a
  fixed test signal (stored as a small `.npy` file): they agree to within 0.0001.
- **Why pad 1 s to 10 s?** Measured in step 05: with only 1 s of input the network's last feature
  map is just 4×4 cells, the edges dominate and the pretrained head saturates (every probability
  near 1). Its zero-shot mAP was at chance level for any input shorter than 5 s, and normal at 5–10
  s. Padding the 1 s window with silence to 10 s looks to the model like an AudioSet clip in which
  only one second sounds, which is perfectly normal for it. Training and live use the same
  padding, and the label still describes the 1 s window. (Costs: about 10× the computation of a
  1 s input, which the GPU barely notices.)
- **Independent probabilities.** Several sounds can be present at once (speech *and* dishes), so
  each class gets its own yes/no probability (sigmoid), not a share of 100 % (softmax).

### 12.3 How a training window is labelled

Training uses the mixtures of Chapter 11. From each rendered 6 s mixture a **1 s window** is cut
at a random position (a new position every epoch). A class counts as **present** in the window if
it is actually sounding there for at least **0.1 s**, measured on the per-class activity the
renderer returns (100 frames per second). A class that is in the mixture but silent during this
second is "absent" for this window; a window with only background has all seven labels 0.

### 12.4 Training

`soundspy train detection` (config: `configs/detection/default.yaml`):

| Setting | Value | Why |
|---|---|---|
| start | pretrained mn10_as + new 7-output head | reuse AudioSet knowledge |
| head initialisation | each class row = mean of its AudioSet rows (e.g. *dishes* = mean of "Dishes, pots, and pans" and "Cutlery, silverware") | the head starts out already meaning the right thing |
| optimiser | AdamW, backbone LR 1e-4, head LR 1e-3, weight decay 1e-4 | the new head learns faster than the pretrained body may change |
| schedule | 1 epoch linear warm-up, then cosine decay to 0 | stable start, gentle end |
| precision | mixed (float16) on the GPU | ~2× faster, less memory |
| batch / epochs | 32 windows / 30 epochs (one window per training recipe per epoch) | |
| loss | binary cross-entropy with `pos_weight` = negatives / positives per class (measured on 1,000 windows: each class is present in 9–15 % of windows) | rare positives count as much as common negatives |
| augmentation | random gain ±6 dB, faint noise (30–50 dB SNR), SpecAugment masks (up to 48 mel bands, 20 frames), mel-range jitter | robustness to level, hiss and missing bands |
| validation | every epoch, on fixed val windows (2 per recipe, seeded) | comparable from epoch to epoch |

Everything about a run lands in `experiments/det_mn10_v1/`: `config.json` (the full resolved
configuration, the git commit, a hash of the data index), `log.csv` (one row per epoch: losses,
val mAP, per-class AP, learning rates, epoch time, GPU memory), `checkpoints/best.pt` (best val
mAP) and `last.pt`, `thresholds.json`, `metrics.json` and `plots/`. A run that was stopped
continues with `--resume`; `--smoke` does 2 quick epochs on 500/200 recipes to test everything end
to end.

**Data loading on Windows.** Rendering mixtures is the slow part, so 4 worker processes render in
parallel while the GPU trains. Each worker is a fresh Python process (Windows *spawns* them), and
each one loads PyTorch (~0.7 GB of memory) and the clip library. Two lessons from step 05
(Chapter 23.24): every worker is limited to one BLAS/OpenMP thread (otherwise each one reserved
buffers for all 12 CPU threads, 1.7 GB per worker), and the loaders abort with a clear message if
no batch arrives within 120 s instead of hanging silently.

**The real run (step 05, `det_mn10_v1`).** 30 epochs on the laptop's RTX 3050 Ti, batch 32,
4 workers: **221 minutes** in total, normally ~6.4 minutes per epoch (one epoch took 24 minutes
because the laptop was busy with something else; no harm, only time). GPU memory peak: 2.2 GB of
4 GB. Validation mAP after epoch 1: 0.375; epoch 5: 0.483; epoch 10: 0.522; epoch 15: 0.537;
epoch 30: **0.548** (the best epoch was the last, but the curve had been flat since about epoch
22, each epoch adding less than 0.003). Training loss fell from 1.86 to 0.71, validation loss from
1.07 to 0.82 and never rose again, so there is no sign of over-fitting. Tuned thresholds are
0.62–0.76. Curves: `experiments/det_mn10_v1/plots/training_curves.png`.

![Training curves](experiments/det_mn10_v1/plots/training_curves.png)

### 12.5 Measuring quality: AP, mAP and F1 in plain words

For each class the detector outputs a probability per window. Two different questions:

- **"Does it rank correctly?"** Sort all test windows by the speech probability. A good detector
  puts the windows with speech at the top. **AP** (average precision) summarises how clean that
  ranking is across every possible cut-off: 1.0 is perfect; a detector that guesses randomly
  scores about the share of windows that contain the class (here ~0.1). **mAP** is the average AP
  over the seven classes, each counting equally.
- **"Are its yes/no answers right?"** At a chosen threshold: **precision** = of the windows it
  called "speech", how many really had speech; **recall** = of the windows with speech, how many it
  found; **F1** = a single number combining both (their harmonic mean).

**Thresholds are tuned on the validation set**, per class, to maximise F1, and saved in
`thresholds.json`. The test set is never used for any choice; it is scored once, at the end, with
those thresholds. Test windows are a fixed, seeded set (2 per recipe), so numbers stay comparable
between runs.

**The reference: zero-shot.** The untouched pretrained model can already answer our question:
take its 527 AudioSet outputs and, for each class, use the highest probability among the AudioSet
categories that belong to it (`F(c)` from Chapter 11.6, e.g. any of 150 music sub-genres for
*music*). Its thresholds are tuned on val the same way. Fine-tuning has to beat this to be worth it.

### 12.6 Results

Test set (4,000 fixed windows from 2,000 test mixtures; never used for any choice), thresholds
from val. `soundspy evaluate detection --split test` prints this table:

| Class | Positive windows | AP fine-tuned | AP zero-shot | F1 fine-tuned | F1 zero-shot | ROC-AUC fine-tuned |
|---|---|---|---|---|---|---|
| speech | 424 | **0.732** | 0.664 | **0.658** | 0.643 | 0.910 |
| music | 456 | **0.524** | 0.489 | **0.567** | 0.522 | 0.889 |
| footsteps | 542 | **0.624** | 0.432 | **0.577** | 0.459 | 0.891 |
| dishes | 326 | **0.517** | 0.381 | **0.488** | 0.394 | 0.882 |
| chair_scrape | 493 | **0.438** | 0.234 | **0.434** | 0.268 | 0.796 |
| alarm | 468 | **0.662** | 0.434 | **0.583** | 0.470 | 0.889 |
| traffic | 519 | **0.415** | 0.229 | **0.431** | 0.330 | 0.822 |
| **mean** | | **0.559** | 0.409 | **0.534** | 0.441 | 0.869 (zero-shot 0.771) |

Validation (the set the thresholds were tuned on): mAP 0.548 fine-tuned vs 0.391 zero-shot, F1
0.547 vs 0.436. Test and val agree, so the thresholds were not just fitted to val.

What this means:

- **Fine-tuning helps every class**, most where AudioSet's categories did not match ours well:
  chair scraping (0.23 → 0.44), traffic (0.23 → 0.42), alarm (0.43 → 0.66), footsteps
  (0.43 → 0.62). Speech and music were already well known to the pretrained model and gain least.
- **Speech is the best class, traffic and chair scraping the weakest.** Road noise and a
  scraping chair are broadband, noisy sounds that resemble a lot of background.
- **In yes/no terms:** at the tuned thresholds about half of the reported detections are correct
  and about half of the windows containing a class are found (mean precision 0.51, recall 0.57).
  This is a hard task as defined: a 1 s window, mixed with background at varying loudness, and a
  sound that touches the window for only 0.1 s already counts as "present".
- **The most common confusions** (`metrics.json`, `false_positives`): *music* reported in windows
  that hold an *alarm* (137 cases: melodic alarms and ringtones) or a scraping chair (90);
  *traffic* reported on background only (112). In the precision/recall plot
  (`plots/precision_recall_test.png`), the zero-shot model is actually more precise than the
  fine-tuned one at the very top of the music ranking; everywhere else the fine-tuned curve lies
  above.

![Precision / recall per class, test](experiments/det_mn10_v1/plots/precision_recall_test.png)

### 12.7 Live detection: scheduling, worker thread, smoothing

The `Detector` is a side-observer in the `SEMANTIC` chain (next to the Analyzer):

1. Every 20 ms chunk is mixed to mono and written into a **ring buffer** holding the last second
   at the microphone's own rate (48 kHz).
2. Every **0.25 s** of audio (counted in samples, so it does not matter how the chunks are cut),
   the current second is handed to an **inference thread**. That thread resamples it to 32 kHz and
   runs the model (on the GPU in half precision if there is one, otherwise on the CPU with one
   thread, `live.cpu_threads`). The thread warms the model up itself before the session starts
   (Chapter 23.27).
3. **The audio path never waits.** If the inference thread is still busy with the previous window,
   the new one is **skipped** and counted. The session status reports the model, device, mean
   inference time and the skipped count; the session bar shows them.
4. The results are **smoothed** so the chips do not flicker: an exponential moving average
   (`p = 0.5·new + 0.5·previous`) and **hysteresis**: a class switches *on* at its tuned threshold
   and *off* only when it falls 0.1 below it.
5. Each result becomes a `DetectionFrame` (probabilities for all classes + the active ones) sent to
   the page as a `detections` WebSocket message, four per second.

If no trained model is configured (or it fails to load), the session simply starts **without**
the detector and the page shows a warning explaining why. Everything else keeps working.

**Measured on the laptop (step 05),** 60 s sessions with the laptop microphone and Listen on
(muted, so the speakers could not feed back into the microphone; the output stream still runs
and counts underruns), the trained model:

| | GPU (RTX 3050 Ti, half precision) | CPU (1 thread) |
|---|---|---|
| inference time per window (incl. resampling) | 28 ms | 71 ms |
| skipped windows | 0 of 238 | 0 of 238 |
| RTF p95 of the audio chain | 0.07 | 0.10 |
| output underruns / input overflows / dropped chunks | 0 / 0 / 0 | 0 / 0 / 0 |
| listening delay | 84 ms | 84 ms |
| backend CPU use | 21 % of a core | 38 % of a core |
| session start (model load + warm-up) | ~6 s | ~2.5 s |

Two problems were found and fixed during these measurements (Chapter 23.27 and 20.28): the first
3 s of every GPU session were skipped, and CPU mode kept ~7 cores busy. The numbers above are
after the fixes.

- **Quiet room:** nothing switched on in 60 s. *Traffic* hovered highest (mean probability
  0.2–0.3, peaks 0.62–0.66 against a threshold of 0.76): steady fan and hum noise looks a bit like
  distant road noise.
- **A val mixture played in a loop** (speech from 3.67 to 5.37 s of each 6 s loop): *speech*
  switched on about 0.3 s after the speech began and off 0.6–1 s after it ended; *chair scraping*
  flickered on briefly in 3 % of the frames. GPU and CPU gave the same result.
- **Live and offline agree:** a mixture where the model is wrong (traffic + dishes reported as
  footsteps) is wrong in the same way with `soundspy detect`, so the errors come from the model,
  not from the live scheduling.
- **The owner's check in a real room** (laptop microphone, step 05): talking, clinking a cup and
  cutlery, walking, dragging a chair, music from a phone and an alarm from YouTube each lit the
  right class; silence lit nothing; the timeline scrolled smoothly, the chips did not flicker,
  selection and Detect worked, no windows were skipped. Two expected side effects:
  **classical music also lights "alarm"** (sustained tonal instruments resemble beeps and sirens;
  the same music/alarm confusion as on the test set), and **songs with vocals also light
  "speech"** (singing and rap are voice).

### 12.8 On the website

- **Class chips** show each class's live (smoothed) probability as a bar and a percentage, and
  light up green while the class is "on".
- Chips are now **selectable** (several at once). The selection is sent to the backend as the
  target (`set_target`); it changes nothing audible yet: *"Selected sounds will be enhanced from
  step 06"*. *(Step 06: it is audible now. The hint reads "Keeping N sound type(s); everything
  else is turned down by 30 dB", or "Select one or more sounds to enhance them" when nothing is
  selected. Chapter 13.)*
- A **detection timeline** shows the last 30 s: one lane per class, filled where it was active,
  brighter where more certain, with "now" at the right edge like the spectrogram.
- The session bar shows the detector's device, inference time and skipped windows.
- Each saved recording has a **Detect** button: the whole file is analysed with the same windows
  and smoothing, and a small timeline per class appears under it (`POST
  /api/recordings/{id}/detect`; recordings longer than 2 minutes are refused with a hint to use
  `soundspy detect`).

### 12.9 Offline: `soundspy detect`

```powershell
soundspy detect data\raw\speech_20261114-163256-131.wav [--experiment det_mn10_v1] [--json out.json]
```

prints, per class, the stretches where it was active with their mean probability, e.g.
`speech  0.50-5.00 s (p 0.89)`. Stretches of one class that touch or overlap are merged
(Chapter 23.29).

With the trained model, on the owner's recordings: the 5 s speech test → speech 0.50–5.00 s; the
62 s UI test (mostly talking) → five speech stretches (2.25–7.75, 9.00–11.25, 21.25–32.75,
34.75–48.00, 51.25–53.00 s) and one short music stretch (42.75–44.25 s); the two quiet probe
recordings → nothing. A 62 s file takes about 10 s including loading the model.

### 12.10 Limitations

- Trained only on synthetic mixtures of clean dataset clips; real rooms (reverberation, the laptop
  microphone's colour) differ. The owner's own recordings (`record-dataset`) are the way to close
  that gap.
- Some classes have few distinct clips (chair scraping 233 training clips, dishes 422, footsteps
  295), and validation/test sets for some are small (traffic test: 26 clips), so their scores are
  noisy.
- Overlapping AudioSet definitions (horns, sirens, bicycle bells) were excluded from training, so
  those sounds are not reliably reported as "alarm" or "traffic".
- 1 s windows every 0.25 s: a short sound (a single clink) is reported for up to a second, with
  about 0.25–0.5 s delay after smoothing.

---

## 13. Extraction (baseline) — keeping the selected sounds

**Exists today** (step 06). Location: `backend/src/soundspy/extraction/` (`mask_baseline.py`,
`activity.py`, `offline.py`, `metrics.py`, `base.py`), `dsp/stft.py`, `engine/detection_state.py`,
`io/runs.py`, `training/evaluate_extraction.py`, the CLI commands `extract`, `evaluate extraction`
and `db rebuild`, and on the Semantic tab: the attenuation slider, Bypass, the input/output
spectrogram, and the Process button with its A/B player.

Until now selecting a sound type changed nothing you could hear. From this step it does: **the
selected sounds are kept, everything else is turned down** (by 30 dB unless you move the
slider). This chapter describes the first method that does it, the *baseline*. It needs no
training. It is deliberately simple, it is measured honestly, and its limits are the reason for
the neural extractor of step 07 (Chapter 14), which plugs into the same places and is compared with it there.

### 13.1 The idea in plain words

The baseline cannot hear *which part* of the sound is speech. It only knows, from the detector
(Chapter 12), *whether* speech is present right now. So it works like a careful sound engineer
with one hand on a volume fader:

1. **A gate.** While none of the selected sounds is present, the fader is down: everything is
   30 dB quieter. When a selected sound starts, the fader comes up.
2. **Noise reduction while the fader is up.** During the quiet stretches the method *listens to
   what "everything else" sounds like* (the hum, the hiss, the room). While the selected sound
   plays, it removes that learnt background from the sound, frequency by frequency, and keeps
   what stands out above it.

What it **cannot** do: separate two things that sound at the same moment in the same
frequencies. If you select speech while music plays, the gate opens for the speech and the music
comes through with it. The noise reduction only removes what is *steady* or what it has heard
during the pauses; a song changes all the time. This limit is measured in section 13.7, not
hidden.

### 13.2 The picture

```
                       detector (Chapter 12)
                       "speech is active"  ──►  DetectionState  (shared, thread-safe)
                                                      │
 microphone ─► mean of the channels ─► STFT ─► for every frame (5.3 ms) and every frequency bin:
                      │                              │
                      │                 target inactive ─► gain = floor  (−30 dB)
                      │                 target active   ─► gain = Wiener gain (keeps what is
                      │                                    above the learnt background)
                      │                 in between      ─► smooth ramps (open 10 ms, hold
                      │                                    300 ms, close 200 ms)
                      │                              │
                      │                      inverse STFT ─► output ─► headphones
                      │                                         │
                      └─► same delay, no processing ─► "dry" ───┘ (played instead when Bypass is on)
```

### 13.3 The STFT: working on frequencies, frame by frame

The method changes loudness *per frequency*, so the sound is first split into frequencies. The
**short-time Fourier transform** (STFT) does that for short overlapping pieces (*frames*):
1,024 samples each (21 ms at 48 kHz), a new one every 256 samples (5.3 ms). Each frame becomes
513 numbers, one per frequency band (*bin*). After the gains are applied, the inverse transform
turns the frames back into sound and overlaps them again.

`dsp/stft.py` is built so that **doing nothing gives the input back exactly**: with a square-root
Hann window before and after, the overlapping frames add up to exactly 1. A test checks this for
blocks of 1, 100, 960 and 4,096 samples: the output equals the input to within 0.00001, delayed by
a fixed **1,024 samples (21.3 ms)**. That delay is the price of the method (a sample can only be
finished once every frame that contains it has arrived) and it is added to the listening delay
(Chapter 10.5).

### 13.4 The three parts in detail

**1. Is the target active?** One question per frame, answered by an *activity source*
(`extraction/activity.py`):

| Source | Used for | What it knows |
|---|---|---|
| `LiveDetectorActivity` | the live session | the detector's current decision. It lags: on about 0.3 s after a sound starts, off 0.6–1 s after it ends |
| `TimelineActivity` | offline processing (Process, `soundspy extract`) | decisions for the whole file, worked out beforehand with windows **centred** on each moment, so there is no lag |
| `OracleActivity` | evaluation only | the truth, from the mixture recipe |

Live, the detector and the extractor share one small object, the `DetectionState`: the detector
writes its newest decision into it, the extractor reads it. It is created per session by
`engine/factory.py` and handed to both (no global variable).

**2. The gate.** Its position moves smoothly between closed (floor gain,
`10^(−attenuation/20)`; 30 dB → 0.032) and open: it **opens in 10 ms**, **holds for 300 ms**
after the target stops (so word endings are not chopped), then **closes over 200 ms**. The
movement is linear in dB and spread over overlapping frames, so there are no clicks (a test
checks that the output never jumps more than the input does).

**3. The Wiener gain while the gate is open.**

- *Noise estimate.* While the target is **inactive**, the power in each frequency bin is averaged
  (`noise_alpha` 0.95): this is what "not the target" sounds like. Live, the detector is late, so
  a frame only counts as noise once the target has stayed inactive for another 0.5 s after it
  (`noise_guard_s`); otherwise the first 0.3 s of every word would be learnt as noise and removed.
- *Fallback: minimum statistics.* If no inactive frame has been seen for 5 s (or there is no
  detector at all), the estimate is the minimum of the smoothed power over the last 1.5 s. It
  follows whatever stays steady. For exactly that reason it is only the fallback: measured in
  step 06, it treats a **steady target** (a held note, a continuous alarm, road noise) as noise
  and removes it (Chapter 23.30).
- *The gain.* From the noise estimate comes the signal-to-noise ratio per bin (smoothed over time
  with the "decision-directed" rule of Ephraim and Malah, `dd_alpha` 0.98), and from it the Wiener
  gain `snr / (1 + snr)`: near 1 where the sound is far above the background, near 0 where it is
  not. It never goes below the floor gain.

All numbers are in `configs/extraction/mask_baseline.yaml` and are checked when the app starts.

**Special cases**

- **Nothing selected** → pass-through. The sound still goes through the STFT with gain 1, so the
  delay stays the same and selecting a class does not make the sound jump; the change is ramped.
- **No detection model** → no gating. The gate stays open and only the minimum-statistics noise
  reduction works; the page shows a warning that selecting sounds cannot switch the rest off.
- `extraction.method: passthrough` in `configs/extraction/default.yaml` removes the extractor
  (and its 21 ms) altogether.

### 13.5 Live: what changed in the session

- **Selecting chips is audible.** The target (`labels` + `attenuation_db` from the new slider,
  0–40 dB) reaches the extractor between two chunks.
- **Bypass (A/B).** With Bypass on, the headphones get the unprocessed sound instead, *delayed by
  the same 21 ms*, so switching back and forth is instant and nothing shifts in time. Processing
  keeps running in the background.
- **Two spectrograms.** A second analyser watches the chain's *output*. The page can show input,
  output or both stacked: what was removed is visible in the upper picture and missing in the
  lower one.
- **Measured on the laptop** (60 s, laptop microphone, speech selected, Listen on but muted, GPU):

| | |
|---|---|
| processing time per 20 ms chunk | 1.6 ms mean, 3.3 ms p95 → **RTF p95 0.17** |
| listening delay | 105–125 ms (median 125), of which **21.3 ms** is the STFT |
| underruns / overflows / dropped / skipped windows | 0 / 0 / 0 / 0 |
| backend CPU | 24 % of one core |
| quiet room, speech selected | output 30.8 dB below the input |

  With a test loop (6 s: speech from 3.7 to 5.4 s over steady noise): speech selected → the output
  is 3.8 dB below the input while the speech sounds and 29 dB below elsewhere; an absent class
  selected → 30 dB below throughout; nothing selected → unchanged (−0.1 dB).

  Two problems surfaced during these measurements and were fixed: garbage-collection pauses
  that caused an occasional underrun (Chapter 23.31) and uneven pacing of file playback
  (Chapter 23.32).

- **The owner's check** (after step 06, laptop microphone, headphones): selecting sound types,
  Attenuation, Bypass, the input/output spectrogram and Process with the A/B player all worked as
  described in Chapter 2; no problems were reported.

### 13.6 Offline: Process, A/B, `soundspy extract`

- **On the page:** select one or more chips, set the attenuation, press **Process** on a
  recording. A *run* appears under it: *waiting…* → *processing…* → *done*. A finished run has an
  **A/B player**: one play button and one position; the **Original / Processed** switch changes
  what you hear without changing the moment you are at. There is also a download link.
- **On the command line:**

```powershell
soundspy extract data\raw\speech_20261114-163256-131.wav --labels speech [--attenuation-db 30] [--out out.wav]
```

- **How it works.** The detector runs over the whole file first, with windows centred on each
  moment; the extractor then uses that timeline (no lag, unlike live). The result is one channel,
  exactly as long as the recording and aligned with it, which is what makes A/B meaningful.
- **Runs are files.** Each run is `outputs/processed/<run_id>.wav` plus `<run_id>.json` (the
  recording, the classes, method and settings, status, error, timing). As for recordings, the
  JSON sidecar is the truth and the database only indexes it (Chapter 16). Only **one job runs at
  a time**; others wait in order. State changes are pushed to the page over the WebSocket. A
  recording longer than 5 minutes is refused on the page (use the CLI).
- **Measured:** the 61.9 s UI recording was processed in **6.5 s** through the API (model already
  loaded); inside the detected speech the level changed by −0.2 dB, outside it by −30.0 dB.
  `soundspy extract` on the 5 s speech recording took 5.7 s, almost all of it loading the model.

### 13.7 Measuring it: SI-SDR, the five systems, and what the numbers say

**The metrics** (`extraction/metrics.py`). Test mixtures are built from recipes, so the correct
answer (the target alone) is known exactly.

- **SI-SDR** (scale-invariant signal-to-distortion ratio, dB): how much of the output is the
  wanted sound, however loud the output is. Higher is better. **SI-SDRi** is the improvement over
  doing nothing: 0 dB = no better than the unprocessed mixture, negative = worse.
- **SNRi**: the same idea but loudness counts too. SI-SDR does not notice an output that is
  "right but everything 30 dB too quiet"; SNR does.
- **Suppression** (dB): for mixtures where the selected class is *absent*, the correct output is
  silence, so the output level relative to the input is reported. −30 dB is the best this method
  can do (its floor).
- **Target missed**: the share of mixtures whose whole output sits at the floor, i.e. the gate
  never opened although the target was there. SI-SDRi scores that as exactly 0, which looks
  harmless, so it is counted separately.

**The five systems**, on the same 500 test mixtures (fixed, seeded; 48 kHz; attenuation 30 dB;
`soundspy evaluate extraction --method mask_baseline --split test --n 500`, 3 min 56 s):

| System | SI-SDRi mean | SI-SDRi median | SNRi | Target missed | Suppression, target absent |
|---|---|---|---|---|---|
| mixture (no processing) | 0.00 | 0.00 | 0.00 | 0 % | 0.0 |
| baseline + **timeline** (offline detector) | **+2.19** | 0.00 | +4.17 | 28 % | −26.6 |
| baseline + **live** (detector lag reproduced) | **−4.85** | −0.96 | +0.80 | 32 % | −28.7 |
| baseline + **oracle** (true activity) | **+8.09** | +7.25 | +8.77 | 0 % | −30.0 |
| **IRM oracle** (best any masking method could do) | **+24.52** | +18.24 | +24.66 | 0 % | −100 |

(483 mixtures with a target, 17 with the target class absent. Full tables:
`experiments/ext_mask_baseline/metrics.md`; listening examples:
`outputs/eval/ext_mask_baseline/`, described in its `index.md`.)

**Reading the table**

- **The ceiling is far away.** A perfect mask in the same STFT would gain 24.5 dB. Even with
  perfect knowledge of *when* the target sounds, the baseline gains 8.1 dB. The 16 dB between
  them is what "cannot separate overlapping sounds" costs: the gap a neural extractor can close.
  *(Step 07: the neural extractor reaches +3.3 dB, live and offline alike. It beats this
  baseline as it is, but not yet the baseline with perfect timing; Chapter 14.9.)*
- **The detector costs most of the rest.** Oracle − timeline = 5.9 dB: the detector's mistakes.
  In 28 % of the mixtures it never notices the target, so nothing is let through at all.
- **Live is worse than doing nothing on these mixtures (−4.9 dB).** Oracle − live = 12.9 dB. The
  extra 7 dB against the timeline is pure *lag*: the gate opens ~0.3 s after the sound starts and
  stays open ~1 s after it ends. For a short sound (a clink, a single word) that means the gate
  is shut during the sound and open after it, which is worse than not gating. The test mixtures
  are 6 s long with short events, the hardest case for a lagging gate; in a continuous
  conversation the same lag matters much less.
- **When the target is absent, the method does its job:** −27 to −29 dB with the real detector,
  the full −30 dB with the truth.

**Breakdowns** (SI-SDRi, dB)

| | timeline | live | oracle | IRM |
|---|---|---|---|---|
| target sounds < 1 s (196) | +5.4 | −5.6 | +11.5 | +32.6 |
| target sounds 1–3 s (203) | +1.1 | −4.3 | +7.5 | +20.7 |
| target sounds ≥ 3 s (84) | −2.7 | −4.6 | **+1.5** | +15.0 |
| target 5 dB or more below the rest (118) | +3.1 | −4.1 | +8.9 | +24.3 |
| target 5 dB or more above the rest (112) | +0.9 | −6.9 | +6.6 | +24.8 |
| 1 foreground source (165) | +3.8 | −4.4 | +9.8 | +29.3 |
| 3 foreground sources (151) | +0.8 | −5.9 | +7.0 | +20.7 |

- **The longer the target sounds, the less a gate can do.** A target that fills most of the
  mixture leaves the gate open all the time; with the truth as guide the gain shrinks from +11.5
  to +1.5 dB. This is the overlap limit in numbers: while the target sounds, almost everything
  else comes through with it.
- **More sources, less gain,** for every system.
- **By class** (oracle +6.1 to +8.6, timeline −0.4 to +3.8): speech, dishes and alarm do best
  with the real detector (+3.5 to +3.8); chair scraping and traffic, the detector's weakest
  classes, gain nothing (−0.3, −0.4).

**What each part contributes** (an extra check on 120 mixtures, true activity): the gate alone
gives +6.5 dB; gate + Wiener gain +7.9 dB; minimum statistics alone, without any gating, +0.3 dB.
So the gating does most of the work, the noise reduction adds about 1.3 dB, and without a detector
the method is of little use on mixtures like these.

### 13.8 Limitations

- **It does not separate overlapping sounds.** While a selected sound is present, whatever else
  sounds at the same time is let through (only its steady part is reduced). This is the main
  limit and the reason for the neural method (Chapter 14).
- **It is only as good as the detector,** and live the detector is late: short sounds lose their
  beginning and are followed by up to a second of everything else. Offline processing (no lag)
  is clearly better than live (+2.2 vs −4.9 dB on the test mixtures).
- **A missed sound is silenced.** If the detector does not notice the target, the output is 30 dB
  down throughout. A lower attenuation makes this less drastic, at the price of less
  suppression.
- **21 ms more delay** when listening, with or without a selection.
- **One channel.** It works on the mono mix; direction is not used (that is module 3).

---

## 14. Neural extraction — a network that pulls the chosen sounds out

**Exists today** (step 07). Location: `backend/src/soundspy/extraction/` (`neural_tse.py`,
`neural_model.py`, `models/waveformer/`), `dsp/resample.py` (`StreamingResampler`),
`data/cache.py`, `data/reverb.py`, `training/extraction_data.py`, `training/train_extraction.py`,
`training/evaluate_extraction.py`, the settings in `configs/extraction/neural_tse.yaml`, the CLI
commands `data cache`, `train extraction`, `evaluate extraction --method neural_tse` and
`extract --method`, and on the Semantic tab: the **Method** selector, the output level meter and
the new figures in the session bar.

The baseline of Chapter 13 only knows *when* a selected sound is present. While speech and music
play at the same moment it has nothing to offer: its gate is open and both come through. This
chapter describes the second method, which does what the project is really about: a **neural
network that listens to the mixture and gives back only the selected kinds of sound**, even
while other sounds overlap them. It is called *target sound extraction* (TSE). Both methods stay
in the app, and you choose between them with the Method selector, so they can be compared.

### 14.1 The idea in plain words

Think of a person who has listened to thousands of rooms and learnt what speech, music, dishes,
footsteps and so on sound like. You hand this person a recording and say "speech, please". They
hand back a recording that contains the speech and (ideally) nothing else. They do not use a
volume fader, as the baseline does. They rebuild the sound, sample by sample, from what they
hear.

The network is such a listener. It gets two things, 100 times a second:

1. the newest few milliseconds of sound from the microphone, and
2. a list of seven switches, one per sound type, saying which types you selected.

It gives back the same few milliseconds with only the selected types in them. Because it works
on the sound itself, it does not wait for the detector (no 0.3 s reaction time), and it can keep
a voice while removing music that plays at the same moment.

It is not magic, and the measurements in 14.9 show where it fails. The typical mistake of the
baseline is *letting everything through*. The typical mistake of a network is different:
sometimes it **extracts the wrong sound**, or damages the right one. This is why the result is
mixed back with a little of the original (14.8).

### 14.2 The network: Waveformer

Sound Spy does not invent its own network. It uses **Waveformer** (Veluri, Chan, Itani, Chen,
Yoshioka, Gollakota; ICASSP 2023), from the same research group as SemanticHearing. It was
built for exactly this job: extracting named sounds *while they happen*, on a small computer. Its
code is open (MIT licence) and the authors publish a trained copy. The model code was adapted
into `extraction/models/waveformer/` (Chapter 26 lists what was taken and changed).

```
 sound in, 44,100 samples per second
     │
     ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────┐
 │ ENCODER  "what is in the sound?"                                                         │
 │  1 convolution: every 32 samples (0.7 ms) become one "latent frame" of 512 numbers       │
 │  10 causal convolution layers, each looking twice as far back as the one before          │
 │  (1, 2, 4 … 512 frames): together they remember about 1.5 s of the past, never the future│
 └──────────────────────────────────────────────────────────────────────────────────────────┘
     │ 512 numbers per frame                 seven switches: [speech, music, …]  (1 = wanted)
     │                                              │
     │                                   LABEL NETWORK: two small layers turn the switches
     │                                   into 512 numbers "what to listen for"
     ▼                                              │
     ×  ◄───────────────────────────────────────────┘   (multiplied, number by number)
     │
     ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────┐
 │ DECODER  "which parts belong to the wanted sounds?"                                      │
 │  a small transformer layer; it works on 13 frames at a time (9.4 ms) and may look at the │
 │  previous 13. Its result is a MASK: a number per frame and feature saying how much to keep│
 └──────────────────────────────────────────────────────────────────────────────────────────┘
     │ mask × encoder output
     ▼
 1 transposed convolution: back from frames to samples   ─►  sound out: only the wanted types
```

Some words from the picture:

- **Causal** means the network never uses the future. It has to: live, the future has not
  happened yet. The only exception is a *lookahead* of 32 samples (0.7 ms), which is the price
  of its first layer.
- A **convolution** slides a small pattern-matcher along the sound. A *dilated* one skips
  samples, so a few layers can cover a long stretch of time cheaply.
- A **transformer** layer compares every frame in its window with every other one and decides
  what belongs together. It is the same building block large language models are made of, here
  in a very small size.
- The network has **3.86 million** adjustable numbers (*parameters*): 15 MB as a file. A language
  model has thousands of times more. Small is the point: it must finish 9.4 ms of sound in well
  under 9.4 ms.

**Streaming.** The network works in *steps* of 416 samples (9.4 ms at 44.1 kHz). Between steps it
keeps three small memories (the encoder's past, the decoder's past, the unfinished end of the
output). Running it step by step with these memories gives the same output as running it once
over the whole recording. This is tested (`tests/extraction/test_waveformer.py`), and it is what
makes the same code usable live and offline (Rule 5).

**Was the port faithful?** The original code was run once on the authors' own example file
(`Sample.wav`, asking for "Computer_keyboard"), and Sound Spy's adapted copy was run on the same
file with the same published weights: the two outputs are **identical** (difference 0.0), and the
step-by-step streaming version differs from them by less than one millionth. Two small files in
`tests/fixtures/` hold outputs of the *original* code, so the tests keep checking this without
needing the original code or any download.

### 14.3 Teaching it Sound Spy's classes: fine-tuning

The published network knows 41 sound classes (from a dataset called FSDKaggle2018: "Cello",
"Bark", "Computer_keyboard"…). Sound Spy has seven, and they are different ones. Training a
network from nothing takes days on a strong graphics card. **Fine-tuning** is the shortcut:
start from the published network, which already knows how to pull a named sound out of a
mixture, and train it further on Sound Spy's own material until it knows the new names.

Only one part has to be rebuilt: the first layer of the label network, which had 41 inputs and
now needs 7. Each new input starts from what the network already knew about similar classes:

| Sound Spy class | Starts from the published classes | Why |
|---|---|---|
| speech | *(none: the average of all 41)* | the published network was never taught speech |
| music | 16 instruments (guitar, cello, flute, drums, saxophone, violin…) | music is made of them |
| footsteps | Knock | short dull impacts |
| dishes | Keys_jangling, Chime, Shatter | bright, ringing, clinking sounds |
| chair_scrape | Squeak, Drawer_open_or_close | friction and furniture |
| alarm | Telephone, Microwave_oven | ringing and beeping |
| traffic | Bus | the only vehicle among the 41 |

The mapping is a setting (`label_init` in `neural_tse.yaml`), not code. During training that new
layer learns four times faster than the rest of the network (learning rates 0.001 and 0.00025):
it has the most to learn, while the rest should change carefully.

### 14.4 The training material: a cache on D:, and simulated rooms

The mixtures are the *recipes* of Chapter 11: 20,000 for training, 2,000 for validation, 2,000
for the final test. Every recipe says which clips are mixed, when, how loud, and which classes
are "selected". From a recipe the code renders the **mixture** (what the microphone would hear)
and the **target** (only the selected classes): the network gets the first and is asked to
produce the second.

**Why a cache.** Rendering one recipe takes about 50 ms; the graphics card needs far less than
that to learn from it. In step 05 the training spent most of its time waiting for mixtures. So
this time every recipe is rendered **once** and stored on drive D:
(`soundspy data cache --task extraction`):

| | Recipes | Size on D: | Time to build |
|---|---|---|---|
| train | 20,000 | 21.2 GB | 14.6 min |
| val | 2,000 | 2.1 GB | 1.6 min |
| test | 2,000 | 2.1 GB | 1.6 min |

The files are plain arrays in *shards* of 256 recipes, stored as 16-bit floating point numbers
(half the size of the usual 32-bit, and precise enough). A shard is written under a temporary
name and renamed when it is complete, so an interrupted build continues where it stopped, and
running the command again when everything is there does nothing. Ten processes render in
parallel. Before starting, the command checks that the disk has room (and keeps 10 GB free).
Training then reads the shards through *memory mapping*: the operating system loads only the
parts that are used.

**Simulated rooms.** The dataset clips are recorded close and dry. A real room adds
*reverberation*: the sound also arrives a little later from the walls. A network that has only
heard dry sound can stumble over that. So **half of the recipes are rendered inside a simulated
room** (`data/reverb.py`, using the `pyroomacoustics` library): a box-shaped room of random size
(3–10 m by 3–8 m, 2.4–4 m high), a random reverberation time between 0.2 and 0.8 s, the
microphone at a random spot and **every sound at its own random position**. Each sound is passed
through the room's *impulse response* (how the room answers a hand clap from that position)
before the sounds are added up.

Two details matter:

- The target is the **reverberant** target. The network is asked to remove the *other sounds*,
  not the room. (Removing reverberation is a different, harder task.)
- Which recipes get a room, and which room, depends only on a seed and the recipe's name. A
  rebuilt cache is therefore identical.

On top of that, every time a training example is used it gets a random volume change (±6 dB, the
same for mixture and target) and a little white or pink hiss on the mixture only, as a real
microphone would add. The training does not use whole 6 s examples but random **3 s crops**
(a whole example does not fit the 4 GB graphics card at a useful batch size); 80 % of the crops
are placed where the target actually sounds.

### 14.5 What "good" means to the network: the loss

Training means: show an example, measure how wrong the output is with one number (the *loss*),
and nudge the 3.86 million parameters so that the number gets smaller. What that number rewards
decides what the network becomes.

**When the target sounds**, the loss is built from two measures, both in decibels, both "higher
is better":

- **SNR** compares output and target sample by sample. Shape *and* loudness must be right.
- **SI-SNR** (the same thing as the SI-SDR of Chapter 13.7) ignores loudness and looks only at
  the shape. It is what the evaluation reports.

The loss is `−(0.2 × SNR + 0.8 × SI-SNR)`. The small share of SNR is there for one reason: it
fixes the *loudness* of the output, which SI-SNR does not care about and the attenuation slider
needs (14.8).

**When the target is absent** (you asked for speech and nobody speaks) the right answer is
silence, and both measures break down (they divide by the target's energy, which is zero). For
these examples the loss is simply **how loud the output is compared with the input, in dB**. An
output 20 dB quieter than the input scores −20. Below −40 dB there is no further reward,
otherwise the easiest way to improve the score would be to push already inaudible outputs
towards minus infinity instead of learning anything. This term counts half as much as the
others.

**A lesson from the first run.** The published network was trained with the weights the other
way round (0.9 × SNR + 0.1 × SI-SNR), and the first full training run of this step copied that.
It stopped after 9 of 32 rounds at +2.6 dB. Looking at the 500 validation examples one by one
explained why: the typical example was fine, but about one in eight was *worse than doing
nothing*, some by 20–40 dB. In those, the network had extracted the **wrong sound with full
confidence**. SNR hardly punishes that: if you are not sure, SNR's safe choice is to output
*something quiet*, and a wrong quiet sound costs about as much as silence. SI-SNR punishes it
by 20–40 dB. A short side-by-side test (same starting point, 6,000 examples each) confirmed it:
the SI-SNR-heavy loss gained almost 1 dB on the validation set where the old one gained nothing.
The second run used it.

### 14.6 The training run

`soundspy train extraction` (in its own PowerShell window, writing to a log file):

| | First run | Second run (the one in use) |
|---|---|---|
| Loss for examples with a target | 0.9 × SNR + 0.1 × SI-SNR | **0.2 × SNR + 0.8 × SI-SNR** |
| Stops after … rounds without improvement | 6 | 10 |
| Rounds (*epochs*) done, of 32 planned | 9 | 26 |
| Best round | 3 | 16 |
| Validation SI-SDRi of the best round | +2.60 dB | **+3.72 dB** |
| Time | 1 h 58 min | 5 h 24 min |
| Kept in | deleted on 2026-11-17 (owner's decision; its numbers are in this table) | `experiments/tse_waveformer_v1/` |

How a run works, in the numbers of the second one:

- **Batch size.** Before training, the program tries 1, 2, 3, 4, 6… examples per batch with one
  real training step each and measures the memory. Four 3-second crops need 2.26 GB of the
  card's 4 GB; six would need 3.37 GB, more than the 72 % the program allows itself (beyond the
  card's real memory Windows silently switches to memory that is a hundred times slower,
  Chapter 23.37). So the batch is **4**, and the corrections of **4 batches are added up**
  before the parameters are changed: each update still learns from 16 examples.
- **One round** = 10,000 crops (2,500 batches, 625 updates), about **12 minutes** plus 45 s of
  validation. The rounds rotate through the material: two rounds together use each of the 20,000
  recipes exactly once, every time with a different crop, volume and hiss.
- **Learning rate.** It rises during the first round (*warm-up*), then falls smoothly towards
  zero along a cosine curve: big steps first, fine adjustments at the end.
- **Validation** after every round on the same 500 whole 6-second validation examples, measured
  the way the final evaluation measures (on the output with the rest 30 dB down). The round with
  the best score is kept as `best.pt` (15 MB); `last.pt` also holds what is needed to continue
  with `--resume`.

The curve of the second run (`plots/training_curves.png` in its folder):

| Round | 1 | 2 | 5 | 9 | 12 | **16** | 20 | 24 | 26 |
|---|---|---|---|---|---|---|---|---|---|
| Training loss (lower is better) | −1.4 | −4.2 | −5.1 | −5.6 | −5.8 | −6.2 | −6.4 | −6.6 | −6.8 |
| Validation SI-SDRi (dB) | +2.44 | +3.00 | +3.41 | +3.42 | +3.60 | **+3.72** | +3.51 | +3.70 | +3.62 |

Two things can be read from it. After **one** round the new loss was already where the first run
ended after nine. And from about round 12 the validation score stops rising (+3.5 to +3.7) while
the training loss keeps falling: the network goes on getting better at the 20,000 mixtures it
trains on without getting better at new ones. That is called *overfitting*, and it says that
**more rounds would not help; more varied training material would**. The run stopped itself at
round 26, ten rounds after its best.

On the validation set the kept network improves a typical example by 3.3 dB (the median) and the
best tenth by more than 13 dB, while 41 of 479 examples are still worse than −5 dB (it was 52 in
the first run). By class, for examples with a single selected class: speech +8.1 dB, dishes
+7.1, music +4.5, alarm +3.3, footsteps +3.2, traffic +3.2, chair scraping +2.0.

All training together (two runs, a 10-minute trial and a 20-minute comparison of the two losses)
took about **7.9 hours** of the 12 allowed. Following a run and continuing an interrupted one:

```powershell
Get-Content experiments\tse_waveformer_v1\log.csv -Wait     # one line per round
soundspy train extraction --resume                          # continue after a stop
```

### 14.7 Running it live

**In the audio path.** The extractor is a `Processor` like the baseline (Chapter 7): it gets a
20 ms chunk and returns a 20 ms chunk. Inside, three things have to line up.

```
 chunk from the microphone (960 samples at 48,000 per second)
   │
   ├─► resample 48 → 44.1 kHz ─► waiting line ─► whenever 416 new samples are there (9.4 ms):
   │                                              one step of the network (with its memories)
   │                                                        │
   │                                    resample 44.1 → 48 kHz
   │                                                        │ "estimate"
   └─► delay line (exactly as long as the path above) ──────┤ "dry"
                                                            ▼
                         output = estimate + g × (dry − estimate)        (14.8)
```

- **Two sample rates.** The laptop's microphone delivers 48,000 samples per second, the network
  was trained at 44,100. Retraining it at 48,000 would throw away what it knows (every learnt
  filter would mean a different frequency). Instead, a **streaming resampler** converts on the
  way in and back on the way out (`dsp/resample.py`). "Streaming" means it keeps the last
  samples of each chunk, so the result is the same as converting the whole recording at once,
  with no clicks at the chunk borders (tested to 0.00001). The length of the second filter is
  chosen so that both filters together delay the sound by a *whole* number of samples (22), which
  keeps Bypass and the measurements exactly aligned.
- **Steps and chunks do not match.** A chunk is 20 ms, a network step 9.4 ms. A small waiting
  line collects the samples; each chunk usually triggers two steps, sometimes three, and they go
  through the network in one call.
- **Constant delay.** The extractor always answers with exactly as many samples as it got. For
  that it runs a fixed 509 samples (10.6 ms) behind: the two filters (22 samples), the worst-case
  wait for a full step, and the network's lookahead. Bypass plays the dry signal with the same
  delay, so switching does not shift the sound.
- **Nothing selected** = the delayed input, untouched; the network is not run. Selecting and
  deselecting fades over 20 ms instead of switching (no click).
- **Changing the selection** keeps the network's memory by default: its memory is about the
  *sound*, not about the selection. (`on_target_change: reset` clears it instead.)
- **Where it runs.** On the graphics card if there is one, otherwise on the processor. The
  network is "warmed up" on the audio thread before the first chunk (the first use of the
  graphics card from a new thread takes seconds, as found in step 05; `Processor.start()` is the
  new hook for this).
- **The detector keeps running** next to it. The network does not need it, but the chips and the
  timeline still show what is being heard. While the network uses the graphics card, the
  detector runs on the processor (see the measurements below).

**Measured on the laptop** (microphone, Neural, speech selected at 15 dB, Listen on but muted,
60 seconds each; RTF p95 must stay under 0.8 and the underrun counter at 0):

| Where the models run | RTF p95 (typical / worst second) | Network time per call | Underruns | Delay to the ear | Processor use |
|---|---|---|---|---|---|
| extractor on the graphics card, detector on the processor (**the default**) | **0.63** / 0.72 | 7.5 ms | **0** | 95–115 ms | 71 % of one core |
| extractor on the processor (4 threads), detector on the graphics card | 0.61 / 0.71 | 7.6 ms | 1 | 105–155 ms | 3.5 cores |
| everything on the processor (no graphics card) | 0.69 / 0.81 | 8.1 ms | 1 | 95–135 ms | 3.8 cores |
| *for comparison:* Baseline method (detector on the graphics card) | 0.13 | – | 0 | 115–125 ms | 23 % of one core |

Getting there took two findings, both now built in:

1. **The two models must not share the graphics card.** In the first measurement both ran on
   it: 5 underruns in a minute and a delay that crept up to 225 ms. The detector's recognition
   takes 25 ms, four times a second; an extractor call that arrives during one has to wait, and
   20 ms is all it has. With the detector on the processor (one thread, about 80 ms per
   recognition, well inside its 250 ms rhythm) the problem is gone. So when the extractor uses
   the graphics card and the detector's device is left on `auto`, the detector is started on
   the processor (`detector_on_cpu` in `neural_tse.yaml`).
2. **Most of a call was bookkeeping, not arithmetic.** One call of this small network is about
   200 tiny operations, and PyTorch spends more time organising them than computing. *Tracing*
   the step once (TorchScript) removes much of that: exactly the same numbers, **8.4 → 6.0 ms**
   per call on the graphics card (11.2 → 8.4 ms on the processor). The traced step is made in
   `start()`, checked against the plain one, and run without TorchScript's own run-time
   optimiser, which had caused single calls of 48, 60 and 208 ms during the first second
   (`compile` in `neural_tse.yaml`).

**The delay budget** with the default setting (about 105 ms, as with the baseline):

| Part | Time |
|---|---|
| microphone and its driver | ~22 ms |
| collecting one 20 ms chunk | 20 ms |
| the extractor's own delay (filters 0.5 ms + waiting for a full step up to 9.4 ms + lookahead 0.7 ms) | 10.6 ms |
| speaker queue (its cushion also absorbs the ~9 ms of computing) | ~30 ms |
| speaker and its driver | ~22 ms |

The extractor itself adds only 10.6 ms, half of the baseline's 21.3 ms. A session reports
*running* after about 3 s; the first time after the backend was started, sound flows a few
seconds later, while the network's steps are being traced. Without a graphics card the app still runs, but close to its limit: expect an
occasional crackle on a slower processor. (`block_chunks` in the settings lets the network
process more steps per call, 9.4 ms of extra delay each; on this laptop it was not needed.)

Offline, a one-minute recording is processed in about 7 s with either method (12 s the first
time, while the model loads).

### 14.8 The attenuation slider, and choosing the method

**Residual mixing.** The network's output is not played as it is. It is mixed with what it
removed:

```
output = estimate + g × (input − estimate)        g = 10^(−attenuation / 20)
```

With the slider at 15 dB (`g` = 0.18) everything the network removed comes back 15 dB quieter;
at 0 dB (`g` = 1) the output is the input; at 40 dB almost only the estimate is left. This is the
same meaning the slider has for the baseline ("how much is everything else turned down"), and it
has two more benefits: you stay aware of your surroundings, and when the network makes a
mistake, the piece it wrongly removed is still faintly there.

**The live default is now 15 dB** (it was 30 in step 06). The evaluation keeps 30 dB so that
the numbers can be compared with step 06; the table in 14.9 also has a row at 15 dB.

**Choosing the method.** The Method selector on the Semantic tab offers **Neural** and
**Baseline**. The choice belongs to a session: it can be changed while stopped, not while
running, because the two methods have different delays (10.6 ms and 21.3 ms) and swapping in
mid-stream would make the sound jump. The Process button has its own "Process with" choice, so
the same recording can be processed with both and compared in the A/B players. If Neural is
chosen but its trained file is missing, the session starts with the baseline and a message says
so; nothing breaks.

### 14.9 Results against the baseline

`soundspy evaluate extraction --method neural_tse` runs every system on the **same 500 test
mixtures** as in step 06 (48 kHz, everything else 30 dB down, outputs lined up with the input
before measuring). The score is SI-SDRi: how much closer to the correct answer the output is than
the untouched mixture, in dB. 0 = no better than doing nothing.

| System | SI-SDRi, average | SI-SDRi, typical (median) | SNRi | Target missed | Quiet when the target is absent |
|---|---|---|---|---|---|
| nothing (the mixture itself) | 0.0 | 0.0 | 0.0 | – | 0 dB |
| Baseline, live (as you hear it) | **−4.85** | −0.96 | +0.80 | 32 % | −28.7 dB |
| Baseline, offline (Process) | +2.19 | 0.00 | +4.17 | 28 % | −26.6 dB |
| Baseline with the *true* timing (oracle) | +8.09 | +7.25 | +8.77 | 0 % | −30.0 dB |
| **Neural, whole file at once** | **+3.28** | +3.32 | +5.51 | 0 % | −12.5 dB |
| **Neural, streaming as live** | **+3.28** | +3.32 | +5.51 | 0 % | −12.5 dB |
| Neural, streaming, slider at 15 dB | +3.32 | +3.17 | +5.06 | 2 % | −8.5 dB |
| ideal mask (cannot be built) | +24.52 | +18.24 | +24.66 | 0 % | – |

The same 500 mixtures **in simulated rooms** (every sound through its own room echo, a fixed
set of rooms never used in training):

| System | SI-SDRi, average | typical (median) | Target missed |
|---|---|---|---|
| Baseline, live | −5.20 | −0.17 | 36 % |
| Baseline, offline | +0.70 | 0.00 | 33 % |
| Baseline with the true timing | +7.72 | +6.94 | 0 % |
| **Neural, streaming** | **+2.92** | +3.11 | 0 % |

**What the numbers say, in plain words.**

- **Live, the difference is large.** What you hear with the baseline is on average *worse* than
  doing nothing (−4.85 dB), because its gate opens late. Neural is at +3.28 dB: **8 dB better**,
  and better in 375 of the 483 mixtures with a target. This is the comparison that matters for
  the app, because live is how it is used.
- **Streaming equals offline.** Processed piece by piece exactly as in a live session, the
  network gives the same result as when it gets the whole file (the largest difference over all
  483 mixtures is 0.02 dB). The baseline loses 7 dB between offline and live; Neural loses
  nothing.
- **Against the baseline's offline mode the lead is small**: +3.28 against +2.19 dB on average.
  The *typical* mixture tells more: the baseline's median is 0.00 (in half of the mixtures it
  achieves nothing or less), Neural's is +3.32. Neural is the better one in 278 of 483 mixtures
  (58 %).
- **It never misses the target outright** (0 %, against 28–32 % for the baseline, whose detector
  simply does not notice the sound).
- **It is still far from what is possible.** A gate with *perfect* timing reaches +8.09 dB, and
  Neural does not come near it. The reason is visible in the spread: the best tenth of the
  mixtures gains more than 11.6 dB, but 51 of 483 (about one in nine) end up more than 5 dB
  *worse* than untouched. Those are the cases where it keeps the wrong sound, and they pull the
  average down.
- **It is poor at staying quiet.** When the selected sound is not there at all, the output is
  only 12.5 dB quieter than the input (the baseline: 27–29 dB of the possible 30). With the
  slider at the live default of 15 dB it is 8.5 dB. There are only 17 such mixtures among the
  500, so this number is rough, but the direction is clear.
- **Rooms cost little.** In simulated rooms Neural loses 0.4 dB (+2.92), the baseline's offline
  mode loses 1.5 dB (+0.70). The simulated rooms in the training material did their job.
- **The slider setting hardly changes the score** (+3.32 dB at 15 dB against +3.28 at 30): what
  is lost in suppression is won back by milder mistakes.

**By selected class** (SI-SDRi in dB; a mixture with two selected classes counts for both):

| Class | Baseline live | Baseline offline | **Neural** | Neural, in rooms |
|---|---|---|---|---|
| footsteps | −4.08 | +1.07 | **+4.41** | +3.98 |
| dishes | −5.26 | +3.46 | **+3.84** | +3.49 |
| alarm | −4.63 | +3.80 | +3.75 | +3.43 |
| speech | −6.66 | +3.74 | +3.60 | +3.45 |
| music | −5.30 | +0.64 | **+2.65** | +2.06 |
| traffic | −4.86 | −0.39 | **+2.46** | +2.08 |
| chair scraping | −3.02 | −0.32 | +0.16 | −0.05 |

Neural clearly beats the offline baseline where sounds are long or continuous (music, traffic)
or short and frequent (footsteps); for speech, alarm and dishes the two are level; chair scraping
is where neither works.

**Other cuts through the same results:**

| | Baseline offline | **Neural** |
|---|---|---|
| one class selected (334 mixtures) | +3.16 | **+3.89** |
| two or more classes selected (149) | +0.03 | **+1.91** |
| one foreground sound in the mixture (165) | +3.76 | +4.19 |
| two (167) / three (151) | +1.87 / +0.83 | +2.66 / +2.95 |
| target much quieter than the rest (118) | +3.09 | +3.89 |
| target much louder than the rest (112) | +0.94 | +2.79 |
| target sounds for less than 1 s (196) | +5.44 | +5.07 |
| 1–3 s (203) / more than 3 s (84) | +1.06 / −2.66 | +2.37 / +1.27 |

The baseline is only good when the target is short (a gate is the right tool for a short sound in
a pause). Neural keeps a positive score in every row, also for long targets, several selected
classes and crowded mixtures: the situations a gate cannot handle by design.

**Listen for yourself.** `outputs\eval\ext_neural_tse\` holds ten test mixtures, each with the
input, the correct answer and every system's output; `index.md` there lists them with their
scores. Two are worth hearing first because they were the examples of step 06: `test-000030`
(speech with dishes: Neural +6.6 dB, baseline live −9.1) and `test-000036` (quiet speech under
footsteps and dishes: here Neural *fails*, −3.4 dB, while the offline baseline reaches +10.7).
The full tables are in `experiments\ext_neural_tse\metrics.md`, and every mixture's numbers in
`per_recipe.csv` next to it.

### 14.10 Limitations

- **A modest gain, unevenly spread.** +3.3 dB on average; large for many mixtures, negative for
  about one in nine. It is clearly better than the baseline live, only slightly better than the
  baseline's offline mode, and well below what a perfect gate would reach.
- **It sometimes keeps the wrong sound.** A network that rebuilds the sound can be confidently
  wrong, which a volume fader cannot. Residual mixing (15 dB by default) softens this.
- **It does not go quiet when the selected sound is absent** as firmly as the baseline does
  (about 12 dB instead of nearly 30).
- **Two sounds of the same type cannot be told apart.** "Speech" means every voice: yours, the
  neighbour's and the singer in the song. Choosing one voice among several is what the Spatial
  and Sources tabs are for (direction, modules 3–5).
- **Chair scraping barely works**, and music and traffic are weaker than the other classes.
- **It learnt from only 20,000 mixtures** of dataset clips, and it began to overfit after 12
  rounds. More and more varied material is the most promising way to improve it; a longer
  training of the same material is not.
- **Not yet verified by long listening in real rooms.** All numbers come from synthetic
  mixtures; the simulated rooms help, but a real cafeteria is a different thing. No recordings
  from the owner's own microphone were used for training.
- **It needs the graphics card to be comfortable.** On the processor alone it runs at about
  0.7–0.8 of real time on this laptop and uses almost four cores. And while it uses the
  graphics card, nothing else should (a training run in the background would make it stutter).
- **One channel.** Like the baseline it works on the mono mix; the second microphone is not used
  yet.

---

## 15. The backend's web interface — REST and WebSocket

**Exists today**, partly. Location: `backend/src/soundspy/api/`.

### 15.1 The addresses

| Address | What it does | Status |
|---|---|---|
| `GET /api/health` | "Are you alive?" Answers `{"status": "ok", "version": "0.1.0"}` | **works** |
| `GET /api/config` | the class list (ID, display name, description), which tabs are available and why not, the active device profile. *Step 07:* also the extraction methods (name, display name, whether each can run and why not), the method a session would really use and the default attenuation (15 dB) | **works** |
| `GET /api/devices` | list microphones and speakers, and which one the active profile resolves to | **works** (since step 02) |
| `GET /api/session` | the current session status (idle if none) | **works** (step 03) |
| `POST /api/session/start` / `stop` | start or stop the live session; body: mode, source (`mic`, or a `file` path inside the project, looped), optional `monitor` | **works** (step 03): 409 if already running, 400 for a bad mode or source, 503 if the sound library is unavailable; stop can be repeated safely |
| `PUT /api/session/target` | change what to listen to (a `TargetSpec`); checked against the class list, stored, passed to the session | **works** (step 03), no audible effect until step 05; unknown class → 422. *(Step 05: the class chips send it over the WebSocket as `set_target`; still no audible effect until step 06.)* *(Step 06: audible. `attenuation_db` comes from the new slider.)* |
| `PUT /api/session/bypass` | `{"enabled": true}`: play the unprocessed sound (same delay) instead of the output, for A/B listening | **works** (step 06); remembered between sessions |
| `PUT /api/session/method` | `{"method": "neural_tse"}` or `"mask_baseline"`: the extraction method sessions are started with (Chapter 14.8) | **works** (step 07); remembered between sessions; **409** while a session runs (the methods have different delays). The start request also accepts `method` for that one session |
| `PUT /api/session/monitor` | Listen on/off, volume (−40…+12 dB), mute | **works** (step 03) |
| `POST /api/session/record/start` / `stop` | record what the session hears; stop returns the new recording | **works** (step 03); 409 without a session, or when (not) recording |
| `GET /api/recordings` | list recordings, newest first (re-reads `data/raw` first, Chapter 9.6) | **works** (since step 02) |
| `GET /api/recordings/{id}`, `/{id}/audio` | view one recording; download its sound as `audio/wav` (the page's Play button) | **works** (step 03); 404 if missing |
| `POST /api/recordings/{id}/detect` | offline detection of a recording: probabilities over time and the active segments per class (Chapter 12.8) | **works** (step 05); 413 over 2 minutes, 503 without a trained model |
| `POST /api/recordings/{id}/process` | extract sound types from a saved recording: `{labels, attenuation_db?, method?}`. Answers at once (202) with a *run* in state `pending`; a background worker processes it, one job at a time | **works** (step 06): 404 file gone, 413 over 5 minutes, 422 no or unknown classes, 503 without a detection model | *Step 07:* `method` may be `neural_tse` or `mask_baseline` (default: the configured method if it can run, else the baseline); 503 names the model that is missing; `attenuation_db` defaults to 15. |
| `GET /api/recordings/{id}/runs` | the processing runs of a recording, newest first | **works** (step 06) |
| `GET /api/runs/{run_id}`, `/{run_id}/audio` | one run (status, timing, where the target was found); its processed sound as `audio/wav` | **works** (step 06): 404 unknown, 409 audio of an unfinished or failed run |
| `WS /ws/session` | the live channel (below) | **works**: status, live pictures, warnings, errors; accepts `set_target`. *Step 05:* also `detections` (4 per second: every class's probability + the active ones), and the status carries a `detector` block (model, device, hop, inference time, skipped windows). *Step 06:* live pictures carry `source` (`input` / `output`); the status carries `bypass` and an `extraction` block (method, its delay, whether gating works, whether the target is active now); `run` messages report every state change of a processing run; the page may send `set_bypass` | *Step 07:* accepts `set_method` (answered with `error` while a session runs); the status carries `extraction_method` (the chosen method) and, for the neural method, the device, the model's name and the time of one model call; the output waveform is no longer sent (the output has level meters). |

**Why have addresses that only say "not implemented"?** Because the complete
shape of the interface is then visible from day one. The automatic
documentation page at http://127.0.0.1:8765/docs lists every address, the
website can be written against the real shape, and each later step fills in
an address that already exists. The 501 reply says *which* step will
implement it: `{"detail": "not implemented (step-03)"}`.
*(Since step 06 no address answers 501 any more: the last placeholder, `process`, is real.)*

**`GET /api/devices` in detail.** It answers with the active profile's name, every input and
output device (index, name, host API, channel count, native rate, whether it is the system
default) and `resolved_input`: the device, channels and sample rate the profile turns into. If the
profile can't be matched, for example because the named microphone is unplugged, the answer is
still **200**, with `resolved_input: null` and an `error` sentence saying why, so a page can show
the reason instead of a generic failure. Only when the sound library itself can't be loaded does
it answer **503**.

### 15.2 The live channel: `/ws/session`

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

*Since step 03, while a session runs:*

| Message | How often | Contents |
|---|---|---|
| `status` | once a second, and on every change (start, stop, error, listen, record) | state, device, format, latency, RTF, counters, listening and recording state |
| `levels` | 20 per second | per microphone: RMS, peak, clip flag |
| `waveform` | 25 per second | 64 min/max pairs of the mixed sound |
| `spectrogram` | 25 per second | 128 mel bands in dB, with the frequency range |
| `warning` | when needed | e.g. `rtf_high`: processing is getting too slow |
| `error` | when needed | the reply to a bad client message |

The 2-second heartbeat only runs while no session is active. `set_target` from the page is now
accepted: it is checked against the class list, stored and answered with a `status`.

Each browser connection has its **own queue** of at most 64 messages. If a tab can't keep up, the
oldest `levels` / `waveform` / `spectrogram` messages for that tab are thrown away; `status`,
`warning` and `error` are never thrown away.

### 15.3 One envelope for every message

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
  `set_target` from the website. *(Step 03 added `waveform` and `warning`.)*
- **`t`** is the number of seconds since the session started (0 when idle).
- **`payload`** holds the actual contents. Its shape depends on the type.

With a single envelope, the website can sort every incoming message by its
`type`, and a new kind of message never breaks the old ones. The website
simply ignores types it doesn't know yet, with a warning in the browser's
console.

### 15.4 Who may talk to it

- The backend listens on **127.0.0.1**, so only programs on this same laptop
  can reach it. It is a desktop app, not a public website.
- **CORS** allows exactly one other web address, the development website at
  `http://localhost:5173`. Tests check that a different address is refused.
- There are no passwords and no rate limiting yet (see Chapter 25).

---

## 16. The database

**Exists today** (the tables are created, nothing writes to them yet).
Location: `backend/src/soundspy/api/db/`. File: `data/soundspy.db`.

*Corrected after step 02:* the `recordings` table is now filled. It is an
**index** of the sidecar files in `data/raw/`, rebuilt from them on startup and on
every `GET /api/recordings` (Chapter 9.6). `processing_runs` stays empty until
step 06.

*Rewritten in step 06: the database is a disposable index.* Everything the database knows is
also written in a small JSON file next to the audio it describes: a recording's sidecar in
`data/raw/`, a processing run's sidecar in `outputs/processed/`. The rule, now in `CLAUDE.md`:
**never store information only in the database.** Consequences:

- The database can be deleted or rebuilt at any time without losing anything:
  `soundspy db rebuild` drops every table, creates them again and re-reads the sidecars.
- No migration tool is needed. The code carries a **schema version** (currently 2); the database
  stores the version it was built with. If they differ when the backend starts, it rebuilds the
  database by itself. This happened on the development laptop at the first start after step 06:
  the old database was replaced and its 12 recordings re-indexed from their sidecars.
- The tables below changed accordingly (section 16.2, corrected).

### 16.1 What it stores, and what it doesn't

The database stores **information about** audio, not the audio itself. Sound
files stay on disk as WAV files. Each one gets a matching **JSON "sidecar"**
file with its details, plus a database row pointing to its path. Databases are
good at searching and linking records, while sound files are large and are
better left as plain files.

### 16.2 The two tables

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

*Corrected in step 06.* The tables as they are now:

- `recordings` has one more column, **`missing`**: true while the recording's sidecar is still
  there but its WAV file is not. Such a recording stays in the list (greyed on the page, with
  its runs) but cannot be played or processed; if the file comes back, the mark is cleared.
  A recording whose **sidecar** is gone too is removed from the index: nothing on disk describes
  it any more, and keeping the row would be information that exists only in the database.
- `processing_runs` mirrors the run sidecars (`io/runs.py`):

| Column | Meaning |
|---|---|
| `run_id` | the sidecar's name, e.g. `run_20261115-192114-361`; the stable ID used in addresses |
| `recording_id`, `recording_path` | which recording; the ID is empty if that recording is no longer indexed, the path is always kept |
| `labels`, `attenuation_db` | what was kept and how far the rest was turned down |
| `method`, `activity_mode` | `mask_baseline`; `timeline` (how the target was located) |
| `status` | `pending`, `running`, `done` or `failed` |
| `error` | what went wrong, if it failed |
| `output_path`, `duration_s`, `processing_seconds` | the processed WAV, its length, how long processing took |
| `info` | what the method reports, e.g. the stretches where it found the target |

- A third table, **`meta`**, holds the schema version.
- Runs **outlive their recording**: they have their own sidecars, so deleting a recording no
  longer deletes them (the old rule "deleting a recording also deletes its runs" is gone).
- Runs left *pending* or *running* by a backend that was closed are marked `failed`
  ("interrupted") at the next start.

### 16.3 Raw recordings are never changed

A recording is saved at its original sample rate with **all** its channels,
and it is never edited afterwards. Converted copies (resampled, or reduced to
fewer channels) are made fresh when needed, or stored separately in
`data/processed/`. That way any recording can always be processed again later
with a better method.

### 16.4 How the database starts up

When the backend starts, it creates any missing tables. There is no
"migration" tool yet, meaning a tool that records step by step how a
database's structure changes over time. The decision log notes that one
(Alembic) should be added as soon as the table structure starts changing.

*Corrected in step 06:* the table structure did change, and no migration tool was added, because
none is needed: a database whose schema version is not the code's is dropped and rebuilt from
the sidecars at start-up (see the top of this chapter). Start-up order: check the schema version
(rebuild if needed) → mark interrupted runs → index recordings, then runs.

---

## 17. The website

**Exists today** as a shell. Location: `frontend/`.

### 17.1 What you see

- A header with the app name and a **connection light**.
- Three tabs: **Semantic**, **Spatial**, **Sources**.
- **Semantic** lists the seven sound types as buttons, using their display
  names, with the description as a tooltip. The buttons are greyed out and
  marked *"Selection available in step 03"*.
- **Spatial** and **Sources** say *"Requires a microphone array"*, because
  the backend reports them unavailable with the laptop profile.
- A dark colour theme. The final visual design comes later.

*Since step 03* the Semantic tab is the live session (Chapter 10.1): a session bar (Start/Stop,
source, live figures), Listen with volume and mute (behind a headphone confirmation), Record with
label and notes, the level meters, spectrogram and waveform, the class buttons (now marked
*"Sound selection arrives with detection (step 05)"*) and the recordings list with Play buttons.
Warnings and errors appear as small notices in the bottom-right corner that can be dismissed.

*Since step 06* (Chapter 13.5–13.6): next to Listen there are an **Attenuation** slider
(0–40 dB) and a **Bypass (A/B)** button; the session bar shows the extraction method and the
delay it adds; the spectrogram has an **Input / Output / Both** switch; every recording has a
**Process** button (active once a sound type is selected), its runs are listed under it with
their status, and a finished run has an **A/B player** (Original / Processed, one play position)
and a download link. A recording whose file is gone is shown greyed out with "file missing".

*Since step 07* (Chapter 14.7–14.8): a **Method** selector (Neural / Baseline) next to the attenuation slider, usable while stopped; a method that cannot run is greyed out with the reason as its tooltip. The slider starts at **15 dB** (the value comes from the backend). Next to the input level meters there is an **output level meter**, so the effect of the extraction is visible at a glance. The session bar shows the method by its name, where its model runs (`cuda` or `cpu`), the delay it adds and, for Neural, the **model time** per call. The recordings list has a **Process with** choice, and every run shows the method it was made with.

### 17.2 The connection light

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

### 17.3 How the code is organised

| Folder / file | Job |
|---|---|
| `src/api/types.ts` | TypeScript descriptions of what the backend sends, mirroring the backend's own definitions |
| `src/api/client.ts` | a small helper for calling `/api/...` addresses |
| `src/api/ws.ts` | the self-reconnecting WebSocket, and the code that checks every incoming message |
| `src/store/appStore.ts` | the shared state: backend up or down, WebSocket state, open tab, configuration, last status |
| `src/hooks/useSessionSocket.ts` | opens the WebSocket when the app starts and puts what arrives into the store |
| `src/components/ConnectionIndicator.tsx` | the connection light |
| `src/components/viz/` | empty for now; live waveform, spectrogram and source-map components go here. *Since step 03:* `LevelMeters`, `Waveform`, `Spectrogram` (canvas drawing), `liveFrames.ts` (the buffer the live messages go into, outside React), `useCanvasLoop.ts` (the shared animation loop) and `scales.ts` (dB, mel and colour scales) |
| `src/components/session/` | *(step 03)* `SessionBar`, `MonitorControls` (+ `headphones.ts`), `RecordControls`, `RecordingsList`; *(step 06)* `ExtractionControls` (attenuation, Bypass; *step 07:* the Method selector), `ABPlayer` |
| `src/components/Toasts.tsx` | *(step 03)* dismissible warnings and errors |
| `src/tabs/` | one component per tab |
| `src/App.tsx` | puts the header, tabs and panels together |

The backend's descriptions of its messages and the website's are currently
two hand-written copies that must agree. The plan is to generate the
website's copy automatically from the backend's published description
(`npm run gen:api` is a placeholder for that).

### 17.4 The development proxy

During development, the website is served at `localhost:5173` and the backend
runs at `127.0.0.1:8765`. The website's server **forwards** every `/api/...`
and `/ws/...` request to the backend. To the browser everything then comes
from one address, which keeps things simple and avoids CORS problems. The
forwarding goes to `127.0.0.1` rather than `localhost`, on purpose
(Chapter 23).

---

## 18. The command line tool

**Exists today.** The backend also installs a command called `soundspy`, used
for developer jobs that don't need the website: running the server, recording
test clips, preparing data and training models.

| Command | What it will do | Status |
|---|---|---|
| `soundspy serve` | start the backend (address and port from settings; `--reload` restarts it when code changes) | **works** |
| `soundspy devices` | list microphones and speakers, marking the system default (`*`) and the one the profile uses (`>>`); `--json` for machine-readable output | **works** (step 02) |
| `soundspy check-mic` | record a few seconds and report the levels, channel comparison and warnings (Chapter 9.8); `--meter` for a live level meter | **works** (step 02) |
| `soundspy record` | record a clip into `data/raw` as WAV + sidecar (`--seconds`, `--label`, `--notes`, `--out`) | **works** (step 02) |
| `soundspy record-dataset` | guided recording of labelled clips into `data/custom` (Chapter 11.12) | **works** (step 04) |
| `soundspy data` | `paths`, `download`, `cleanup-archives`, `prepare`, `stats`, `mix`, `preview` (Chapter 11); *step 07:* `cache --task extraction [--split all] [--sample-rate 44100] [--workers N] [--force]` renders the mixtures once to D: (Chapter 14.4) | **works** (step 04) |
| `soundspy train` | `train detection [--name] [--epochs] [--smoke] [--resume] [--workers]` (Chapter 12.4); *step 07:* `train extraction [--name tse_waveformer_v1] [--epochs N] [--smoke] [--resume] [--batch-size N]` fine-tunes the neural extractor (Chapter 14.6) | **works** (step 05) |
| `soundspy evaluate` | `evaluate detection [--experiment] [--split test]`: fine-tuned vs zero-shot (Chapter 12.5); `evaluate extraction --method mask_baseline [--split test] [--n 500] [--sample-rate 48000]`: the five-system comparison (Chapter 13.7) | **works** (step 05; extraction: step 06) | *Step 07:* `evaluate extraction --method neural_tse` adds the neural extractor (whole-file, streaming, and at the live attenuation) to the same table and repeats it in simulated rooms (Chapter 14.9). |
| `soundspy detect` | per-class timeline of a recording (Chapter 12.9) | **works** (step 05) |
| `soundspy extract` | `extract <wav> --labels speech [music …] [--attenuation-db 15] [--method neural_tse|mask_baseline] [--out out.wav]`: keep the named sound types of a recording, as a processing run in `outputs/processed` (Chapter 13.6) | **works** (step 06) |
| `soundspy db` | `db rebuild`: drop the database and index the sidecar files again (Chapter 16) | **works** (step 06) |

Placeholder commands print `not implemented yet (step-XX)` and exit normally.
*(Since step 06 there are no placeholder commands left.)*

All audio commands accept `--profile <name>` and `--device <index or name>`. A
problem with the setup (no such device, invalid profile, a stalled microphone)
prints one line starting with `error:` and exits with code 1, rather than a long
Python error.

---

## 19. Following a request from start to finish

This chapter follows three everyday actions through every part of the program: opening the page
(18.1), selecting a sound type while listening (18.2, since step 06) and processing a saved
recording (18.3, since step 06).

### 19.1 Opening the page

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

### 19.2 You select "Speech" while listening (since step 06)

A session is running with Listen on. You click the **Speech** button.

**Step 1: the page builds the target.** `ClassChips` adds `speech` to the selection (kept in the
order of the class list, not the click order) and builds a `TargetSpec`: `labels: ["speech"]`,
`attenuation_db: 30` (from the slider). It sends it over the WebSocket as a `set_target`
message. If the WebSocket is down, it uses `PUT /api/session/target` instead.

**Step 2: the backend checks and stores it.** `api/ws.py` checks every label against the class
list (an unknown one is answered with an `error` message). The `SessionManager` remembers the
target (so the next session starts with it) and hands it to the running `Session`, which only
*notes* it. The reply is a fresh `status` message.

**Step 3: the change waits for a chunk boundary.** The session's worker thread processes the
sound in 20 ms pieces. Before the next piece, it applies the noted target to every stage of the
chain. The extractor sees new labels, forgets the background it had learnt for the old
selection, and starts to fade from "pass everything" to "gate" over 10 ms.

**Step 4: every 20 ms piece now takes this path** (all on the worker thread, about 1.6 ms):

1. The **input analyser** looks at the raw piece (both microphones) and produces level, waveform
   and spectrogram events marked `source: "input"`.
2. The **detector** writes the piece into its one-second memory. Every 0.25 s it hands that
   second to its own **inference thread**, which runs the network on the graphics card (~28 ms).
   The answer is picked up by a later piece: it becomes a `detections` message and is written
   into the shared **`DetectionState`**.
3. The **channel reducer** mixes the two microphones into one.
4. The **extractor** cuts that into overlapping frames (a new one every 5.3 ms, so 3–4 per
   piece). For each frame it asks the `DetectionState` whether speech is active, moves the gate,
   updates its picture of the background, computes one gain per frequency, applies them and turns
   the frames back into 960 samples of sound, 21 ms late.
5. The chain also keeps the reducer's output, delayed by the same 21 ms: the **dry** signal for
   Bypass.
6. The **output analyser** looks at the processed piece and produces events marked
   `source: "output"`.

**Step 5: the sound goes to the headphones.** If Listen is on, the session hands the processed
piece (or the dry one, if Bypass is on) to the `SpeakerSink`. It applies the volume, copies the one
channel to both ears, passes everything through the safety limiter and puts it in a small queue.
The sound card's own thread takes it from there.

**Step 6: the page follows.** All events travel from the worker thread to the web server's event
loop and out over the WebSocket: the spectrogram columns go into two separate histories (input,
output), the detection frame makes the Speech button green, and once a second a `status`
message updates the session bar.

**In time:** you start talking at 0 s. The button turns green after about 0.3–0.5 s; the gate is
open 10 ms later; you hear your voice about 0.12 s after each sound reaches the microphone. When
you stop, the button goes back after 0.6–1 s, the gate holds for 0.3 s and fades over 0.2 s.

### 19.3 You press "Process" on a recording (since step 06)

Speech is selected and the attenuation is 30 dB. You press **Process** on a 62-second recording.

**Step 1: the request.** The page sends `POST /api/recordings/12/process` with
`{"labels": ["speech"], "attenuation_db": 30}`.

**Step 2: the checks.** The route refuses with a clear message if the recording does not exist or
its file is gone (404), if it is longer than 5 minutes (413), if there is no detection model
(503) or if a label is unknown (422).

**Step 3: the run is created, as files first.** `extraction/offline.start_run` writes
`outputs/processed/run_20261115-192114-361.json` with status `pending`, the recording, the
selection and every setting of the method. The database index is updated from that file, the job
is put in the queue, and the backend answers at once (202) with the run. A `run` message tells
every open page about it, and the row *waiting…* appears.

**Step 4: the worker picks it up.** One background thread (`soundspy-runs`) handles one job at a
time. It rewrites the file with status `running`, re-indexes it and sends another `run` message:
*processing…*.

**Step 5: detection over the whole file.** The recording is read (both channels). The detector
judges every quarter-second using the second **centred** on it, so each decision uses sound
from just before and just after. The answers are smoothed forwards and backwards (so the
smoothing does not shift them in time), and the live thresholds decide where speech is
active. The result is a timeline: speech at 2.4–7.1 s, 9.4–10.6 s and so on.

**Step 6: the same extractor as live.** The two channels are mixed into one, and the recording is
fed through the **same** `MaskBaselineExtractor` the live session uses, in blocks of 4,096
samples. The only difference is where it gets its answer to "is speech active?": the timeline
instead of the live detector. The last 21 ms are flushed out and the first 21 ms dropped, so
the result is exactly as long as the recording and lines up with it sample for sample.

**Step 7: the result is saved.** The processed sound is written to
`run_….wav`, then the description file is rewritten with status `done`, the time taken (6.5 s)
and where speech was found. The worker re-indexes it and sends the last `run` message.

**Step 8: the player appears.** The row turns *done* and shows the A/B player. Its *Original*
side plays `GET /api/recordings/12/audio`, its *Processed* side `GET /api/runs/run_…/audio`.
Because the two files line up exactly, switching keeps you at the same moment.

**If the backend is closed in the middle**, the job's file still says `running`; at the next
start it is marked *failed (interrupted)*. **If the database is deleted**, nothing is lost: the
next start rebuilds it from the description files, runs included.

---

## 20. What comes next — the plan, step by step

Everything in this chapter is **planned**, not built. Concepts are explained
here so the later chapters will make sense when they arrive.

| Step | What gets built | Tab |
|---|---|---|
| 02 | microphone capture, device listing, recording to disk, level measurement, `ChannelReducer` — **done** (Chapter 9) | – |
| 03 | the live engine (threads, queues, playback), live level meters and spectrogram on the website, a record button — **done** (Chapter 10) | 1 |
| 04 | datasets: AudioSet mapping, downloading ESC-50 and FSD50K, own recordings, splits, synthetic mixtures — **done** (Chapter 11) | – |
| 05 | sound detection: fine-tuning a pretrained model, live detections on the website — **done** (Chapter 12) | 1 |
| 06 | first extractor (classical masking) + quality scores; clean up saved recordings and compare before/after — **done** (Chapter 13) | 1 |
| 07 | neural extractor that runs live — **done** (Chapter 14) | 1 |
| 08 | full evaluation, robustness, documentation for the university report | 1 |
| modules 3–4 | simulated array, direction finding, beamforming, Spatial tab | 2 |
| module 5 | source tracking, Sources tab | 3 |

### 20.1 Recognising sounds (step 05)

A **pretrained** audio model already knows a lot about sound in general. It
will be **fine-tuned** to answer seven yes/no questions at once, one per
class. Its input is a log-mel spectrogram of a short window of sound, and its
output is seven probabilities. Run continuously on the live stream, it
produces `Detection` events: *"speech, 92%, from 12.0 s to 13.0 s"*. Which
pretrained model to use has not been decided yet.

### 20.2 Keeping the chosen sounds — two ways (steps 06 and 07)

Both methods sit behind the same `Extractor` interface, so either can be
plugged into the engine.

**The classical baseline (step 06): detector-guided masking.** Convert the
sound to a spectrogram. Use the detector to estimate which time-and-frequency
cells belong to the chosen sounds, and build a mask: close to 1 for cells to
keep, small for the rest. The value is never below the attenuation floor, so
nothing drops to unnatural total silence. Multiply, then convert back to
sound. It is transparent and easy to reason about, and it gives a reference
score that the neural method has to beat.

*(Built in step 06: Chapter 13 describes how it really works. In short: the detector only says
*when* the chosen sounds are present, not *which cells* are theirs, so the method is a gate plus
noise reduction. The reference scores the neural method has to beat, on 500 test mixtures:
SI-SDRi +2.2 dB offline and −4.9 dB live; with perfect knowledge of the timing +8.1 dB; the
ceiling for any masking method +24.5 dB.)*

**The neural method (step 07): label-conditioned target sound extraction.**
A neural network receives two things: the mixed sound, and a multi-hot vector
saying which classes to keep. It outputs only those classes. The design
follows **SemanticHearing**'s *Waveformer* model, which is built to work in
small chunks with a memory of what came before, so it can run live with low
latency.

*(Built in step 07: Chapter 14 describes how it really works. The published Waveformer was
fine-tuned for the seven classes; it scores +3.3 dB SI-SDRi on the 500 test mixtures, the same
live and offline, against −4.9 dB live and +2.2 dB offline for the baseline.)*

### 20.3 Training data from mixtures (step 04)

*(Built in step 04: Chapter 11 describes how it really works.)*

There is no recording of "cafeteria speech, with the exact speech-only
version alongside". So training examples are **made**: take a clean clip of
a chosen class (the *foreground*), add clips of other sounds (the
*background*) at a controlled loudness, and keep the clean foreground as the
known answer. Datasets with different label vocabularies can all contribute,
because every clip's label is translated through the AudioSet ontology into
the project's seven classes. The data is split carefully, so that no original
recording ends up in both training and test, which would be leakage.

### 20.4 Measuring quality

- **Detection:** how well the probabilities rank "present" above "absent" for
  each class (mean average precision), and how often the yes/no decisions are
  right (F1 score).
- **Extraction:** **SI-SDR improvement**, measured on synthetic mixtures where
  the clean answer is known. A positive improvement means the output is
  closer to the clean target than the raw mixture was.

### 20.5 Space (modules 3–5)

With the 4-microphone array, the delays between microphones reveal where each
sound comes from (direction of arrival). **Beamforming** then focuses
listening in a chosen direction before the extractor runs. In Tab 3, a
**tracker** combines "what" (detections) with "where" (directions) into
persistent sources the user can tap on a map. Until the hardware arrives,
**simulated** array recordings (made with a room-acoustics library called
pyroomacoustics) will let this work start without it.

### 20.6 A note on reused code

If any code from SemanticHearing (MIT licence) is adapted in future, the
adapted file keeps its original copyright notice and the reuse gets recorded
in a third-party notices file. So far none has been adapted.

*Updated in step 02:* there is no separate notices file any more. Adapted code is
recorded in Chapter 26, "Third-party code".

---

## 21. Running everything

All commands are for **PowerShell** on Windows 11.

### 21.1 What you need

- **Python 3.11.** On this machine, plain `python` is version 3.14, which is
  too new for the project, so the backend is always created with the `py
  -3.11` launcher.
- **Node.js** (version 22 or newer; this machine has 24) for the website.
- Ports **8765** and **5173** free.

### 21.2 First-time setup

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

### 21.3 Starting it

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

### 21.4 What you should see

- At http://127.0.0.1:8765/docs, an automatically generated page listing
  every backend address. You can try them out from there.
- At http://localhost:5173, the three tabs, a **green "connected"** light
  showing the version, and the seven sound types on the Semantic tab.
- Stop the backend and the light turns red within about 5 seconds. Start it
  again and it turns green by itself.
- *(Step 03)* Press **Start** on the Semantic tab: the meters, spectrogram and waveform move
  with the room sound, and the bar shows the latency (about 94 ms here) and RTF (about 0.06).
  Tick **Listen** only with headphones on.

### 21.5 Using the microphone

```powershell
# in backend/, with the venv active
soundspy devices                              # which microphone the profile uses
soundspy check-mic                            # 5 s test + report (Chapter 9.8)
soundspy check-mic --meter                    # live level meter, Ctrl+C to stop
soundspy record --seconds 10 --label speech   # saved in data/raw/
```

Before recording anything worth keeping, switch off Windows audio enhancements /
Voice Clarity for the microphone (Chapter 9.10) and check that `check-mic` no
longer warns about noise suppression.

---

## 22. Testing everything

### 22.1 Running the checks

```powershell
# backend (in backend/, with the venv active)
pytest              # 540 tests (after step 07; 482 after step 06), about 105 seconds
ruff check .        # code-quality check

# website (in frontend/)
npm run test        # 46 tests (after step 07; 36 after step 06); run it from PowerShell,
                    # not Git Bash (Chapter 23.26)
npm run lint        # code-quality check
npm run typecheck   # TypeScript type check
npm run build       # proves the production build works
```

### 22.2 Tests never need hardware

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

*Since step 03* it also has **speakers that listen**: a fake output stream pulls sound through the
callback like a real sound card and keeps everything it "played", so tests can check what would
have reached the headphones. For exact tests it can also be advanced one callback at a time
instead of running on its own. A test fixture fails any test that leaves a session or audio
thread running.

Run one area with, for example, `pytest tests/io`. *(Step 03 removed the empty `__init__.py`
files from the test folders: with them, starting pytest inside `tests/io` made that folder pose
as Python's own `io` module, Chapter 23.13.)*

### 22.3 What the backend tests cover

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
| chunker, chain builder (step 03) | blocks of any size become exact chunks with nothing lost; the Semantic chain is reducer → extractor stand-in, analyzer on the side; Spatial/Sources refused; a named stage can be swapped |
| analyzer (step 03) | 20 / 25 / 25 events per second whatever the chunk size (37 frames to 1 s); a 1 kHz tone peaks in the right mel band; silence is the floor; levels equal `dsp/levels.py`; waveform min/max exact; values rounded to 0.1 dB |
| speaker output (step 03) | prebuffer before playing; an underrun plays silence and is counted once; oldest chunks dropped when full; mono copied to both ears; gain and mute; **a +7 dBFS input with +12 dB gain never exceeds the −1 dBFS ceiling**; the limiter recovers; more channels than the output is refused |
| session (step 03) | starts and stops cleanly with no thread left; emits every event type; a target change is applied by the worker between chunks; the recording is exactly the raw input, a contiguous stretch of the source; stopping finishes a recording; a crashing stage gives `state="error"`; slow processing raises a warning; monitoring plays the chain's output |
| session API and WebSocket (step 03) | start / 409 / stop twice / restart; microphone session; bad mode, missing file and paths outside the project → 400; no sound library → 503; unknown class → 422; gain limits; record through the API, then the audio route serves the WAV; the WebSocket delivers status, levels, waveform and spectrogram during a session; a slow client loses only picture messages; shutting the app down stops the session |
| settings (step 03) | every new section loads; each kind of invalid value, a missing key, an unknown key and a missing section give a clear error |
| recordings (step 02) | file name and every sidecar field; no overwrite; the API indexes new files, lists newest first and forgets deleted ones |
| levels (step 02) | a full-scale sine reads −3.01 dBFS RMS and 0 dBFS peak; clipping, digital silence and noise floor on known signals; a 7-sample delay is found with the right sign; identical channels are recognised |
| channel rule (step 02) | a scan of the source code fails if channels are indexed (`[:, …]`) or averaged anywhere except the four allowed files; proven by planting a violation and watching it fail |
| audio commands (step 02) | `devices --json` and the table markers; `check-mic` report, meter and warnings; `record` writes WAV + sidecar |

| training data (step 04) | built on a tiny fake ontology (with a two-parent node), fake ESC-50 and FSD50K trees with smeared labels, and generated WAVs: graph walks never loop; FSD50K names map through mids; unknown class names come with candidates; every role case (pure, smeared, mixed, foreground + background, overlap); official splits, sessions, no group in two splits; silent head and tail found; a second `prepare` reads no audio; recipes are repeatable and keep every rule; uniform class drawing; ~configured rate of empty targets; mixture = target + rest exactly; target vs. rest within 0.1 dB; one shared peak factor; empty target is silence; resampling keeps durations exact; previews written |
| downloads (step 04) | a local web server that drops the connection mid-file: the download resumes with a Range request; a wrong checksum is an error and leaves nothing behind; not enough space stops before any request; `.done` makes a second run do nothing; split zips without 7-Zip stop with the install hint; counts are compared with the CSVs; unsafe zip entries are refused |
| record-dataset (step 04) | with the fake microphone: keys (record, redo, discard, skip, quit), sidecars with `class` and `session_id`, numbering continues across runs |
| no PyTorch (step 04) | importing the whole data pipeline loads no `torch` |
| data fixes (step 05) | explicit ESC-50 decisions (`null` rejected); greedy validation rebalance moves whole uploaders, only train → val, deterministic, logged, never touches test or ESC-50 |
| detection model (step 05) | front-end equals the reference implementation on a fixed signal (stored fixture); head rows = mean AudioSet rows; zero-shot = max over F(c) rows; short windows padded to the context; checkpoint round trip |
| window labels, datasets (step 05) | `min_active_s` edge cases, background-only windows; the same (recipe, epoch) gives the same window, another epoch another; **two spawned DataLoader workers** deliver exactly the batches of direct loading |
| training (step 05) | with a tiny stub backbone on the fake dataset: one epoch, then `--resume` continues at epoch 2; refuses to overwrite a run; CUDA required unless CPU is allowed; evaluation writes fine-tuned + zero-shot metrics and plots |
| metrics (step 05) | AP, ROC-AUC, precision/recall/F1 against hand-computed values; threshold tuning finds the best F1; false-positive co-occurrence table |
| live detector (step 05) | hop times identical for 160/960/4000-sample chunks; windows are exactly the last second; a busy model skips windows while the audio path never waits; stereo mixed down; EMA + hysteresis; missing checkpoint → session starts with a warning |
| detection API (step 05) | `detections` messages during a file session (stub model), the status's detector block, the no-model warning, `/detect` (200, 404, 413, 503) |
| loader watchdog (step 05) | a timed-out or dead DataLoader worker becomes a clear error with advice, other errors pass through |
| Waveformer port (step 07) | the re-implemented positional encoding equals a stored output of the original module; a tiny model with stored weights reproduces the **original code's** stored output; step-by-step streaming equals the whole signal; changing the input later never changes earlier output (causal); the parameter count matches the published checkpoint |
| label layer, checkpoints (step 07) | each new class column is the mean of its mapped published columns (or of all 41); two learning-rate groups cover every parameter; a published-style file with NumPy numbers loads in the safe mode; junk files and missing files give clear errors |
| streaming resampler (step 07) | any chunk sizes give the one-shot result (48 ↔ 44.1 ↔ 16 kHz, two channels); equals SciPy's result shifted by its delay; reset; 48 → 44.1 → 48 kHz returns a tone with a whole-sample delay of 22 |
| neural extractor (step 07) | with a tiny random model on the CPU: streaming in any chunk sizes equals the whole file, with and without resampling; the reported delay is exact; nothing selected = the delayed input bit for bit and the model is not run; the residual-mixing formula at 0 / 15 / 30 dB; changing only the slider keeps the model's state; keep vs reset on a selection change; fades on select and deselect; compiled (traced) steps give the same output and a wrong trace is not used; missing or broken model files are reported |
| cache, rooms, loss, training (step 07) | a reverberant render is still target + rest; rooms are the same for the same recipe; shards are written, memory-mapped and equal a fresh render; a second build renders nothing, a deleted shard is rebuilt alone, other settings are refused, a full disk is noticed first; crops are repeatable and contain the target; both loss terms, the −40 dB floor and its gradients; one training round, `--resume`, the experiment's files, the rotating sampler, the smoke command |
| method selection (step 07) | neural without a model falls back to the baseline with a warning (also for a broken file); `/api/config` lists the methods; `PUT /api/session/method` and `set_method` (409 / error while running); a whole neural session through the API reports device and model time; `/process` and `soundspy extract` with either method, 503 when the asked-for model is missing; the start hook runs once on the audio thread before the first chunk; the detector moves to the processor when the extractor is on the graphics card |
| neural evaluation (step 07) | all eight systems on the fake dataset; streaming equals offline; the reverberant copy is repeatable; listening examples are aligned |
### 22.4 What the website tests cover

*Step 03 added:* live messages go to the store (status) or to `liveFrames` (pictures), warnings
and errors become notices without duplicates; history is trimmed to 5 s / 10 s and cleared when
a new session starts; peak hold lasts about a second; **Listen asks for headphones the first time,
can't continue until the box is ticked, and remembers the answer**; the session bar shows the live
figures and starts a microphone or a looped-file session; a failed start shows a notice; the
recordings list builds the right audio address; the three canvases mount and draw.

*Step 07 added:* the Method selector sends the choice, is disabled during a session, greys out a
method that cannot run (with the reason) and shows a refused change as a notice; the session bar
shows *Neural · cuda · +10.6 ms* and the model time; the attenuation slider starts at the
backend's default (15 dB) until it is moved; the output level meter keeps its own level, peak
hold and clip memory; an output waveform that still arrives is ignored; Process sends the chosen
method, follows the session's method by default, skips an unavailable one, and every run shows
its method.

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

### 22.5 A manual check worth repeating

With both programs running, stop the backend, wait for the red light, then
start it again and watch it turn green without reloading the page. This was
done for step 01 at the network level: the health check, the configuration,
and the WebSocket heartbeat and error replies were all confirmed through the
website's proxy, with the backend up, stopped, and restarted.

**After any change to audio capture**, run `soundspy devices`, `soundspy check-mic`
and a short `soundspy record` on the real laptop. Then start `soundspy serve` and
confirm the recording appears at `GET /api/recordings`. Tests can only prove the
code behaves correctly with the *fake* sound card. The two biggest surprises of
step 02, start-up silence and noise suppression (Chapter 23.8–17.9), only showed
up on real hardware.

---

## 23. Problems found and fixed

Each of these was real. Most were silent: nothing crashed, and something was
simply wrong. They are the most instructive part of this document.

### 23.1 Port 8000 was secretly taken

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

### 23.2 Windows paths came out scrambled

**Symptom:** a test checking where the database file goes failed: the path
came back as `C%3A%5CUsers%5C...`.

**Cause:** the database library (SQLAlchemy 2.1), when turning a database
address back into text, **percent-encodes** the colon and backslashes of a
Windows path. Those are the codes web addresses use for special characters.

**Fix:** pass the address around as the library's own `URL` object, never as
text. The database file now lands exactly where intended:
`data/soundspy.db` at the project root.

### 23.3 The newest TypeScript was too new

TypeScript 7 (a rewrite of the TypeScript compiler) was the latest release,
but the code-checking tool `typescript-eslint` only supports versions below
6.1. The project therefore uses **TypeScript 6.0**, a deliberate choice
recorded in the decision log. Upgrade when the checker catches up.

### 23.4 The installer refused a file outside its folder

The backend's package description pointed at `../README.md`, one folder up.
The Python packaging tool refuses to include files from outside the package's
own folder, so the first install failed. The reference was removed.

### 23.5 The engine's own guard caught a mistake in a test

A test set up a chain for 48,000 Hz audio, then fed it a 16,000 Hz chunk by
accident. The chain refused the chunk, as designed (Chapter 7.2). The test
was wrong, not the code, and the guard proved its worth on day one.

### 23.6 Dataset names were about to reach the user

The first version of the website showed AudioSet category names such as
"Dishes, pots, and pans" as class descriptions. Those are dataset labels, not
friendly text. Each class now has a separate display name and description,
and the backend **no longer sends** the AudioSet names to the website at all.
A test makes sure they stay out.

### 23.7 Known and harmless

- **One test warning:** the web library (Starlette 1.10) suggests a newer
  testing helper (`httpx2`) instead of the one used (`httpx`). Nothing is
  broken. It is noted for a later tidy-up.
- **"Extraneous" packages in `npm ls`:** these are optional helper packages
  that the styling and build tools use on Windows. They are harmless.
- **A leftover `.venv` folder at the project root** belongs to the project's
  old name (SoundFocus, step 00). It is no longer used and can be deleted.

*The entries below were found in step 02, mostly on real hardware.*

### 23.8 Every new recording started with half a second of fake silence

**Symptom:** the first real `check-mic` reported a noise floor of −120 dBFS,
the lowest value the app can display.

**Cause:** measured directly, the first **500 ms** of a freshly opened WASAPI
stream on the laptop are **exact zeros**: 24,000 frames at 48 kHz, every time.
After that, real sound flows. The quietest 10 % of the recording was simply that
silent start.

**Fix:** after opening a microphone, leading frames that are exactly zero are
discarded, for at most 1 second (Chapter 9.4). A real microphone never produces
exact zeros, and the 1-second cap means a truly muted microphone still reaches
the report and gets flagged.

**Lesson:** a measurement that looks impossibly good deserves suspicion, not
celebration.

### 23.9 The laptop's sound is being "cleaned" before Sound Spy sees it

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
off is a setting on the laptop, not code (Chapter 9.10). Until it's done, any
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

### 23.10 The same microphone wasn't recognised as itself

**Symptom:** in a test, the app chose the MME microphone instead of its WASAPI
twin.

**Cause:** MME cuts names to 31 characters, and on this name the cut fell just
after a space: "Microphone Array (Intel® Smart ". The matching code removed
surrounding spaces *before* cutting the other name to 31 characters, so one side
was 30 characters long and the other 31, and they never matched.

**Fix:** cut first, then remove spaces. The test fake reproduces exactly this
name, so the mistake can't come back unnoticed.

### 23.12 A warning that cried wolf: isolated zeros are normal

**Symptom:** after the enhancements were switched off, every `check-mic` still
warned that 1.4–5 % of samples were "exact digital silence", although the noise
floor now looked perfectly honest.

**Investigation, on the real recording:**
- the exact zeros came in about 3,200 tiny stretches, none longer than
  5 samples (0.1 ms). A noise gate would silence whole milliseconds;
- every sample was a whole multiple of 2.354/32768: a 16-bit converter plus
  +7.44 dB of Windows gain (Chapter 9.8);
- the value 0 occurred exactly as often as its neighbours ±1 (1.40 % vs 1.34 % /
  1.42 %). That is ordinary rounding of a quiet signal, not something forcing
  values to zero.

**Fix:** "digital silence" now counts only **stretches of at least 5 ms**. The
same recording now reports no warnings, and a test reproduces a quiet 16-bit
signal with isolated zeros to make sure it stays that way.

**Lesson:** a check written from one bad recording needs a second, good recording
before it can be trusted not to fire on healthy input.

### 23.11 Times without a time zone

SQLite stores dates as plain text without a time zone. A recording made at
16:05 in Romania and one made at 14:30 UTC would sort wrongly if each were stored
in its own local time. All recording times are therefore stored in UTC, and the
API marks them as UTC when it sends them.

*The entries below were found in step 03.*

### 23.13 A test folder pretending to be part of Python

**Symptom:** running only the storage tests (`pytest tests/io/test_storage.py`) failed with
`cannot import name 'StringIO' from 'io'`, while the full test run passed.

**Cause:** the test folder `tests/io/` contained an empty `__init__.py`, which makes a folder a
Python *package*. When pytest started inside that folder, it registered it under its plain name,
`io`, which is the name of one of Python's own core modules. From then on, anything asking for
Python's `io` got the test folder instead.

**Fix:** the empty `__init__.py` files were removed from all test folders (pytest's import mode
doesn't need them); folders with no tests yet keep a `.gitkeep` so git still tracks them. Running
`pytest tests/io` now works.

**Lesson:** a test suite has to work the way people actually run it, including one folder at a
time, not only as a whole.

### 23.14 The channel-reduction guard flagged three innocent lines

The scan that enforces "channels are only reduced in `ChannelReducer`" (Chapter 4) found
`[:, 0]` and `.sum(axis=1)` in three new files. None of them reduced audio channels: one took the
already-mono output of the `ChannelReducer`, one checked the rows of the mel filter table, one
reshaped a gain curve. Rather than weaken the guard, the three lines were rewritten
(`.ravel()`, `.max(axis=1)`, `.reshape(-1, 1)`). The guard stays strict, so a real violation
still fails the build.

### 23.15 "Listen" seemed to ignore the volume for a moment

**Symptom:** a test changed the volume and the server's reply still showed the old value.

**Cause:** by design, a change is applied at the start of the next 20 ms chunk (Chapter 10.3), and
the reply was built before that moment.

**Fix:** the status now reports the requested setting while it waits to be applied, so the page
shows what the user just chose.

### 23.16 A test that listened to silence

The first monitoring test played a file whose two channels were exact opposites (channel 1 =
−channel 0) and expected to hear something. The chain mixes the channels to one, and opposites
cancel to perfect silence. The code was right; the test file was wrong. Real stereo recordings
rarely cancel, but it's a reminder that a mono mix *can* lose sound when microphones pick it up
in opposite phase. That's one reason the analyzer and the recordings always keep every channel.

### 23.17 Latency vs clicks: the prebuffer trade-off

Measured on the laptop, 30 seconds each: with a prebuffer of 2 chunks the typical latency is 94 ms
and there were **no** underruns; with 1 chunk it's 84 ms but there was **one** underrun (a click).
The default stays at 2. Chapter 10.5 explains the parts of the delay.

### 23.18 Plugging in headphones broke the next session

**Symptom (owner's test):** everything worked, including Listen with headphones. Then a file
session started without an output device ("no output device is available for monitoring" when
Listen was clicked). After a backend restart, even the microphone failed to open with
*"Unanticipated host error … WDM-KS error"*. A few minutes later, both devices opened normally
again.

**Cause:** the sound library (PortAudio) reads the device list once and keeps the handles it found.
Plugging headphones into a Realtek jack makes Windows reconfigure its audio endpoints, and opening
a stream through the old handles can then fail, briefly even for a freshly started backend while
Windows is still switching. The "WDM-KS" in the message was misleading: PortAudio attaches the
*last* host error it recorded, which came from its WDM-KS device scan, not from the WASAPI stream
that actually failed. Sound Spy never chose a WDM-KS device; `soundspy devices` resolved to WASAPI
index 9 throughout.

**Fix:**
- every session start re-reads the device list first (only when no session is running, because
  re-initialising PortAudio would cut off open streams);
- a failed open is retried once after another re-read;
- a failure that persists gets a message explaining the likely reason;
- a session that starts without an output device says why in a notice, and disables Listen with
  an explanation;
- `soundspy serve` now prints Sound Spy's own log messages, which were invisible before, so such
  problems leave a trace in the backend window;
- `GET /api/devices` also re-reads the list while idle.

Tests reproduce the stale-handle failure with the fake sound card; on the real laptop a
microphone → file → microphone sequence now gets the speakers every time.

### 23.19 The Listen box wouldn't tick while no session was running

**Symptom (owner's test):** Listen unticked during a session, session stopped, Listen ticked
again: the box stayed unticked until the next Start (which then did start listening).

**Cause:** while idle, the backend stored the choice for the next session correctly, but answered
with the status of the session that had just ended, which still said "not listening". The page
showed that answer.

**Fix:** while no session runs, the status reports the stored Listen, volume and mute settings, so
the box shows what the user chose and the next session starts with it. A test replays the exact
sequence.


### 23.20 A whole test folder was silently skipped

**Symptom:** the first run of the new data tests reported *28 skipped*, none run.

**Cause:** the test setup skips tests marked `data` (those that need downloaded datasets) unless
`--run-data` is given. It recognised the mark by looking in a list of *keywords* that pytest also
fills with **folder names**, and the new tests live in `tests/data/`.

**Fix:** it now looks only at real marks. All 57 data tests run; the skip still works for tests
that really are marked.

### 23.21 A car horn looked like "traffic plus something else"

**Symptom:** a clip labelled `Vehicle_horn, Car, Vehicle, Alarm` came out as "class mixed with
other sounds" instead of being reported as an overlap of *alarm* and *traffic*.

**Cause:** FSD50K adds every parent of a label, and the horn has **two** parent chains (under
*Car* and under *Alarm*). `Alarm` is not an ancestor of the traffic class, so the spec's rule did
not tolerate it.

**Fix:** labels that are ancestors of another label *of the same clip* are ignored before the
rule is applied (Chapter 11.6). The horn is now correctly an alarm + traffic overlap, and every
other clip gets the same role as before.

### 23.22 7-Zip was missing

FSD50K's audio comes as a split zip that only 7-Zip can unpack, and the laptop had no 7-Zip. The
downloader downloads everything first and then stops with the install command; the owner agreed
to install it (`winget install 7zip.7zip`), after which the same command continued with the
extraction without downloading anything again.

### 23.23 FSD50K's validation split shared uploaders with training

**Symptom:** the first real `prepare` stopped with *"926 leakage group(s) span several splits"*.

**Cause:** FSD50K guarantees that dev and eval have no uploader in common, but its train/val
split inside dev is only a proposal and does share uploaders: 2,379 of the 4,170 val clips come
from someone who also uploaded training clips. The leakage check did its job.

**Fix:** each uploader is moved wholly to the side that already has most of their dev clips
(Chapter 11.8). 2,010 clips change side, the test split is untouched, and `prepare` reports the
number every time.

### 23.24 The first training run hung: memory-hungry DataLoader workers

**Symptom (owner):** `soundspy train detection --smoke` started its DataLoader workers and then
nothing happened: 8 Python worker processes, no `log.csv` after 8 minutes, GPU idle. The log's
last lines: *"OpenBLAS error: Memory allocation still failed after 10 retries, giving up."*

**Diagnosis:** the same run with `--workers 0` finished normally (1.2 min), so the data pipeline
itself was fine. A measurement of one worker process explained the rest: on Windows each worker is
*spawned* as a fresh Python, which loads numpy, scipy and torch again, and each of those starts a
thread pool sized to all 12 CPU threads, with per-thread buffers: **1.67 GB of committed memory and
29 threads per worker**. Eight such workers (4 for training + 4 for validation) plus the main
process with the GPU context used up the free commit (RAM + page file); OpenBLAS inside a worker
could not allocate, the worker stalled, and the DataLoader, which had no timeout, waited forever.
Not the cause: the dataset pickles fine, and the entry point is protected (`if __name__ ==
"__main__"` in pip's launcher).

**Fix:** workers get one BLAS/OpenMP thread each (rendering is single-threaded anyway; the
parallelism comes from the processes): **0.95 GB and 7 threads per worker**, of which ~0.7 GB is
simply importing PyTorch. Validation uses half as many workers, the loader that measures class
frequencies is released before training starts, and every loader has a 120 s watchdog: no batch →
a clear error suggesting fewer workers instead of a silent hang. With 4 workers the smoke run then
finished normally (first batch after 12 s). A test runs two real spawned workers.

### 23.25 The pretrained model saw "everything" in one-second clips

**Symptom:** the first zero-shot check gave AP equal to chance for every class; every probability
was ~1.0.

**Cause:** EfficientAT was trained on 10 s clips. Given 1 s, its last feature map is only 4 × 4
cells, the convolution padding dominates, the pooled features shift and the AudioSet head
saturates. Measured zero-shot mAP: 0.10–0.12 for 1–3 s of input, 0.41 for 5–10 s.

**Fix:** the 1 s window is padded with silence to 10 s before the model (`input_context_s`),
identically in training and live (Chapter 12.2).

### 23.26 Frontend tests "failed to find the current suite" in Git Bash

**Symptom:** `npm run test` failed in every file at `tests/setup.ts` with *"Vitest failed to find
the current suite"*, even for untouched tests; it had passed in the morning.

**Cause:** the shell, not the code. Run from Git Bash, the project folder is reported as `c:\…`
(lowercase drive letter) and Vitest ends up loading itself twice under two spellings of the same
path. From PowerShell (as in the morning) all tests run normally.

**Fix:** run the frontend checks from PowerShell (noted in Chapter 22.1).

### 23.27 Every GPU session skipped its first 3 seconds of detection

**Symptom:** each live session on the GPU reported exactly 12 skipped windows, though one window
takes only ~28 ms and a new one is due every 250 ms.

**Diagnosis:** feeding the detector in real time and logging *when* windows were skipped showed all
12 in the first 3 s. The first inference took **3.1 s**, every later one 15–60 ms. The model had
been warmed up at session start, but on the session's thread. CUDA's libraries (cuDNN, cuBLAS) set
up their state separately for every thread on its first call, and the inference thread is a
different thread, so it paid ~3 s on its first real window.

**Fix:** the inference thread warms the model up itself, and the session start waits for it (at
most 60 s). Afterwards: first window 16 ms, 0 skipped. A test checks that the warm-up runs on the
`soundspy-detector` thread and is finished when the detector is configured.

### 23.28 Detection on the CPU kept seven cores busy

**Symptom:** with `live.device: cpu`, the backend used **680 % of one core** (about 7 of the 12
CPU threads), for a model that needs 44 ms every 250 ms.

**Cause:** PyTorch splits each inference over all cores, and between inferences its idle worker
threads keep spinning ("busy-waiting") so they can react fast. Measured with 12 / 4 / 2 / 1 threads:
40 / 36 / 45 / 68 ms per window, and 971 / 286 / 112 / 25 % CPU.

**Fix:** the inference thread limits itself to `live.cpu_threads` (1) PyTorch threads. The limit
applies only to that thread, so offline `detect` still uses all cores. Afterwards: 71 ms per window
(28 % of the time between windows), backend CPU 38 % of a core, 0 skipped.

### 23.29 Overlapping stretches in `soundspy detect`

**Symptom:** `detect` on the 62 s UI recording printed `speech 21.25-26.75 s` and
`26.25-32.75 s`, overlapping.

**Cause:** a stretch starts one window (1 s) before its first "on" result. After a single
"off" result in between, the next stretch therefore started before the previous one ended.

**Fix:** stretches of the same class that overlap or touch are merged: `21.25-32.75 s`. Test added.

### 23.30 The noise estimate removed a steady target

**Symptom:** in the first check of the extractor, a constant tone that was the *target* came out
7 dB *worse* than it went in, although the gate was open.

**Cause:** minimum statistics estimates the background as "whatever each frequency never drops
below for 1.5 s". A tone that lasts longer than that never drops, so it is taken for background
and removed. The same would happen to a held note, a continuous alarm or steady road noise,
exactly the things a user may select. The first version used the larger of the two noise
estimates (detector-gated average and minimum statistics), so the fallback was always active.

**Fix:** the detector-gated average ("what it sounded like while the target was absent") is used
whenever it is fresh; minimum statistics only when no target-free moment was seen for 5 s, or
when there is no detector. With the gated estimate the same tone in noise went from −1.8 dB to
+22.7 dB SI-SDR.

### 23.31 An occasional click: Python's memory clean-up stopped the audio thread

**Symptom:** with extraction on, about one output underrun every few minutes (the step requires
none in 60 s), although processing a 20 ms chunk takes under 2 ms.

**Diagnosis:** a log of garbage collections in the running backend showed full collections of
**45–87 ms**. Python stops *every* thread during a collection, and a full one scans all
long-lived objects: about 140,000 here (PyTorch, the model, the web framework). The output queue
cushions about 40 ms, so one collection during a session is one underrun. Step 06 doubled the
number of live messages, and with it the garbage, so it showed up now.

**Fix:** when a session starts, one collection is run on purpose and everything alive is then
*frozen* (`gc.freeze()`): excluded from later scans. A full collection during a session then
takes about a millisecond. When the session ends, the freeze is lifted. Afterwards: no long
collection inside any session, and 0 underruns in 8 consecutive sessions (2 × 60 s microphone,
6 × 24 s file). A test checks that a collection during a session takes under 20 ms.

### 23.32 File playback arrived in a 30 ms / 10 ms rhythm

**Symptom:** sessions that play a *file* (not the microphone) still had a rare underrun; the log
showed chunks arriving alternately 30 ms and 10 ms apart instead of every 20 ms.

**Cause:** the file source paced itself with `time.monotonic()`, which on Windows with
Python 3.11 only advances every ~15.6 ms.

**Fix:** it uses the high-resolution clock (`time.perf_counter()`). Before, 306 of about 7,800
file-session chunks arrived more than 30 ms after the previous one; afterwards none.

### 23.33 Two threads indexing at once, and a file read while it is replaced

**Symptom:** one test of the processing runs failed about once in three runs of the whole
suite: asking for a run's status sometimes answered "no such run".

**Cause:** two things that only happen when the background worker and a web request meet.
(1) Both re-read the run sidecars to update the database; both found the same new sidecar and
both tried to insert it. (2) On Windows a file cannot be opened in the instant it is being
replaced. The worker rewrote a run's sidecar exactly while the request read it; the unreadable
file was treated as "gone" and the run's entry was deleted from the index.

**Fix:** indexing is done by one thread at a time (a lock); sidecar reads and rewrites are
serialised inside the process and retried briefly if another process holds the file; and an
index entry is only removed when its sidecar really no longer exists. Three tests cover it
(six threads indexing at once; one thread rewriting a sidecar 300 times while three read it).


### 23.34 The published network file was refused as unsafe

**Symptom:** loading the downloaded Waveformer weights failed with *"Unsupported global:
numpy.core.multiarray.scalar"*.

**Cause:** a `.pt` file can contain arbitrary Python objects, and loading one can run code. Sound
Spy therefore loads every weight file in PyTorch's *weights only* mode, which accepts numbers and
arrays and refuses everything else. The published file also stores a few bookkeeping numbers
(its best validation score, for example) as NumPy objects, which that mode does not know.

**Fix:** the safe mode is kept, and exactly the harmless NumPy number types are added to its
allow-list for this one load. The file is also checked against a fixed SHA-256 checksum right
after the download, so a changed file is noticed.

### 23.35 The cache refused its own files on the second run

**Symptom:** running `soundspy data cache` a second time, which should do nothing, stopped with
*"the cache was made with other settings"*.

**Cause:** the cache stores a *signature* of the settings it was made with and compares it on
every run. In memory some settings are tuples, `(0.2, 0.8)`; written to the JSON file and read
back they are lists, `[0.2, 0.8]`. Python does not consider the two equal.

**Fix:** the signature is passed through JSON once before it is compared, so both sides have the
same form. A test builds a cache twice and checks that the second run renders nothing.

### 23.36 The simulated rooms were too dry

**Symptom:** a room asked to reverberate for 0.5 s measured only 0.29 s.

**Cause:** the simulation follows the sound's reflections from wall to wall, up to a maximum
number of bounces (the *order*). With 12 bounces the late part of the echo was simply missing.

**Fix:** 40 bounces: 0.53 s measured for 0.5 s asked, at 133 ms of computing per room, which
the one-time cache build can afford.

### 23.37 A batch that "fits" the graphics card but takes 24 seconds

**Symptom:** with 8 examples per batch the training did not fail, it crawled: 24 s per step
instead of 0.3 s.

**Cause:** the batch needed 4.5 GB and the card has 4 GB. On Windows, PyTorch does not stop with
"out of memory" in that case: the graphics driver quietly moves the overflow into ordinary
(*shared*) memory, which is perhaps a hundred times slower. Step 05 met the same effect.

**Fix:** before training, the code tries growing batch sizes with one real training step each
and **measures the peak memory**; the largest one that stays under 72 % of the card is used
(4 examples), and gradients are added up over 4 batches so that each update still sees 16.

### 23.38 A helper script that started itself again and again

**Symptom:** a short trial training, started from a small script, hung without output.

**Cause:** on Windows a new worker process starts by *importing the script that launched it*. The
script had its commands at the top level instead of under `if __name__ == "__main__":`, so every
data-loading worker ran the whole script again (including the line that deleted the experiment
folder).

**Fix:** the guard was added. The `soundspy` commands were never affected: they start from a
proper entry point. The lesson is noted in `CLAUDE.md`.

### 23.39 The first full training stopped early, with a hidden tail of bad results

**Symptom:** the first 32-round training stopped by itself after round 9 at +2.6 dB on the
validation set, barely better than the simple baseline, although the loss was still falling.

**Cause:** two things together. (1) The loss was the one the published network had been trained
with, 90 % SNR and 10 % SI-SNR. Looking at the 500 validation examples one by one showed a
healthy typical case (the median, and +12 to +14 dB for the best tenth) but about **one example
in eight worse than doing nothing**, some by 20–40 dB: the network had extracted the *wrong
sound* with full confidence. SNR barely punishes that, SI-SNR punishes it severely, and the
reported score is an average of SI-SNR values, so those few examples ate the gain of all the
others. (2) Early stopping watched exactly that jumpy average with a patience of 6 rounds,
during the phase where the learning rate is still high.

**Fix:** a 20-minute side-by-side test from the same starting weights (6,000 examples each)
settled it: the old loss stayed at +2.0…+2.1 dB, a loss of 20 % SNR and 80 % SI-SNR reached
+2.95…+2.99 dB. The second run uses that loss, a patience of 10, and measures validation the
way the final evaluation does (on the output with the rest 30 dB down). It reached +3.72 dB
(26 rounds, best at round 16). The first run's folder was deleted on 2026-11-17 (owner's
decision); its numbers are in the table of Chapter 14.6. Chapter 14.5–14.6 tell the story.

### 23.40 Two trainings on the 4 GB card at once

**Symptom:** the second run's first steps took 1.2–4.3 s instead of 0.3 s.

**Cause:** it was started a few seconds before the side-by-side test had finished its last
validation. Together the two needed more than 4 GB, and both fell into the slow shared memory
of 23.37, each slowing the other down.

**Fix:** both were stopped and the training started again alone, one minute later. Rule for the
laptop: **one job on the graphics card at a time**, and that includes a running live session
while something trains.

### 23.41 A memory-mapped file cannot be deleted while it is open

**Symptom:** a test that deletes one cache shard, to check that the build resumes, failed with
*"the file is being used by another process"*.

**Cause:** the cache is read through memory mapping (Chapter 14.4), and Windows does not allow
deleting a file while any mapping of it is alive, even from the same program.

**Fix:** `ExtractionCache.close()` releases the mappings, and the cache reader now also checks
that every shard file is really there (a missing one is reported as "incomplete, run the cache
command again" instead of a crash later). Rebuilding with `--force` therefore requires that no
training is reading the cache.


### 23.42 Two models on one graphics card: clicks and a growing delay

**Symptom:** the first live minute with the neural extractor had 5 underruns, RTF spikes up to
1.8, and the delay to the ear crept from 105 ms to 225 ms and stayed there.

**Cause:** the extractor (a call every 20 ms) and the detector (a 25 ms recognition four times a
second) both ran on the graphics card, which does one thing at a time. An extractor call that
arrived during a recognition had to wait for it and missed its 20 ms. After such a stall the
chunks arrive in a burst, the speaker queue fills up, and the app has no mechanism to shrink it
again, so the delay stays high.

**Fix:** when the neural extractor uses the graphics card, the detector runs on the processor
(one thread, about 80 ms per recognition, no skipped windows). Measured after the change: 0
underruns, RTF p95 0.63, delay 95–115 ms. The setting is `detector_on_cpu` in
`neural_tse.yaml`; an explicit `detection.live.device: cuda` is still respected.

### 23.43 A faster network call that stuttered in its first second

**Symptom:** tracing the network's step with TorchScript made the average call 30 % faster, but
in a live session single calls took 48, 60 and even 208 ms right after the start: 2 underruns
and again a delay stuck at 225 ms.

**Cause:** TorchScript watches how a traced function is used and re-optimises it on the fly
during its first real calls. Each re-optimisation takes tens to hundreds of milliseconds,
unpredictably, in the middle of the audio path.

**Fix:** the traced step is run with that run-time optimiser switched off
(`torch.jit.optimized_execution(False)`). The speed-up stays, because it comes from skipping
Python's bookkeeping between 200 small operations, not from the optimiser: 8.4 → 6.0 ms per call
on the graphics card, no slow first calls. Tracing itself happens in `start()`, before the first
chunk, and a traced step is compared with the plain one before it is used.

---

## 24. Design decisions and why

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

*Revisited in step 06, as planned.* A recording whose WAV is gone is now **marked missing**, not
deleted, and its runs stay. No migration tool was needed for the new column: the database is
rebuilt from the sidecars whenever its schema version changes (Chapter 16). One refinement: if
the recording's *sidecar* is gone as well, the row is removed, because the database must never
be the only place something is written down. The runs survive in both cases; they have their own
sidecars.

**Why a gate with noise reduction, and not a mask per frequency for the chosen sound?**
(step 06) The detector says *when* a class is present, not *which frequencies* are its. Without
training, timing is the only information available, so the baseline gates in time and removes
only the background it could learn while the target was absent. Anything smarter needs a model
that has learnt what the classes sound like: step 07.

**Why does the sound go through the STFT even when nothing is selected?** (step 06) So the
delay is always the same. If pass-through skipped the STFT, selecting a class would add 21 ms in
the middle of a sentence: a jump you would hear.

**Why is the bypass signal delayed?** (step 06) For a fair, seamless A/B. Both versions reach
the headphones equally late, so switching changes only the processing, not the moment you hear.

**Why are processing runs files with sidecars, like recordings?** (step 06) Same reasons: they
can be copied or backed up as plain files, the database can be rebuilt from them, and the
command line can create them without any database code.

**Why three "activity sources"?** (step 06) They separate two questions the evaluation must
answer: how good is the method when it knows exactly when the target sounds (the *oracle*), and
how much is lost because the detector errs (*timeline*) and lags (*live*)?

**Why 20 ms chunks?** (step 03) Short enough to keep the delay low (one chunk is one of the
delays you hear), long enough that the computer handles 50 of them a second with almost no
effort (RTF 0.06). Detection and extraction models of later steps may need longer windows; they
can collect several chunks internally without changing the engine.

**Why separate input and output streams with their own queues?** (step 03) The microphone and
the headphones may be different devices with different clocks. Two independent streams joined by
a small queue (cushion before playing, drop the oldest when too full) tolerate that drift, where
a single combined stream would need both sides on one device and one clock.

**Why is the safety limiter always on?** (step 03) The laptop's microphone already exceeds full
scale and the volume adds up to +12 dB. A sudden clap or a dropped plate could otherwise reach the
ears at full blast. Hearing safety isn't a setting.

**Why is Listen off by default, behind a headphone question?** (step 03) Through laptop speakers
the sound feeds back and howls within a second, which is unpleasant and could startle people
nearby. Listening has to be a deliberate choice.

**Why record the raw input rather than what you hear?** (step 03) What you hear will change as the
processing improves; the raw input can always be processed again by a better method. It is also
exactly what `soundspy record` saves, so recordings from the page and from the command line are
interchangeable.


**Why is the big data on D: and not in the project?** C: has about 31 GB free, and unpacking
FSD50K needs about 60 GB for a while. Both are partitions of the same fast SSD, so nothing is lost
by using D:. The project folder stays small and backed-up-able; the datasets can always be
downloaded again.

**Why recipes instead of rendered mixtures?** A recipe is a few hundred bytes, exactly repeatable,
and can be rendered at any sample rate. Rendered audio would be ~25 GB fixed at one rate, and the
model's sample rate is only chosen in step 05.

**Why tolerate ancestor labels but exclude everything else?** FSD50K labels every clip with all
parents of its true label, so parents carry no new information. Anything else on the clip is a
real second sound, and a "speech" example that also contains music would teach the extractor the
wrong answer.

**Why draw classes uniformly?** FSD50K has many times more music than footsteps. Drawing clips at
random would let the common classes dominate; drawing the class first makes every class equally
frequent in the mixtures.

**Why no resampling cache yet?** Resampling only the needed piece of each clip on the fly is
fast enough for previews and costs no disk space. A cache can be added once step 05 fixes the
sample rate and training speed is measured. *(Step 05: rendering costs ~20 ms per recipe at 32 kHz,
60 % of it resampling; with 4 workers one training epoch takes ~6–7 minutes and the GPU waits for
data. A cache of clips at 32 kHz would make training faster; not needed to finish this step.)*

**Why EfficientAT mn10_as?** Small enough (4.9 M parameters) for live inference on a 4 GB laptop
GPU and acceptable on the CPU, pretrained on all of AudioSet (so it already knows our sounds), MIT
licensed. Larger models score slightly higher on AudioSet but would not leave room for the
extractor of step 07 running at the same time.

**Why 1 s windows every 0.25 s, padded to 10 s?** One second is long enough to recognise most
sounds and short enough to follow them; a new answer four times a second feels live. The padding
is needed because the pretrained network does not work on short inputs (Chapter 23.25).

**Why tune thresholds on val, never on test?** Any choice made by looking at the test set makes
the test score optimistic. Val decides thresholds and the best epoch; test is scored once.

**Why an inference thread that skips windows when busy?** The audio path must never wait for the
model: a late model would mean clicks in the headphones. Skipping a window only means one fewer
detection, and the skipped count shows whether the model keeps up.

**Why are the PyTorch datasets in `training/` and not in `data/`?** So the whole data pipeline
(downloads, index, recipes, rendering) keeps working without PyTorch; a test checks it.

**Why CUDA 12.6 wheels?** The newest PyTorch build that pairs with torchaudio (torch 2.11.0 +
torchaudio 2.11.0; torchaudio is in maintenance mode and has no builds for newer torch) on a CUDA
12.x runtime, as the owner asked; the laptop's driver (621.24) runs it.

**Why fine-tune a published network instead of training our own?** (step 07) Waveformer already
knows how to pull a named sound out of a mixture in real time; it was trained for days on far
more material than the laptop could handle. Fine-tuning reuses all of that and only teaches it
our seven names, in hours on a 4 GB card. Its code and weights are MIT-licensed, so they may be
adapted (Chapter 26).

**Why does the network keep its own sample rate (44.1 kHz) when the microphone gives 48 kHz?**
(step 07) Everything the network learnt is tied to its sample rate: a filter that means "1 kHz"
at 44.1 kHz would mean "1.09 kHz" at 48. So the sound is converted on the way in and back on the
way out, by a resampler that keeps its state between chunks and adds only 22 samples of delay.

**Why mix the result with the original (residual mixing)?** (step 07) Three reasons. The
attenuation slider keeps one meaning for both methods. You stay aware of what is around you. And
when the network makes a mistake, the part it wrongly removed is still faintly there instead of
gone.

**Why is the live default 15 dB, when the evaluation uses 30?** (step 07) It was the owner's
decision for this step. A milder setting keeps some of the room audible and makes the
extractor's mistakes less harsh, while 30 dB exposes every one of them. The evaluation keeps
30 dB so that its numbers can be compared with step 06, and also reports the 15 dB setting.

**Why render the training mixtures once to disk?** (step 07) Rendering took longer than learning.
With the 25 GB cache on D: the graphics card is the only thing the training waits for. The
recipes stay the source of truth: the cache can be deleted and rebuilt identically at any time.

**Why add simulated rooms, and why is the target reverberant?** (step 07) Dataset clips are dry;
your room is not. Hearing the same sounds in thousands of random rooms makes the network less
surprised by a real one. The target keeps the room's echo because the task is "remove the other
sounds", not "remove the room": asking for both would make the task much harder and the result
would sound unnatural.

**Why can the method not be changed while a session runs?** (step 07) Neural adds 10.6 ms of
delay and the baseline 21.3 ms. Swapping in mid-stream would repeat or skip 10 ms of sound, and
the two would need to be warmed up and kept running side by side. Stopping, choosing and
starting takes a few seconds and is always clean.

**Why does a missing neural model fall back to the baseline instead of refusing to start?**
(step 07) Same rule as for a missing detector in step 05: a missing file must never stop you from
listening. The page says which method is really running.

**Why is the loss mostly SI-SNR with a little SNR?** (step 07) SI-SNR is what is measured and it
punishes extracting the wrong sound; SNR alone lets the network "play safe" by going quiet. The
small SNR share is kept because only it fixes how loud the output is, which residual mixing
needs (Chapter 14.5, 23.39).

**Why does the detector move to the processor when Neural runs?** (step 07) A graphics card
does one job at a time. The extractor is in the audio path and has 20 ms per chunk; the detector
only feeds the buttons and can take 80 ms without anyone noticing. So the one that cannot wait
gets the card to itself (Chapter 23.42).

**Why trace the network's step, and why without TorchScript's optimiser?** (step 07) For a small
network called fifty times a second, the time goes into organising operations, not computing
them; tracing removes much of that and changes no number. The optimiser is left off because a
live session needs every call to take the same time more than it needs the average to be low
(Chapter 23.43).

---

## 25. Known limitations

These are understood and accepted for now, not oversights.

1. **No audio is processed yet.** Step 01 built the frame. The sound path
   starts in step 02. *(After step 02: sound can be captured and recorded, but
   nothing processes or plays it yet. The live session comes in step 03.)*
   *(After step 03: a live session runs and plays the sound, but nothing is
   recognised or removed yet. That begins with steps 05–06.)*
   *(After step 05: sounds are recognised live (Chapter 12), but nothing is removed yet; the
   selected classes take effect from step 06.)*
   *(After step 06: selected sound types are kept and the rest is turned down, live and for
   saved recordings (Chapter 13), with the limits of items 22–24 below.)*
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
   *(Step 04: all names exist. Open question for the owner: "traffic" uses AudioSet's
   "Vehicle", which also contains aircraft, boats and trains; "Motor vehicle (road)" would
   match the description "road traffic" more closely. See item 15.)*
7. **Nothing on the website has been checked for appearance in a real
   browser** beyond a quick look. Automated tests can't see layout.
8. **Laptop microphones on Windows may have "audio enhancements" switched
   on** (noise suppression, automatic volume). Step 02 needs to detect or
   avoid these, for the same reason the browser is not trusted with audio.
   *(Step 02: confirmed on the development laptop. `check-mic` now detects it
   (Chapter 9.8), but switching it off is a Windows setting the user has to
   change (Chapter 9.10). The app can't do it for them.)*
9. **The laptop's built-in microphone is effectively mono** (step 02): both
   channels are identical, so it carries no direction information.
   *(Corrected after the retest: that was the Windows enhancements. With them off
   the laptop delivers two real microphones, about 4+ cm apart. That is still too
   small and fixed an array for the Spatial/Sources tabs, which need the
   ReSpeaker.)*
10. **Microphones plugged in while the backend runs aren't seen** (step 02).
    PortAudio reads the device list once, at start-up, so restart the backend
    (or re-run the command) after plugging in a USB microphone.
    *(Step 03: no longer needed. Every Start, and the devices list while idle,
    re-read the devices. A device plugged in **during** a running session is
    picked up at the next Start.)*
11. **The website doesn't show devices or recordings yet** (step 02). They exist
    in the API (`/api/devices`, `/api/recordings`). The page that uses them comes
    with the live session in step 03.
    *(Step 03: recordings are now listed and playable on the page. Choosing a
    microphone or output device on the page is still not possible; the profile
    and `output.device` in `default.yaml` decide.)*
12. **The listening delay is about 94 ms** (step 03; 84 ms measured in step 05). Noticeable when you hear
    your own voice, natural for other people's. It is mostly the sound system's own
    buffers (Chapter 10.5); a smaller cushion saves ~10 ms but caused a click in
    testing. Bluetooth headphones add far more; use wired ones.
13. **Only one session at a time, one output device**, and the page can't choose
    between microphones yet (step 03).

14. **Some training data is licensed for non-commercial use only** (step 04): all of ESC-50 and
    part of FSD50K. Fine for the university project; must be reviewed before any public or
    commercial use (the per-clip licence is in the clip index).
15. **Class definitions overlap** (step 04): car horns and emergency sirens are both "alarm" and
    "traffic"; bicycle bells are also "music". Such clips are excluded, so these sounds are
    under-represented in both classes. ESC-50's church bells count as music (AudioSet puts bells
    under Music), and airplanes, helicopters and trains count as traffic while "traffic" uses
    "Vehicle". *(Step 05: traffic is now "Motor vehicle (road)", so aircraft and trains are
    background; horns and sirens remain alarm + traffic overlaps.)*
16. **Several classes have few foreground clips** (step 04): see the coverage table in
    Chapter 11.13. Own recordings (`record-dataset`) are the way to add more.
17. **Mixtures are mono** (step 04). Spatial mixtures for the array come with module 3.
18. **Validation is thin for some classes** (step 04): after making FSD50K's val split
    uploader-disjoint, dishes has 17 validation clips, chair scraping 19, footsteps 30 and speech
    45. Validation scores for those classes will be noisy. *(Step 05: rebalanced to ≥ 40 val
    clips per class (Chapter 11.14). The traffic **test** split has only 26 clips.)*
19. **Detection is trained on synthetic mixtures only** (step 05): clean dataset clips added
    together, without the room's echo or the laptop microphone's sound. Real-world accuracy may be
    lower than the test numbers; own recordings help (Chapter 12.10).
20. **Training is limited by data loading** (step 05): ~6–7 minutes per epoch with 4 workers,
    the GPU mostly waiting. Every worker costs ~1 GB of memory (mostly PyTorch itself).
21. **Music can also light "alarm" or "speech"** (step 05): classical music sets off "alarm",
    vocals in songs set off "speech". In step 06 this means selecting *speech* will also keep
    the singing in music, and an alarm target may let some orchestral music through.
22. **The baseline extractor does not separate overlapping sounds** (step 06). While a selected
    sound is present, whatever else sounds at the same time comes through with it; only steady
    background is reduced. With perfect timing it gains 8 dB where a perfect mask would gain
    24 dB (Chapter 13.7). *(Step 07: the Neural method does separate overlapping sounds; see
    items 26–30 for its own limits.)*
23. **Live extraction lags behind the sound** (step 06). The gate follows the detector, which
    switches on ~0.3 s late and off ~1 s late: short sounds lose their beginning and are followed
    by a second of everything else. On the test mixtures live extraction scores *below* doing
    nothing (−4.9 dB SI-SDRi); offline processing, which has no lag, scores +2.2 dB. A sound the
    detector misses altogether is turned down like everything else (about 30 % of the test
    mixtures).
24. **Extraction adds 21 ms of delay** (step 06), with or without a selection, and works on
    the mono mix only. *(Step 07: the Neural method adds 10.6 ms; still mono.)*
25. **A reloaded page does not show the remembered selection** (step 06). The backend keeps the
    last selection and attenuation between sessions, but there is no way yet for the page to ask
    for them, so after a reload the buttons look unselected while the sound is still processed.
    Selecting again brings the two back in step (Chapter 2.12).
26. **The neural extractor's gain is modest and uneven** (step 07): +3.3 dB SI-SDRi on the test
    mixtures; large for many, but in about one mixture in nine the output is more than 5 dB
    *worse* than the input, because the wrong sound was kept. A gate with perfect timing would
    reach +8.1 dB (Chapter 14.9).
27. **Neural does not go quiet when the selected sound is absent** (step 07): about 12 dB of
    suppression at the 30 dB setting, 8.5 dB at the default 15 dB (the baseline: 27–29 dB).
28. **Sounds of the same type cannot be told apart** (step 07): "speech" is every voice in the
    room, including singing. Choosing one source among several of a kind needs direction
    (modules 3–5).
29. **The neural extractor learnt from 20,000 synthetic mixtures only** (step 07) and began to
    overfit after 12 rounds; chair scraping barely works (+0.2 dB). No recordings from the
    owner's microphone and no real rooms were used. More varied material is the way forward.
30. **Neural needs the graphics card to be comfortable, and needs it alone** (step 07): on the
    processor it uses almost four cores at RTF p95 0.7–0.8; and a training run or another
    GPU program next to a live session makes it stutter. The detector therefore runs on the
    processor during a Neural session (about 80 ms per recognition instead of 25).
31. **After a stall, the listening delay stays high** (step 07, found while measuring): when
    chunks arrive in a burst the speaker queue fills up to its maximum (8 chunks) and nothing
    drains it back to its normal 2, so the delay stays at about 225 ms until the session is
    restarted. It did not happen in the final configuration, but any overload can cause it.

---

## 26. Third-party code

Sound Spy is original code. It uses ordinary libraries installed with `pip` and `npm` (NumPy,
SciPy, sounddevice, soundfile, FastAPI, React and so on), each under its own licence; those are
dependencies, not adapted code, and are not listed here.

The rule for **adapted** code, meaning code copied from another project and modified: the
adapted file keeps its original copyright and licence header, and the component is listed in the
table below with where it came from, its licence, and which of our files contain it.

| Component | Source | Licence | Adapted files |
|---|---|---|---|
| **EfficientAT** (MobileNetV3 audio tagger `mn10_as` and its mel front-end), step 05 | https://github.com/fschmid56/EfficientAT (Florian Schmid; model code itself adapted from torchvision's `mobilenetv3.py`, BSD-3-Clause) | MIT (full text in `detection/backbones/efficientat/LICENSE`) | `backend/src/soundspy/detection/backbones/efficientat/{model,block_types,utils}.py`, the `EfficientATMel` front-end in `backend/src/soundspy/dsp/features.py` |
| **Waveformer** (causal target-sound-extraction network, default `E512/D256` configuration), step 07 | https://github.com/vb000/Waveformer (Bandhav Veluri, Justin Chan, Malek Itani, Tuochao Chen, Takuya Yoshioka, Shyamnath Gollakota; ICASSP 2023), file `src/training/dcc_tf.py`. Published weights and configuration: `https://targetsound.cs.washington.edu/files/default_ckpt.pt` and `default_config.json` (downloaded to `D:/SoundSpyData/models/waveformer/`, SHA-256 pinned in `configs/extraction/neural_tse.yaml`; never stored in the repository) | MIT, © 2022 Bandhav Veluri (the notice is kept in the adapted file and in `extraction/models/waveformer/LICENSE`) | `backend/src/soundspy/extraction/models/waveformer/model.py` |

**Changes we made:** only the `mlp` head is kept; `torchvision`'s `ConvNormActivation` is replaced by an identical local class (same layer order, so the pretrained weights load unchanged); no download or printing inside the model; the front-end uses the non-deprecated STFT call (same numbers, checked by a test) and caches its mel filterbank. **Weights:** `mn10_as_mAP_471.pt` from the project's GitHub release v0.0.1, downloaded to `D:/SoundSpyData/models/efficientat/` and checked against a pinned SHA-256; never committed. **Data file:** `configs/detection/audioset_labels.csv` (the 527 AudioSet labels, from AudioSet, CC BY 4.0, as distributed with EfficientAT).

**Changes we made to Waveformer:** only the model is taken (no training loop, dataset or metric code); `speechbrain`'s `PositionalEncoding` is replaced by a small local class with the same buffer name and values (proved equal by a stored output of the original, `tests/fixtures/waveformer_posenc_reference.npy`); `torchmetrics` is not used; `stream_sizes` and `stream_step` were added so that a caller can run whole steps with explicit state; the first layer of the label network is rebuilt for Sound Spy's classes when fine-tuning (in `extraction/neural_model.py`, not in the adapted file). A stored input/output pair of the *original* code on a tiny random model (`tests/fixtures/waveformer_tiny_reference.npz`) keeps the port honest.

**Where the idea came from.** The extraction design follows the **SemanticHearing** project
(Veluri et al., UIST 2023, https://github.com/vb000/SemanticHearing, MIT licence): a
label-conditioned Waveformer, chunked streaming inference, an AudioSet class mapping and
synthetic training mixtures. No code was taken from SemanticHearing itself; the model code
comes from the same group's **Waveformer** repository, which has its own row above.


**Data, not code.** The datasets downloaded in step 04 (AudioSet ontology, ESC-50, FSD50K) are
data used for training, not adapted code. Their sources and licences are recorded in each
dataset's `SOURCE.json` and described in Chapter 11.3.
---

## 27. Quick reference

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

### Training data (step 04)

```powershell
soundspy data paths                              # where everything lives (C: / D:)
soundspy data download ontology esc50            # then: soundspy data download fsd50k
soundspy data prepare                            # clip index + activity maps (parallel)
soundspy data stats                              # coverage table and warnings
soundspy data mix --split all                    # recipes -> D:/SoundSpyData/mixtures/*.jsonl
soundspy data preview --split val --n 5          # listen: outputs/previews/
soundspy data cleanup-archives                   # delete the zips once satisfied
soundspy record-dataset --classes speech background --clips 10 --seconds 5
```

### Detection (step 05)

```powershell
soundspy train detection --smoke                 # 2-epoch end-to-end check
soundspy train detection                         # full run (hours) -> experiments\det_mn10_v1\
soundspy train detection --resume                # continue after a stop
soundspy evaluate detection --split test         # fine-tuned vs zero-shot table
soundspy detect data\raw\<file>.wav              # per-class timeline of a recording
Get-Content experiments\det_mn10_v1\log.csv -Wait   # follow training
```

### Extraction (steps 06-07)

```powershell
soundspy data cache --task extraction --split all   # render the mixtures once (~25 GB on D:)
soundspy train extraction --smoke                   # 1 short epoch, end-to-end check
soundspy train extraction                           # full run (hours) -> experiments\tse_waveformer_v1\
soundspy train extraction --resume                  # continue after a stop
Get-Content experiments\tse_waveformer_v1\log.csv -Wait   # follow training
soundspy evaluate extraction --method neural_tse    # neural vs baseline vs oracles
soundspy extract data\raw\<file>.wav --labels speech [--method mask_baseline] [--attenuation-db 15]
soundspy db rebuild                                 # rebuild the index from the sidecars
```

### Addresses

```
GET  /api/health                        is the backend alive, which version
GET  /api/config                        classes, which tabs are available, device profile
GET  /api/devices                       devices + what the profile resolves to (503 = no PortAudio)
GET  /api/recordings                    recordings, newest first (re-indexes data/raw)
GET  /api/recordings/{id}[/audio]       one recording / its WAV (audio/wav)
GET  /api/session                       current session status
POST /api/session/start | stop          {mode, source: {type: mic|file, path, loop}, monitor}
PUT  /api/session/target                TargetSpec: {labels, attenuation_db} (422 unknown class)
PUT  /api/session/monitor               {enabled, gain_db -40..12, muted}
PUT  /api/session/bypass                {enabled}: hear the unprocessed sound (A/B)
PUT  /api/session/method                 {method: neural_tse | mask_baseline} (409 while running)
POST /api/session/record/start | stop   {label, notes} / returns the new recording
POST /api/recordings/{id}/detect        offline detection timeline (413 > 2 min, 503 no model)
POST /api/recordings/{id}/process       {labels, attenuation_db?, method?} -> a run, 202 (413 > 5 min,
                                        422 no/unknown class, 503 no model)
GET  /api/recordings/{id}/runs          its processing runs, newest first
GET  /api/runs/{run_id}[/audio]         one run / its processed WAV (409 if not done)
WS   /ws/session                        status 1/s, levels 20/s (input and output), waveform 25/s (input),
                                        spectrogram 25/s (input and output), detections 4/s, run,
                                        warning, error; heartbeat every 2 s when idle;
                                        accepts set_target, set_bypass and set_method
```

### The numbers worth remembering

| | |
|---|---|
| Backend address | `127.0.0.1:8765` |
| Website (development) | `localhost:5173` |
| Database | `data/soundspy.db` |
| Sound classes | 7 |
| Default suppression of unchosen sounds | 15 dB live since step 07 (to about 18 % of its size); 30 dB in the evaluation (about 3 %) |
| Engine chunk | 1,024 frames (~21 ms at 48 kHz); *since step 03:* 20 ms (960 frames at 48 kHz) |
| Listening delay (laptop, wired) | about 94 ms (22 in + 20 chunk + ~30 queue + 22 out); *since step 06:* 105–125 ms (+21.3 ms STFT); *since step 07, Neural:* 95–115 ms (+10.6 ms) |
| Processing load (laptop) | RTF p95 0.06; ~12 % of one CPU core; *since step 06 (detector on GPU + extractor):* RTF p95 0.17; ~24 %; *since step 07, Neural on the GPU + detector on the CPU:* RTF p95 0.63, network 7.5 ms per call, ~71 % of one core, 0 underruns in 60 s |
| Extractor (baseline) | STFT 1,024 / 256 at 48 kHz; gate 10 ms open, 300 ms hold, 200 ms close; SI-SDRi on 500 test mixtures: +2.2 dB offline, −4.9 dB live, +8.1 dB with true timing, +24.5 dB ideal mask |
| Extractor (neural) | Waveformer E512/D256, 3.86 M parameters, 44.1 kHz, steps of 416 samples (9.4 ms), lookahead 32; delay 509 samples (10.6 ms) at 48 kHz; SI-SDRi on 500 test mixtures: +3.28 dB live and offline (median +3.32), +2.92 dB in simulated rooms; 0 % targets missed; 12.5 dB quieter when the target is absent |
| Neural training | 20,000 mixtures (25.4 GB cache on D:, half reverberant), 3 s crops, batch 4 × 4, ~12 min per round of 10,000 crops; second run 26 rounds, 5.4 h, best round 16: validation +3.72 dB |
| Offline processing | 62 s recording in 6.5 s (step 07: 6.8 s with either method, 12 s the first time); at most 5 minutes per recording on the page; one job at a time |
| Speaker cushion / max queue / limiter | 2 chunks / 8 chunks / −1 dBFS ceiling, 50 ms release |
| Heartbeat / health check | every 2 s / every 5 s |
| Python / Node | 3.11 / 24 |
| Laptop microphone | WASAPI index 9, 48 kHz, 2 real mics (≥ ~4.3 cm apart), 16-bit + 7.44 dB gain |
| Honest noise floor (enhancements off) | about −65 dBFS |
| WASAPI start-up silence (discarded) | 500 ms with enhancements on, none with them off |
| Mic stall timeout / live chunk queue | 2 s / 32 chunks, oldest dropped first |
| Backend tests / website tests | 264 / 18 (after step 03); 344 / 18 (after step 04); 387 / 23 (after step 05); 482 / 36 (after step 06); 540 / 46 (after step 07) |
| Detector | EfficientAT mn10_as, 4.88 M parameters, 32 kHz, 1 s window padded to 10 s, a result every 0.25 s |
| Training data | `D:/SoundSpyData/` (external, processed, mixtures, models) |
| Mixtures | 6 s, 20,000 / 2,000 / 2,000 recipes (train / val / test), seeds 1 / 2 / 3 |

### Files that matter most

| File | Why |
|---|---|
| `backend/configs/classes.yaml` | the sound classes — IDs, display names, descriptions, AudioSet mapping |
| `backend/configs/default.yaml` | paths, active microphone profile, port, database location |
| `backend/src/soundspy/core/` | the shared data types every part relies on |
| `backend/src/soundspy/engine/` | the chunk-by-chunk processing frame; `session.py` runs it live, `factory.py` builds the chain |
| `backend/src/soundspy/io/speaker_sink.py` | listening: speaker queue, limiter, underrun counting |
| `backend/src/soundspy/analysis/analyzer.py` | the data behind the live pictures |
| `backend/src/soundspy/api/services/session_manager.py` | starts sessions and passes their events to the web server |
| `frontend/src/components/viz/` | the live canvases and their frame buffer (`liveFrames.ts`) |
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
| `backend/configs/data/default.yaml` | every training-data setting (splits, mixture rules) |
| `backend/configs/data/esc50_to_audioset.yaml` | ESC-50 labels → AudioSet mids |
| `backend/src/soundspy/data/roles.py` | the foreground / background / excluded rule |
| `backend/src/soundspy/data/mixtures.py` | recipes and rendering |
| `backend/src/soundspy/data/download.py` | resumable, checked downloads |
| `backend/configs/detection/default.yaml` | every detection setting (model, window, live, training) |
| `backend/src/soundspy/detection/model.py` | the detection network, head initialisation, zero-shot |
| `backend/src/soundspy/detection/inference.py` | live `Detector`, smoothing, offline timeline |
| `backend/src/soundspy/training/train.py` | the training loop |
| `backend/configs/extraction/` | `default.yaml` (method, default attenuation, evaluation), `mask_baseline.yaml`, `neural_tse.yaml` |
| `backend/src/soundspy/extraction/mask_baseline.py` | the baseline: gate + Wiener filter |
| `backend/src/soundspy/extraction/neural_tse.py` | the neural extractor live and offline: resampling, steps, delay, residual mixing |
| `backend/src/soundspy/extraction/models/waveformer/model.py` | the network itself (adapted, MIT) |
| `backend/src/soundspy/extraction/neural_model.py` | download, label-layer adaptation, checkpoints |
| `backend/src/soundspy/data/cache.py`, `data/reverb.py` | the pre-rendered training examples and the simulated rooms |
| `backend/src/soundspy/training/train_extraction.py`, `extraction_data.py` | the fine-tuning loop, the crops and the loss |
| `backend/src/soundspy/training/evaluate_extraction.py` | the comparison of all extraction systems |
| `docs/architecture.md` | the exact technical rules and message formats |
| `docs/decisions.md` | every important decision, dated, with its reason |
