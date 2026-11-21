# Sound Spy: a 5-minute demo

For the presentation of the Semantic tab. One presenter, the laptop, **closed headphones** on
the presenter (the audience watches the screen and hears the result through the laptop
speakers only in step 4, at low volume), a phone with a classical piece and a song with
vocals, and a looping recording as the fallback for a noisy room.

## Before the audience arrives (5 min)

1. Windows audio enhancements **off** for the microphone (they were off for every measurement).
2. Start the app: `.\start.ps1` in the project folder. The page opens at
   `http://127.0.0.1:8765/`; wait until the session bar no longer says *Preparing models*
   (about 10 s after a fresh start).
3. Press **Start**, put the headphones on, tick **Listen** (confirm "I'm using headphones"),
   volume about −6 dB. Check that you hear yourself. **Stop**.
4. In **Listen**: method **Hybrid (recommended)**, Gate on, attenuation **40 dB**.
5. Fallback ready: in *Recordings and processing* there is a recording of you talking over
   music (`test_music_and_speech_...`); in the session bar's **Source** choose it to play it in a
   loop instead of the microphone.

## The demo

| Time | Do | The audience sees / hears |
|---|---|---|
| 0:00 | One sentence: "Sound Spy lets you choose what you hear: the chosen sound types stay, everything else is turned down, live." Point at the three areas: Listen, Sounds, Visuals. | The page, the session bar idle. |
| 0:20 | **Start**. Talk normally. | *Sounds*: the Speech chip fills while you talk (live probability). *Visuals*: input and output meters move together, the spectrogram scrolls. Session bar: route, gate, delay. |
| 0:50 | Play **classical music** from the phone, about a metre from the laptop, and keep talking. Click **Speech**. | Speech chip selected; the lower (output) spectrogram keeps the voice's harmonics, the music's bands fade. Session bar: *speech enhancer · Gate open* while you talk, *Gate closed* in your pauses. |
| 1:30 | Pause for two or three seconds, then talk again. | In the pause the output meter drops to the bottom and the output spectrogram goes dark; the first word comes back at once (the gate reopens within milliseconds). |
| 2:00 | Press **Space** (Bypass), talk a few seconds, press **Space** again. Unplug one headphone ear for the audience or describe it. | *Bypass: unprocessed* in the session bar: the music is back, at the same delay; Space again: only the voice. |
| 2:40 | Deselect Speech, select **Music**. | Route changes to *Waveformer*; now the music stays and the voice is turned down (the delay drops, the Waveformer route needs 10.6 ms instead of 40). |
| 3:20 | Play the **song with vocals**, select **Speech** again. | Honest limitation: the singer is a voice too, so the song's vocals stay (the evaluation's *song* rows). |
| 3:50 | Open *Details* in the session bar. | Real-time factor well below 1, 0 underruns, the delay estimate (and the measured value, if `soundspy measure-latency` was run). |
| 4:20 | **Stop**. Open *Recordings and processing*, process a recording with Speech, play **Original / Processed** with the A/B player. | The same processing offline, switchable sample-aligned. |
| 4:50 | Close: "The numbers behind this are in the evaluation: in the real room the pauses are 33-42 dB quieter with classical music, speech is cut in 0.3 % of its frames." | `docs/evaluation.md`, the pause figure. |

## If something goes wrong

- **Noisy room** (other people talking, the detector hears speech everywhere): use the looping
  recording as the **Source** (step 5 above): the same demo, repeatable, no microphone needed.
- **No sound in the headphones**: Listen ticked? Mute off (key **M**)? Volume not at −40? The
  right **Output** device in the Listen area (**Rescan devices** after plugging the headphones
  in, with the session stopped)?
- **The session stops with an error** (headphones or microphone unplugged): plug back in, press
  **Restart**.
- **"Preparing models…" for long**: the first start after launching the app loads the models
  (about 10 s); it only happens once per launch.
- **Howling**: Listen is on with the laptop speakers. Headphones, or Listen off.
- **The page says the backend is not reachable**: the `start.ps1` window was closed; start it
  again, the page reconnects by itself.
