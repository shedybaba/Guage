# Demo video script (about 3 minutes)

Everything on screen is the real app answering real questions. Nothing is mocked and no result is pre-written.
The numbers below are what the pendulum clips gave during testing; read them off the screen on the day, because
results can vary slightly between runs.

## Before recording (10 minutes)

- Footage: if you have no clips of your own, use the freely licensed ones in the table at the end of this file (all from Wikimedia Commons). Show the credit line on screen or in the video description. Film your own if you can: phone propped still, object fully visible, good light, slow motion for anything fast.
- Tracking by SAM can take from 15 seconds to several minutes depending on Meta's load (one run took 8 minutes). Run every question in advance so it is cached; a cached question answers in about 15 seconds.
- Run every question once beforehand so tracking is cached and the take is fast. Keep a second browser tab with the finished result as a safety net.
- Quiet room, microphone close. Test the microphone button once (a mis-hear on camera is fine: it shows why you check the words).
- Close other tabs and notifications. Browser zoom at 110% so the numbers are readable.
- Check the health page says the key is configured (`/api/health`).

## The script

**0:00 to 0:25. Hook (screen: home page, then the upload page)**
> "Measuring motion from video means hours in manual tools, or chatbots that guess numbers. Gauge measures. You upload a video, ask a question in plain English, and every number comes from code, with its uncertainty and the proof on screen."

**0:25 to 1:40. Live demo, pendulum**
1. Drag the pendulum clip into the upload box. (Say: "any phone video, MP4 or MOV".)
2. Tap the microphone and say: *"What is the swing period?"* Stop. Point at the words in the box: "I always get to check the words before anything runs, because one mis-heard number would ruin a measurement." Press Ask.
3. While progress streams: "SAM 3.1 is tracking the object frame by frame. This log is Gauge checking its own work."
4. Result: read the answer with its uncertainty ("0.86 seconds, plus or minus 0.002"). Press **Show on the video**: the mask, number and path follow the bob. Point at the chart: dots are measurements, the line is the fit, the shaded part is the stretch it used. "It ignored the first four seconds because that was set-up motion, and it says so."
5. Follow-up, typed or spoken: *"What value of g does that give? The string is 0.50 metres."* Gauge reads the length from the sentence, converts units in code, and shows g with its uncertainty against 9.81. (If the string length on the clip does not match what you say, the answer will honestly say it is NOT consistent. Use the real length of your pendulum.)
6. Say: "The follow-up cost nothing in tracking: the video was already tracked."

**1:40 to 2:10. Self-checking (screen: the "How Gauge checked its work" list)**
- Use a clip where the object is hidden for a moment (the brass pendulum clip has a real occlusion). Show the line: "lost the object for 27 frames, re-acquired it with a box prompt from the last good frame, and checked it landed where it should."
- Then ask something it cannot measure (for example the temperature of the room: Gauge has no tool for it). Show the refusal: "When it can't measure, it says so and says why."

**2:10 to 2:40. Under the hood (screen: the architecture diagram from the README)**
> "Three Meta models. SAM 3.1 finds and follows the objects, with confidence and box prompts for repairs. Muse Spark reads the question and three frames to plan, and writes the explanation, but code checks every number it says. Muse Voice Transcribe turns speech into the question. The physics runs as plain Python with uncertainty, tested on simulated motion with known answers. And everything here is a plain API, so other apps can plug it in with one call."

Show `/developers` for two seconds.

**2:40 to 3:00. Impact**
> "Labs without sensors, researchers who track motion by hand, engineers checking a machine: every camera becomes an instrument. Everything you saw was built during this hackathon."

## If something goes wrong on the day

- **Tracking is slow:** cut away to the cached tab; say "cached from my earlier run" only if it is true.
- **The microphone fails:** type the question. The box is always usable.
- **A result looks odd:** read the warnings on screen out loud. Honest warnings are the point of the product.
- **Do not** edit numbers, splice answers from different runs, or hide a failure.

## Footage that was tested on the live site (Nov 4)

Files are in `samples/` (git-ignored, never committed). Say the credit on screen if the clip appears in the video.

| Clip file | What it is and credit | Question to ask | Result seen |
|---|---|---|---|
| `dl_p60_close.mp4` | Close-up of a 60 cm pendulum, cut from "Fadenpendelgeschwindigkeit (2 grad).webm" by Rhetos, CC BY-SA 4.0 (Wikimedia Commons) | "What is the swing period of the pendulum?" then "What value of g does that give? The string is 0.60 metres." | T = 1.59 ± 0.03 s (the video's own narration says about 1.54 s); g = 9.4 ± 0.3 m/s², consistent with 9.81. The best accuracy showcase |
| `dl_oscillator.mp4` | Cart on a spring with a ruler, "Oscillatory motion acceleration.ogv" by Pieter Kuiper (Wikimedia Commons, attribution licence) | "What is the oscillation frequency?" then "What is the spring constant?" (Gauge pauses and asks for the mass) | f = 0.432 ± 0.004 Hz; the peaks in the data are 2.3 s apart, which agrees. The flat turning points are a real wave that is not a perfect sine, and the answer says the fit is only fair |
| `newtons_cradle.mp4`, `pendulum.mp4`, `pendulum2.mp4` | The first clips used during development | "How many balls are there?" (7); "What is the swing period?" (0.858 s and 1.180 s) | see README |
| `dl_lentil.mp4` | Lentil microgreens time-lapse, CC BY 2.0 (Wikimedia Commons) | "How much did the sprouts grow?" | Several sprouts are tracked; Gauge measures the one seen in most frames and says it ignored the others. Weak as a demo |

Known limits found while testing (say them if asked, do not hide them):
- Dense scenes with more than 16 objects (a bottling line with a panning camera) cannot be counted: SAM follows at most 16 objects per frame.
- Tiny, blurred objects (a ball a few pixels across in a dark room) are lost, and Gauge answers "no fall was found" instead of a number.
- A clip filmed from a moving or zooming camera is reported, not corrected.

## Still worth filming (if there is time)

- A mass on a spring with a known mass, slow motion: frequency and spring constant.
- Parts moving across the frame on a belt: count and rate per minute.
- A dropped ball next to a metre stick, slow motion: free fall and g.
- A fan or wheel with a sticker on it: rotation speed in rpm.
