## Inspiration

Modern studying is saturated with distraction. Existing desktop blockers are blunt instruments: they block entire apps or domains without understanding context.

But tools like YouTube or Chrome are not inherently distractions — they can be productive or unproductive depending on intent. We wanted a system that distinguishes between the two dynamically, rather than blocking blindly.

---

## What it does

### 1. Context-aware intelligent blocking

DeBrainrot selectively blocks applications based on what they are actually providing at that moment.

Instead of blocking YouTube or a browser outright, the system evaluates whether the current content aligns with the user’s declared task. If it supports the task, it remains accessible. If it deviates into distraction, it is blocked.

Blocking decisions are conditioned on:
- The user’s chosen focus task  
- Screenshot-based behavioural signals  
- LLM-powered semantic validation  

This allows nuanced enforcement rather than static blacklists.

### 2. Agent-generated learning roadmaps

We use LLM agents to generate structured study roadmaps.

Goals are transformed into:
- Missions  
- Quests  
- Progressive milestones  

The experience is gamified to maintain engagement, but intentionally restrained to avoid becoming another distraction. The tone is light and humorous while preserving productivity focus.

---

## How we built it

DeBrainrot Desktop is a cross-platform productivity RPG built with:

- **Electron** for the native desktop shell  
- **React + TypeScript** for the frontend  
- **React Query** for API state management  
- **HashRouter** for Electron-compatible routing  
- **Tailwind CSS + Radix UI (shadcn-style)** with custom retro styling  

The UI uses pixel typography, scanlines, and game-style progress elements to create a levelling-up experience rather than a conventional todo list.

### Backend Architecture

We integrated with a **Cloudflare Workers** backend:

`debrainrot-backend.brndndiaz.workers.dev`

The desktop client manages authentication and session state, then communicates with Worker endpoints for:

- Profiles  
- Missions  
- Quests  
- Stats  
- Streaks  
- Integrations  
- Notification preferences  

Mission generation is asynchronous. After mission creation, the client triggers quest generation and polls progress so tasks appear incrementally.

Tooling:
- Electron Forge + Webpack  
- TypeScript across the stack  
- Clean separation between Electron (main/preload), frontend logic, and backend Workers  

---

## Intelligent Blocking System

Blocking works in three stages:

### 1. Screenshot Dissimilarity Filtering

We intermittently capture screenshots.

A fast hybrid image dissimilarity metric is computed between consecutive frames.  
If two frames differ significantly, we infer a potential context switch and send the latest screenshot for deeper evaluation.

This:
- Minimises unnecessary LLM calls  
- Reduces latency  
- Improves responsiveness  

Occasionally, screenshots are sampled regardless of the metric to provide fail-safe coverage if the dissimilarity filter misses a transition.

#### Hybrid Comparison (RGB / RGBA)

Our dissimilarity function is based on a hybrid structural–colour comparison approach inspired by:  
https://github.com/ChrisRega/image-compare

### By structure: "Hybrid Comparison"

- Splitting the image to YUV colorspace according to T.871
- Processing the Y channel with MSSIM
- Comparing U and V channels via RMS
- Recombining the differences to a nice visualization image
- RGB Score is calculated as: $\mathrm{score}=\mathrm{avg}_{x,y}\left(
  \mathrm{min}\left[\Delta \mathrm{MSSIM}(Y,x,y),\sqrt{(\Delta RMS(U,x,y))^2 + (\Delta RMS(V,x,y))^2}\right]\right)$
- RGBA can either be premultiplied with a specifiable background color using `rgba_blended_hybrid_compare`
- Otherwise, for `rgba_hybrid_compare` the $\alpha$ channel is also compared using MSSIM and taken into account.
- The average alpha of each pixel $\bar{\alpha}(x,y) = 1/2 (\alpha_1(x,y) + \alpha_2(x,y))$ is then used as a linear
  weighting factor
- RGBA Score is calculated as: $\mathrm{score}=\mathrm{avg}_{x,y}\left(1/\bar{\alpha} \cdot
  \mathrm{min}\left[\Delta \mathrm{MSSIM}(Y,x,y),\sqrt{(\Delta RMS(U,x,y))^2 + (\Delta RMS(V,x,y))^2}, \Delta \mathrm{RMS}(\alpha,x,y)\right]
  \right)$
- Edge cases RGBA: $\mathrm{score} \in (0, 1)$ and $\mathrm{score} = 1.0$ if $\bar{\alpha} = 0.0$
- This allows for a good separation of color differences and structure differences for both RGB and RGBA


**Diff image interpretation:**

- RGB:  
  - Red → structure differences  
  - Green/Blue → colour differences  
  - Higher saturation → larger difference  

- RGBA:  
  - Same as RGB  
  - Alpha channel contains inverse alpha differences  
  - Minimum alpha clamped at 0.1 to preserve visibility  
  - Heavily translucent regions reflect difficulty separating colour and structural differences  

---

### 2. LLM-Based Task Validation

We built a Cloudflare Workers (TypeScript) endpoint:

`POST /v1/focus/validate`

It receives:
- Task context  
- Base64-encoded screenshot  

The Worker sends a compact prompt + image to Gemini via Cloudflare AI Gateway, then returns a normalised JSON verdict:

```json
{
  "productive": boolean,
  "confidence": number,
  "task_alignment": string,
  "sensitive_data": boolean
}