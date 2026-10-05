# Movie-review judge study (PREFER human evaluation)

Two-part study in one page: Part 1 aspect intrusion (10 items), Part 2 PREFER vs generic summary (10 items, plus a 1 to 5 rating of each summary). Every question is required, there is no back button, and participants never see whether they were right.

```
index.html   the study (host on GitHub Pages)  — contains question text only, no answers
Code.gs      backend (Google Apps Script)      — holds the answer keys, grades, writes to a Google Sheet
analyze.py   statistics for the paper           — reads the Sheet tabs exported as CSV
```

## 1. Backend (Google Sheet + Apps Script), about 5 minutes
1. Create a new Google Sheet (e.g. "PREFER judge results"). Keep it private.
2. Extensions > Apps Script. Replace the contents of `Code.gs` with this repo's `Code.gs`. Save.
3. Pick `setup` in the function dropdown and click Run. Authorize when asked
   (Advanced > Go to project > Allow). Four tabs appear: participants, intrusion, personalization, raw.
4. Deploy > New deployment > gear icon > Web app.
   Execute as: **Me**. Who has access: **Anyone**. Deploy, and copy the URL ending in `/exec`.
5. Open that URL in a browser; you should see `{"ok":true,"status":"study backend is running"}`.

If you later change `Code.gs`, use Deploy > Manage deployments > edit (pencil) > Version: New version.
That keeps the same URL. "New deployment" would create a different URL.

## 2. Frontend (GitHub Pages)
1. In `index.html`, set `const ENDPOINT = "https://script.google.com/macros/s/.../exec";`
2. Create a GitHub repository, upload `index.html` (only this file needs to be public).
3. Settings > Pages > Source: Deploy from a branch > `main` / root > Save.
4. After a minute the study is live at `https://<username>.github.io/<repo>/`.

## 3. Pilot before sharing
Run through it once with participant ID `test1`. Check that one row appears in `participants` and ten in each of
`intrusion` and `personalization`. `analyze.py` drops IDs starting with `test` by default.

## What gets recorded
Per judge: ID, timestamps, duration, Part 1 accuracy, PREFER win rate, browser.
Per Part 1 answer: shown order of the four sentences, chosen sentence (original option number and its aspect),
correct option, is_correct, response time. Per Part 2 answer: which summary was on the left, choice,
chose_personalized, both ratings (also mapped to PREFER/generic), simulated aspect mass, response time.
Question order, sentence order and left/right placement are randomized per judge.
A resubmission of the same session is ignored; a reused participant ID is flagged.
If sending fails, the page keeps answers in the browser, lets the judge retry, and offers a JSON download.

## 4. Analysis
In the Sheet, File > Download > CSV for the tabs participants, intrusion, personalization.
```
python analyze.py --exclude test,pilot --min-minutes 4
```
