/**
 * Backend for the movie-review study (GitHub Pages frontend -> this web app -> Google Sheet).
 *
 * Setup: create a Google Sheet, open Extensions > Apps Script, paste this file as Code.gs,
 * run setup() once, then Deploy > New deployment > Web app
 *   (Execute as: Me, Who has access: Anyone). Copy the /exec URL into index.html (ENDPOINT).
 *
 * The answer keys live only here, never in the browser.
 */

const KEYS = {
 "intrusion": {
  "1": {
   "correct": 4,
   "target": "story_plot",
   "intruder": "general_movie_opinion",
   "option_aspects": [
    "story_plot",
    "story_plot",
    "story_plot",
    "general_movie_opinion"
   ]
  },
  "2": {
   "correct": 1,
   "target": "filmmaking_direction",
   "intruder": "story_plot",
   "option_aspects": [
    "story_plot",
    "filmmaking_direction",
    "filmmaking_direction",
    "filmmaking_direction"
   ]
  },
  "3": {
   "correct": 3,
   "target": "acting_performances",
   "intruder": "music_sound",
   "option_aspects": [
    "acting_performances",
    "acting_performances",
    "music_sound",
    "acting_performances"
   ]
  },
  "4": {
   "correct": 3,
   "target": "music_sound",
   "intruder": "humor",
   "option_aspects": [
    "music_sound",
    "music_sound",
    "humor",
    "music_sound"
   ]
  },
  "5": {
   "correct": 3,
   "target": "ending",
   "intruder": "general_movie_opinion",
   "option_aspects": [
    "ending",
    "ending",
    "general_movie_opinion",
    "ending"
   ]
  },
  "6": {
   "correct": 4,
   "target": "general_movie_opinion",
   "intruder": "expectations_disappointment",
   "option_aspects": [
    "general_movie_opinion",
    "general_movie_opinion",
    "general_movie_opinion",
    "expectations_disappointment"
   ]
  },
  "7": {
   "correct": 1,
   "target": "history_realism",
   "intruder": "acting_performances",
   "option_aspects": [
    "acting_performances",
    "history_realism",
    "history_realism",
    "history_realism"
   ]
  },
  "8": {
   "correct": 2,
   "target": "fandom_franchise",
   "intruder": "rewatchability",
   "option_aspects": [
    "fandom_franchise",
    "rewatchability",
    "fandom_franchise",
    "fandom_franchise"
   ]
  },
  "9": {
   "correct": 2,
   "target": "tv_episodes_seasons",
   "intruder": "overall_praise",
   "option_aspects": [
    "tv_episodes_seasons",
    "overall_praise",
    "tv_episodes_seasons",
    "tv_episodes_seasons"
   ]
  },
  "10": {
   "correct": 4,
   "target": "family_kids",
   "intruder": "story_plot",
   "option_aspects": [
    "family_kids",
    "family_kids",
    "family_kids",
    "story_plot"
   ]
  }
 },
 "personalization": {
  "1": {
   "personalized": "B",
   "aspect": "family_kids",
   "asin": "B004APVI1G",
   "pers_aspect_mass": 0.9862894415855408,
   "generic_aspect_mass": 0.011037399992346764
  },
  "2": {
   "personalized": "B",
   "aspect": "humor",
   "asin": "B00BNWWYE4",
   "pers_aspect_mass": 0.9914739727973938,
   "generic_aspect_mass": 0.11708640307188034
  },
  "3": {
   "personalized": "B",
   "aspect": "horror_scares",
   "asin": "B00DZP1C9K",
   "pers_aspect_mass": 0.9902125597000122,
   "generic_aspect_mass": 0.0830705389380455
  },
  "4": {
   "personalized": "B",
   "aspect": "acting_performances",
   "asin": "B000EHSVNW",
   "pers_aspect_mass": 0.99336838722229,
   "generic_aspect_mass": 0.047095887362957
  },
  "5": {
   "personalized": "A",
   "aspect": "story_plot",
   "asin": "B0748PJSBJ",
   "pers_aspect_mass": 0.971768856048584,
   "generic_aspect_mass": 0.014601774513721466
  },
  "6": {
   "personalized": "B",
   "aspect": "music_sound",
   "asin": "B00065EAZU",
   "pers_aspect_mass": 0.9945353865623474,
   "generic_aspect_mass": 0.08956516534090042
  },
  "7": {
   "personalized": "A",
   "aspect": "ending",
   "asin": "B00QK4AOPG",
   "pers_aspect_mass": 0.9789078235626221,
   "generic_aspect_mass": 0.013071998953819275
  },
  "8": {
   "personalized": "A",
   "aspect": "history_realism",
   "asin": "B00V3293UC",
   "pers_aspect_mass": 0.9884286522865295,
   "generic_aspect_mass": 0.10344982147216797
  },
  "9": {
   "personalized": "B",
   "aspect": "disc_format_quality",
   "asin": "B0009K8LCA",
   "pers_aspect_mass": 0.9955819845199585,
   "generic_aspect_mass": 0.0029500010423362255
  },
  "10": {
   "personalized": "A",
   "aspect": "tv_episodes_seasons",
   "asin": "B00T6KIK4S",
   "pers_aspect_mass": 0.9967958331108093,
   "generic_aspect_mass": 0.04460204765200615
  }
 }
};

const SHEETS = {
  participants: ["received_at", "participant_id", "session_id", "version", "started_at", "finished_at",
    "duration_s", "intrusion_correct", "intrusion_n", "intrusion_accuracy", "prefer_wins", "pers_n",
    "prefer_win_rate", "repeat_participant_id", "user_agent", "viewport"],
  intrusion: ["received_at", "participant_id", "session_id", "question", "position", "target_aspect",
    "intruder_aspect", "display_order", "chosen_option", "chosen_aspect", "correct_option", "is_correct", "rt_ms"],
  personalization: ["received_at", "participant_id", "session_id", "question", "position", "aspect", "asin",
    "personalized_option", "left_shown", "personalized_on_left", "choice", "chose_personalized",
    "rating_A", "rating_B", "rating_personalized", "rating_generic",
    "pers_aspect_mass", "generic_aspect_mass", "rt_ms"],
  raw: ["received_at", "session_id", "json"]
};

function setup() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  Object.keys(SHEETS).forEach(function (name) {
    let sh = ss.getSheetByName(name) || ss.insertSheet(name);
    if (sh.getLastRow() === 0) { sh.appendRow(SHEETS[name]); sh.setFrozenRows(1); }
  });
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function doGet() { return json_({ ok: true, status: "study backend is running" }); }

function doPost(e) {
  const lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    setup();
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    const now = new Date();
    let p;
    try { p = JSON.parse(e.postData.contents); } catch (err) { return json_({ ok: false, error: "Answers could not be read." }); }

    const problem = validate_(p);
    if (problem) {
      ss.getSheetByName("raw").appendRow([now, String(p && p.session_id), "REJECTED: " + problem + " | " + e.postData.contents]);
      return json_({ ok: false, error: problem });
    }

    const part = ss.getSheetByName("participants");
    const existing = part.getLastRow() > 1 ? part.getRange(2, 2, part.getLastRow() - 1, 2).getValues() : [];
    if (existing.some(function (r) { return r[1] === p.session_id; })) return json_({ ok: true, duplicate: true });
    const repeatPid = existing.some(function (r) { return String(r[0]) === String(p.participant_id); });

    ss.getSheetByName("raw").appendRow([now, p.session_id, JSON.stringify(p)]);

    // Part 1: intrusion
    let nCorrect = 0;
    const iRows = p.intrusion.map(function (r) {
      const k = KEYS.intrusion[String(r.q)];
      const ok = r.choice === k.correct ? 1 : 0;
      nCorrect += ok;
      return [now, p.participant_id, p.session_id, r.q, r.position, k.target, k.intruder,
        r.display_order.join(" "), r.choice, k.option_aspects[r.choice - 1], k.correct, ok, r.rt_ms];
    });
    appendRows_(ss.getSheetByName("intrusion"), iRows);

    // Part 2: personalization
    let nWins = 0;
    const pRows = p.personalization.map(function (r) {
      const k = KEYS.personalization[String(r.q)];
      const win = r.choice === k.personalized ? 1 : 0;
      nWins += win;
      const rP = k.personalized === "A" ? r.rating_A : r.rating_B;
      const rG = k.personalized === "A" ? r.rating_B : r.rating_A;
      return [now, p.participant_id, p.session_id, r.q, r.position, k.aspect, k.asin, k.personalized,
        r.left, r.left === k.personalized ? 1 : 0, r.choice, win, r.rating_A, r.rating_B, rP, rG,
        k.pers_aspect_mass, k.generic_aspect_mass, r.rt_ms];
    });
    appendRows_(ss.getSheetByName("personalization"), pRows);

    const dur = (new Date(p.finished_at) - new Date(p.started_at)) / 1000;
    part.appendRow([now, p.participant_id, p.session_id, p.version, p.started_at, p.finished_at, dur,
      nCorrect, iRows.length, nCorrect / iRows.length, nWins, pRows.length, nWins / pRows.length,
      repeatPid ? 1 : 0, p.user_agent, p.viewport]);

    return json_({ ok: true });
  } finally {
    lock.releaseLock();
  }
}

function appendRows_(sh, rows) {
  if (rows.length) sh.getRange(sh.getLastRow() + 1, 1, rows.length, rows[0].length).setValues(rows);
}

function validate_(p) {
  if (!p || !p.session_id || !p.participant_id) return "Missing participant or session ID.";
  const iq = Object.keys(KEYS.intrusion), pq = Object.keys(KEYS.personalization);
  if (!Array.isArray(p.intrusion) || p.intrusion.length !== iq.length) return "Part 1 is incomplete.";
  if (!Array.isArray(p.personalization) || p.personalization.length !== pq.length) return "Part 2 is incomplete.";
  const seen1 = {}, seen2 = {};
  for (const r of p.intrusion) {
    if (!KEYS.intrusion[String(r.q)] || seen1[r.q]) return "Part 1 has an unexpected question.";
    seen1[r.q] = 1;
    if (!(r.choice >= 1 && r.choice <= 4)) return "Part 1 has an invalid answer.";
  }
  for (const r of p.personalization) {
    if (!KEYS.personalization[String(r.q)] || seen2[r.q]) return "Part 2 has an unexpected question.";
    seen2[r.q] = 1;
    if (r.choice !== "A" && r.choice !== "B") return "Part 2 has an invalid choice.";
    if (!(r.rating_A >= 1 && r.rating_A <= 5 && r.rating_B >= 1 && r.rating_B <= 5)) return "Part 2 has an invalid rating.";
  }
  return null;
}
