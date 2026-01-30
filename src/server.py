"""
FastAPI server for the Akinator game.
Run from repo root: python main.py or uvicorn src.server:app --reload
"""
import os
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.model import AkinatorEngine

app = FastAPI(title="Akinator Clone")

# Global engine (load GPU tensors once at startup)
engine = None


@app.on_event("startup")
def startup():
    global engine
    engine = AkinatorEngine()


# --- Pydantic models (same file, student-style) ---


class AnswerRequest(BaseModel):
    session_id: str
    answer_value: float  # 1.0=Yes, 0.0=No, 0.5=Unknown


class GuessRequest(BaseModel):
    session_id: str


class StartGameResponse(BaseModel):
    session_id: str
    question: str


class AnswerResponse(BaseModel):
    question: str
    top_guess: Optional[str] = None
    top_guess_probability: Optional[float] = None


class GuessResponse(BaseModel):
    name: str
    probability: float


# Expert-system: resolve category on Yes (never ask again), cap questions per category
YES_THRESHOLD = 0.7  # answer_value >= this -> resolve that category

# In-memory session state
# session_id -> {current_probs, asked_set, current_question_idx, resolved_categories, asked_count_per_category}
sessions = {}


# --- Akinator-style playable frontend at / ---

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Akinator Clone</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Quicksand:wght@500;600;700&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: 'Quicksand', system-ui, sans-serif;
      margin: 0;
      padding: 1rem;
      min-height: 100vh;
      background: linear-gradient(160deg, #1a0a2e 0%, #16213e 40%, #0f3460 100%);
      color: #f0e6ef;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    .container { max-width: 520px; width: 100%; }
    header { text-align: center; margin-bottom: 1rem; }
    header h1 {
      font-size: 1.75rem;
      font-weight: 700;
      margin: 0 0 0.25rem 0;
      color: #f4d03f;
      background: linear-gradient(135deg, #f4d03f 0%, #e8a838 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    @supports (background-clip: text) {
      header h1 { -webkit-text-fill-color: transparent; }
    }
    header p { margin: 0; color: rgba(240,230,239,0.85); font-size: 0.95rem; }
    .genie-area {
      height: 140px;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 0.5rem;
    }
    .genie-area.thinking .genie-orb { animation: pulse 1s ease-in-out infinite; }
    .genie-orb {
      width: 100px;
      height: 100px;
      border-radius: 50%;
      background: radial-gradient(circle at 30% 30%, #f4d03f, #e8a838 40%, #6b2d5c 70%, #2d1b4e);
      box-shadow: 0 0 40px rgba(244,208,63,0.4), 0 0 80px rgba(107,45,92,0.3), inset -10px -10px 20px rgba(0,0,0,0.2);
    }
    .lamp-base {
      width: 60px;
      height: 24px;
      margin-top: -8px;
      background: linear-gradient(180deg, #8b7355 0%, #5c4a32 100%);
      border-radius: 0 0 30px 30px;
      box-shadow: inset 0 2px 4px rgba(255,255,255,0.2), 0 4px 8px rgba(0,0,0,0.3);
    }
    @keyframes pulse { 0%, 100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.05); opacity: 0.9; } }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    .speech-wrap { position: relative; margin-bottom: 1.25rem; }
    .speech-bubble {
      background: rgba(255,255,255,0.96);
      color: #1a0a2e;
      padding: 1.25rem 1.5rem;
      border-radius: 20px;
      border-bottom-left-radius: 6px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.25);
      position: relative;
      animation: fadeIn 0.3s ease-out;
    }
    .speech-bubble::before {
      content: '';
      position: absolute;
      left: 24px;
      top: -10px;
      border: 10px solid transparent;
      border-bottom-color: rgba(255,255,255,0.96);
    }
    .question-num {
      display: block;
      font-size: 0.8rem;
      font-weight: 600;
      color: #6b2d5c;
      margin-bottom: 0.5rem;
    }
    .question-text { font-size: 1.15rem; line-height: 1.5; font-weight: 500; }
    .answers {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      justify-content: center;
      margin-bottom: 1rem;
    }
    .answers button {
      font-family: 'Quicksand', system-ui, sans-serif;
      font-size: 1rem;
      font-weight: 600;
      padding: 0.65rem 1rem;
      border: none;
      border-radius: 12px;
      cursor: pointer;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .answers button:hover { transform: scale(1.03); box-shadow: 0 4px 12px rgba(0,0,0,0.25); }
    .btn-yes { background: linear-gradient(180deg, #2ecc71, #27ae60); color: #fff; }
    .btn-no { background: linear-gradient(180deg, #e74c3c, #c0392b); color: #fff; }
    .btn-dont { background: linear-gradient(180deg, #5c4a6a, #4a3d55); color: #e8e0e8; }
    .btn-probably { background: linear-gradient(180deg, #58d68d, #2ecc71); color: #fff; }
    .btn-probably-not { background: linear-gradient(180deg, #ec7063, #e74c3c); color: #fff; }
    .guess-row { text-align: center; margin-bottom: 1rem; }
    .btn-guess {
      font-family: 'Quicksand', system-ui, sans-serif;
      font-size: 1rem;
      font-weight: 600;
      padding: 0.7rem 1.5rem;
      background: linear-gradient(135deg, #f4d03f, #e8a838);
      color: #1a0a2e;
      border: none;
      border-radius: 12px;
      cursor: pointer;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .btn-guess:hover { transform: scale(1.03); box-shadow: 0 6px 20px rgba(244,208,63,0.4); }
    .guess-result {
      margin-top: 1rem;
      padding: 1.25rem;
      background: linear-gradient(135deg, rgba(107,45,92,0.6), rgba(45,27,78,0.8));
      border-radius: 16px;
      border: 1px solid rgba(244,208,63,0.3);
      text-align: center;
      animation: fadeIn 0.3s ease-out;
    }
    .guess-result strong { color: #f4d03f; }
    .start-card {
      text-align: center;
      padding: 2rem 1.5rem;
      background: rgba(107,45,92,0.25);
      border-radius: 20px;
      border: 1px solid rgba(244,208,63,0.2);
    }
    .btn-start {
      font-family: 'Quicksand', system-ui, sans-serif;
      font-size: 1.2rem;
      font-weight: 700;
      padding: 0.9rem 2rem;
      background: linear-gradient(135deg, #f4d03f, #e8a838);
      color: #1a0a2e;
      border: none;
      border-radius: 14px;
      cursor: pointer;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .btn-start:hover { transform: scale(1.05); box-shadow: 0 8px 24px rgba(244,208,63,0.4); }
    .hidden { display: none !important; }
    .error { color: #e74c3c; margin-top: 0.75rem; text-align: center; font-weight: 500; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Akinator Clone</h1>
      <p>Think of a person. I will ask questions and try to guess who it is.</p>
    </header>
    <div id="start" class="start-card">
      <button class="btn-start" onclick="startGame()">Start game</button>
    </div>
    <div id="game" class="hidden">
      <div class="genie-area" id="genieArea">
        <div>
          <div class="genie-orb"></div>
          <div class="lamp-base"></div>
        </div>
      </div>
      <div class="speech-wrap">
        <div class="speech-bubble">
          <span class="question-num" id="questionNum">Question 1</span>
          <div class="question-text" id="question"></div>
        </div>
      </div>
      <div class="answers">
        <button class="btn-yes" onclick="answer(1)">Yes</button>
        <button class="btn-no" onclick="answer(0)">No</button>
        <button class="btn-dont" onclick="answer(0.5)">Don't know</button>
        <button class="btn-probably" onclick="answer(0.75)">Probably</button>
        <button class="btn-probably-not" onclick="answer(0.25)">Probably not</button>
      </div>
      <div class="guess-row">
        <button class="btn-guess" onclick="guess()">That's who I was thinking of</button>
      </div>
      <div id="guessResult" class="guess-result hidden"></div>
    </div>
    <div id="error" class="error hidden"></div>
  </div>
  <script>
    let sessionId = null;
    let questionCount = 1;
    function api(path, body) {
      return fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
        .then(function(r) {
          if (!r.ok) return r.json().then(function(j) { throw new Error(j.detail || r.statusText); });
          return r.json();
        });
    }
    function showError(msg) {
      var el = document.getElementById('error');
      el.textContent = msg;
      el.classList.remove('hidden');
    }
    function hideError() {
      document.getElementById('error').classList.add('hidden');
    }
    function setThinking(on) {
      var area = document.getElementById('genieArea');
      if (on) area.classList.add('thinking'); else area.classList.remove('thinking');
    }
    function updateQuestionNum() {
      document.getElementById('questionNum').textContent = 'Question ' + questionCount;
    }
    function startGame() {
      hideError();
      questionCount = 1;
      setThinking(true);
      api('/start_game', {}).then(function(res) {
        setThinking(false);
        sessionId = res.session_id;
        document.getElementById('start').classList.add('hidden');
        document.getElementById('game').classList.remove('hidden');
        document.getElementById('guessResult').classList.add('hidden');
        updateQuestionNum();
        document.getElementById('question').textContent = res.question;
      }).catch(function(e) {
        setThinking(false);
        showError(e.message);
      });
    }
    function answer(value) {
      hideError();
      setThinking(true);
      api('/answer', { session_id: sessionId, answer_value: value }).then(function(res) {
        setThinking(false);
        questionCount += 1;
        updateQuestionNum();
        document.getElementById('question').textContent = res.question || 'No more questions. Click "That\'s who I was thinking of" for my guess.';
        if (res.top_guess) {
          document.getElementById('guessResult').innerHTML = 'I think it\'s <strong>' + res.top_guess + '</strong> (' + Math.round(res.top_guess_probability * 100) + '%)';
          document.getElementById('guessResult').classList.remove('hidden');
        }
      }).catch(function(e) {
        setThinking(false);
        showError(e.message);
      });
    }
    function guess() {
      hideError();
      setThinking(true);
      api('/guess', { session_id: sessionId }).then(function(res) {
        setThinking(false);
        document.getElementById('guessResult').innerHTML = 'I guess: <strong>' + res.name + '</strong> (' + Math.round(res.probability * 100) + '%)';
        document.getElementById('guessResult').classList.remove('hidden');
      }).catch(function(e) {
        setThinking(false);
        showError(e.message);
      });
    }
  </script>
</body>
</html>
"""

# Prefer static file if present (avoids encoding/truncation issues)
_static_index = os.path.join(os.path.dirname(__file__), "..", "static", "index.html")
if os.path.isfile(_static_index):
    with open(_static_index, "r", encoding="utf-8") as _f:
        INDEX_HTML = _f.read()


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=INDEX_HTML)


def _asked_mask(asked_set, M):
    return [1 if i in asked_set else 0 for i in range(M)]


@app.post("/start_game", response_model=StartGameResponse)
def start_game():
    session_id = uuid.uuid4().hex
    # Init state from engine prior (keep on CPU; move to device when calling engine)
    current_probs = engine.prior_probs.cpu().clone()
    asked_set = set()
    asked = _asked_mask(asked_set, engine.M)
    idx, text = engine.get_next_question(
        current_probs.to(engine.device),
        asked,
    )
    if idx is None:
        raise HTTPException(status_code=500, detail="No questions available")
    sessions[session_id] = {
        "current_probs": current_probs,
        "asked_set": asked_set,
        "current_question_idx": idx,
        "resolved_categories": set(),
        "asked_count_per_category": {},
    }
    return StartGameResponse(session_id=session_id, question=text)


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    state = sessions[req.session_id]
    q_idx = state["current_question_idx"]
    if q_idx < 0:
        raise HTTPException(status_code=400, detail="No more questions; use /guess to get the answer")
    current_probs = state["current_probs"].to(engine.device)
    # Bayesian update
    new_probs = engine.update_belief(current_probs, q_idx, req.answer_value)
    state["current_probs"] = new_probs.cpu().clone()
    state["asked_set"].add(q_idx)

    # Expert-system: resolve category on Yes/Probably (never ask that category again)
    cat = engine.get_category_of_question(q_idx)
    if cat is not None:
        state["asked_count_per_category"][cat] = state["asked_count_per_category"].get(cat, 0) + 1
        if req.answer_value >= YES_THRESHOLD:
            state["resolved_categories"].add(cat)

    # Next question (expert-system: resolved + cap per category + no same category twice)
    asked = _asked_mask(state["asked_set"], engine.M)
    idx, text = engine.get_next_question(
        new_probs,
        asked,
        last_asked_idx=q_idx,
        resolved_categories=state["resolved_categories"],
        asked_count_per_category=state["asked_count_per_category"],
    )
    state["current_question_idx"] = idx if idx is not None else -1
    # Top guess if confident
    top = engine.get_top_guess(new_probs, threshold=0.8)
    top_name, top_prob = (top[0], top[1]) if top else (None, None)
    return AnswerResponse(
        question=text or "",
        top_guess=top_name,
        top_guess_probability=top_prob,
    )


@app.post("/guess", response_model=GuessResponse)
def guess(req: GuessRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    state = sessions[req.session_id]
    current_probs = state["current_probs"].to(engine.device)
    top = engine.get_top_guess(current_probs, threshold=0.0)
    if not top:
        raise HTTPException(status_code=500, detail="No guess available")
    return GuessResponse(name=top[0], probability=top[1])
