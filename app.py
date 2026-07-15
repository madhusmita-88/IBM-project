# =============================================================================
#  NutriWise AI - Personalized Nutrition Coach
#  Powered by IBM watsonx.ai Granite Models
#  Multi-Agent Architecture: Nutrition Knowledge | Diet Planner |
#                             Health Advisory | Meal Analysis
# =============================================================================

import os
import json
import requests
from flask import Flask, request, jsonify, render_template_string

# Auto-load IBM watsonx.ai credentials from a .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; env vars can also be set manually

app = Flask(__name__)

# =============================================================================
#  IBM watsonx.ai Configuration
#  Reads credentials from environment variables:
#    WATSONX_API_KEY      - your IBM Cloud API key
#    WATSONX_PROJECT_ID   - your watsonx.ai project ID
#    WATSONX_URL          - inference endpoint (defaults to Dallas region)
# =============================================================================

WATSONX_API_KEY    = os.environ.get("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")
WATSONX_URL        = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# IBM IAM token endpoint
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"


def get_iam_token(api_key: str) -> str:
    """
    Exchange an IBM Cloud API key for a short-lived IAM bearer token.
    This token is used to authenticate every call to the watsonx.ai REST API.
    """
    resp = requests.post(
        IAM_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={api_key}",
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def generate_response(prompt: str, max_tokens: int = 900) -> str:
    """
    Core IBM watsonx.ai call.
    Sends the given prompt to the IBM Granite model and returns the
    generated text.  All four agents funnel through this single function,
    making it easy to swap models or tune parameters in one place.

    Model used: ibm/granite-3-3-8b-instruct
    """
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        return (
            "[WARN] IBM watsonx.ai credentials are not configured. "
            "Please set the WATSONX_API_KEY and WATSONX_PROJECT_ID "
            "environment variables and restart the application."
        )

    try:
        # Step 1 - obtain a fresh IAM bearer token
        token = get_iam_token(WATSONX_API_KEY)

        # Step 2 - build the watsonx.ai text-generation request payload
        payload = {
            "model_id": "ibm/granite-3-3-8b-instruct",
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "repetition_penalty": 1.1,
            },
            "project_id": WATSONX_PROJECT_ID,
        }

        # Step 3 - call the watsonx.ai inference endpoint
        endpoint = f"{WATSONX_URL}/ml/v1/text/generation?version=2023-05-29"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        # Step 4 - extract and return the generated text
        result = response.json()
        return result["results"][0]["generated_text"].strip()

    except requests.exceptions.HTTPError as e:
        return f"[ERROR] API Error: {str(e)}"
    except Exception as e:
        return f"[ERROR] Unexpected error: {str(e)}"


# =============================================================================
#  AGENT 1 - Nutrition Knowledge Agent
#  Answers general nutrition and food-science questions using IBM Granite.
# =============================================================================

def nutrition_knowledge_agent(question: str) -> str:
    """
    Routes a user's nutrition question to IBM watsonx.ai and returns
    a clear, educational answer.
    """
    prompt = f"""You are NutriWise, an expert nutrition scientist and registered dietitian.
Answer the following nutrition question in a clear, educational, and friendly tone.
Structure your answer with:
- A brief direct answer (2-3 sentences)
- Key nutritional facts (use bullet points)
- Practical tips for including this in a healthy diet

Question: {question}

Answer:"""
    # --- IBM watsonx.ai call via Agent 1 ---
    return generate_response(prompt)


# =============================================================================
#  AGENT 2 - Diet Planner Agent
#  Generates a full personalized meal plan based on user profile data.
# =============================================================================

def diet_planner_agent(profile: dict) -> str:
    """
    Builds a structured prompt from the user's health profile and asks
    IBM Granite to generate a full one-day personalised meal plan with
    macro targets.
    """
    prompt = f"""You are NutriWise Diet Planner, a certified sports nutritionist and diet expert.
Create a detailed one-day personalized meal plan for the following person:

Profile:
- Age: {profile.get('age')} years
- Gender: {profile.get('gender')}
- Height: {profile.get('height')} cm
- Weight: {profile.get('weight')} kg
- Dietary Preference: {profile.get('diet_type')}
- Activity Level: {profile.get('activity_level')}
- Fitness Goal: {profile.get('goal')}

Generate a complete structured response with the following sections:

## 📊 Daily Nutritional Targets
Provide:
- Estimated Daily Calorie Target (kcal)
- Protein Recommendation (grams)
- Carbohydrate Recommendation (grams)
- Fat Recommendation (grams)

## 🌅 Breakfast
List 3-4 specific food items with approximate portions.

## 🌞 Mid-Morning Snack
List 2-3 items.

## 🍽️ Lunch
List 4-5 specific food items with approximate portions.

## 🍎 Evening Snack
List 2-3 items.

## 🌙 Dinner
List 3-4 specific food items with approximate portions.

## 💧 Hydration & Supplements
Briefly mention water intake and any relevant supplements.

## ✅ Tips for Success
3 personalized tips based on the goal.

Meal Plan:"""
    # --- IBM watsonx.ai call via Agent 2 ---
    return generate_response(prompt, max_tokens=1100)


# =============================================================================
#  AGENT 3 - Health Advisory Agent
#  Provides disease-specific dietary guidance using IBM Granite.
# =============================================================================

def health_advisory_agent(conditions: list) -> str:
    """
    Receives a list of health conditions and produces tailored dietary
    recommendations via IBM watsonx.ai, including mandatory disclaimer.
    """
    conditions_str = ", ".join(conditions) if conditions else "General Health"

    prompt = f"""You are NutriWise Health Advisor, a clinical nutrition expert specializing in
disease-specific dietary management.

The user has the following health condition(s): {conditions_str}

Provide detailed, evidence-based dietary and lifestyle recommendations structured as:

## 🟢 Foods to Include
List 8-10 specific foods that are beneficial, with a brief reason for each.

## 🔴 Foods to Avoid
List 6-8 foods or food categories to strictly limit or avoid, with reasons.

## 🏃 Healthy Habits & Lifestyle Recommendations
List 5-6 actionable lifestyle tips tailored to these conditions.

## 📋 Sample Daily Eating Pattern
Suggest a brief outline for breakfast, lunch, and dinner.

## [WARN] Important Disclaimer
Always end with: "This is educational information only. Please consult a qualified
healthcare professional or registered dietitian before making any dietary changes,
especially for managing medical conditions."

Health Advisory:"""
    # --- IBM watsonx.ai call via Agent 3 ---
    return generate_response(prompt, max_tokens=1000)


# =============================================================================
#  AGENT 4 - Meal Analysis Agent
#  Analyses a user's free-text meal diary and returns nutritional feedback.
# =============================================================================

def meal_analysis_agent(meal_text: str) -> str:
    """
    Parses the user's free-text meal description, sends it to IBM Granite,
    and returns a structured nutritional assessment with improvement tips.
    """
    prompt = f"""You are NutriWise Meal Analyst, an expert nutritionist specialising in
dietary assessment and meal optimisation.

Analyse the following meal diary entry and provide a comprehensive nutritional review:

--- Meal Diary ---
{meal_text}
--- End of Diary ---

Provide a structured analysis with:

## 📊 Overall Nutritional Assessment
Rate the overall diet quality (Excellent / Good / Fair / Needs Improvement) and
give a 2-3 sentence summary.

## ✅ Nutritional Strengths
List 4-5 things the person is doing well in their diet.

## [WARN] Nutritional Gaps & Deficiencies
List potential nutrient gaps or imbalances observed.

## 🔄 Healthier Alternatives
For any less healthy items, suggest specific healthier swaps.

## 💡 Personalised Improvement Recommendations
List 5 actionable, specific recommendations to improve this diet.

## 🎯 Estimated Nutritional Balance
Give a rough estimate of whether the meals provide adequate:
- Protein
- Carbohydrates
- Healthy Fats
- Fibre
- Key Vitamins & Minerals

Meal Analysis:"""
    # --- IBM watsonx.ai call via Agent 4 ---
    return generate_response(prompt, max_tokens=1000)


# =============================================================================
#  AGENT ORCHESTRATOR
#  Routes each user request to the correct specialist agent.
# =============================================================================

def orchestrator(agent_name: str, payload: dict) -> str:
    """
    Central dispatcher. Examines the requested agent and delegates to
    the appropriate agent function.

    Supported agents:
      'nutrition_knowledge' -> nutrition_knowledge_agent()
      'diet_planner'        -> diet_planner_agent()
      'health_advisory'     -> health_advisory_agent()
      'meal_analysis'       -> meal_analysis_agent()
    """
    if agent_name == "nutrition_knowledge":
        return nutrition_knowledge_agent(payload.get("question", ""))

    elif agent_name == "diet_planner":
        return diet_planner_agent(payload)

    elif agent_name == "health_advisory":
        return health_advisory_agent(payload.get("conditions", []))

    elif agent_name == "meal_analysis":
        return meal_analysis_agent(payload.get("meal_text", ""))

    else:
        return "[ERROR] Unknown agent. Please select a valid feature."


# =============================================================================
#  HTML TEMPLATE  (Bootstrap 5 - all markup inside app.py)
# =============================================================================

BASE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>NutriWise AI - Personalized Nutrition Coach</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet"/>
  <style>
    :root {
      --brand-green:#2e7d32;--brand-light:#e8f5e9;--brand-accent:#43a047;
      --sidebar-w:260px;--sidebar-bg:#1b5e20;--text-muted-custom:#546e7a;
    }
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:"Segoe UI",system-ui,sans-serif;background:#f0f4f0;color:#1a2e1a;min-height:100vh;display:flex}

    /* ── Sidebar ── */
    #sidebar{width:var(--sidebar-w);min-height:100vh;background:var(--sidebar-bg);
             display:flex;flex-direction:column;flex-shrink:0;position:fixed;top:0;left:0;z-index:100;
             transition:transform .3s ease}
    #sidebar .brand{padding:24px 20px 16px;border-bottom:1px solid rgba(255,255,255,.15)}
    #sidebar .brand h4{color:#fff;font-weight:700;font-size:1.15rem;line-height:1.3}
    #sidebar .brand small{color:#a5d6a7;font-size:.75rem}
    #sidebar nav{padding:12px 0;flex:1}
    #sidebar nav a{display:flex;align-items:center;gap:10px;padding:11px 22px;
                   color:#c8e6c9;text-decoration:none;font-size:.9rem;border-left:3px solid transparent;
                   transition:all .2s}
    #sidebar nav a:hover,#sidebar nav a.active{background:rgba(255,255,255,.1);
      color:#fff;border-left-color:#69f0ae}
    #sidebar nav a i{font-size:1.1rem;width:20px;text-align:center}
    #sidebar .powered{padding:14px 20px;border-top:1px solid rgba(255,255,255,.12);
                      color:#81c784;font-size:.72rem;text-align:center}

    /* ── Main content ── */
    #main{margin-left:var(--sidebar-w);flex:1;display:flex;flex-direction:column;min-height:100vh}
    .topbar{background:#fff;border-bottom:1px solid #dce8dc;padding:14px 28px;
            display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:90}
    .topbar h5{font-weight:600;color:var(--brand-green);margin:0}
    .topbar .badge-ibm{background:#0530ad;color:#fff;font-size:.7rem;padding:4px 10px;border-radius:20px}
    .content{padding:28px;flex:1}

    /* ── Cards ── */
    .card{border:none;border-radius:14px;box-shadow:0 2px 12px rgba(0,0,0,.07)}
    .card-header-green{background:linear-gradient(135deg,var(--brand-green),var(--brand-accent));
                       color:#fff;border-radius:14px 14px 0 0!important;padding:16px 22px}
    .agent-card{transition:transform .2s,box-shadow .2s;cursor:pointer}
    .agent-card:hover{transform:translateY(-4px);box-shadow:0 8px 24px rgba(46,125,50,.18)}
    .agent-icon{width:56px;height:56px;border-radius:14px;display:flex;align-items:center;
               justify-content:center;font-size:1.6rem;margin-bottom:14px}

    /* ── Chat / output area ── */
    .output-box{background:#fff;border:1px solid #dce8dc;border-radius:12px;padding:20px;
                min-height:180px;white-space:pre-wrap;line-height:1.75;font-size:.92rem;
                color:#263238;max-height:540px;overflow-y:auto}
    .output-box h2,
    .output-box h3{color:var(--brand-green);margin-top:14px;margin-bottom:6px;font-size:1rem;font-weight:700}
    .output-box ul{padding-left:18px}
    .output-box li{margin-bottom:4px}

    /* ── Forms ── */
    .form-label{font-weight:600;font-size:.85rem;color:#37474f}
    .form-control,.form-select{border-color:#c8e6c9;font-size:.9rem}
    .form-control:focus,.form-select:focus{border-color:var(--brand-accent);
      box-shadow:0 0 0 .2rem rgba(67,160,71,.2)}
    .btn-nutriwise{background:var(--brand-green);color:#fff;border:none;border-radius:8px;
                   padding:10px 24px;font-weight:600;font-size:.9rem;transition:background .2s}
    .btn-nutriwise:hover{background:#1b5e20;color:#fff}
    .btn-nutriwise:disabled{background:#9e9e9e}

    /* ── Condition checkboxes ── */
    .condition-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    .condition-item{background:#f1f8e9;border:1px solid #c5e1a5;border-radius:10px;
                    padding:10px 14px;cursor:pointer;transition:all .2s}
    .condition-item:hover,.condition-item.selected{background:#e8f5e9;border-color:var(--brand-accent)}
    .condition-item input{display:none}

    /* ── Spinner ── */
    .spinner-wrap{display:none;text-align:center;padding:28px}

    /* ── Home hero ── */
    .hero{background:linear-gradient(135deg,#1b5e20 0%,#2e7d32 50%,#388e3c 100%);
          color:#fff;border-radius:16px;padding:40px;margin-bottom:28px}
    .hero h1{font-size:2rem;font-weight:800}
    .hero p{opacity:.88;font-size:1rem}

    /* ── Responsive ── */
    @media(max-width:768px){
      #sidebar{transform:translateX(-100%)}
      #sidebar.open{transform:translateX(0)}
      #main{margin-left:0}
      .condition-grid{grid-template-columns:1fr}
    }

    /* ── Markdown-style output formatting ── */
    .md-output h2{font-size:1rem;font-weight:700;color:#2e7d32;margin:16px 0 6px;
                  padding-bottom:4px;border-bottom:1px solid #e8f5e9}
    .md-output strong{color:#1b5e20}
    .md-output ul{padding-left:20px;margin-bottom:10px}
    .md-output li{margin-bottom:3px}
  </style>
</head>
<body>

<!-- ═══════════════ SIDEBAR ═══════════════ -->
<div id="sidebar">
  <div class="brand">
    <h4><i class="bi bi-activity me-2"></i>NutriWise AI</h4>
    <small>Personalized Nutrition Coach</small>
  </div>
  <nav>
    <a href="/"         class="{% if page=='home'    %}active{% endif %}"><i class="bi bi-house-fill"></i>Home</a>
    <a href="/nutrition-chat" class="{% if page=='chat'    %}active{% endif %}"><i class="bi bi-chat-dots-fill"></i>Nutrition Chat</a>
    <a href="/diet-planner"   class="{% if page=='planner' %}active{% endif %}"><i class="bi bi-calendar2-heart-fill"></i>Diet Planner</a>
    <a href="/health-advisor" class="{% if page=='advisor' %}active{% endif %}"><i class="bi bi-heart-pulse-fill"></i>Health Advisor</a>
    <a href="/meal-analyzer"  class="{% if page=='analyzer'%}active{% endif %}"><i class="bi bi-search-heart-fill"></i>Meal Analyzer</a>
    <a href="/about"          class="{% if page=='about'   %}active{% endif %}"><i class="bi bi-info-circle-fill"></i>About</a>
  </nav>
  <div class="powered">
    <i class="bi bi-cpu-fill me-1"></i>Powered by IBM watsonx.ai<br/>Granite Models
  </div>
</div>

<!-- ═══════════════ MAIN ═══════════════ -->
<div id="main">
  <div class="topbar">
    <h5><i class="bi bi-activity me-2"></i>{{ title }}</h5>
    <span class="badge-ibm"><i class="bi bi-cpu me-1"></i>IBM watsonx.ai · Granite</span>
  </div>
  <div class="content">
    {{ body | safe }}
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
/* ── Markdown-to-HTML lightweight renderer ── */
function renderMarkdown(text){
  return text
    .replace(/## (.*)/g,'<h2>$1</h2>')
    .replace(/### (.*)/g,'<h3>$1</h3>')
    .replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>')
    .replace(/\*(.*?)\*/g,'<em>$1</em>')
    .replace(/^- (.*)/gm,'<li>$1</li>')
    .replace(/(<li>.*<\/li>)/gs, s => '<ul>'+s+'</ul>')
    .replace(/\n{2,}/g,'<br/><br/>')
    .replace(/\n/g,'<br/>');
}

/* ── Generic async AI call ── */
async function callAgent(endpoint, body, outputId, spinnerId, btnId){
  const out = document.getElementById(outputId);
  const sp  = document.getElementById(spinnerId);
  const btn = document.getElementById(btnId);
  out.innerHTML = '';
  sp.style.display = 'block';
  btn.disabled = true;
  try {
    const res  = await fetch(endpoint, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const data = await res.json();
    out.innerHTML = '<div class="md-output">'+renderMarkdown(data.result || data.error)+'</div>';
  } catch(e){
    out.innerHTML = '<span class="text-danger">Network error. Please try again.</span>';
  } finally {
    sp.style.display = 'none';
    btn.disabled = false;
    out.scrollIntoView({behavior:'smooth',block:'nearest'});
  }
}
</script>
{{ extra_js | safe }}
</body>
</html>"""


def render_page(page: str, title: str, body: str, extra_js: str = "") -> str:
    return render_template_string(
        BASE_HTML,
        page=page,
        title=title,
        body=body,
        extra_js=extra_js,
    )


# =============================================================================
#  FLASK ROUTES
# =============================================================================

# ── Home Page ──────────────────────────────────────────────────────────────
@app.route("/")
def home():
    body = """
    <div class="hero">
      <div class="row align-items-center">
        <div class="col-md-8">
          <h1><i class="bi bi-activity me-3"></i>NutriWise AI</h1>
          <p class="mt-2 mb-3" style="font-size:1.05rem">Your AI-powered Personalized Nutrition Coach —
          built on IBM watsonx.ai Granite Models with a four-agent architecture.</p>
          <a href="/nutrition-chat" class="btn btn-light btn-sm fw-600 me-2">
            <i class="bi bi-chat-dots me-1"></i>Ask a Nutrition Question</a>
          <a href="/diet-planner" class="btn btn-outline-light btn-sm">
            <i class="bi bi-calendar2-heart me-1"></i>Create My Meal Plan</a>
        </div>
        <div class="col-md-4 text-end d-none d-md-block" style="font-size:4rem;opacity:.5">🥗</div>
      </div>
    </div>

    <!-- Agent Cards -->
    <div class="row g-4 mb-4">
      <div class="col-md-6 col-lg-3">
        <a href="/nutrition-chat" class="text-decoration-none">
          <div class="card agent-card h-100 p-3">
            <div class="agent-icon" style="background:#e8f5e9;color:#2e7d32">🧠</div>
            <h6 class="fw-700">Nutrition Knowledge</h6>
            <p class="text-muted small mb-0">Ask any nutrition or food science question and receive
            evidence-based answers from our AI expert.</p>
          </div>
        </a>
      </div>
      <div class="col-md-6 col-lg-3">
        <a href="/diet-planner" class="text-decoration-none">
          <div class="card agent-card h-100 p-3">
            <div class="agent-icon" style="background:#e3f2fd;color:#1565c0">📅</div>
            <h6 class="fw-700">Diet Planner</h6>
            <p class="text-muted small mb-0">Get a complete personalised meal plan with calorie and
            macro targets tailored to your profile and goal.</p>
          </div>
        </a>
      </div>
      <div class="col-md-6 col-lg-3">
        <a href="/health-advisor" class="text-decoration-none">
          <div class="card agent-card h-100 p-3">
            <div class="agent-icon" style="background:#fce4ec;color:#c62828">❤️</div>
            <h6 class="fw-700">Health Advisor</h6>
            <p class="text-muted small mb-0">Receive disease-specific dietary guidance for diabetes,
            hypertension, heart disease, and more.</p>
          </div>
        </a>
      </div>
      <div class="col-md-6 col-lg-3">
        <a href="/meal-analyzer" class="text-decoration-none">
          <div class="card agent-card h-100 p-3">
            <div class="agent-icon" style="background:#fff3e0;color:#e65100">🔍</div>
            <h6 class="fw-700">Meal Analyzer</h6>
            <p class="text-muted small mb-0">Enter today's meals and get an in-depth nutritional
            assessment with personalised improvement tips.</p>
          </div>
        </a>
      </div>
    </div>

    <!-- Stats row -->
    <div class="row g-3">
      <div class="col-6 col-md-3">
        <div class="card text-center p-3">
          <div style="font-size:1.8rem">4</div>
          <div class="text-muted small">AI Agents</div>
        </div>
      </div>
      <div class="col-6 col-md-3">
        <div class="card text-center p-3">
          <div style="font-size:1.8rem">🤖</div>
          <div class="text-muted small">IBM Granite Model</div>
        </div>
      </div>
      <div class="col-6 col-md-3">
        <div class="card text-center p-3">
          <div style="font-size:1.8rem">∞</div>
          <div class="text-muted small">Personalised Answers</div>
        </div>
      </div>
      <div class="col-6 col-md-3">
        <div class="card text-center p-3">
          <div style="font-size:1.8rem">🔒</div>
          <div class="text-muted small">Secure &amp; Private</div>
        </div>
      </div>
    </div>
    """
    return render_page("home", "Home - NutriWise AI", body)


# ── Nutrition Chat Page ────────────────────────────────────────────────────
@app.route("/nutrition-chat")
def nutrition_chat():
    body = """
    <div class="row g-4">
      <div class="col-lg-5">
        <div class="card h-100">
          <div class="card-header-green">
            <h6 class="mb-0 fw-700"><i class="bi bi-chat-dots me-2"></i>Ask Nutrition Questions</h6>
            <small style="opacity:.85">Powered by IBM watsonx.ai · Agent 1</small>
          </div>
          <div class="card-body p-4">
            <label class="form-label">Your nutrition question</label>
            <textarea id="q_input" class="form-control" rows="4"
              placeholder="e.g. What are the benefits of oats?&#10;Which foods are rich in protein?&#10;Is paneer healthy for muscle gain?"></textarea>

            <div class="mt-3">
              <p class="form-label mb-2">Quick questions:</p>
              <div class="d-flex flex-wrap gap-2">
                <button class="btn btn-sm btn-outline-success" onclick="setQ('What are the health benefits of eating oats daily?')">Oats benefits</button>
                <button class="btn btn-sm btn-outline-success" onclick="setQ('Which plant-based foods are rich in protein?')">Plant protein</button>
                <button class="btn btn-sm btn-outline-success" onclick="setQ('What foods contain Vitamin B12?')">Vitamin B12</button>
                <button class="btn btn-sm btn-outline-success" onclick="setQ('Is paneer healthy for muscle gain?')">Paneer &amp; muscles</button>
                <button class="btn btn-sm btn-outline-success" onclick="setQ('What are superfoods and do they really work?')">Superfoods</button>
              </div>
            </div>

            <div class="spinner-wrap" id="chat_spinner">
              <div class="spinner-border text-success" role="status"></div>
              <p class="mt-2 text-muted small">IBM Granite is thinking...</p>
            </div>
            <button class="btn btn-nutriwise mt-3 w-100" id="chat_btn" onclick="askNutrition()">
              <i class="bi bi-send me-2"></i>Get AI Answer
            </button>
          </div>
        </div>
      </div>
      <div class="col-lg-7">
        <div class="card h-100">
          <div class="card-header-green">
            <h6 class="mb-0 fw-700"><i class="bi bi-robot me-2"></i>NutriWise AI Response</h6>
          </div>
          <div class="card-body p-4">
            <div class="output-box" id="chat_output">
              <span class="text-muted">Your AI-generated nutrition answer will appear here...</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    """
    js = """
    <script>
    function setQ(q){ document.getElementById('q_input').value = q; }
    function askNutrition(){
      const q = document.getElementById('q_input').value.trim();
      if(!q){ alert('Please enter a question.'); return; }
      callAgent('/api/nutrition-knowledge', {question:q}, 'chat_output', 'chat_spinner', 'chat_btn');
    }
    </script>"""
    return render_page("chat", "Nutrition Chat - Agent 1", body, js)


# ── Diet Planner Page ──────────────────────────────────────────────────────
@app.route("/diet-planner")
def diet_planner():
    body = """
    <div class="row g-4">
      <div class="col-lg-5">
        <div class="card">
          <div class="card-header-green">
            <h6 class="mb-0 fw-700"><i class="bi bi-calendar2-heart me-2"></i>Your Health Profile</h6>
            <small style="opacity:.85">Powered by IBM watsonx.ai · Agent 2</small>
          </div>
          <div class="card-body p-4">
            <div class="row g-3">
              <div class="col-6">
                <label class="form-label">Age (years)</label>
                <input type="number" id="dp_age" class="form-control" placeholder="25" min="10" max="100"/>
              </div>
              <div class="col-6">
                <label class="form-label">Gender</label>
                <select id="dp_gender" class="form-select">
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
              <div class="col-6">
                <label class="form-label">Height (cm)</label>
                <input type="number" id="dp_height" class="form-control" placeholder="170" min="100" max="250"/>
              </div>
              <div class="col-6">
                <label class="form-label">Weight (kg)</label>
                <input type="number" id="dp_weight" class="form-control" placeholder="70" min="30" max="300"/>
              </div>
              <div class="col-12">
                <label class="form-label">Dietary Preference</label>
                <select id="dp_diet" class="form-select">
                  <option value="Vegetarian">Vegetarian</option>
                  <option value="Vegan">Vegan</option>
                  <option value="Non-Vegetarian">Non-Vegetarian</option>
                  <option value="Eggetarian">Eggetarian</option>
                  <option value="Gluten-Free">Gluten-Free</option>
                </select>
              </div>
              <div class="col-12">
                <label class="form-label">Activity Level</label>
                <select id="dp_activity" class="form-select">
                  <option value="Sedentary (little or no exercise)">Sedentary</option>
                  <option value="Lightly Active (1-3 days/week)">Lightly Active</option>
                  <option value="Moderately Active (3-5 days/week)">Moderately Active</option>
                  <option value="Very Active (6-7 days/week)">Very Active</option>
                  <option value="Super Active (physical job + training)">Super Active</option>
                </select>
              </div>
              <div class="col-12">
                <label class="form-label">Fitness Goal</label>
                <select id="dp_goal" class="form-select">
                  <option value="Weight Loss">Weight Loss</option>
                  <option value="Weight Gain">Weight Gain</option>
                  <option value="Muscle Gain">Muscle Gain</option>
                  <option value="General Wellness">General Wellness</option>
                  <option value="Athletic Performance">Athletic Performance</option>
                  <option value="Improve Energy Levels">Improve Energy Levels</option>
                </select>
              </div>
            </div>
            <div class="spinner-wrap" id="dp_spinner">
              <div class="spinner-border text-success" role="status"></div>
              <p class="mt-2 text-muted small">IBM Granite is creating your meal plan...</p>
            </div>
            <button class="btn btn-nutriwise mt-3 w-100" id="dp_btn" onclick="generatePlan()">
              <i class="bi bi-stars me-2"></i>Generate My Meal Plan
            </button>
          </div>
        </div>
      </div>
      <div class="col-lg-7">
        <div class="card h-100">
          <div class="card-header-green">
            <h6 class="mb-0 fw-700"><i class="bi bi-journal-check me-2"></i>Your Personalised Meal Plan</h6>
          </div>
          <div class="card-body p-4">
            <div class="output-box" id="dp_output">
              <span class="text-muted">Fill in your profile and click "Generate My Meal Plan" to receive a personalised plan from IBM Granite...</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    """
    js = """
    <script>
    function generatePlan(){
      const age    = document.getElementById('dp_age').value;
      const gender = document.getElementById('dp_gender').value;
      const height = document.getElementById('dp_height').value;
      const weight = document.getElementById('dp_weight').value;
      if(!age||!height||!weight){ alert('Please fill in Age, Height, and Weight.'); return; }
      callAgent('/api/diet-planner', {
        age, gender, height, weight,
        diet_type: document.getElementById('dp_diet').value,
        activity_level: document.getElementById('dp_activity').value,
        goal: document.getElementById('dp_goal').value
      }, 'dp_output', 'dp_spinner', 'dp_btn');
    }
    </script>"""
    return render_page("planner", "Diet Planner - Agent 2", body, js)


# ── Health Advisor Page ────────────────────────────────────────────────────
@app.route("/health-advisor")
def health_advisor():
    conditions = [
        ("Diabetes",        "🩸"),
        ("Hypertension",    "💊"),
        ("Obesity",         "⚖️"),
        ("Heart Disease",   "❤️"),
        ("PCOS",            "🌸"),
        ("High Cholesterol","🔬"),
    ]
    cond_html = ""
    for name, icon in conditions:
        cond_html += f"""
        <div class="condition-item" id="cond_{name.replace(' ','_')}" onclick="toggleCond(this,'{name}')">
          <div class="d-flex align-items-center gap-2">
            <span style="font-size:1.3rem">{icon}</span>
            <span class="fw-600" style="font-size:.88rem">{name}</span>
          </div>
        </div>"""

    body = f"""
    <div class="row g-4">
      <div class="col-lg-5">
        <div class="card">
          <div class="card-header-green">
            <h6 class="mb-0 fw-700"><i class="bi bi-heart-pulse me-2"></i>Select Health Conditions</h6>
            <small style="opacity:.85">Powered by IBM watsonx.ai · Agent 3</small>
          </div>
          <div class="card-body p-4">
            <p class="text-muted small mb-3">Select one or more conditions to receive personalised dietary guidance:</p>
            <div class="condition-grid mb-3" id="conditions_grid">
              {cond_html}
            </div>
            <div id="selected_display" class="text-muted small mb-3">No conditions selected.</div>
            <div class="spinner-wrap" id="ha_spinner">
              <div class="spinner-border text-success" role="status"></div>
              <p class="mt-2 text-muted small">IBM Granite is generating health recommendations...</p>
            </div>
            <button class="btn btn-nutriwise w-100" id="ha_btn" onclick="getAdvice()">
              <i class="bi bi-shield-heart me-2"></i>Get Health Recommendations
            </button>
            <div class="alert alert-warning mt-3 p-2" style="font-size:.78rem">
              <i class="bi bi-exclamation-triangle me-1"></i>
              Educational use only. Always consult a qualified healthcare professional.
            </div>
          </div>
        </div>
      </div>
      <div class="col-lg-7">
        <div class="card h-100">
          <div class="card-header-green">
            <h6 class="mb-0 fw-700"><i class="bi bi-clipboard2-pulse me-2"></i>Health Advisory Report</h6>
          </div>
          <div class="card-body p-4">
            <div class="output-box" id="ha_output">
              <span class="text-muted">Select health conditions and click "Get Health Recommendations" for AI-powered dietary guidance...</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    """
    js = """
    <script>
    let selectedConditions = [];
    function toggleCond(el, name){
      el.classList.toggle('selected');
      if(selectedConditions.includes(name)){
        selectedConditions = selectedConditions.filter(c => c !== name);
      } else {
        selectedConditions.push(name);
      }
      const disp = document.getElementById('selected_display');
      disp.textContent = selectedConditions.length
        ? 'Selected: ' + selectedConditions.join(', ')
        : 'No conditions selected.';
    }
    function getAdvice(){
      if(!selectedConditions.length){ alert('Please select at least one health condition.'); return; }
      callAgent('/api/health-advisory', {conditions: selectedConditions},
                'ha_output', 'ha_spinner', 'ha_btn');
    }
    </script>"""
    return render_page("advisor", "Health Advisor - Agent 3", body, js)


# ── Meal Analyzer Page ─────────────────────────────────────────────────────
@app.route("/meal-analyzer")
def meal_analyzer():
    body = """
    <div class="row g-4">
      <div class="col-lg-5">
        <div class="card">
          <div class="card-header-green">
            <h6 class="mb-0 fw-700"><i class="bi bi-search-heart me-2"></i>Enter Today's Meals</h6>
            <small style="opacity:.85">Powered by IBM watsonx.ai · Agent 4</small>
          </div>
          <div class="card-body p-4">
            <label class="form-label">Describe your meals (free text)</label>
            <textarea id="ma_input" class="form-control" rows="10"
              placeholder="Breakfast:&#10;2 rotis with ghee&#10;1 cup dal&#10;1 glass milk&#10;&#10;Lunch:&#10;2 cups rice&#10;Paneer butter masala&#10;Salad&#10;&#10;Snack:&#10;Samosa (2)&#10;Chai&#10;&#10;Dinner:&#10;Chapati (2)&#10;Mixed vegetable curry&#10;Curd"></textarea>

            <div class="mt-3">
              <p class="form-label mb-2">Load sample meal:</p>
              <div class="d-flex flex-wrap gap-2">
                <button class="btn btn-sm btn-outline-success" onclick="loadSample('veg')">Indian Vegetarian</button>
                <button class="btn btn-sm btn-outline-success" onclick="loadSample('nonveg')">Non-Vegetarian</button>
                <button class="btn btn-sm btn-outline-success" onclick="loadSample('western')">Western Diet</button>
              </div>
            </div>

            <div class="spinner-wrap" id="ma_spinner">
              <div class="spinner-border text-success" role="status"></div>
              <p class="mt-2 text-muted small">IBM Granite is analysing your meals...</p>
            </div>
            <button class="btn btn-nutriwise mt-3 w-100" id="ma_btn" onclick="analyzeMeal()">
              <i class="bi bi-bar-chart me-2"></i>Analyse My Meals
            </button>
          </div>
        </div>
      </div>
      <div class="col-lg-7">
        <div class="card h-100">
          <div class="card-header-green">
            <h6 class="mb-0 fw-700"><i class="bi bi-clipboard2-data me-2"></i>Nutritional Analysis Report</h6>
          </div>
          <div class="card-body p-4">
            <div class="output-box" id="ma_output">
              <span class="text-muted">Enter your meals and click "Analyse My Meals" to receive a detailed nutritional breakdown...</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    """
    js = r"""
    <script>
    const samples = {
      veg: `Breakfast:\n2 idli with sambar\n1 cup filter coffee with milk\n\nLunch:\n1.5 cups rice\nDal tadka\nPalak sabzi\nCurd\nPickle\n\nSnack:\n1 banana\nHandful of almonds (10 pieces)\n\nDinner:\n3 chapati\nChana masala\nGreen salad`,
      nonveg: `Breakfast:\n3 scrambled eggs\n2 slices white bread toast\n1 cup tea with sugar\n\nLunch:\n2 cups biryani (chicken)\nRaita\n\nSnack:\nPacket of chips\nCold drink (Pepsi 300ml)\n\nDinner:\n2 chapati\nChicken curry\nDal`,
      western: `Breakfast:\nBowl of cornflakes with full-fat milk\nOrange juice (packaged, 200ml)\n\nLunch:\nChicken burger with fries\nDiet Coke\n\nSnack:\nChocolate bar\nCoffee with cream and sugar\n\nDinner:\nPasta with Alfredo sauce\nGarlic bread (2 slices)\nSoda`
    };
    function loadSample(type){ document.getElementById('ma_input').value = samples[type].replace(/\\n/g,'\n'); }
    function analyzeMeal(){
      const meal = document.getElementById('ma_input').value.trim();
      if(!meal){ alert('Please enter your meal details.'); return; }
      callAgent('/api/meal-analysis', {meal_text: meal}, 'ma_output', 'ma_spinner', 'ma_btn');
    }
    </script>"""
    return render_page("analyzer", "Meal Analyzer - Agent 4", body, js)


# ── About Page ─────────────────────────────────────────────────────────────
@app.route("/about")
def about():
    body = """
    <div class="row g-4">
      <div class="col-12">
        <div class="hero" style="padding:32px">
          <h2 class="mb-2"><i class="bi bi-info-circle me-2"></i>About NutriWise AI</h2>
          <p class="mb-0">A multi-agent AI nutrition assistant built on IBM watsonx.ai Granite Models.
          Designed for IBM hackathons, SkillsBuild showcases, and AI project demonstrations.</p>
        </div>
      </div>

      <!-- Architecture -->
      <div class="col-12">
        <div class="card">
          <div class="card-header-green">
            <h6 class="mb-0 fw-700"><i class="bi bi-diagram-3 me-2"></i>Multi-Agent Architecture</h6>
          </div>
          <div class="card-body p-4">
            <div class="row g-3">
              <div class="col-md-6">
                <div class="card border-success h-100">
                  <div class="card-body p-3">
                    <h6 class="fw-700 text-success"><span class="badge bg-success me-2">Agent 1</span>Nutrition Knowledge Agent</h6>
                    <p class="small text-muted mb-1">Answers general nutrition questions using IBM Granite.</p>
                    <small class="text-muted"><strong>Input:</strong> Free-text nutrition question<br/>
                    <strong>Output:</strong> Educational AI-generated answer with bullet points</small>
                  </div>
                </div>
              </div>
              <div class="col-md-6">
                <div class="card border-primary h-100">
                  <div class="card-body p-3">
                    <h6 class="fw-700 text-primary"><span class="badge bg-primary me-2">Agent 2</span>Diet Planner Agent</h6>
                    <p class="small text-muted mb-1">Generates personalised meal plans based on user profile.</p>
                    <small class="text-muted"><strong>Input:</strong> Age, gender, height, weight, goal, diet preference<br/>
                    <strong>Output:</strong> Full daily meal plan + macro targets</small>
                  </div>
                </div>
              </div>
              <div class="col-md-6">
                <div class="card border-danger h-100">
                  <div class="card-body p-3">
                    <h6 class="fw-700 text-danger"><span class="badge bg-danger me-2">Agent 3</span>Health Advisory Agent</h6>
                    <p class="small text-muted mb-1">Provides disease-specific dietary guidance.</p>
                    <small class="text-muted"><strong>Input:</strong> Selected health conditions<br/>
                    <strong>Output:</strong> Foods to include/avoid + lifestyle tips</small>
                  </div>
                </div>
              </div>
              <div class="col-md-6">
                <div class="card border-warning h-100">
                  <div class="card-body p-3">
                    <h6 class="fw-700 text-warning"><span class="badge bg-warning text-dark me-2">Agent 4</span>Meal Analysis Agent</h6>
                    <p class="small text-muted mb-1">Analyses a free-text meal diary and returns feedback.</p>
                    <small class="text-muted"><strong>Input:</strong> Free-text meal description<br/>
                    <strong>Output:</strong> Nutritional assessment + improvement suggestions</small>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Orchestrator -->
      <div class="col-md-6">
        <div class="card h-100">
          <div class="card-header-green">
            <h6 class="mb-0 fw-700"><i class="bi bi-arrow-left-right me-2"></i>Agent Orchestrator</h6>
          </div>
          <div class="card-body p-3">
            <p class="small text-muted">A central <code>orchestrator()</code> function receives each
            request, identifies the target agent, and delegates to the appropriate specialist function.
            All agents share a single <code>generate_response()</code> function that calls the IBM
            watsonx.ai REST API.</p>
            <pre class="bg-light p-3 rounded" style="font-size:.78rem">
orchestrator(agent, payload)
  ├── nutrition_knowledge_agent()
  ├── diet_planner_agent()
  ├── health_advisory_agent()
  └── meal_analysis_agent()
        └── generate_response(prompt)
              └── IBM watsonx.ai Granite</pre>
          </div>
        </div>
      </div>

      <!-- IBM watsonx.ai -->
      <div class="col-md-6">
        <div class="card h-100">
          <div class="card-header-green">
            <h6 class="mb-0 fw-700"><i class="bi bi-cpu me-2"></i>IBM watsonx.ai Integration</h6>
          </div>
          <div class="card-body p-3">
            <ul class="small text-muted ps-3">
              <li class="mb-2"><strong>Model:</strong> ibm/granite-3-3-8b-instruct</li>
              <li class="mb-2"><strong>Auth:</strong> IBM IAM Bearer Token (auto-refreshed)</li>
              <li class="mb-2"><strong>Endpoint:</strong> watsonx.ai Inference REST API v1</li>
              <li class="mb-2"><strong>Config:</strong> Environment variables — <code>WATSONX_API_KEY</code>,
              <code>WATSONX_PROJECT_ID</code>, <code>WATSONX_URL</code></li>
              <li><strong>Decoding:</strong> Greedy · Temperature 0.7 · Repetition Penalty 1.1</li>
            </ul>
          </div>
        </div>
      </div>

      <!-- Tech Stack -->
      <div class="col-12">
        <div class="card">
          <div class="card-header-green">
            <h6 class="mb-0 fw-700"><i class="bi bi-stack me-2"></i>Technology Stack</h6>
          </div>
          <div class="card-body p-3">
            <div class="row g-2 text-center">
              <div class="col-6 col-md-2">
                <div class="card border-0 bg-light p-2"><small class="fw-600">Python 3.10+</small></div>
              </div>
              <div class="col-6 col-md-2">
                <div class="card border-0 bg-light p-2"><small class="fw-600">Flask</small></div>
              </div>
              <div class="col-6 col-md-2">
                <div class="card border-0 bg-light p-2"><small class="fw-600">Bootstrap 5.3</small></div>
              </div>
              <div class="col-6 col-md-2">
                <div class="card border-0 bg-light p-2"><small class="fw-600">JavaScript ES6</small></div>
              </div>
              <div class="col-6 col-md-2">
                <div class="card border-0 bg-light p-2"><small class="fw-600">IBM watsonx.ai</small></div>
              </div>
              <div class="col-6 col-md-2">
                <div class="card border-0 bg-light p-2"><small class="fw-600">IBM Granite LLM</small></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    """
    return render_page("about", "About NutriWise AI", body)


# =============================================================================
#  API ENDPOINTS  (called by JavaScript fetch() in the browser)
# =============================================================================

@app.route("/api/nutrition-knowledge", methods=["POST"])
def api_nutrition_knowledge():
    """Agent 1 API: receives a question, routes through orchestrator."""
    data = request.get_json(force=True)
    result = orchestrator("nutrition_knowledge", data)
    return jsonify({"result": result})


@app.route("/api/diet-planner", methods=["POST"])
def api_diet_planner():
    """Agent 2 API: receives user profile, routes through orchestrator."""
    data = request.get_json(force=True)
    result = orchestrator("diet_planner", data)
    return jsonify({"result": result})


@app.route("/api/health-advisory", methods=["POST"])
def api_health_advisory():
    """Agent 3 API: receives list of conditions, routes through orchestrator."""
    data = request.get_json(force=True)
    result = orchestrator("health_advisory", data)
    return jsonify({"result": result})


@app.route("/api/meal-analysis", methods=["POST"])
def api_meal_analysis():
    """Agent 4 API: receives free-text meal diary, routes through orchestrator."""
    data = request.get_json(force=True)
    result = orchestrator("meal_analysis", data)
    return jsonify({"result": result})


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  NutriWise AI - Personalized Nutrition Coach")
    print("  Powered by IBM watsonx.ai Granite Models")
    print("=" * 60)
    print()
    if not WATSONX_API_KEY:
        print("[WARN]  WARNING: WATSONX_API_KEY is not set.")
    if not WATSONX_PROJECT_ID:
        print("[WARN]  WARNING: WATSONX_PROJECT_ID is not set.")
    print()
    print("  Set environment variables before starting:")
    print("    set WATSONX_API_KEY=<your_api_key>")
    print("    set WATSONX_PROJECT_ID=<your_project_id>")
    print("    set WATSONX_URL=https://us-south.ml.cloud.ibm.com")
    print()
    print("  Starting Flask server at http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
