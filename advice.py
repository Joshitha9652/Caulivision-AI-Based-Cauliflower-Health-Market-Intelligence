"""LLM + offline agronomy advice for cauliflower on black soil."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


SYSTEM_PROMPT = """You are an agronomist helping Andhra Pradesh cauliflower farmers.
Soil is BLACK SOIL (regur). Keep language simple. Mix short Telugu farmer words in brackets where useful.
Reply ONLY valid JSON with keys:
precautions (array of 4-6 short strings),
measures (array of 4-6 short strings),
black_soil (array of 4-5 short strings about drainage, moisture, fertilizer on black soil),
pesticides (array of objects with name, use_for, dose, note).
Prefer IPM first. Recommend only commonly used India-legal crop protection (copper oxychloride, mancozeb, metalaxyl+mancozeb, streptocycline, neem oil, trichoderma). Include safety: mask, waiting period, do not spray at harvest. Never invent banned chemicals."""


def build_advice(disease: dict, quality: dict) -> dict:
    payload = {
        "crop": "cauliflower",
        "soil": "black soil",
        "disease_name": disease.get("name"),
        "disease_severity": disease.get("severity"),
        "disease_summary": disease.get("summary"),
        "quality_grade": quality.get("grade"),
        "quality_summary": quality.get("summary"),
        "spots_count": quality.get("spots_count"),
        "color_score": quality.get("color_score"),
    }
    llm = _try_llm(payload)
    if llm:
        llm["source"] = "llm"
        return llm
    fallback = offline_advice(disease, quality)
    fallback["source"] = "offline"
    return fallback


def _try_llm(payload: dict) -> dict | None:
    groq = os.getenv("GROQ_API_KEY", "").strip()
    openai = os.getenv("OPENAI_API_KEY", "").strip()
    user = (
        "Give farmer advice for this cauliflower sample:\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    if groq:
        data = _http_json(
            "https://api.groq.com/openai/v1/chat/completions",
            {
                "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                "temperature": 0.3,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            },
            {"Authorization": f"Bearer {openai}"},
        )
        if data:
            return _parse_content(data["choices"][0]["message"]["content"])
    if openai:
        data = _http_json(
            "https://api.openai.com/v1/chat/completions",
            {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "temperature": 0.3,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            },
            {"Authorization": f"Bearer {openai}"},
        )
        if data:
            return _parse_content(data["choices"][0]["message"]["content"])
    return None


def _http_json(url: str, body: dict, headers: dict) -> dict | None:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
        return None


def _parse_content(text: str) -> dict | None:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    needed = ("precautions", "measures", "black_soil", "pesticides")
    if not all(k in data for k in needed):
        return None
    return data


def offline_advice(disease: dict, quality: dict) -> dict:
    name = disease.get("name") or "Healthy"
    grade = quality.get("grade") or "Grade B"
    common_precautions = [
        "Plant on raised beds so black soil water does not sit around roots (nillu nilavadu).",
        "Keep 45–60 cm spacing for air flow and to reduce leaf wetness.",
        "Avoid overhead watering in the evening; water at the base in the morning.",
        "Remove and bury badly infected leaves away from the field.",
        "Wash tools after working in a sick patch so bacteria do not spread.",
    ]
    common_soil = [
        "Black soil holds water. Make 20–25 cm furrows or raised beds before planting.",
        "Add 8–10 tonnes well-rotted FYM / compost per acre before last ploughing.",
        "Do not flood irrigate. Give light irrigation when the top 4–5 cm feels dry.",
        "Add gypsum 200 kg/acre if soil is very sticky, to improve tilth.",
        "Use balanced NPK (example 80:60:60 kg/acre) in 2–3 splits; extra nitrogen makes soft, disease-prone curds.",
    ]
    disease_map = {
        "Healthy": {
            "measures": [
                "Continue weekly scouting of lower leaves and curd.",
                "Spray 3 ml neem oil per litre every 10–12 days as a preventive.",
                "Keep field weeds low so humidity does not rise.",
                f"Harvest this {grade} curd at full compactness and send to market quickly.",
            ],
            "pesticides": [
                {
                    "name": "Neem oil 1500 ppm",
                    "use_for": "Preventive sucking pests and mild fungus",
                    "dose": "3 ml per litre water",
                    "note": "Safe near harvest. Spray evening.",
                },
                {
                    "name": "Trichoderma viride",
                    "use_for": "Soil-borne rot prevention in black soil",
                    "dose": "1 kg with 50 kg FYM per acre",
                    "note": "Apply at planting or with irrigation. Biological, not a chemical spray.",
                },
            ],
        },
        "Downy Mildew": {
            "measures": [
                "Remove yellow lower leaves and do not leave them in the field.",
                "Improve drainage immediately; standing water on black soil worsens mildew.",
                "Avoid dense planting and stop evening sprinkler irrigation.",
                "Start a protective fungicide spray at first yellow patches, then repeat after 7–10 days if weather stays wet.",
            ],
            "pesticides": [
                {
                    "name": "Mancozeb 75% WP",
                    "use_for": "Downy mildew protective spray",
                    "dose": "2–2.5 g per litre",
                    "note": "Spray both leaf sides. Waiting period about 7–10 days before harvest.",
                },
                {
                    "name": "Metalaxyl 8% + Mancozeb 64% WP",
                    "use_for": "Active downy mildew in humid weather",
                    "dose": "2 g per litre",
                    "note": "Do not use more than 2 sprays per crop. Wear mask and gloves.",
                },
                {
                    "name": "Neem oil",
                    "use_for": "Support spray / early stage",
                    "dose": "3 ml per litre",
                    "note": "Can be rotated with fungicides. Not enough alone in severe attack.",
                },
            ],
        },
        "Black Rot": {
            "measures": [
                "This is bacterial. Do not use only fungicides; they will not cure it.",
                "Rogue badly blackened plants and destroy them.",
                "Stop working in the field when leaves are wet.",
                "Use copper spray after removing infected tissue, and keep drainage open.",
            ],
            "pesticides": [
                {
                    "name": "Copper oxychloride 50% WP",
                    "use_for": "Black rot / bacterial leaf spots",
                    "dose": "3 g per litre",
                    "note": "Spray at 7–10 day interval. Do not mix with streptocycline in the same tank unless label allows.",
                },
                {
                    "name": "Streptocycline",
                    "use_for": "Bacterial black rot (early stage)",
                    "dose": "0.1 g (100 ppm) per litre, often with copper",
                    "note": "Use only as recommended. Stop well before harvest. Follow local agriculture officer advice.",
                },
            ],
        },
        "Bacterial Spot Rot": {
            "measures": [
                "Cut away rotting curd parts with a clean knife and remove them from the plot.",
                "Reduce irrigation until the surface of black soil dries a little.",
                "Keep harvested heads in shade; rot spreads fast in heat.",
                "Disinfect crates and knives with 1% bleaching powder water.",
            ],
            "pesticides": [
                {
                    "name": "Copper oxychloride 50% WP",
                    "use_for": "Bacterial spot / soft rot check",
                    "dose": "3 g per litre",
                    "note": "Cover curd and wrapper leaves. Waiting period 5–7 days.",
                },
                {
                    "name": "Bleaching powder (field sanitation)",
                    "use_for": "Soil and tool hygiene",
                    "dose": "5 kg/acre near plant base in wet patches, or 1% for tools",
                    "note": "This is sanitation, not a leaf spray on edible curd.",
                },
            ],
        },
        "Physical Damage": {
            "measures": [
                "Handle curds with two hands; do not throw into crates.",
                "Check for larvae / borers inside damaged holes.",
                "Harvest in cool hours and line crates with paper or leaves.",
                "Sell Grade C damaged heads locally the same day.",
            ],
            "pesticides": [
                {
                    "name": "Neem seed kernel extract / neem oil",
                    "use_for": "Mild caterpillar and sucking pest pressure",
                    "dose": "3–5 ml neem oil per litre",
                    "note": "If borers are many, consult AO for a permitted insecticide; do not spray near harvest.",
                },
                {
                    "name": "Pheromone traps (DBM)",
                    "use_for": "Diamondback moth monitoring",
                    "dose": "8–10 traps per acre",
                    "note": "Non-chemical. Helps decide if a spray is needed.",
                },
            ],
        },
    }
    block = disease_map.get(name, disease_map["Healthy"])
    precautions = common_precautions[:]
    if grade == "Grade C":
        precautions.append("Do not send badly spotted curds to distant markets; they will reject and you lose transport cost.")
    if grade == "Grade A" and name == "Healthy":
        precautions.append("Keep irrigation even for 3–4 days before harvest so curds stay compact and white.")
    return {
        "precautions": precautions,
        "measures": block["measures"],
        "black_soil": common_soil,
        "pesticides": block["pesticides"],
    }
