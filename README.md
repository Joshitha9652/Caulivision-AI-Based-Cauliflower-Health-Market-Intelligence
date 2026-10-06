# 🥦 Smart Cauliflower Prediction (AICW)

An AI-powered Computer Vision, Quality Grading, and Black Soil Agronomic LLM Advisory platform for cauliflower produce management and Andhra Pradesh APMC Mandi market intelligence.

---

## 🤝 Project Partners & Collaboration (Co-Founders)

Developed under the **AICW (Artificial Intelligence and Career for Women)** initiative in collaboration with:
- **Microsoft** — Cloud Computing & AI Technology Partner
- **Edunet Foundation** — Implementation & Agritech Skilling Partner
- **APSSDC** — Andhra Pradesh State Skill Development Corporation
- **SAP** — Enterprise Software & Cloud Innovation Partner

---

## 👥 Project Team & Mentorship

- **Project Guide:** **Abdulaziz** (Project Mentor & Technical Guide)
- **Project Team Members:**
  1. **Ch. Joshitha** — Team Lead & Machine Learning Engineer (Model Architecture & Training)
  2. **Drakshayani.R** — Computer Vision & Data Specialist (Feature Engineering & Dataset Analytics)
  3. **CH.Adhi Lakshmi** — Agronomy Integration & UI/UX Specialist (LLM Advisory Workflows & Interface)

---

## 📌 Core Features & Web Application Workflow

### 1. 🏠 Home Page (Project Overview & Team)
- Comprehensive introduction to cauliflower cultivation challenges in India & Andhra Pradesh.
- Highlighting economic losses due to **Bacterial Spot Rot**, **Black Rot**, and **Downy Mildew**.
- Prominent Industrial Co-Founder cards (Microsoft, Edunet, APSSDC, SAP) placed under the top navigation.
- Dedicated Team & Guide showcase cards.

### 2. 🔬 Disease Prediction and Quality Prediction
- The disease and quality panels each have their own image upload and live-camera scan controls.
- Capture a photo with the camera or upload an image, then run the prediction for that panel.
- The disease panel shows only the top disease label and its calibrated model score.
- The quality panel shows the predicted `Healthy`/`Damaged` condition and score, plus its `Grade A`, `Grade B`, or `Grade C` prediction and score.
- The project quality dataset labels damaged produce as Grade B or Grade C; the predictor follows that rule rather than reporting damaged produce as Grade A.
- The site can be switched between English and Telugu; the selected language is remembered between pages.
- Class percentages are calibrated estimates from the project data and have not been independently validated on field-collected photos. Confirm important diagnoses with an agricultural expert.

### 3. 🤖 Below Predictions: Integrated LLM Agronomic Advisory Engine
Both predicted inputs (Disease + Quality) and the farm condition (**Black Soil / నల్ల రేగడి నేల**) are fed into the LLM Advisory:
1. **🛡️ Precautions (ముందస్తు జాగ్రత్తలు):** Seed sanitation, crop rotation, field hygiene, certified seed protocols.
2. **⚡ Immediate Measures (తక్షణ నివారణ చర్యలు):** Rogueing infected heads, pruning leaves, avoiding overhead sprinkler splashing.
3. **🌱 Special Black Soil Management (నల్ల రేగడి నేల ప్రత్యేక సలహా):**
   - High clay content (>50%) and water-logging mitigation.
   - Raised bed configuration (**Broad Bed and Furrow - BBF** system).
   - Strict drip irrigation scheduling (avoiding water standing around root collars).
   - Gypsum application (1.5 - 2.0 t/ha) to improve soil flocculation and supply calcium.
   - Summer deep ploughing for soil solarization.
4. **🧪 Recommended Pesticides, Fungicides, Bactericides & Dosages (ఏ పురుగు మందులు వాడాలి):**
   - **Bacterial Spot Rot / Black Rot:** Copper Oxychloride 50 WP (2.5 - 3.0 g/L) + Streptocycline 100 ppm (1 g / 10 L water), or Kasugamycin 3% SL (2 ml/L).
   - **Downy Mildew:** Metalaxyl 8% + Mancozeb 64% WP (Ridomil Gold) @ 2.0 - 2.5 g/L, or Cymoxanil + Mancozeb (2.5 g/L).
   - **Healthy Curds (Biological):** Neem Oil 1500 ppm @ 3 ml/L, Trichoderma viride & Pseudomonas fluorescens @ 5 g/L.
   - Exact application timing, safety PPE, and Pre-Harvest Interval (PHI).
5. **🗣️ Summary in Simple Telugu (రైతుల కొరకు ముఖ్యమైన సలహాలు)**
6. **💬 Interactive AI Agronomist Query Box:** Allows farmers or evaluators to ask custom questions in English or Telugu with instant answers (works 100% offline or with optional Gemini API key).

### 4. 📍 AP Mandi Directory & Analytics
- Complete coverage of all 26 Andhra Pradesh districts and 44 APMC Mandis & Rythu Bazars.
- The `/market` dashboard can fetch current cauliflower mandi prices and arrival-date history from the official data.gov.in (AGMARKNET) resource. It converts reported rupees per quintal to rupees per kilogram and caches successful responses for 15 minutes.
- Set `DATA_GOV_IN_API_KEY` in the server environment to enable the official feed. Without a key, the dashboard clearly labels the bundled mandi prices as reference data and does not fabricate daily price trends.
- Each mandi card can estimate a farmer's selling price with the trained `cauliflower_price_model.pkl`. The model uses quantity to sell and mandi modal/minimum/maximum price as inputs and predicts estimated rupees per kilogram plus estimated sale total.
- This is a project-dataset estimate, not a guaranteed quote. `generate_datasets.py` creates the 1,000-row price dataset from randomized mandi references and simulated premiums; use verified transaction records to improve real-world prediction quality.
- The analyze page can estimate pesticide product amounts for the selected field area. For doses stated per litre, enter the actual spray-water litres used per acre; per-acre rates are multiplied by the selected acres. Unrecognized dose formats are not calculated. These are arithmetic estimates from the displayed advice, not a replacement for the product label or local agriculture officer guidance.
- Direct inspection and visualization of all 4 master CSV datasets:
  - `01_cauliflower_size_dataset.csv`
  - `02_cauliflower_damage_dataset.csv`
  - `03_cauliflower_quality_dataset.csv`
  - `04_cauliflower_price_dataset.csv`

---

## 🚀 How to Run the Web Application

The browser UI for Home (`/`) and Analyze (`/analyze`) listens on **port 5000**.
`ERR_CONNECTION_REFUSED` means this process is not running.

```bash
pip install -r requirements.txt
python webapp.py
```
Then open `http://127.0.0.1:5000/` (home), `http://127.0.0.1:5000/analyze` (disease and quality prediction panels), or `http://127.0.0.1:5000/market` (mandi prices).

To enable live mandi prices, set the `DATA_GOV_IN_API_KEY` environment variable before starting the app. Get a key from [data.gov.in](https://data.gov.in/). If the official service is unavailable, the page reports the feed warning and shows only clearly labelled reference prices.

On `/market`, enter the planned cauliflower quantity in a mandi card and select **Predict selling price** to get the model's per-kg estimate and estimated total.

### Retrain Models (Optional):
```bash
python train_model.py
```

---

## 📦 Project Structure

```
cauliflower_separate_datasets(1)/
├── webapp.py                          # Browser application (Home and disease/quality prediction)
├── templates/                         # Home and prediction pages
├── static/                            # Styles and English/Telugu language support
├── llm_advisor.py                     # Black Soil agronomic LLM advisory engine & Q&A assistant
├── model_utils.py                     # Image feature extraction, calibrated model training & grading
├── train_model.py                     # End-to-end training pipeline for all models
├── ap_locations.py                    # Andhra Pradesh 26 districts & 44 APMC mandis database
├── generate_datasets.py               # Master datasets generator
├── Cauliflower_256x256/               # Curd image datasets
│   ├── Original Cauliflower Dataset/  # 4 classes: Bacterial Spot Rot, Black Rot, Downy Mildew, Healthy Fruit
│   ├── Healthy/                       # 527 healthy images
│   └── Damaged/                       # 362 damaged images
├── models/                            # Trained model artifacts (.pkl)
│   ├── cauliflower_disease_model.pkl  # Calibrated disease model
│   ├── disease_encoder.pkl            # Disease label encoder
│   ├── cauliflower_quality_model.pkl  # Calibrated quality classifier (Grade A/B/C)
│   ├── quality_encoder.pkl            # Quality label encoder
│   ├── cauliflower_size_model.pkl     # 100% accuracy size classifier
│   └── cauliflower_price_model.pkl    # AP mandi price regressor
├── 01_cauliflower_size_dataset.csv    # 1,000 physical size records
├── 02_cauliflower_damage_dataset.csv  # 1,000 disease and damage records
├── 03_cauliflower_quality_dataset.csv # 1,000 quality grade records (Grade A/B/C)
├── 04_cauliflower_price_dataset.csv   # 1,000 Andhra Pradesh mandi price records
└── README.md                          # Project documentation
```
