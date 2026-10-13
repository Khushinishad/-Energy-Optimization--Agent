# Deployment Guide: GitHub + Streamlit Community Cloud (free)

Goal: put the project on GitHub and get a public live link for your final submission.

> **Why your first deployment failed:** `ModuleNotFoundError: matplotlib` appears when Streamlit Cloud cannot
> find `requirements.txt`. It only looks at the **root of the repo** (or next to the main file). The old zip
> had the project inside an extra folder, so the repo root had no `requirements.txt`.
> **Rule:** the folder that contains `requirements.txt`, `src/` and `app/` must be the **root** of the GitHub repo.

**Main file path for Streamlit Cloud: `app/dashboard.py`**

---

## Step 0. Unzip correctly

1. Unzip `energy-optimization-agent-deploy.zip`.
2. Open the extracted `energy-optimization-agent` folder. Inside you must see:
   `README.md`, `requirements.txt`, `src`, `app`, `data`, `models`, `reports`, `docs`, `.streamlit`.
3. This folder is your project root. Do not push its parent folder.

## Step 1. (Recommended) Test on your computer

Open Command Prompt **inside the project root** and run:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/dashboard.py
```

macOS/Linux: use `python3 -m venv venv` and `source venv/bin/activate`.
The dashboard opens at http://localhost:8501. If it works here, it will work on Streamlit Cloud.
(The results are already included, so you do not need to retrain. To retrain: `python -m src.train_pipeline`.)

## Step 2. Create the GitHub repository

1. Sign up or log in at https://github.com.
2. Click **+** (top right) then **New repository**.
3. Repository name: `energy-optimization-agent`.
4. Choose **Public** (Streamlit's free tier deploys from public repos).
5. **Do NOT** tick "Add a README", ".gitignore" or "license" (the project already has them).
6. Click **Create repository**. Keep this page open.

## Step 3. Push the project

### Option A: Git commands (recommended)

Install Git from https://git-scm.com/downloads if `git --version` does not work. Then, **inside the project root**:

```
git config --global user.name "Your Name"
git config --global user.email "your-github-email@example.com"

git init
git add .
git commit -m "AI Energy Optimization Agent - final project"
git branch -M main
git remote add origin https://github.com/<your-username>/energy-optimization-agent.git
git push -u origin main
```

Replace `<your-username>` with your GitHub username. On the first push, a browser window asks you to sign in
to GitHub; approve it. GitHub no longer accepts your account password in the terminal. If asked for a
password, use a Personal Access Token (GitHub, Settings, Developer settings, Personal access tokens) instead.

### Option B: Upload from the browser (no Git needed)

1. On the empty repository page click **uploading an existing file**.
2. Open the project root folder on your computer, select **everything inside it**
   (`src`, `app`, `data`, `models`, `reports`, `docs`, `.streamlit`, `requirements.txt`, `README.md`, ...)
   and drag it into the browser. Do not upload the zip itself.
3. Wait until all files finish uploading (about 40 files; each is well under GitHub's size limits).
4. Write the commit message "Initial commit" and click **Commit changes**.

### Check the repository

On GitHub you must see `requirements.txt`, `src/`, `app/` **directly on the first page** of the repo,
not inside another folder. If you see only one folder, you uploaded the wrong level. Fix it before continuing.

## Step 4. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and click **Continue with GitHub**. Authorise Streamlit to read your repositories.
2. Click **Create app** (or **New app**), then **Deploy a public app from GitHub**.
3. Fill in:

| Field | Value |
|---|---|
| Repository | `<your-username>/energy-optimization-agent` |
| Branch | `main` |
| **Main file path** | **`app/dashboard.py`** |
| App URL (optional) | choose a short name, for example `energy-optimization-agent` |

4. Open **Advanced settings** and choose **Python 3.11 or 3.12**. No secrets are needed.
5. Click **Deploy**. The first build takes a few minutes while packages install.
6. When it finishes you get a link such as `https://<your-app-name>.streamlit.app`. **Copy it.**

Wording on the Streamlit page may differ slightly, but the three important values (repo, branch `main`,
main file `app/dashboard.py`) stay the same.

## Step 5. Verify the live app

Open the link and check every tab: **Forecast, Anomalies, Usage patterns, RL agent, Tips**.
In the RL agent tab, pick a different test day and confirm the chart updates.
Take screenshots of each tab for your report.

## Step 6. Add your links to the README

Edit the README on GitHub (pencil icon) or locally, then replace the two placeholders in section 11:

```
**Live demo:** https://<your-app-name>.streamlit.app
**GitHub repository:** https://github.com/<your-username>/energy-optimization-agent
```

If you edited locally, publish the change:

```
git add .
git commit -m "Add live demo link"
git push
```

Every push to `main` redeploys the app automatically.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: matplotlib` (or numpy, pandas, sklearn) | `requirements.txt` is not at the repo root. Fix the folder level (Step 3, "Check the repository"), push, then **Manage app, Reboot app** |
| `ModuleNotFoundError: src` | Main file path must be exactly `app/dashboard.py` and the `src` folder must be at the repo root |
| Main file not found | Path is case-sensitive: `app/dashboard.py` (lowercase). Check the folder level on GitHub |
| App shows "First run: generating data..." | Normal if the `reports` folder was not uploaded. Wait about a minute, it builds everything itself. Better: upload the `reports`, `data`, `models` folders too |
| `git push` rejected or "remote contains work" | You ticked "Add a README" when creating the repo. Run `git pull origin main --allow-unrelated-histories`, then `git push` |
| `git push` asks for a password | Use the browser sign-in, or a Personal Access Token |
| `git is not recognized` | Install Git, then reopen Command Prompt |
| App went to sleep | Free apps sleep after inactivity. Open the link and press **Yes, get this app back up**. Open it once before your evaluation |
| Build fails after you edit `requirements.txt` | Keep one package per line, no typos. Open **Manage app** (bottom right) to read the full log |
| Want to see the real error | **Manage app** (bottom right of the live app), then the logs |

## Final submission checklist

- [ ] GitHub repository link (public, with README showing the live link)
- [ ] Live Streamlit app link, opened and tested
- [ ] Project report (outline in `docs/project_report_outline.md`)
- [ ] Lean Canvas PDF (`docs/Lean_Canvas_AI_Energy_Optimization_Agent.pdf`)
- [ ] Presentation slides (optional) and screenshots of all five dashboard tabs
- [ ] Results table from `reports/summary.json` and charts from `reports/figures/`
- [ ] Limitations stated honestly: synthetic data, simulated savings, approximate CO2 factor

### Ready-to-use project description (edit as needed)

> **AI-Powered Energy Optimization Agent (SDG 13: Climate Action).** An AI/ML system that forecasts hourly
> electricity demand (Gradient Boosting, R² about 0.92), detects wasteful or faulty usage (Isolation Forest),
> discovers daily usage patterns (K-Means), and learns an air-conditioner control policy with reinforcement
> learning (Q-learning). In simulation it reduced AC energy by about 42% versus an always-on habit while
> keeping occupants comfortable, and reports the CO2 avoided. Built with Python, scikit-learn and Streamlit using
> only free tools, deployed on GitHub and Streamlit Community Cloud.

### 2-minute explanation for a viva or demo

1. **Problem:** wasted electricity raises bills and emissions, and people only see a monthly bill.
2. **Data:** hourly smart-meter style data (synthetic here, UCI real data supported).
3. **Models:** forecasting, anomaly detection, clustering, reinforcement learning.
4. **Impact:** kWh saved, cost saved, CO2 avoided, shown live on the dashboard.
5. **Limitations and future work:** real building data, IoT control, deep RL.
