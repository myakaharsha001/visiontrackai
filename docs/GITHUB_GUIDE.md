# GitHub guide (from zero)

I (the assistant) have **not** created or pushed any repository. Replace `<YOUR-USERNAME>` below.

## 1. One-time setup
1. Create an account at <https://github.com/signup>.
2. Install Git for Windows: <https://git-scm.com/download/win>. Then in PowerShell:
```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

## 2. Create the empty repository on GitHub
GitHub → **New repository** → name `visiontrack-ai-object-detection` → description *Real-time object detection with YOLO + OpenCV, logged to MySQL, shown in a Streamlit dashboard* → **Public** (or Private) → **do NOT** tick README/.gitignore/license (the project already has them) → Create.

## 3. Pre-flight checks (do these BEFORE committing)
```powershell
cd <project folder>
git init
git status --short                     # .env must NOT appear
git check-ignore -v .env               # should print the .gitignore rule that ignores it
Select-String -Path *.py,*.md,*.sql,*.example -Pattern "your-real-password" # search for your real password; expect no matches
Test-Path requirements.txt, database_schema.sql, README.md, .env.example, .gitignore, LICENSE   # all True
```
Also confirm `database_schema.sql` still contains `CHANGE_ME_TO_A_STRONG_PASSWORD` (not your real password).
`yolov8n.pt` and `.uploads/`, `logs/` are ignored on purpose — the app downloads the weights itself, so nothing needed is excluded.

## 4. Commit and push
```powershell
git add .
git status                             # review the list: no .env, no .venv
git commit -m "VisionTrack AI: real-time object detection platform"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/visiontrack-ai-object-detection.git
git push -u origin main
```
GitHub no longer accepts account passwords for `git push`; the first push opens a browser login (Git Credential Manager), or use a personal access token.

## 5. Verify online
* Open the repo URL in a **private/incognito window** (checks public visibility).
* The file list shows `main.py`, `detector.py`, `database.py`, `ui.py`, `requirements.txt`, `database_schema.sql`, README renders.
* There is **no** `.env` file. Use the repo search box for `DB_PASSWORD` — only `.env.example`/docs should match, with placeholder values.
* If you ever committed a secret: change that password immediately (removing the file from a later commit does not remove it from history).

## 6. Add screenshots later
```powershell
git add screenshots README.md
git commit -m "Add screenshots"
git push
```
