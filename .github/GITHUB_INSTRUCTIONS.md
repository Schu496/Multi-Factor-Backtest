# How to Publish This Project to GitHub
## Step-by-step guide for VS Code on Windows

---

## Prerequisites

Before starting, make sure you have:
- A GitHub account (free at github.com)
- Git installed on your computer (download at git-scm.com)
- VS Code open with your project folder

To check if Git is installed, open the VS Code terminal and type:
```
git --version
```
You should see something like `git version 2.x.x`. If you get an error, install Git first.

---

## Step 1 — Create a New GitHub Repository

1. Go to **github.com** and sign in
2. Click the **+** icon in the top right corner
3. Select **"New repository"**
4. Fill in the details:
   - **Repository name:** `multi-factor-backtest`
   - **Description:** `A professional 5-factor equity strategy backtest in Python (S&P 500, 2010-2024)`
   - **Visibility:** Public (so recruiters can see it)
   - **DO NOT** check "Add a README file" — we already have one
   - **DO NOT** check "Add .gitignore" — we already have one
5. Click **"Create repository"**
6. GitHub will show you a page with setup instructions — **keep this page open**, you will need the URL

---

## Step 2 — Configure Git with Your Identity

Open the VS Code terminal and run these two commands (replace with your details):

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

This tells Git who you are for commit messages. Only needs to be done once per computer.

---

## Step 3 — Initialize Git in Your Project Folder

In the VS Code terminal, make sure you are in your project root folder.
You should see something like:
```
PS C:\Users\YourName\...\MultiFactor Equity Strategy Backtest\All Files>
```

Then run:
```bash
git init
```

You should see: `Initialized empty Git repository in ...`

---

## Step 4 — Check What Will Be Uploaded

Run this to see all files Git will track:
```bash
git status
```

You should see many files listed in red under "Untracked files."

Check that your `.gitignore` is working correctly by confirming these folders
are NOT listed (they should be excluded):
- `data/raw/`
- `data/processed/`
- `venv/`
- `reporting/output/`

If those folders appear, your `.gitignore` is not in the right place.

---

## Step 5 — Stage All Files

This tells Git which files to include in your first commit:
```bash
git add .
```

The `.` means "add everything" (that isn't in .gitignore).

Run `git status` again — everything should now appear in green.

---

## Step 6 — Create Your First Commit

A "commit" is a saved snapshot of your project:
```bash
git commit -m "Initial commit: Multi-factor equity strategy backtest (2010-2024)"
```

You should see output listing all the files that were committed.

---

## Step 7 — Connect to Your GitHub Repository

Go back to the GitHub page from Step 1. Copy the repository URL.
It will look like: `https://github.com/YOUR_USERNAME/multi-factor-backtest.git`

Then run:
```bash
git remote add origin https://github.com/YOUR_USERNAME/multi-factor-backtest.git
```

Replace `YOUR_USERNAME` with your actual GitHub username.

---

## Step 8 — Push to GitHub

```bash
git branch -M main
git push -u origin main
```

Git will ask for your GitHub username and password.

> **Important:** GitHub no longer accepts your account password here.
> You need a **Personal Access Token** instead. Here is how to get one:
>
> 1. Go to github.com → click your profile photo → Settings
> 2. Scroll down to **"Developer settings"** (bottom of left sidebar)
> 3. Click **"Personal access tokens"** → **"Tokens (classic)"**
> 4. Click **"Generate new token (classic)"**
> 5. Give it a name like "VS Code push"
> 6. Set expiration to 90 days
> 7. Check the **"repo"** checkbox
> 8. Click **"Generate token"**
> 9. **Copy the token immediately** — you will never see it again
>
> When Git asks for your password, paste the token instead.

---

## Step 9 — Verify on GitHub

Go to `https://github.com/YOUR_USERNAME/multi-factor-backtest`

You should see your project with:
- The README displaying with all the strategy results
- All your Python files organized in folders
- The green "Code" button at the top right

---

## Step 10 — Add the Tearsheet PDF (Optional but Recommended)

The tearsheet PDF is excluded by `.gitignore` by default (it is in `reporting/output/`).
To share it on GitHub, copy it to the root folder first:

```bash
copy reporting\output\tearsheet.pdf tearsheet.pdf
git add tearsheet.pdf
git commit -m "Add strategy tearsheet PDF"
git push
```

Then update your README to link to it:
```markdown
[View Full Tearsheet PDF](tearsheet.pdf)
```

---

## Updating GitHub After Changes

Every time you make changes and want to update GitHub:
```bash
git add .
git commit -m "Brief description of what you changed"
git push
```

---

## Making Your Profile Stand Out

Once the repo is live, consider these final touches:

1. **Pin the repo** to your GitHub profile:
   - Go to your profile page
   - Click "Customize your pins"
   - Select `multi-factor-backtest`

2. **Add topics** to the repository for discoverability:
   - Go to your repo → click the gear icon next to "About"
   - Add topics: `python`, `quantitative-finance`, `backtesting`,
     `factor-investing`, `pandas`, `matplotlib`, `finance`

3. **Write a good About description:**
   `"5-factor equity strategy backtest on S&P 500 (2010-2024). CAGR 11.75%, Sharpe 0.61, Alpha +1.14%. Built with Python, pandas, matplotlib."`

---

## Troubleshooting

**"fatal: not a git repository"**
→ Run `git init` first (Step 3)

**"remote origin already exists"**
→ Run `git remote remove origin` then redo Step 7

**"Authentication failed"**
→ Make sure you are using a Personal Access Token, not your password (Step 8)

**"error: src refspec main does not match any"**
→ Run `git add .` and `git commit -m "first commit"` before pushing

**Large file warnings**
→ Your data CSV files are excluded by .gitignore. If you see warnings,
  check that `data/processed/` appears in your `.gitignore`
