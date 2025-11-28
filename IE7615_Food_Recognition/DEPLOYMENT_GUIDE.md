# Deployment Guide

## Step 1: Prepare for GitHub

### Files to Upload

**Essential (upload to GitHub)**:
- All .py files
- README.md files
- requirements.txt
- .gitignore
- Small data files (<100MB):
  - ingredient_vocab.json (4KB)
  - ingredients.xlsx (26KB)
  - cooccurrence_prob.npy (485KB)

**Large files (DO NOT upload directly)**:
- efficientnet_best.pth (134MB) - too large

## Step 2: Handle Large Model File

### Option A: Git LFS (Recommended if <200MB total)
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "deployment/*.pth"
git lfs track "deployment/*.npy"

# Add and commit
git add .gitattributes
git commit -m "Configure Git LFS"
```

### Option B: External Hosting (If model too large)

1. Upload model to Google Drive/Dropbox
2. Get sharing link
3. Modify app.py to download on startup:
```python
import gdown

@st.cache_resource
def download_model():
    url = "YOUR_GOOGLE_DRIVE_LINK"
    output = "efficientnet_best.pth"
    if not Path(output).exists():
        gdown.download(url, output, quiet=False)
    return output
```

## Step 3: Upload to GitHub
```bash
cd /home/jen.che/IE7615_Food_Recognition

# Initialize git
git init

# Add files
git add .

# Commit
git commit -m "Initial commit: Food ingredient recognition system"

# Create repo on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/IE7615_Food_Recognition.git
git branch -M main
git push -u origin main
```

## Step 4: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click "New app"
3. Select your GitHub repository
4. Set:
   - Main file: `deployment/app.py`
   - Python version: 3.12
5. Click "Deploy"

## Step 5: Test Deployment

Once deployed, test:
- Upload sample image
- Check ingredient detection
- Try different thresholds
- Enable uncertainty estimation
- Test portion size input
- Verify nutrition calculation

## Notes

- First run will be slow (model loading)
- Uncertainty mode adds ~30s per prediction
- Model runs on CPU in cloud deployment
