# Streamlit Deployment

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files Required

- `app.py` - Main application
- `efficientnet_best.pth` - Model weights (134MB)
- `ingredient_vocab.json` - Vocabulary
- `ingredients.xlsx` - Nutrition database
- `cooccurrence_prob.npy` - Co-occurrence matrix
- `requirements.txt` - Dependencies

## Deployment to Streamlit Cloud

### Option 1: With Model File (if <200MB total)

1. Push all files to GitHub
2. Deploy directly from share.streamlit.io

### Option 2: Model Hosted Separately (Recommended)

If model file too large:

1. Upload `efficientnet_best.pth` to cloud storage (Google Drive, Dropbox)
2. Modify `app.py` to download model on startup
3. Deploy to Streamlit Cloud

## Notes

- Model runs on CPU (deployment)
- First prediction may be slow (model loading)
- Uncertainty estimation adds ~30 seconds per image
