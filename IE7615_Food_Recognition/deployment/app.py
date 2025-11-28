"""
Food Nutrition Analysis System
AI-powered ingredient detection with manual portion input
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import timm
from torchvision import transforms
from pathlib import Path
import json
import gdown
import os

# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="Food Nutrition Analyzer",
    page_icon="🍽️",
    layout="wide"
)

st.title("AI-Powered Food Nutrition Analyzer")
st.markdown("### Analyze ingredients and nutrition from food images")

# ============================================================
# Download Model from Google Drive
# ============================================================

@st.cache_resource
def download_model_if_needed():
    """Download model from Google Drive if not exists"""
    model_path = "efficientnet_best.pth"
    
    if not Path(model_path).exists():
        with st.spinner('Downloading model from Google Drive (134MB, first time only)...'):
            # Google Drive file ID
            file_id = "1iTgcoJ4DJVWDorzFWTzG_YjHaCbfR2kq"
            url = f"https://drive.google.com/uc?id={file_id}"
            
            try:
                gdown.download(url, model_path, quiet=False)
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                st.stop()
    
    return model_path

# ============================================================
# Load System
# ============================================================

@st.cache_resource
def load_system():
    """Load model and data"""
    
    # Download model if needed
    model_path = download_model_if_needed()
    
    # Other paths (relative)
    vocab_path = "ingredient_vocab.json"
    nutrition_path = "ingredients.xlsx"
    cooccur_path = "cooccurrence_prob.npy"
    
    # Vocabulary
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)['vocab']
    
    # Nutrition DB
    nutrition_df = pd.read_excel(nutrition_path)
    nutrition_df.columns = [c.lower().replace(' ','_').replace('(','').replace(')','') 
                           for c in nutrition_df.columns]
    
    # Model (EfficientNet-B3 + MFB)
    class FoodModel(nn.Module):
        def __init__(self, nc):
            super().__init__()
            self.backbone = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0, global_pool='')
            
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(1536, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.15),
                nn.Linear(512, nc)
            )
            
        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features)
    
    device = torch.device('cpu')
    
    model = FoodModel(len(vocab))
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Co-occurrence
    if Path(cooccur_path).exists():
        cooccur = np.load(cooccur_path)
    else:
        cooccur = None
    
    return model, vocab, nutrition_df, cooccur, device

with st.spinner('Loading system...'):
    model, vocab, nutrition_df, cooccur_prob, device = load_system()

st.success(f'System ready! (Device: {device})')

# ============================================================
# Functions
# ============================================================

def predict_with_uncertainty(model, img_tensor, device, n=20):
    """MC Dropout uncertainty estimation"""
    for m in model.classifier.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    
    preds = []
    with torch.no_grad():
        for _ in range(n):
            out = model(img_tensor)
            preds.append(torch.sigmoid(out).cpu().numpy())
    
    model.eval()
    preds = np.array(preds)
    return preds.mean(axis=0)[0], preds.std(axis=0)[0]

def get_nutrition(ing, df):
    """Get ingredient nutrition per gram"""
    m = df[df['ingr'].str.lower() == ing.lower()]
    if len(m) == 0:
        m = df[df['ingr'].str.lower().str.contains(ing.lower(), na=False)]
    if len(m) == 0:
        return {'cal': 1.5, 'protein': 0.08, 'carb': 0.15, 'fat': 0.05}
    
    r = m.iloc[0]
    return {
        'cal': float(r.get('cal/g', 1.5)),
        'protein': float(r.get('proteing', 0.08)),
        'carb': float(r.get('carbg', 0.15)),
        'fat': float(r.get('fatg', 0.05))
    }

def calc_nutrition(ings, weight_g, df):
    """Calculate total nutrition from weight"""
    wt_per = weight_g / max(len(ings), 1)
    
    tot = {'cal': 0, 'protein': 0, 'carb': 0, 'fat': 0, 'wt': weight_g}
    
    for ing in ings:
        n = get_nutrition(ing, df)
        tot['cal'] += n['cal'] * wt_per
        tot['protein'] += n['protein'] * wt_per
        tot['carb'] += n['carb'] * wt_per
        tot['fat'] += n['fat'] * wt_per
    
    return tot

def get_advice(nutr):
    """Generate dietary advice"""
    adv = []
    c, p = nutr['cal'], nutr['protein']
    
    if c < 300:
        adv.append("Light meal - good for snacking")
    elif c < 600:
        adv.append("Moderate meal - suitable for regular meals")
    else:
        adv.append("High calorie - consider portion control")
    
    if p > 25:
        adv.append("High protein - excellent for muscle building")
    elif p > 15:
        adv.append("Good protein content")
    elif p < 10:
        adv.append("Low protein - add protein sources")
    
    if nutr['carb'] > 60:
        adv.append("High carbs - good for energy")
    
    if nutr['fat'] > 25:
        adv.append("High fat - be mindful of portions")
    
    return adv if adv else ["Balanced meal"]

# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Settings")

threshold = st.sidebar.slider(
    "Prediction Threshold",
    0.1, 0.8, 0.25, 0.05,
    help="Lower = detect more ingredients"
)

use_uncertainty = st.sidebar.checkbox(
    "Show Uncertainty Scores",
    value=False,
    help="MC Dropout (slower but shows confidence)"
)

use_cooccur = st.sidebar.checkbox(
    "Use Co-occurrence Boosting",
    value=False,
    help="Boost related ingredients based on learned patterns"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
st.sidebar.info("""
**Model**: EfficientNet-B3 + MFB  
**Training**: 2,792 dishes  
**Classes**: 249 ingredients  
**F1 Score**: 0.786  
**Optimal Threshold**: 0.25

**Features**:
- Threshold Optimization
- MC Dropout Uncertainty
- Co-occurrence Learning
- Nutritional Reasoning
""")

# ============================================================
# Main Interface
# ============================================================

tab1, tab2, tab3 = st.tabs(["Upload & Analyze", "About", "Performance"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Food Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a photo of your food"
        )
        
        if uploaded_file:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption="Your Dish", use_column_width=True)
    
    with col2:
        st.subheader("System Capabilities")
        st.markdown("""
        **1. Ingredient Detection**  
        Identifies 249 types of ingredients
        
        **2. Confidence Scores**  
        Shows prediction certainty (optional)
        
        **3. Co-occurrence Learning**  
        Uses ingredient relationships
        
        **4. Nutrition Analysis**  
        Calculates macros from portion size
        
        **5. Health Advice**  
        Provides dietary recommendations
        """)
    
    # Analysis
    if uploaded_file is not None:
        st.markdown("---")
        
        if st.button("Analyze Food", type="primary"):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Preprocess
            status_text.text("Preprocessing image...")
            progress_bar.progress(20)
            
            tfm = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            img_tensor = tfm(img).unsqueeze(0).to(device)
            
            # Predict
            status_text.text("Detecting ingredients...")
            progress_bar.progress(50)
            
            if use_uncertainty:
                mean_probs, std_probs = predict_with_uncertainty(
                    model, img_tensor, device, 20
                )
            else:
                with torch.no_grad():
                    out = model(img_tensor)
                    mean_probs = torch.sigmoid(out).cpu().numpy()[0]
                    std_probs = np.zeros_like(mean_probs)
            
            # Co-occurrence boosting
            if use_cooccur and cooccur_prob is not None:
                status_text.text("Applying co-occurrence learning...")
                progress_bar.progress(70)
                
                detected = np.where(mean_probs > threshold)[0]
                boosted = mean_probs.copy()
                
                for idx in detected:
                    related = cooccur_prob[idx]
                    for j in range(len(vocab)):
                        if j != idx and related[j] > 0.3:
                            boosted[j] = min(1.0, boosted[j] + related[j] * 0.15)
                
                mean_probs = boosted
            
            # Get predictions
            pred_idx = np.where(mean_probs > threshold)[0]
            pred_ings = [vocab[i] for i in pred_idx]
            
            progress_bar.progress(100)
            status_text.text("Complete!")
            
            progress_bar.empty()
            status_text.empty()
            
            # ============================================================
            # Results
            # ============================================================
            
            st.success("Analysis Complete!")
            st.markdown("---")
            
            # Ingredients
            st.subheader(f"Detected Ingredients ({len(pred_ings)})")
            
            if len(pred_ings) > 0:
                sorted_idx = pred_idx[np.argsort(mean_probs[pred_idx])[::-1]]
                
                # Display in columns
                cols = st.columns(2)
                
                for i, idx in enumerate(sorted_idx):
                    col = cols[i % 2]
                    
                    ing = vocab[idx]
                    conf = mean_probs[idx]
                    unc = std_probs[idx]
                    
                    if unc < 0.05:
                        level = "Very Confident"
                        indicator = "[+++]"
                    elif unc < 0.10:
                        level = "Confident"
                        indicator = "[++]"
                    elif unc < 0.20:
                        level = "Uncertain"
                        indicator = "[+]"
                    else:
                        level = "Very Uncertain"
                        indicator = "[?]"
                    
                    with col:
                        if use_uncertainty:
                            st.markdown(f"{indicator} **{ing}**: {conf:.0%} (±{unc:.1%}) - *{level}*")
                        else:
                            st.markdown(f"**{ing}**: {conf:.0%}")
            else:
                st.warning("No ingredients detected. Try lowering the threshold.")
            
            # Manual Portion Input
            st.markdown("---")
            st.subheader("Portion Size Input")
            
            st.markdown("Enter the portion size manually:")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                portion_method = st.radio(
                    "Select input method:",
                    ["Common portions", "Custom weight"],
                    horizontal=True
                )
            
            if portion_method == "Common portions":
                portion_options = {
                    "Small bowl (150g)": 150,
                    "Medium bowl (250g)": 250,
                    "Large bowl (400g)": 400,
                    "Small plate (200g)": 200,
                    "Medium plate (350g)": 350,
                    "Large plate (500g)": 500,
                    "Fist size (100g)": 100,
                    "Palm size (120g)": 120,
                    "Two palms (240g)": 240
                }
                
                selected_portion = st.selectbox(
                    "Choose portion size:",
                    list(portion_options.keys())
                )
                
                weight_g = portion_options[selected_portion]
                
            else:
                weight_g = st.number_input(
                    "Enter weight in grams:",
                    min_value=10,
                    max_value=1000,
                    value=200,
                    step=10
                )
            
            with col2:
                st.metric("Total Weight", f"{weight_g}g")
                st.caption(f"~{weight_g / max(len(pred_ings), 1):.0f}g per ingredient")
            
            # Calculate nutrition
            if len(pred_ings) > 0:
                nutrition = calc_nutrition(pred_ings, weight_g, nutrition_df)
                advice = get_advice(nutrition)
                
                # Nutrition display
                st.markdown("---")
                st.subheader("Nutritional Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Calories", f"{nutrition['cal']:.0f} kcal")
                with col2:
                    st.metric("Protein", f"{nutrition['protein']:.1f} g")
                with col3:
                    st.metric("Carbs", f"{nutrition['carb']:.1f} g")
                with col4:
                    st.metric("Fat", f"{nutrition['fat']:.1f} g")
                
                # Macronutrient chart
                st.markdown("#### Macronutrient Distribution")
                
                p_cal = nutrition['protein'] * 4
                c_cal = nutrition['carb'] * 4
                f_cal = nutrition['fat'] * 9
                total_cal = p_cal + c_cal + f_cal
                
                if total_cal > 0:
                    chart_data = pd.DataFrame({
                        'Nutrient': ['Protein', 'Carbs', 'Fat'],
                        'Calories': [p_cal, c_cal, f_cal],
                        'Percentage': [
                            f"{p_cal/total_cal*100:.0f}%",
                            f"{c_cal/total_cal*100:.0f}%",
                            f"{f_cal/total_cal*100:.0f}%"
                        ]
                    })
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.bar_chart(chart_data.set_index('Nutrient')['Calories'])
                    
                    with col2:
                        st.dataframe(chart_data[['Nutrient', 'Percentage']], 
                                   hide_index=True)
                
                # Dietary advice
                st.markdown("---")
                st.subheader("Dietary Recommendations")
                
                for adv in advice:
                    st.markdown(f"- {adv}")
                
                # Technical details
                with st.expander("Technical Details"):
                    st.markdown(f"""
                    **Model Configuration**:
                    - Architecture: EfficientNet-B3 + MFB Class Weighting
                    - Device: {device}
                    - Threshold: {threshold}
                    - MC Dropout Samples: {20 if use_uncertainty else 'Disabled'}
                    - Co-occurrence Boosting: {'Enabled' if use_cooccur else 'Disabled'}
                    
                    **Prediction Statistics**:
                    - Detected Ingredients: {len(pred_ings)}
                    - Average Confidence: {mean_probs[pred_idx].mean():.1%}
                    - Average Uncertainty: {std_probs[pred_idx].mean():.1%}
                    
                    **Portion Input**:
                    - Method: Manual Input
                    - Total Weight: {weight_g}g
                    - Weight per Ingredient: {weight_g / max(len(pred_ings), 1):.1f}g
                    
                    **Nutrition Calculation**:
                    - Database: 555 ingredients
                    - Method: weight × nutrient_percentage_per_gram
                    - Total Calories: {nutrition['cal']:.0f} kcal
                    """)
    else:
        st.info("Upload a food image to begin analysis")

with tab2:
    st.header("About This System")
    
    st.markdown("""
    ### System Overview
    
    This AI system analyzes food images to provide:
    - Multi-label ingredient detection (249 classes)
    - Confidence scores with uncertainty quantification
    - Co-occurrence learning for improved predictions
    - Nutritional analysis with manual portion input
    - Personalized dietary recommendations
    
    ### Key Features
    
    **1. Threshold Optimization**
    - Data-driven threshold: 0.25
    - Improves F1 score by 0.9%
    
    **2. MC Dropout Uncertainty**
    - 20 forward passes
    - Quantifies prediction confidence
    
    **3. Co-occurrence Learning**
    - Learns from 2,792 training dishes
    - Boosts related ingredients
    
    **4. Nutritional Reasoning**
    - 555 ingredients in database
    - Manual portion size input
    
    **5. MFB Class Weighting**
    - Median Frequency Balancing
    - Handles class imbalance
    
    ### Dataset
    
    **Nutrition5K** (Google Research)
    - 3,490 dishes total
    - 2,792 training samples
    - RGB + Depth images
    """)

with tab3:
    st.header("System Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Metrics")
        
        metrics_df = pd.DataFrame({
            'Metric': ['F1 Score', 'Precision', 'Recall', 'Exact Match', 'Hamming Acc'],
            'Value': [0.786, 0.781, 0.824, 0.370, 0.985],
        })
        
        st.dataframe(metrics_df, hide_index=True)
        
        st.markdown("---")
        st.subheader("Threshold Analysis")
        
        threshold_data = pd.DataFrame({
            'Threshold': [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
            'F1': [0.753, 0.778, 0.786, 0.785, 0.780, 0.769, 0.749],
        })
        
        st.line_chart(threshold_data.set_index('Threshold')['F1'])
    
    with col2:
        st.subheader("Feature Impact")
        
        features_df = pd.DataFrame({
            'Feature': [
                'Base Model (B0)',
                'Upgrade to B3',
                '+ MFB Weighting',
                '+ Threshold Opt',
                '+ MC Dropout',
                '+ Co-occurrence'
            ],
            'F1 Score': [0.777, 0.783, 0.786, 0.786, 0.786, 0.790],
        })
        
        st.dataframe(features_df, hide_index=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>IE7615 Deep Learning Project | Northeastern University | 2025</p>
</div>
""", unsafe_allow_html=True)
