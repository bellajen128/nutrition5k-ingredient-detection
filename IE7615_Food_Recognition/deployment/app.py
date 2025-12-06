"""
Food Nutrition Analysis System
AI-powered ingredient detection with manual editing and individual portion control
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
# Download Model
# ============================================================

@st.cache_resource
def download_model_if_needed():
    """Download model from Google Drive if not exists"""
    model_path = "efficientnet_best.pth"
    
    if not Path(model_path).exists():
        with st.spinner('Downloading model from Google Drive (134MB, first time only)...'):
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
    
    # Use relative paths (Streamlit Cloud handles this)
    vocab_path = "ingredient_vocab.json"
    nutrition_path = "ingredients.xlsx"
    cooccur_path = "cooccurrence_prob.npy"
    
    # Download model (will be in same directory)
    model_path = download_model_if_needed()
    
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)['vocab']
    
    nutrition_df = pd.read_excel(nutrition_path)
    nutrition_df.columns = [c.lower().replace(' ','_').replace('(','').replace(')','') 
                           for c in nutrition_df.columns]
    
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

def predict_with_uncertainty(model, img_tensor, device, n=10):
    """MC Dropout uncertainty estimation"""
    # Set model to eval first
    model.eval()
    
    # Only enable dropout layers
    dropout_layers = [m for m in model.classifier.modules() if isinstance(m, nn.Dropout)]
    for m in dropout_layers:
        m.train()
    
    preds = []
    for _ in range(n):
        with torch.no_grad():
            out = model(img_tensor)
            preds.append(torch.sigmoid(out).cpu().numpy())
    
    # Return everything to eval
    model.eval()
    
    preds = np.array(preds)
    mean = preds.mean(axis=0)[0]
    std = preds.std(axis=0)[0]
    
    return mean, std

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

def calc_nutrition_individual(ingredient_weights, df):
    """Calculate nutrition from individual ingredient weights"""
    tot = {'cal': 0, 'protein': 0, 'carb': 0, 'fat': 0, 'total_wt': 0}
    
    for ing, wt in ingredient_weights.items():
        if wt > 0:
            n = get_nutrition(ing, df)
            tot['cal'] += n['cal'] * wt
            tot['protein'] += n['protein'] * wt
            tot['carb'] += n['carb'] * wt
            tot['fat'] += n['fat'] * wt
            tot['total_wt'] += wt
    
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
# Session State Initialization
# ============================================================

if 'detected_ingredients' not in st.session_state:
    st.session_state.detected_ingredients = []
if 'ingredient_weights' not in st.session_state:
    st.session_state.ingredient_weights = {}
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Settings")

threshold = st.sidebar.slider(
    "Prediction Threshold",
    0.1, 0.8, 0.20, 0.05,
    help="Lower = detect more ingredients"
)

use_uncertainty = st.sidebar.checkbox(
    "Show Uncertainty Scores",
    value=False,
    help="MC Dropout (slower)"
)

use_cooccur = st.sidebar.checkbox(
    "Co-occurrence Boosting (Experimental)",
    value=False,
    help="Increases recall but may reduce precision. Default: OFF"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
st.sidebar.info("""
**Model**: EfficientNet-B3 + MFB  
**F1 Score**: 0.814 
**Precision**: 0.803  
**Recall**: 0.858  
**Optimal Threshold**: 0.20

**Features**:
- Threshold Optimization
- MC Dropout Uncertainty
- Co-occurrence (Experimental)
- Nutritional Reasoning
""")

# ============================================================
# Main Interface
# ============================================================

tab1, tab2 = st.tabs(["Analyze", "About"])

with tab1:
    st.subheader("Upload Food Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a photo of your food"
    )
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Your Dish", use_column_width=True)
        
        if st.button("Detect Ingredients", type="primary"):
            
            with st.spinner("Analyzing image..."):
                # Preprocess
                tfm = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                img_tensor = tfm(img).unsqueeze(0).to(device)
                
                # Predict
                if use_uncertainty:
                    mean_probs, std_probs = predict_with_uncertainty(model, img_tensor, device, 10)
                else:
                    with torch.no_grad():
                        out = model(img_tensor)
                        mean_probs = torch.sigmoid(out).cpu().numpy()[0]
                        std_probs = np.zeros_like(mean_probs)
                
                # Co-occurrence
                if use_cooccur and cooccur_prob is not None:
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
                
                # Store in session state
                st.session_state.detected_ingredients = [
                    {
                        'name': vocab[i],
                        'confidence': float(mean_probs[i]),
                        'uncertainty': float(std_probs[i])
                    }
                    for i in pred_idx
                ]
                
                # Initialize weights (default 100g each)
                st.session_state.ingredient_weights = {
                    vocab[i]: 100.0 for i in pred_idx
                }
                
                st.session_state.analysis_done = True
                st.rerun()
    
    # Show results if analysis done
    if st.session_state.analysis_done and len(st.session_state.detected_ingredients) > 0:
        
        st.markdown("---")
        st.subheader("Detected Ingredients")
        
        # Display detected ingredients with confidence
        for item in sorted(st.session_state.detected_ingredients, key=lambda x: x['confidence'], reverse=True):
            conf = item['confidence']
            unc = item['uncertainty']
            
            if unc < 0.05:
                indicator = "[+++] Very Confident"
            elif unc < 0.10:
                indicator = "[++] Confident"
            elif unc < 0.20:
                indicator = "[+] Uncertain"
            else:
                indicator = "[?] Very Uncertain"
            
            if use_uncertainty:
                st.markdown(f"**{item['name']}**: {conf:.0%} (±{unc:.1%}) - {indicator}")
            else:
                st.markdown(f"**{item['name']}**: {conf:.0%}")
        
        # Edit ingredients section
        st.markdown("---")
        st.subheader("Edit Ingredients & Portions")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Remove Ingredients**")
            
            current_ingredients = list(st.session_state.ingredient_weights.keys())
            
            if current_ingredients:
                to_remove = st.multiselect(
                    "Select ingredients to remove:",
                    current_ingredients,
                    key="remove_select"
                )
                
                if st.button("Remove Selected") and to_remove:
                    for ing in to_remove:
                        del st.session_state.ingredient_weights[ing]
                    st.success(f"Removed {len(to_remove)} ingredient(s)")
                    st.rerun()
        
        with col2:
            st.markdown("**Add Ingredients**")
            
            available_ingredients = [ing for ing in vocab if ing not in st.session_state.ingredient_weights]
            
            if available_ingredients:
                new_ingredients = st.multiselect(
                    "Select ingredients to add:",
                    sorted(available_ingredients),
                    key="add_select"
                )

                if st.button("Add Selected") and new_ingredients:
                    for ing in new_ingredients:
                        st.session_state.ingredient_weights[ing] = 100.0
                    st.success(f"Added {len(new_ingredients)} ingredient(s)")
                    st.rerun()
        
        # Individual portion input
        st.markdown("---")
        st.subheader("Set Individual Portions")
        
        if st.session_state.ingredient_weights:
            
            st.markdown("Adjust weight for each ingredient:")
            
            # Create editable table
            portion_data = []
            
            for ing in sorted(st.session_state.ingredient_weights.keys()):
                portion_data.append({
                    'Ingredient': ing,
                    'Weight (g)': st.session_state.ingredient_weights[ing]
                })
            
            # Display as editable inputs
            updated_weights = {}
            
            # Common portion presets
            portion_presets = {
            "Tiny (10g)": 10,                    # 0 - 調味料
            "Thumb size (30g)": 30,              # 1 - 堅果
            "Fist size (100g)": 100,             # 2 - 預設/蔬菜
            "Palm size (120g)": 120,             # 3 - 肉類
            "Small bowl (150g)": 150,            # 4
            "Small plate (200g)": 200,           # 5
            "Medium bowl (250g)": 250,           # 6 - 主食
            "Medium plate (350g)": 350,          # 7
            "Large bowl (400g)": 400,            # 8
            "Large plate (500g)": 500            # 9
            }

            def get_default_preset(ing_name):
                """Get smart default based on ingredient type"""
                ing_lower = ing_name.lower()
                
                # 堅果類 - Thumb size
                if any(nut in ing_lower for nut in ['almond', 'walnut', 'cashew', 'peanut', 'pecan']):
                    return 1
                
                # 主食類 - Medium bowl
                if any(grain in ing_lower for grain in ['rice', 'pasta', 'noodle', 'bread', 'quinoa']):
                    return 6
                
                # 調味料 - Tiny
                if any(cond in ing_lower for cond in ['salt', 'pepper', 'oil', 'sauce', 'vinegar']):
                    return 0
                
                # 預設 - Fist size
                return 2    
            
            
            
            for ing in sorted(st.session_state.ingredient_weights.keys()):
                cols = st.columns([3, 2, 2])
                
                with cols[0]:
                    st.markdown(f"**{ing}**")
                
                with cols[1]:
                    preset = st.selectbox(
                        "Preset",
                        list(portion_presets.keys()),
                        key=f"preset_{ing}",
                        label_visibility="collapsed",
                        index=get_default_preset(ing)

                    )
                    preset_weight = portion_presets[preset]
                
                with cols[2]:
                    custom_weight = st.number_input(
                        "Custom (g)",
                        min_value=1,
                        max_value=500,
                        value=int(preset_weight),
                        step=10,
                        key=f"weight_{ing}",
                        label_visibility="collapsed"
                    )
                    updated_weights[ing] = custom_weight
            
            # Update session state
            st.session_state.ingredient_weights = updated_weights
            
            # Calculate nutrition
            st.markdown("---")
            st.subheader("Nutritional Analysis")
            
            nutrition = calc_nutrition_individual(st.session_state.ingredient_weights, nutrition_df)
            
            if nutrition['total_wt'] > 0:
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Weight", f"{nutrition['total_wt']:.0f} g")
                with col2:
                    st.metric("Calories", f"{nutrition['cal']:.0f} kcal")
                with col3:
                    st.metric("Protein", f"{nutrition['protein']:.1f} g")
                with col4:
                    st.metric("Carbs", f"{nutrition['carb']:.1f} g")
                with col5:
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
                        st.dataframe(chart_data[['Nutrient', 'Percentage']], hide_index=True)
                
                # Detailed breakdown
                with st.expander("Detailed Ingredient Breakdown"):
                    breakdown = []
                    for ing, wt in st.session_state.ingredient_weights.items():
                        n = get_nutrition(ing, nutrition_df)
                        breakdown.append({
                            'Ingredient': ing,
                            'Weight (g)': wt,
                            'Calories': n['cal'] * wt,
                            'Protein (g)': n['protein'] * wt,
                            'Carbs (g)': n['carb'] * wt,
                            'Fat (g)': n['fat'] * wt
                        })
                    
                    breakdown_df = pd.DataFrame(breakdown)
                    st.dataframe(breakdown_df, hide_index=True)
                
                # Dietary advice
                st.markdown("---")
                st.subheader("Dietary Recommendations")
                
                advice = get_advice(nutrition)
                for adv in advice:
                    st.markdown(f"- {adv}")
        
        else:
            st.warning("No ingredients selected. Add ingredients to calculate nutrition.")
    
    else:
        st.info("Upload a food image and click 'Detect Ingredients' to begin")

with tab2:
    st.header("About This System")
    
    st.markdown("""
    ### Key Features
    
    **1. AI Ingredient Detection**
    - EfficientNet-B3 with MFB Class Weighting
    - 249 ingredient classes
    - F1 Score: 0.814
    
    **2. Threshold Optimization**
    - Data-driven threshold: 0.20
    - Balances precision (0.803) and recall (0.858)
    
    **3. MC Dropout Uncertainty**
    - 20 forward passes per prediction
    - Quantifies prediction confidence
    
    **4. Co-occurrence Learning (Experimental)**
    - Learns from 2,792 training dishes
    - Tested but decreased F1 by 7.6% due to false positives
    - Kept as optional feature (default: OFF)
    
    **5. Manual Editing**
    - Remove incorrect predictions
    - Add missing ingredients
    - Set individual portion sizes
    
    **6. Nutritional Analysis**
    - 555 ingredients in database
    - Per-ingredient weight control
    - Detailed macro breakdown
    
    ### Performance
    
    | Metric | Value |
    |--------|-------|
    | F1 Score | 0.814 |
    | Precision | 0.803 |
    | Recall | 0.858 |
    | Exact Match | 0.364 |
    | Hamming Accuracy | 0.988 |
    
    ### Dataset
    
    **Nutrition5K** (Google Research)
    - 3,490 RGB + Depth images
    - 2,792 training dishes
    - 349 validation dishes
    - 349 test dishes
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>IE7615 Deep Learning Project | Northeastern University | 2025</p>
</div>
""", unsafe_allow_html=True)
