import re

with open('app.py', 'r') as f:
    content = f.read()

# 找到 load_system 函數
old_paths = '''    # Download model if needed
    model_path = download_model_if_needed()
    
    # Other paths (relative)
    vocab_path = "ingredient_vocab.json"
    nutrition_path = "ingredients.xlsx"
    cooccur_path = "cooccurrence_prob.npy"'''

new_paths = '''    # Download model if needed
    model_path = download_model_if_needed()
    
    # Ensure we're in the right directory
    import os
    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    os.chdir(script_dir)
    
    # Other paths (relative to script location)
    vocab_path = "ingredient_vocab.json"
    nutrition_path = "ingredients.xlsx"
    cooccur_path = "cooccurrence_prob.npy"'''

content = content.replace(old_paths, new_paths)

with open('app.py', 'w') as f:
    f.write(content)

print("✓ Fixed paths")
