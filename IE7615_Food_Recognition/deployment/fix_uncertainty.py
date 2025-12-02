import re

with open('app.py', 'r') as f:
    content = f.read()

# 找到並替換 predict_with_uncertainty 函數
old_func = '''def predict_with_uncertainty(model, img_tensor, device, n=20):
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
    return preds.mean(axis=0)[0], preds.std(axis=0)[0]'''

new_func = '''def predict_with_uncertainty(model, img_tensor, device, n=20):
    """MC Dropout uncertainty estimation"""
    # Enable dropout for uncertainty
    model.train()
    for m in model.classifier.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    
    preds = []
    with torch.no_grad():
        for _ in range(n):
            out = model(img_tensor)
            preds.append(torch.sigmoid(out).cpu().numpy())
    
    # Return to eval mode
    model.eval()
    
    preds = np.array(preds)
    mean = preds.mean(axis=0)[0]
    std = preds.std(axis=0)[0]
    
    return mean, std'''

content = content.replace(old_func, new_func)

with open('app.py', 'w') as f:
    f.write(content)

print("✓ Fixed MC Dropout uncertainty")
