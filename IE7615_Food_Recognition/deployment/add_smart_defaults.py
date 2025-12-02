# 在 app.py 中加入食材分類和智能預設

ingredient_categories = {
    # 主食類 - 預設 Medium bowl (250g)
    'staples': {
        'ingredients': ['white rice', 'brown rice', 'jasmine rice', 'basmati rice', 
                       'pasta', 'spaghetti', 'penne', 'noodles', 'ramen',
                       'quinoa', 'couscous', 'bulgur', 'bread', 'tortilla'],
        'default': 'Medium bowl (250g)',
        'default_weight': 250
    },
    
    # 堅果類 - 預設 Palm size (30g)
    'nuts': {
        'ingredients': ['almonds', 'walnuts', 'cashews', 'peanuts', 'pecans',
                       'pistachios', 'hazelnuts', 'macadamia nuts'],
        'default': 'Palm size (30g)',
        'default_weight': 30
    },
    
    # 蔬菜類 - 預設 Fist size (100g)
    'vegetables': {
        'ingredients': ['spinach', 'lettuce', 'kale', 'arugula', 'broccoli',
                       'carrot', 'tomato', 'cucumber', 'bell pepper', 'onions',
                       'mushroom', 'cabbage', 'chard', 'brussels sprouts'],
        'default': 'Fist size (100g)',
        'default_weight': 100
    },
    
    # 肉類 - 預設 Palm size (120g)
    'proteins': {
        'ingredients': ['chicken', 'beef', 'pork', 'turkey', 'salmon', 'tuna',
                       'shrimp', 'tofu', 'tempeh', 'eggs', 'bacon'],
        'default': 'Palm size (120g)',
        'default_weight': 120
    },
    
    # 調味料 - 預設 Tiny (10g)
    'condiments': {
        'ingredients': ['salt', 'pepper', 'olive oil', 'soy sauce', 'vinegar',
                       'ketchup', 'mustard', 'mayo', 'hot sauce', 'garlic',
                       'ginger', 'herbs', 'spices'],
        'default': 'Tiny (10g)',
        'default_weight': 10
    },
    
    # 其他 - 預設 Medium (100g)
    'others': {
        'default': 'Medium (100g)',
        'default_weight': 100
    }
}

def get_ingredient_category(ingredient_name):
    """Get category and default weight for ingredient"""
    ing_lower = ingredient_name.lower()
    
    for category, data in ingredient_categories.items():
        if category == 'others':
            continue
        if any(item in ing_lower for item in data['ingredients']):
            return category, data['default'], data['default_weight']
    
    # Default
    return 'others', ingredient_categories['others']['default'], ingredient_categories['others']['default_weight']

# 使用範例
category, preset_name, default_weight = get_ingredient_category('almonds')
# Returns: ('nuts', 'Palm size (30g)', 30)
