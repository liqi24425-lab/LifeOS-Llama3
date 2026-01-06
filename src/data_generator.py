import json
import random
import os

# 配置
INPUT_FILE = 'data/seed_data.json'
OUTPUT_FILE = 'data/generated_data_10k.json'
TOTAL_SAMPLES = 10000

# 语气模板
STYLES = [
    "Clinical",   # "Protocol initiated."
    "Coach",      # "No excuses. Get it done."
    "Empathetic", # "Sorry to hear that. Let's adjust."
    "Casual"      # "Here's the plan."
]

TEMPLATES = {
    "schedule": [
        "What is the plan for {day}?",
        "Do I have gym on {day}?",
        "Walk me through {day}.",
        "Schedule check: {day}."
    ],
    "health": [
        "I feel {symptom}.",
        "My {body_part} hurts.",
        "Dealing with {symptom} today."
    ]
}

def load_seed_data():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Please ensure {INPUT_FILE} exists.")
    with open(INPUT_FILE, 'r') as f:
        return json.load(f)

def generate_entry(seed_data):
    """
    核心生成逻辑：随机选择意图、风格和上下文
    """
    category = random.choice(['schedule', 'health'])
    style = random.choice(STYLES)
    
    if category == 'schedule':
        # 从 seed data 提取 workout 数据
        days = list(seed_data['workout_schedule']['weekly_plan'].keys())
        day = random.choice(days)
        plan = seed_data['workout_schedule']['weekly_plan'][day]
        
        instruction = random.choice(TEMPLATES['schedule']).format(day=day.capitalize())
        
        # 根据风格构建回答
        focus = plan.get('focus', 'Rest')
        if style == "Coach":
            response = f"Listen up! It's {day}. Focus: {focus}. Go crush it."
        elif style == "Clinical":
            response = f"Schedule: {day.capitalize()}. Primary Activity: {focus}."
        else:
            response = f"For {day}, we are doing {focus}."
            
        return {"instruction": instruction, "input": "", "output": response}

    elif category == 'health':
        # 从 seed data 提取健康协议
        scenarios = list(seed_data['health_protocols']['scenarios'].keys())
        scenario_key = random.choice(scenarios)
        data = seed_data['health_protocols']['scenarios'][scenario_key]
        
        symptom = scenario_key.replace('_', ' ')
        body_part = symptom.split()[0]
        
        if 'pain' in symptom:
             instruction = f"My {body_part} hurts."
        else:
             instruction = random.choice(TEMPLATES['health']).format(symptom=symptom, body_part=body_part)
        
        # 提取建议
        protocol_text = json.dumps(data['protocol']) # 简化处理，实际可格式化
        source = data.get('source', 'Medical Guidelines')
        
        if style == "Coach":
            response = f"Stop immediately. {protocol_text}"
        else:
            response = f"Protocol for {symptom}: {protocol_text}. (Source: {source})"
            
        return {"instruction": instruction, "input": "Context: Health Issue", "output": response}

def main():
    print(f"🚀 Starting Data Factory... Target: {TOTAL_SAMPLES} samples.")
    seed_data = load_seed_data()
    dataset = []
    
    for _ in range(TOTAL_SAMPLES):
        dataset.append(generate_entry(seed_data))
        
    # 保存
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Generated {len(dataset)} samples at {OUTPUT_FILE}")

if __name__ == "__main__":
    main()