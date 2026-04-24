DATA = [
    {
        "disease": "Cough",
        "keywords": ["cough", "throat", "congestion", "khansi"],
        "dosha": ["Kapha", "Vata"],
        "advice": "Drink warm fluids, avoid cold foods"
    },
    {
        "disease": "Diabetes",
        "keywords": ["fatigue", "urination", "sugar"],
        "dosha": ["Kapha", "Pitta"],
        "advice": "Avoid sugar, follow low-GI diet, exercise regularly"
    },
    {
        "disease": "Hypertension",
        "keywords": ["bp", "pressure", "stress"],
        "dosha": ["Pitta", "Vata"],
        "advice": "Reduce salt, practice meditation"
    }
]

def predict(text: str):
    t = (text or "").lower()
    for row in DATA:
        if any(k in t for k in row["keywords"]):
            return row
    return {
        "disease": "General imbalance",
        "dosha": ["Vata"],
        "advice": "Improve sleep, reduce stress, eat balanced diet"
    }