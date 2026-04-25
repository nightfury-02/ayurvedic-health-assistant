"""Unity Catalog FQNs and AyurGenix Delta column names (sanitized CSV headers)."""

CURATED_FQN = "ayurveda_lakehouse.ayurgenix.ayurgenix_curated"
CHUNKS_FQN = "ayurveda_lakehouse.ayurgenix.pdf_text_chunks"

CURATED_TEXT_COLUMNS = (
    "Disease",
    "Hindi_Name",
    "Marathi_Name",
    "Symptoms",
    "Diagnosis_Tests",
    "Ayurvedic_Herbs",
    "Formulation",
    "Doshas",
    "Herbal_Alternative_Remedies",
    "Diet_and_Lifestyle_Recommendations",
    "Patient_Recommendations",
)

CURATED_DISPLAY_COLUMNS = (
    "Disease",
    "Symptoms",
    "Ayurvedic_Herbs",
    "Formulation",
    "Doshas",
    "Diet_and_Lifestyle_Recommendations",
    "Patient_Recommendations",
)
