import streamlit as st
import pandas as pd # Used for feature vector display example
import numpy as np # Used for dummy data generation

# --- Placeholder Functions ---
# (parse_formula, generate_features, predict_critical_temperature remain the same)
# Note: Added minor improvements to parsing feedback.
def parse_formula(formula_str: str) -> dict:
    """
    Parses a chemical formula string into a dictionary of elements and their counts.
    Placeholder implementation.
    """
    import re
    parsed = {}
    if not isinstance(formula_str, str) or not formula_str.strip():
        # st.info("Please enter a chemical formula.") # Optional feedback
        return {}
    try:
        # Regex: Uppercase letter, optional lowercase, optional digits
        parsing_log = [] # Track parsed parts
        last_index = 0
        formula_cleaned = formula_str.replace(" ", "") # Remove spaces for index tracking

        for match in re.finditer(r"([A-Z][a-z]?)(\d*)", formula_cleaned):
            if match.start() != last_index:
                 # Gap detected, indicates unparsed characters
                 unparsed_segment = formula_cleaned[last_index:match.start()]
                 st.error(f"Invalid characters or format detected near: '{unparsed_segment}' in '{formula_str}'")
                 return {}

            element = match.group(1)
            count_str = match.group(2)
            count = float(count_str) if count_str else 1.0
            parsed[element] = parsed.get(element, 0.0) + count
            parsing_log.append(match.group(0))
            last_index = match.end()

        # Check if the entire string was consumed
        if last_index != len(formula_cleaned):
             unparsed_segment = formula_cleaned[last_index:]
             st.error(f"Could not parse trailing characters: '{unparsed_segment}' in '{formula_str}'")
             return {}

        if not parsed: # Handle cases where regex might match nothing valid
             st.error(f"Could not parse any elements from '{formula_str}'. Check format.")
             return {}

        # Basic validation comparing parsed elements vs original string structure
        # (This is complex; the current checks mainly ensure full string consumption
        # and valid element/number patterns)

        return parsed
    except Exception as e:
        st.error(f"Unexpected error parsing formula '{formula_str}': {e}")
        return {}
# --- End Placeholder ---

def generate_features(parsed_formula: dict) -> tuple[pd.DataFrame | None, list[int], list[float]]:
    """
    Generates features for the superconductor model based on the parsed formula.
    Placeholder implementation.
    """
    if not parsed_formula: return None, [], []
    try:
        elements = list(parsed_formula.keys())
        coefficients = list(parsed_formula.values())
        atomic_numbers_map = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
        }
        atomic_numbers = []
        valid_elements = True
        unknown_elements = []
        for el in elements:
            num = atomic_numbers_map.get(el)
            if num is None: unknown_elements.append(el); valid_elements = False
            else: atomic_numbers.append(num)
        if not valid_elements:
            st.error(f"Unknown element(s) found: {', '.join(unknown_elements)}. Cannot generate features.")
            return None, [], []
        if not atomic_numbers or not coefficients or len(atomic_numbers) != len(coefficients):
             st.error("Mismatch between elements and coefficients after parsing. Cannot generate features.")
             return None, [], []
        num_features = 10
        feature_vector_data = np.random.rand(1, num_features)
        feature_vector_df = pd.DataFrame(feature_vector_data, columns=[f'feature_{i+1}' for i in range(num_features)])
        return feature_vector_df, atomic_numbers, coefficients
    except Exception as e:
        st.error(f"Error generating features: {e}")
        return None, [], []
# --- End Placeholder ---

def predict_critical_temperature(feature_vector: pd.DataFrame, atomic_numbers: list[int], coefficients: list[float]) -> float | None:
    """ Predicts the critical temperature using a placeholder model. """
    if feature_vector is None or not atomic_numbers or not coefficients: return None
    try:
        base_tc = 10.0
        sum_coeffs = sum(coefficients)
        if abs(sum_coeffs) < 1e-9: atomic_sum_effect = 0
        else: atomic_sum_effect = sum(an * c for an, c in zip(atomic_numbers, coefficients)) / sum_coeffs
        random_factor = np.random.rand() * 20
        predicted_tc = base_tc + atomic_sum_effect * 0.5 + random_factor
        predicted_tc = max(0.0, min(predicted_tc, 200.0))
        return predicted_tc
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None
# --- End Placeholder ---


# --- Streamlit App Layout and Logic ---
st.set_page_config(page_title="Superconductor Tc Predictor", page_icon="✨", layout="centered") # Changed icon

# --- Define CSS for the App (Permanent Dark Theme) ---
APP_RADIUS = "8px"  # Slightly larger radius for a softer look
ACCENT_COLOR = "#00CED1" # Vibrant Dark Turquoise
ACCENT_COLOR_HOVER = "#00B0A8" # Slightly darker for hover
ACCENT_COLOR_ACTIVE = "#008F86" # Even darker for active
BG_COLOR = "#1A1A2E" # Dark blue/purple background
COMPONENT_BG_COLOR = "#162447" # Slightly lighter component background
TEXT_COLOR = "#E0E0E0" # Light text
TEXT_COLOR_MUTED = "#A0AEC0" # Muted text (like placeholders)
BORDER_COLOR = "#4A5568" # Subtle border color

app_css = f"""
<style>
    /* --- Base Styles --- */
    body, .stApp {{
        background-color: {BG_COLOR} !important;
        color: {TEXT_COLOR} !important;
        font-family: 'Inter', sans-serif; /* Optional: Use a specific clean font */
    }}
    h1 {{
        color: {ACCENT_COLOR};
        text-align: center;
        font-weight: 700; /* Bolder title */
        margin-bottom: 1.5rem; /* More space below title */
    }}

    /* --- Input Field --- */
    .stTextInput label {{
        color: {ACCENT_COLOR};
        font-weight: 600; /* Semi-bold label */
        margin-bottom: 0.5rem; /* Space between label and input */
    }}
    div[data-testid="stTextInput"] input {{
        border: 1px solid {BORDER_COLOR} !important;
        border-radius: {APP_RADIUS} !important;
        background-color: {COMPONENT_BG_COLOR} !important;
        color: {TEXT_COLOR} !important;
        padding: 12px 15px !important; /* Slightly more padding */
        transition: border-color 0.3s ease, box-shadow 0.3s ease; /* Smooth transition */
    }}
    /* Input focus style */
    div[data-testid="stTextInput"] input:focus {{
        outline: none !important;
        border-color: {ACCENT_COLOR} !important; /* Highlight border on focus */
        box-shadow: 0 0 0 3px rgba(0, 206, 209, 0.3) !important; /* Subtle glow effect */
    }}
    .stTextInput input::placeholder {{
        color: {TEXT_COLOR_MUTED} !important;
        opacity: 1;
    }}

    /* --- Button --- */
    .stButton button {{
        background-color: {ACCENT_COLOR} !important;
        color: #FFFFFF !important; /* White text on accent button */
        border: none !important;
        padding: 12px 24px !important; /* More padding */
        border-radius: {APP_RADIUS} !important;
        font-weight: 600 !important; /* Semi-bold text */
        transition: background-color 0.3s ease, transform 0.1s ease !important;
        width: 100%; /* Make button full width */
        margin-top: 1rem; /* Space above button */
    }}
    .stButton button:hover {{
        background-color: {ACCENT_COLOR_HOVER} !important;
        transform: translateY(-2px); /* Slight lift on hover */
    }}
    .stButton button:active {{
        background-color: {ACCENT_COLOR_ACTIVE} !important;
        transform: translateY(0px); /* Back to normal on click */
    }}
    /* Center the button if needed (depends on Streamlit version) */
    /* div[data-testid="stVerticalBlock"] div[data-testid="stButton"] {{
        display: flex;
        justify-content: center;
    }} */


    /* --- Result Box --- */
    .result-box {{
        background: linear-gradient(135deg, {COMPONENT_BG_COLOR}, {BG_COLOR}); /* Subtle gradient */
        border: 1px solid {ACCENT_COLOR}; /* Accent border */
        border-radius: {APP_RADIUS};
        padding: 25px 30px; /* Generous padding */
        margin-top: 2rem; /* More space above result */
        text-align: center;
        box-shadow: 0 6px 12px rgba(0, 206, 209, 0.15); /* Soft shadow */
        transition: transform 0.3s ease;
    }}
     .result-box:hover {{
         transform: translateY(-3px); /* Slight lift on hover */
     }}
    .result-box strong {{
        color: {ACCENT_COLOR};
        font-size: 2.5em; /* Larger result font */
        font-weight: 700;
        display: block; /* Ensure it takes full width */
        margin-top: 0.5rem; /* Space between label and result */
    }}
    .result-box span {{
        color: {TEXT_COLOR};
        font-size: 1.1em;
    }}

    /* --- Expander --- */
    div[data-testid="stExpander"] {{
        border: 1px solid {BORDER_COLOR} !important;
        border-radius: {APP_RADIUS} !important;
        background-color: {COMPONENT_BG_COLOR} !important;
        overflow: hidden !important;
        margin-top: 1.5rem; /* Space above expander */
    }}
    /* Expander header container */
    div[data-testid="stExpander"] > div:first-child {{
         border: none !important;
         background: none !important;
    }}
     /* Expander header text/icon area */
     div[data-testid="stExpander"] summary {{
        font-weight: 600 !important;
        color: {TEXT_COLOR} !important;
        border-radius: 0 !important;
        padding: 0.75rem 1.25rem !important; /* Adjust padding */
        border-bottom: 1px solid {BORDER_COLOR} !important;
        transition: background-color 0.3s ease;
     }}
      div[data-testid="stExpander"] summary:hover {{
          background-color: rgba(255, 255, 255, 0.05); /* Subtle hover */
      }}
    /* Expander content area */
    div[data-testid="stExpander"] .streamlit-expanderContent div {{
        color: {TEXT_COLOR} !important;
        padding: 1.25rem !important; /* Adjust padding */
        border-top: none !important; /* Ensure no double border */
    }}
    /* Style dataframes inside expander */
     div[data-testid="stExpander"] .stDataFrame {{
         border: 1px solid {BORDER_COLOR};
         border-radius: {APP_RADIUS};
     }}


    /* --- Code Blocks --- */
     code {{
         background-color: {BG_COLOR} !important; /* Match main background */
         color: {ACCENT_COLOR} !important; /* Accent color for code */
         padding: 3px 6px !important;
         border-radius: {APP_RADIUS} !important;
         border: 1px solid {BORDER_COLOR}; /* Subtle border */
         font-size: 0.9em;
    }}

    /* --- General Text & Links --- */
    .stMarkdown, .stWrite, div[data-testid="stText"], div[data-testid="stForm"], .stCaption {{
        color: {TEXT_COLOR} !important;
    }}
     .stCaption {{
         color: {TEXT_COLOR_MUTED} !important; /* Muted color for caption */
         text-align: center;
         margin-top: 2rem;
     }}
    a {{
        color: {ACCENT_COLOR} !important;
        text-decoration: none !important; /* Remove underline */
        transition: color 0.3s ease;
    }}
    a:hover {{
        color: {ACCENT_COLOR_HOVER} !important;
        text-decoration: underline !important; /* Add underline on hover */
    }}

    /* --- Remove Toggle Styling (No longer needed) --- */
    div[data-testid="stToggle"] {{
        display: none !important; /* Hide the toggle completely */
    }}

</style>
"""

# --- Apply Theme ---
st.markdown(app_css, unsafe_allow_html=True)


# --- App Title ---
st.title("Superconductor Tc Predictor")
# Removed the toggle widget

st.markdown("---") # Divider

# --- Input Section ---
# Use a form to group input and button
with st.form("prediction_form"):
    formula_input = st.text_input(
        "Enter Chemical Formula:",
        placeholder="e.g., MgB2, YBa2Cu3O7",
        help="Enter the chemical formula of the material (e.g., H2O, Fe2O3)."
    )
    submitted = st.form_submit_button("✨ Predict Tc ✨") # Changed button text

# --- Processing and Output ---
if submitted and formula_input: # Process only when form is submitted
    st.write("Parsing formula...")
    parsed = parse_formula(formula_input)

    if parsed:
        # Displaying parsed formula immediately after successful parsing
        # st.write(f"Parsed Formula: `{parsed}`") # Optional: Can show in expander instead

        st.write("Generating features...")
        features, atom_nums, coeffs = generate_features(parsed)

        if isinstance(features, pd.DataFrame) and not features.empty:
            with st.expander("View Input Details & Features"): # Changed expander title
                st.write("**Input Interpretation:**")
                st.json(parsed) # Use st.json for better dict display
                st.write("**Atomic Numbers Used:**")
                st.write(f"`{atom_nums}`")
                st.write("**Coefficients Used:**")
                st.write(f"`{coeffs}`")
                st.write("**Example Feature Vector (Dummy Data):**")
                st.dataframe(features)

            st.write("Predicting critical temperature...")
            predicted_tc = predict_critical_temperature(features, atom_nums, coeffs)

            st.markdown("---") # Divider before result
            if predicted_tc is not None:
                # Display result using the custom styled div
                st.markdown(
                    f"""
                    <div class="result-box">
                        <span>Predicted Critical Temperature (Tc) for <strong>{formula_input}</strong>:</span>
                        <strong>{predicted_tc:.2f} K</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.error("Prediction failed after feature generation.")
        elif features is None:
             st.error("Feature generation failed. Cannot proceed.")
        else:
             st.warning("Feature generation resulted in empty data. Cannot predict Tc.")
    # else: # Parsing failed - error message already shown by parse_formula

elif submitted and not formula_input:
     st.warning("Please enter a chemical formula before predicting.")


# --- Footer ---
st.markdown("---")
st.caption("✨ Built with Streamlit | Placeholder Model ✨") # Updated caption
