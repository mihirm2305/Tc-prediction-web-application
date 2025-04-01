import streamlit as st
import pandas as pd # Used for feature vector display example
import numpy as np # Used for dummy data generation

# --- Initialize Session State ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'light' # Default to light mode
if 'theme_toggle' not in st.session_state: # Initialize toggle state if not present
    st.session_state.theme_toggle = (st.session_state.theme == 'dark')


# --- Function to Toggle Theme ---
def toggle_theme():
    # The toggle's value in session_state ('theme_toggle') reflects its NEW state *after* the click
    st.session_state.theme = 'dark' if st.session_state.theme_toggle else 'light'
    # Streamlit automatically reruns on widget interaction, applying the new CSS

# --- Placeholder Functions ---
# (parse_formula, generate_features, predict_critical_temperature remain the same)
def parse_formula(formula_str: str) -> dict:
    """
    Parses a chemical formula string into a dictionary of elements and their counts.
    Placeholder implementation.
    """
    import re
    parsed = {}
    if not isinstance(formula_str, str) or not formula_str.strip():
        return {}
    try:
        # Regex: Uppercase letter, optional lowercase, optional digits
        for match in re.finditer(r"([A-Z][a-z]?)(\d*)", formula_str):
            element = match.group(1)
            count_str = match.group(2)
            count = float(count_str) if count_str else 1.0
            parsed[element] = parsed.get(element, 0.0) + count

        # --- Validation Step (Basic) ---
        # Reconstruct simple formula for basic check
        reconstructed_parts = []
        for element, count in sorted(parsed.items()):
            count_int = int(count)
            if count > 1.0001:
                 reconstructed_parts.append(f"{element}{count_int}")
            elif abs(count - 1.0) < 0.0001:
                 reconstructed_parts.append(element)
        reconstructed = "".join(reconstructed_parts)

        # Count parsed characters
        original_total_chars = 0
        original_parsed_temp = {} # For comparing element sets/counts
        for match in re.finditer(r"([A-Z][a-z]?)(\d*)", formula_str):
             element = match.group(1)
             count_str = match.group(2)
             count = int(count_str) if count_str else 1
             original_parsed_temp[element] = original_parsed_temp.get(element, 0) + count
             original_total_chars += len(match.group(0))

        # Check if all characters were consumed by parsing
        if original_total_chars != len(formula_str.replace(" ", "")):
             st.error(f"Failed to parse the entire formula string '{formula_str}'. Check for invalid characters or format.")
             return {}

        # Check if parsed elements match original elements (useful if regex was too simple)
        if set(parsed.keys()) != set(original_parsed_temp.keys()) or not parsed:
             if parsed: # Only warn if something was parsed but it's inconsistent
                 st.warning(f"Parsing inconsistency detected for '{formula_str}'. Parsed elements: `{list(parsed.keys())}`. Original elements: `{list(original_parsed_temp.keys())}`. Proceeding with parsed data.")
             else: # Complete failure to parse
                 st.error(f"Failed to parse formula '{formula_str}'.")
                 return {}

        return parsed
    except Exception as e:
        st.error(f"Error parsing formula '{formula_str}': {e}")
        return {}
    # --- End Placeholder ---

def generate_features(parsed_formula: dict) -> tuple[pd.DataFrame | None, list[int], list[float]]:
    """
    Generates features for the superconductor model based on the parsed formula.
    Placeholder implementation.
    """
    if not parsed_formula:
        return None, [], []
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
            if num is None:
                unknown_elements.append(el)
                valid_elements = False
            else:
                atomic_numbers.append(num)

        if not valid_elements:
            st.error(f"Unknown element(s) found: {', '.join(unknown_elements)}. Cannot generate features.")
            return None, [], []
        if not atomic_numbers or not coefficients or len(atomic_numbers) != len(coefficients):
             st.error("Mismatch between elements and coefficients after parsing. Cannot generate features.")
             return None, [], []

        # Dummy feature vector
        num_features = 10
        feature_vector_data = np.random.rand(1, num_features)
        feature_vector_df = pd.DataFrame(feature_vector_data, columns=[f'feature_{i+1}' for i in range(num_features)])
        return feature_vector_df, atomic_numbers, coefficients
    except Exception as e:
        st.error(f"Error generating features: {e}")
        return None, [], []
    # --- End Placeholder ---

def predict_critical_temperature(feature_vector: pd.DataFrame, atomic_numbers: list[int], coefficients: list[float]) -> float | None:
    """
    Predicts the critical temperature using a placeholder model.
    """
    if feature_vector is None or not atomic_numbers or not coefficients: return None
    try:
        base_tc = 10.0
        sum_coeffs = sum(coefficients)
        if abs(sum_coeffs) < 1e-9:
             atomic_sum_effect = 0
        else:
             atomic_sum_effect = sum(an * c for an, c in zip(atomic_numbers, coefficients)) / sum_coeffs
        random_factor = np.random.rand() * 20
        predicted_tc = base_tc + atomic_sum_effect * 0.5 + random_factor
        predicted_tc = max(0.0, min(predicted_tc, 200.0))
        return predicted_tc
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None
    # --- End Placeholder ---


# --- Streamlit App Layout and Logic ---
st.set_page_config(page_title="Superconductor Tc Predictor", page_icon="🧊", layout="centered")

# --- Define CSS for Light and Dark Themes ---
CONSISTENT_RADIUS = "6px"
RESULT_BOX_RADIUS = "8px"

# --- Common CSS Rules (Applied in both themes) ---
common_css = f"""
<style>
    /* --- TOGGLE WORKAROUND: Dark container for visibility --- */
    div[data-testid="stToggle"] {{
        background-color: #343A40 !important; /* Dark background always */
        padding: 10px 15px !important;       /* Padding around toggle */
        border-radius: {CONSISTENT_RADIUS} !important; /* Match other radii */
        margin-bottom: 1rem !important;       /* Add some space below */
        border: 1px solid #5A96B3 !important; /* Add a subtle border */
    }}
    /* Ensure label inside dark toggle container is light */
    div[data-testid="stToggle"] label {{
         color: #E0E0E0 !important; /* Light grey text always */
    }}

    /* --- INPUT OUTLINE FIX: Remove default browser outline on focus --- */
    div[data-testid="stTextInput"] input:focus {{
        outline: none !important;
        box-shadow: none !important; /* Also remove potential box-shadow outline */
    }}

    /* Base styles */
    h1 {{ text-align: center; }}
    .stTextInput label {{ font-weight: bold; }}
    .stButton button {{
        border: none !important;
        padding: 10px 20px !important;
        border-radius: {CONSISTENT_RADIUS} !important;
        font-weight: bold !important;
        transition: background-color 0.3s ease !important;
    }}
    .result-box {{
        border-radius: {RESULT_BOX_RADIUS}; padding: 20px; margin-top: 20px;
        text-align: center;
    }}
    .result-box strong {{ font-size: 1.5em; }}
    .result-box span {{ font-size: 1.1em; }}

    /* Expander container */
    div[data-testid="stExpander"] {{
        border-radius: {CONSISTENT_RADIUS} !important;
        overflow: hidden !important;
    }}
    /* Expander header container */
    div[data-testid="stExpander"] > div:first-child {{
         border: none !important;
         background: none !important; /* Let header style handle background */
    }}
     /* Expander header text/icon area */
     div[data-testid="stExpander"] summary {{ /* Changed from header to summary for potentially better targeting */
        font-weight: bold !important;
        border-radius: 0 !important;
        padding: 0.5rem 1rem !important;
        border-bottom: 1px solid; /* Separator line - color set in theme */
     }}
    /* Expander content area */
    div[data-testid="stExpander"] .streamlit-expanderContent div {{
        padding: 1rem !important;
    }}

     code {{
         padding: 2px 5px !important;
         border-radius: {CONSISTENT_RADIUS} !important;
    }}
</style>
"""

# --- Light Theme Specific CSS ---
light_theme_css = f"""
<style>
    body, .stApp {{ background-color: #FFFFFF !important; color: #333333 !important; }}
    h1 {{ color: #0B3D91; }}
    .stTextInput label {{ color: #0B3D91; }}
    div[data-testid="stTextInput"] input {{
        border: 1px solid #83CBEB !important;
        border-radius: {CONSISTENT_RADIUS} !important;
        background-color: #F0F8FF !important;
        color: #333333 !important;
        padding: 10px !important;
    }}
    .stTextInput input::placeholder {{ color: #777777 !important; opacity: 1; }}
    .stButton button {{ background-color: #FFAA5C !important; color: white !important; }}
    .stButton button:hover {{ background-color: #D98B4A !important; }}
    .stButton button:active {{ background-color: #BF7A40 !important; }}
    .result-box {{
        background-color: #C1E5F5; border: 2px solid #83CBEB;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}
    .result-box strong {{ color: #0B3D91; }}
    .result-box span {{ color: #333333; }}
    div[data-testid="stExpander"] {{ border: 1px solid #FFDBBA !important; background-color: #FFF9F3 !important; }}
    div[data-testid="stExpander"] summary {{ color: #0B3D91 !important; border-color: #FFDBBA !important; }}
    div[data-testid="stExpander"] .streamlit-expanderContent div {{ color: #333333 !important; }}
    .stMarkdown, .stWrite, div[data-testid="stText"], div[data-testid="stForm"] {{ color: #333333 !important; }}
     code {{ color: #0B3D91 !important; background-color: #eef !important; }}
</style>
"""

# --- Dark Theme Specific CSS ---
dark_theme_css = f"""
<style>
    body, .stApp {{ background-color: #212529 !important; color: #E0E0E0 !important; }}
    h1 {{ color: #A8D5EF; }}
    .stTextInput label {{ color: #A8D5EF; }}
    div[data-testid="stTextInput"] input {{
        border: 1px solid #5A96B3 !important;
        border-radius: {CONSISTENT_RADIUS} !important;
        background-color: #343A40 !important;
        color: #E0E0E0 !important;
        padding: 10px !important;
    }}
    .stTextInput input::placeholder {{ color: #6C757D !important; opacity: 1; }}
    .stButton button {{ background-color: #FFAA5C !important; color: white !important; }}
    .stButton button:hover {{ background-color: #D98B4A !important; }}
    .stButton button:active {{ background-color: #BF7A40 !important; }}
    .result-box {{
        background-color: #0B3D91; border: 2px solid #83CBEB;
        box-shadow: 0 4px 8px rgba(0,0,0,0.4);
    }}
    .result-box strong {{ color: #C1E5F5; }}
    .result-box span {{ color: #E0E0E0; }}
    div[data-testid="stExpander"] {{ border: 1px solid #A0522D !important; background-color: #343A40 !important; }}
    div[data-testid="stExpander"] summary {{ color: #A8D5EF !important; border-color: #A0522D !important; }}
    div[data-testid="stExpander"] .streamlit-expanderContent div {{ color: #E0E0E0 !important; }}
    .stMarkdown, .stWrite, div[data-testid="stText"], div[data-testid="stForm"] {{ color: #E0E0E0 !important; }}
     code {{ color: #A8D5EF !important; background-color: #343A40 !important; }}
</style>
"""

# --- Apply Selected Theme ---
# Inject common styles first, then theme-specific ones
st.markdown(common_css, unsafe_allow_html=True)
if st.session_state.theme == 'dark':
    st.markdown(dark_theme_css, unsafe_allow_html=True)
else:
    st.markdown(light_theme_css, unsafe_allow_html=True)


# --- App Title ---
st.title("Superconductor Critical Temperature (Tc) Predictor")

# --- Theme Toggle (now styled by common_css) ---
st.toggle(
    "Dark Mode",
    key='theme_toggle',
    value=(st.session_state.theme == 'dark'),
    on_change=toggle_theme,
    help="Switch between light and dark themes"
)

st.markdown("---") # Divider

# --- Input Section ---
formula_input = st.text_input(
    "Enter Chemical Formula:",
    placeholder="e.g., MgB2, YBa2Cu3O7",
    help="Enter the chemical formula of the material."
)

# --- Processing and Output ---
if formula_input:
    st.write("Parsing formula...")
    parsed = parse_formula(formula_input)

    if parsed:
        # Displaying parsed formula immediately after successful parsing
        st.write(f"Parsed Formula: `{parsed}`")

        st.write("Generating features...")
        features, atom_nums, coeffs = generate_features(parsed)

        if isinstance(features, pd.DataFrame) and not features.empty:
            with st.expander("View Generated Features & Input Details"):
                st.write("Input Elements:")
                st.json(parsed) # Use st.json for better dict display
                st.write("Atomic Numbers Used:")
                st.write(f"`{atom_nums}`")
                st.write("Coefficients Used:")
                st.write(f"`{coeffs}`")
                st.write("Example Feature Vector (Dummy Data):")
                st.dataframe(features)

            st.write("Predicting critical temperature...")
            predicted_tc = predict_critical_temperature(features, atom_nums, coeffs)

            st.markdown("---")
            if predicted_tc is not None:
                st.markdown(
                    f"""
                    <div class="result-box">
                        <span>Predicted Critical Temperature (Tc) for <strong>{formula_input}</strong>:</span><br>
                        <strong>{predicted_tc:.2f} K</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.error("Prediction failed after feature generation.")
        elif features is None:
             # Error message likely already shown in generate_features
             st.error("Feature generation failed. Cannot proceed.")
        # This case implies generate_features returned an empty DataFrame/list, not None
        else:
             st.warning("Feature generation resulted in empty data. Cannot predict Tc.")
    # else: # Parsing failed - error message likely already shown

# --- Footer ---
st.markdown("---")
st.caption("Note: This app uses placeholder functions. Replace them with your actual implementation.")
