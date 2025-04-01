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
        # Handle empty or non-string input gracefully
        # st.info("Please enter a chemical formula.") # Optional feedback
        return {}
    try:
        # Improved regex to handle elements like 'Cl' followed by numbers or other elements
        # It looks for an uppercase letter, optionally followed by a lowercase letter,
        # then optionally followed by digits.
        for match in re.finditer(r"([A-Z][a-z]?)(\d*)", formula_str):
            element = match.group(1)
            count_str = match.group(2)
            # Default count is 1.0 if no number follows the element
            count = float(count_str) if count_str else 1.0
            # Add to existing count if element repeats (e.g., H2O + H = H3O)
            parsed[element] = parsed.get(element, 0.0) + count

        # --- Validation Step ---
        # Reconstruct formula from parsed dict for validation
        # Sort by element symbol for consistent comparison
        reconstructed_parts = []
        for element, count in sorted(parsed.items()):
            count_int = int(count)
            # Append count only if it's > 1; handle floating point comparison carefully
            if count > 1.0001: # Use tolerance for float comparison
                 reconstructed_parts.append(f"{element}{count_int}")
            # Append element without count if count is 1
            elif abs(count - 1.0) < 0.0001:
                 reconstructed_parts.append(element)
            # Handle fractional counts if necessary (though less common in basic formulas)
            # else:
            #    reconstructed_parts.append(f"{element}{count}") # Or decide how to handle fractions

        reconstructed = "".join(reconstructed_parts)

        # Also reconstruct the *original* formula in a sorted, simplified way
        original_parsed_temp = {}
        original_total_chars = 0
        for match in re.finditer(r"([A-Z][a-z]?)(\d*)", formula_str):
             element = match.group(1)
             count_str = match.group(2)
             count = int(count_str) if count_str else 1
             original_parsed_temp[element] = original_parsed_temp.get(element, 0) + count
             original_total_chars += len(match.group(0)) # Count characters consumed by regex

        # Check if all characters in the original string were parsed
        if original_total_chars != len(formula_str.replace(" ", "")): # Ignore spaces
             st.error(f"Failed to parse the entire formula string '{formula_str}'. Check for invalid characters or format.")
             return {}

        original_reconstructed_sorted_parts = []
        for element, count in sorted(original_parsed_temp.items()):
             if count > 1:
                 original_reconstructed_sorted_parts.append(f"{element}{count}")
             else:
                 original_reconstructed_sorted_parts.append(element)
        original_reconstructed_sorted = "".join(original_reconstructed_sorted_parts)


        # Compare reconstructed versions
        # This comparison is tricky due to potential order differences (e.g., H2O vs OH2)
        # Comparing the sorted versions is more robust
        if not parsed or reconstructed != original_reconstructed_sorted:
             # Only show warning if parsing actually extracted *something*
             if parsed:
                  st.warning(f"Parsing result may differ from input format or handle complex cases partially: '{formula_str}' -> `{parsed}`. Proceeding with parsed elements.")
             else:
                  st.error(f"Failed to parse formula '{formula_str}'. Please check the input format.")
                  return {} # Return empty only on complete failure

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
        # Using a more comprehensive map
        atomic_numbers_map = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
            'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
            'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
            'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
            'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
            'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
            'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
            'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
            'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
            'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
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
                # Ensure atomic numbers correspond to the order of coefficients
                atomic_numbers.append(num)

        if not valid_elements:
            st.error(f"Unknown element(s) found: {', '.join(unknown_elements)}. Cannot generate features.")
            return None, [], []

        # Ensure lists are not empty before proceeding
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
    if feature_vector is None or not atomic_numbers or not coefficients:
        return None

    try:
        # Dummy prediction
        base_tc = 10.0
        # Ensure coefficients sum is not zero to avoid division by zero
        sum_coeffs = sum(coefficients)
        if abs(sum_coeffs) < 1e-9:
             # Handle case with zero coefficients if necessary, maybe return base_tc or error
             st.warning("Sum of coefficients is zero, prediction might be trivial.")
             atomic_sum_effect = 0
        else:
             atomic_sum_effect = sum(an * c for an, c in zip(atomic_numbers, coefficients)) / sum_coeffs

        random_factor = np.random.rand() * 20
        predicted_tc = base_tc + atomic_sum_effect * 0.5 + random_factor
        predicted_tc = max(0.0, min(predicted_tc, 200.0)) # Clamp prediction
        return predicted_tc
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None
    # --- End Placeholder ---


# --- Streamlit App Layout and Logic ---

st.set_page_config(
    page_title="Superconductor Tc Predictor",
    page_icon="🧊",
    layout="centered"
)

# --- Define CSS for Light and Dark Themes ---
CONSISTENT_RADIUS = "6px"
RESULT_BOX_RADIUS = "8px"

light_theme_css = f"""
<style>
    /* Base styles */
    body, .stApp {{
        background-color: #FFFFFF !important;
        color: #333333 !important;
    }}
    h1 {{ color: #0B3D91; text-align: center; }}

    /* Input field */
    .stTextInput label {{ color: #0B3D91; font-weight: bold; }}
    /* --- BORDER FIX V2: Target input with higher specificity --- */
    div[data-testid="stTextInput"] input {{
        border: 1px solid #83CBEB !important; /* Apply desired border */
        border-radius: {CONSISTENT_RADIUS} !important;
        background-color: #F0F8FF !important;
        color: #333333 !important;
        padding: 10px !important; /* Ensure padding is consistent */
    }}
    .stTextInput input::placeholder {{ color: #777777 !important; opacity: 1; }}
    /* ... other placeholder styles ... */

    /* Button */
    .stButton button {{
        background-color: #FFAA5C !important; color: white !important; border: none !important;
        padding: 10px 20px !important;
        border-radius: {CONSISTENT_RADIUS} !important;
        font-weight: bold !important;
        transition: background-color 0.3s ease !important;
    }}
    .stButton button:hover {{ background-color: #D98B4A !important; }}
    .stButton button:active {{ background-color: #BF7A40 !important; }}

    /* Result box (custom class, less likely to conflict) */
    .result-box {{
        background-color: #C1E5F5; border: 2px solid #83CBEB;
        border-radius: {RESULT_BOX_RADIUS}; padding: 20px; margin-top: 20px;
        text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}
    .result-box strong {{ color: #0B3D91; font-size: 1.5em; }}
    .result-box span {{ color: #333333; font-size: 1.1em; }}

    /* Expander */
    /* --- BORDER FIX V2: Target expander container with higher specificity --- */
    div[data-testid="stExpander"] {{
        border: 1px solid #FFDBBA !important; /* Apply desired border */
        border-radius: {CONSISTENT_RADIUS} !important;
        background-color: #FFF9F3 !important;
        overflow: hidden !important;
    }}
    /* Target header within the specific expander container */
    div[data-testid="stExpander"] > div:first-child {{ /* Usually the header container */
         /* Reset potential default border/background on header parts */
         border: none !important;
         border-bottom: 1px solid #FFDBBA !important; /* Optional: separator line */
         background-color: #FFF9F3 !important; /* Match container background */
    }}
     div[data-testid="stExpander"] header {{ /* Target the <summary> or similar element */
        font-weight: bold !important; color: #0B3D91 !important;
        border-radius: 0 !important; /* Remove radius from header itself */
        padding: 0.5rem 1rem !important; /* Adjust padding as needed */
     }}
    /* Target content area text color */
    div[data-testid="stExpander"] .streamlit-expanderContent div {{
        color: #333333 !important;
        padding: 1rem !important; /* Add padding to content */
    }}

    /* General text elements */
    .stMarkdown, .stWrite, div[data-testid="stText"], div[data-testid="stForm"] {{
         color: #333333 !important;
    }}
     code {{
         color: #0B3D91 !important; background-color: #eef !important;
         padding: 2px 5px !important;
         border-radius: {CONSISTENT_RADIUS} !important;
    }}

    /* --- TOGGLE FIX V4: Target label with higher specificity --- */
    div[data-testid="stApp"] div[data-testid="stToggle"] label {{
        color: #333333 !important; /* Dark grey for light mode */
    }}
</style>
"""

dark_theme_css = f"""
<style>
    /* Base styles */
    body, .stApp {{
        background-color: #212529 !important;
        color: #E0E0E0 !important;
    }}
    h1 {{ color: #A8D5EF; text-align: center; }}

    /* Input field */
    .stTextInput label {{ color: #A8D5EF; font-weight: bold; }}
    /* --- BORDER FIX V2: Target input with higher specificity --- */
    div[data-testid="stTextInput"] input {{
        border: 1px solid #5A96B3 !important; /* Apply desired border */
        border-radius: {CONSISTENT_RADIUS} !important;
        background-color: #343A40 !important;
        color: #E0E0E0 !important;
        padding: 10px !important;
    }}
    .stTextInput input::placeholder {{ color: #6C757D !important; opacity: 1; }}
    /* ... other placeholder styles ... */

    /* Button */
    .stButton button {{
        background-color: #FFAA5C !important; color: white !important; border: none !important;
        padding: 10px 20px !important;
        border-radius: {CONSISTENT_RADIUS} !important;
        font-weight: bold !important;
        transition: background-color 0.3s ease !important;
    }}
    .stButton button:hover {{ background-color: #D98B4A !important; }}
    .stButton button:active {{ background-color: #BF7A40 !important; }}

    /* Result box */
    .result-box {{
        background-color: #0B3D91; border: 2px solid #83CBEB;
        border-radius: {RESULT_BOX_RADIUS}; padding: 20px; margin-top: 20px;
        text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.4);
    }}
    .result-box strong {{ color: #C1E5F5; font-size: 1.5em; }}
    .result-box span {{ color: #E0E0E0; font-size: 1.1em; }}

    /* Expander */
    /* --- BORDER FIX V2: Target expander container with higher specificity --- */
    div[data-testid="stExpander"] {{
        border: 1px solid #A0522D !important; /* Apply desired border */
        border-radius: {CONSISTENT_RADIUS} !important;
        background-color: #343A40 !important;
        overflow: hidden !important;
    }}
     /* Target header within the specific expander container */
    div[data-testid="stExpander"] > div:first-child {{
         border: none !important;
         border-bottom: 1px solid #A0522D !important; /* Optional: separator line */
         background-color: #343A40 !important; /* Match container background */
    }}
     div[data-testid="stExpander"] header {{
        font-weight: bold !important; color: #A8D5EF !important;
        border-radius: 0 !important;
        padding: 0.5rem 1rem !important;
     }}
    /* Target content area text color */
    div[data-testid="stExpander"] .streamlit-expanderContent div {{
        color: #E0E0E0 !important;
        padding: 1rem !important;
    }}

    /* General text elements */
    .stMarkdown, .stWrite, div[data-testid="stText"], div[data-testid="stForm"] {{
         color: #E0E0E0 !important;
    }}
     code {{
         color: #A8D5EF !important; background-color: #343A40 !important;
         padding: 2px 5px !important;
         border-radius: {CONSISTENT_RADIUS} !important;
    }}

    /* --- TOGGLE FIX V4: Target label with higher specificity --- */
    div[data-testid="stApp"] div[data-testid="stToggle"] label {{
        color: #E0E0E0 !important; /* Light grey for dark mode */
    }}
</style>
"""

# --- Apply Selected Theme ---
if st.session_state.theme == 'dark':
    st.markdown(dark_theme_css, unsafe_allow_html=True)
else:
    st.markdown(light_theme_css, unsafe_allow_html=True)


# --- App Title ---
st.title("Superconductor Critical Temperature (Tc) Predictor")

# --- Theme Toggle ---
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
        st.write(f"Parsed Formula: `{parsed}`") # Show parsed dict

        st.write("Generating features...")
        features, atom_nums, coeffs = generate_features(parsed)

        if isinstance(features, pd.DataFrame) and not features.empty:
            with st.expander("View Generated Features & Input Details"):
                st.write("Input Elements:")
                st.json(parsed)
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
        else:
             st.warning("Feature generation resulted in empty data. Cannot predict Tc.")
    # else: # Parsing failed - error message likely already shown

# --- Footer ---
st.markdown("---")
st.caption("Note: This app uses placeholder functions. Replace them with your actual implementation.")

