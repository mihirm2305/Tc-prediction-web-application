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
    # So, if theme_toggle is True, it means the user just switched it ON (to Dark Mode)
    st.session_state.theme = 'dark' if st.session_state.theme_toggle else 'light'
    # Streamlit automatically reruns on widget interaction, applying the new CSS

# --- Placeholder Functions ---
# (parse_formula, generate_features, predict_critical_temperature remain the same)
def parse_formula(formula_str: str) -> dict:
    """
    Parses a chemical formula string into a dictionary of elements and their counts.

    Args:
        formula_str: The chemical formula (e.g., 'H2S', 'MgB2').

    Returns:
        A dictionary mapping element symbols to their counts (e.g., {'H': 2.0, 'S': 1.0}).
        Returns an empty dictionary or raises an error if parsing fails.
    """
    # --- Placeholder Implementation ---
    import re
    parsed = {}
    try:
        for match in re.finditer(r"([A-Z][a-z]*)(\d*)", formula_str):
            element = match.group(1)
            count_str = match.group(2)
            count = float(count_str) if count_str else 1.0
            parsed[element] = parsed.get(element, 0) + count
        # Basic validation check
        reconstructed = "".join(f"{e}{int(c) if c != 1.0 else ''}" for e, c in sorted(parsed.items()))
        original_simplified = formula_str.replace("1.0", "").replace(".0", "")
        original_parsed_temp = {}
        for match in re.finditer(r"([A-Z][a-z]*)(\d*)", original_simplified):
             element = match.group(1)
             count_str = match.group(2)
             count = int(count_str) if count_str else 1
             original_parsed_temp[element] = original_parsed_temp.get(element, 0) + count
        original_reconstructed_sorted = "".join(f"{e}{c if c != 1 else ''}" for e,c in sorted(original_parsed_temp.items()))

        if not parsed or reconstructed != original_reconstructed_sorted:
            # Only show warning if parsing actually extracted *something* but it didn't fully match
            if parsed:
                 st.warning(f"Could not fully parse formula '{formula_str}'. Check format/elements. Using parsed elements: {list(parsed.keys())}")
            # Check if elements or total counts differ significantly even if parsing failed early
            elif not parsed or sum(parsed.values()) != sum(original_parsed_temp.values()) or set(parsed.keys()) != set(original_parsed_temp.keys()):
                 st.error(f"Failed to parse formula '{formula_str}'. Please check the input format.")
                 return {} # Return empty only on significant failure
        # Return parsed even if warning was shown, allowing feature gen attempt
        return parsed
    except Exception as e:
        st.error(f"Error parsing formula '{formula_str}': {e}")
        return {}
    # --- End Placeholder ---

def generate_features(parsed_formula: dict) -> tuple[pd.DataFrame | None, list[int], list[float]]:
    """
    Generates features for the superconductor model based on the parsed formula.

    Args:
        parsed_formula: A dictionary from parse_formula (e.g., {'Mg': 1.0, 'B': 2.0}).

    Returns:
        A tuple containing:
        - feature_vector: A pandas DataFrame or numpy array representing the input features for the model.
        - atomic_numbers: A list of atomic numbers for the elements present.
        - coefficients: A list of coefficients/counts corresponding to the atomic numbers.
        Returns (None, [], []) if feature generation fails.
    """
    # --- Placeholder Implementation ---
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
                # Don't break, collect all unknown elements
            else:
                atomic_numbers.append(num)

        if not valid_elements:
            st.error(f"Unknown element(s) found: {', '.join(unknown_elements)}. Cannot generate features.")
            return None, [], []

        # Dummy feature vector (e.g., 10 features)
        num_features = 10
        # Seed for reproducibility if needed: np.random.seed(42)
        feature_vector_data = np.random.rand(1, num_features)
        feature_vector_df = pd.DataFrame(feature_vector_data, columns=[f'feature_{i+1}' for i in range(num_features)])

        return feature_vector_df, atomic_numbers, coefficients
    except Exception as e:
        st.error(f"Error generating features: {e}")
        return None, [], []
    # --- End Placeholder ---

def predict_critical_temperature(feature_vector: pd.DataFrame, atomic_numbers: list[int], coefficients: list[float]) -> float | None:
    """
    Predicts the critical temperature using the PyTorch model.

    Args:
        feature_vector: The feature vector from generate_features.
        atomic_numbers: List of atomic numbers.
        coefficients: List of coefficients.

    Returns:
        The predicted critical temperature (Tc) in Kelvin, or None if prediction fails.
    """
    # --- Placeholder Implementation ---
    if feature_vector is None or not atomic_numbers or not coefficients:
        return None

    try:
        # Dummy prediction: Simulate some dependency on input
        # Example: base_tc + sum_of_atomic_numbers_weighted
        base_tc = 10.0
        atomic_sum_effect = sum(an * c for an, c in zip(atomic_numbers, coefficients)) / (sum(coefficients) + 1e-6) # Weighted average effect
        random_factor = np.random.rand() * 20 # Add some noise
        predicted_tc = base_tc + atomic_sum_effect * 0.5 + random_factor
        # Ensure prediction is within a reasonable range (e.g., 0K to 200K)
        predicted_tc = max(0.0, min(predicted_tc, 200.0))
        return predicted_tc
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None
    # --- End Placeholder ---


# --- Streamlit App Layout and Logic ---

# Page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Superconductor Tc Predictor",
    page_icon="🧊",
    layout="centered"
)

# --- Define CSS for Light and Dark Themes ---

# Consistent radius value
CONSISTENT_RADIUS = "6px"
RESULT_BOX_RADIUS = "8px" # Slightly larger for emphasis

light_theme_css = f"""
<style>
    /* Main background and default text color */
    body, .stApp {{
        background-color: #FFFFFF !important; /* White background */
        color: #333333 !important; /* Default dark grey text color */
    }}

    /* Title style */
    h1 {{
        color: #0B3D91;
        text-align: center;
    }}

    /* Input label */
    .stTextInput label {{
        color: #0B3D91;
        font-weight: bold;
    }}

     /* Input box styling */
    .stTextInput input {{
        border: 1px solid #83CBEB;
        border-radius: {CONSISTENT_RADIUS}; /* Consistent radius */
        padding: 10px;
        background-color: #F0F8FF;
        color: #333333;
    }}

    /* Style the placeholder text in the input box */
    .stTextInput input::placeholder {{ color: #777777 !important; opacity: 1; }}
    .stTextInput input::-webkit-input-placeholder {{ color: #777777 !important; opacity: 1; }}
    .stTextInput input::-moz-placeholder {{ color: #777777 !important; opacity: 1; }}
    .stTextInput input:-ms-input-placeholder {{ color: #777777 !important; opacity: 1; }}
    .stTextInput input::-ms-input-placeholder {{ color: #777777 !important; opacity: 1; }}

    /* Button styling */
    .stButton button {{
        background-color: #FFAA5C; color: white; border: none;
        padding: 10px 20px;
        border-radius: {CONSISTENT_RADIUS}; /* Consistent radius */
        font-weight: bold;
        transition: background-color 0.3s ease;
    }}
    .stButton button:hover {{ background-color: #D98B4A; }}
    .stButton button:active {{ background-color: #BF7A40; }}

    /* Result display area */
    .result-box {{
        background-color: #C1E5F5; border: 2px solid #83CBEB;
        border-radius: {RESULT_BOX_RADIUS}; /* Slightly larger radius */
        padding: 20px; margin-top: 20px;
        text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}
    .result-box strong {{ color: #0B3D91; font-size: 1.5em; }}
    .result-box span {{ color: #333333; font-size: 1.1em; }}

    /* Styling for expander */
    .stExpander {{
        border: 1px solid #FFDBBA;
        border-radius: {CONSISTENT_RADIUS}; /* Consistent radius */
        background-color: #FFF9F3;
        overflow: hidden; /* Ensures content respects border radius */
    }}
     .stExpander header {{
        font-weight: bold; color: #0B3D91;
        border-top-left-radius: {CONSISTENT_RADIUS}; /* Match container radius */
        border-top-right-radius: {CONSISTENT_RADIUS}; /* Match container radius */
     }}
    .stExpander .streamlit-expanderContent div {{ color: #333333; }}


    /* Ensure text written via st.write has good contrast */
    .stMarkdown, .stWrite, div[data-testid="stText"], div[data-testid="stForm"] {{
         color: #333333 !important;
    }}
     code {{
         color: #0B3D91; background-color: #eef;
         padding: 2px 5px;
         border-radius: {CONSISTENT_RADIUS}; /* Consistent radius */
    }}
    /* --- FIX V3: Targeting toggle container for light mode --- */
    div[data-testid="stToggle"] {{
        color: #333333 !important; /* Ensure text within toggle container is dark grey */
    }}
    /* Explicitly target label again just in case container doesn't cascade */
    div[data-testid="stToggle"] label {{
         color: #333333 !important;
    }}
</style>
"""

dark_theme_css = f"""
<style>
    /* Main background and default text color */
    body, .stApp {{
        background-color: #212529 !important; /* Dark background */
        color: #E0E0E0 !important; /* Default light grey text color */
    }}

    /* Title style */
    h1 {{
        color: #A8D5EF; /* Lighter blue for title */
        text-align: center;
    }}

    /* Input label */
    .stTextInput label {{
        color: #A8D5EF; /* Lighter blue */
        font-weight: bold;
    }}

     /* Input box styling */
    .stTextInput input {{
        border: 1px solid #5A96B3; /* Desaturated dark blue border */
        border-radius: {CONSISTENT_RADIUS}; /* Consistent radius */
        padding: 10px;
        background-color: #343A40; /* Darker grey background */
        color: #E0E0E0; /* Light text */
    }}

    /* Style the placeholder text in the input box */
    .stTextInput input::placeholder {{ color: #6C757D !important; opacity: 1; }}
    .stTextInput input::-webkit-input-placeholder {{ color: #6C757D !important; opacity: 1; }}
    .stTextInput input::-moz-placeholder {{ color: #6C757D !important; opacity: 1; }}
    .stTextInput input:-ms-input-placeholder {{ color: #6C757D !important; opacity: 1; }}
    .stTextInput input::-ms-input-placeholder {{ color: #6C757D !important; opacity: 1; }}


    /* Button styling */
    .stButton button {{
        background-color: #FFAA5C; color: white; border: none; /* Keep orange */
        padding: 10px 20px;
        border-radius: {CONSISTENT_RADIUS}; /* Consistent radius */
        font-weight: bold;
        transition: background-color 0.3s ease;
    }}
    .stButton button:hover {{ background-color: #D98B4A; }}
    .stButton button:active {{ background-color: #BF7A40; }}

    /* Result display area */
    .result-box {{
        background-color: #0B3D91; border: 2px solid #83CBEB; /* Dark blue bg, light blue border */
        border-radius: {RESULT_BOX_RADIUS}; /* Slightly larger radius */
        padding: 20px; margin-top: 20px;
        text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.4); /* Darker shadow */
    }}
    .result-box strong {{ color: #C1E5F5; font-size: 1.5em; }} /* Light blue text */
    .result-box span {{ color: #E0E0E0; font-size: 1.1em; }} /* Light grey text */

    /* Styling for expander */
    .stExpander {{
        border: 1px solid #A0522D; /* Darker orange border */
        border-radius: {CONSISTENT_RADIUS}; /* Consistent radius */
        background-color: #343A40; /* Darker grey background */
        overflow: hidden; /* Ensures content respects border radius */
    }}
    .stExpander header {{
        font-weight: bold; color: #A8D5EF; /* Lighter blue header */
        border-top-left-radius: {CONSISTENT_RADIUS}; /* Match container radius */
        border-top-right-radius: {CONSISTENT_RADIUS}; /* Match container radius */
    }}
    .stExpander .streamlit-expanderContent div {{ color: #E0E0E0; }} /* Light grey content */

    /* Ensure text written via st.write has good contrast */
    .stMarkdown, .stWrite, div[data-testid="stText"], div[data-testid="stForm"] {{
         color: #E0E0E0 !important;
    }}
     code {{
         color: #A8D5EF; background-color: #343A40; /* Light blue text on dark grey */
         padding: 2px 5px;
         border-radius: {CONSISTENT_RADIUS}; /* Consistent radius */
    }}
     /* --- FIX V3: Targeting toggle container for dark mode --- */
    div[data-testid="stToggle"] {{
        color: #E0E0E0 !important; /* Ensure text within toggle container is light grey */
    }}
     /* Explicitly target label again just in case container doesn't cascade */
    div[data-testid="stToggle"] label {{
         color: #E0E0E0 !important;
    }}
</style>
"""

# --- Apply Selected Theme ---
# Apply CSS based on the current theme state
if st.session_state.theme == 'dark':
    st.markdown(dark_theme_css, unsafe_allow_html=True)
else:
    st.markdown(light_theme_css, unsafe_allow_html=True)


# --- App Title ---
st.title("Superconductor Critical Temperature (Tc) Predictor")

# --- Theme Toggle ---
# Place toggle below title, use callback to update state
st.toggle(
    "Dark Mode",
    key='theme_toggle', # Assign key to access state in callback
    value=(st.session_state.theme == 'dark'), # Set initial visual state based on session state
    on_change=toggle_theme, # Call function when toggled
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
    # 1. Parse the formula
    st.write("Parsing formula...")
    parsed = parse_formula(formula_input)

    if parsed: # Proceed only if parsing didn't return an empty dict
        st.write(f"Parsed Formula: `{parsed}`") # Show parsed dict even if warning was issued

        # 2. Generate features
        st.write("Generating features...")
        features, atom_nums, coeffs = generate_features(parsed)

        # Check if features is a DataFrame before proceeding
        if isinstance(features, pd.DataFrame) and not features.empty:
            # Optionally display features in an expander
            with st.expander("View Generated Features & Input Details"):
                st.write("Input Elements:")
                st.json(parsed) # Display parsed elements/counts nicely
                st.write("Atomic Numbers Used:")
                st.write(f"`{atom_nums}`")
                st.write("Coefficients Used:")
                st.write(f"`{coeffs}`")
                st.write("Example Feature Vector (Dummy Data):")
                st.dataframe(features)

            # 3. Predict Tc
            st.write("Predicting critical temperature...")
            predicted_tc = predict_critical_temperature(features, atom_nums, coeffs)

            # 4. Display Result
            st.markdown("---") # Divider
            if predicted_tc is not None:
                # Use markdown with the custom class for styling
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
                # Error during prediction step
                st.error("Prediction failed after feature generation.")
        # Handle case where feature generation returns None or empty list/DataFrame explicitly
        elif features is None:
             # Error message already shown in generate_features
             st.error("Feature generation failed. Cannot proceed with prediction.")
        # This case should be less likely now if generate_features handles errors
        else:
             st.warning("Feature generation resulted in empty data. Cannot predict Tc.")
    # else: # Parsing failed and returned {} - error message already shown in parse_formula
        # st.info("Enter a valid chemical formula to proceed.")


# Add a footer (optional)
st.markdown("---")
st.caption("Note: This app uses placeholder functions for parsing, feature generation, and prediction. Replace them with your actual implementation.")
