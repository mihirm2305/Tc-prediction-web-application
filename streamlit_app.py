import streamlit as st
import pandas as pd # Used for feature vector display example
import numpy as np # Used for dummy data generation
import time # To simulate work for progress bar
import torch
import torch.nn as nn
import os
import re # Added for formula parsing
from collections import defaultdict # Added for parsing

# --- Constants from Training (MUST MATCH TRAINING SCRIPT) ---
MAX_ATOMIC_LEN = 10 # Max sequence length used for padding during training
NUM_ELEMENTS = 93   # Max atomic number + 1, used for embedding layer size
TOP_FEATURES = [    # The exact list and order of features used for training
    'MagneticType1_weighted_std',
    'MagneticType1_weighted_mean',
    'Valence_weighted_std',
    'SpecificFusionHeat_weighted_std',
    'AtomicMass_weighted_std',
    'AtomicNumber_weighted_std',
    'MagneticType2_weighted_mean', # Note: Ensure this exists in your elemental data
    'SpecificFusionHeat_weighted_mean',
    'Block_weighted_std',
    'Block_weighted_mean'
]
NUM_TOP_FEATURES = len(TOP_FEATURES)
EMBEDDING_SIZE = 64
HIDDEN_SIZE_FEATURES = 64
HIDDEN_SIZE_1 = 96
HIDDEN_SIZE_2 = 64
P_DROP = 0.2
MODEL_PATH = 'web_model.pth' # Relative path to the saved model
ELEMENTAL_DATA_PATH = 'elemental_properties_cleaned.csv' # Path to elemental data

# --- Element Cache for Validation ---
# Based on NUM_ELEMENTS = 93 (max atomic number = 92, Uranium)
element_cache = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92
    # Elements beyond 92 are excluded based on NUM_ELEMENTS=93
}

# --- Formula Parsing Function (Adapted from Notebook) ---
def parse_formula(formula_str: str) -> dict:
    """
    Parses a chemical formula string, handling brackets and validating elements.

    Args:
        formula_str (str): The chemical formula string (e.g., "YBa2Cu3O7", "Ca(OH)2").

    Returns:
        dict: Dictionary mapping element symbols to float coefficients, or {} on failure.
    """
    if not isinstance(formula_str, str) or not formula_str.strip():
        return {}

    formula_cleaned = formula_str.replace(" ", "")

    # Basic check for allowed characters (letters, numbers, parentheses, brackets, dots)
    if not re.fullmatch(r'^[a-zA-Z0-9\(\)\[\]\.]+$', formula_cleaned):
          st.error(f"Formula '{formula_str}' contains invalid characters.")
          return {}

    element_counts = defaultdict(float)
    i = 0
    formula_length = len(formula_cleaned)
    parsed_elements_in_scope = set() # Track unique elements found at this level/scope

    while i < formula_length:
        char = formula_cleaned[i]

        # --- Handle Brackets Recursively ---
        if char in ('(', '['):
            bracket_type = char
            closing_bracket = ')' if bracket_type == '(' else ']'
            bracket_level = 1
            j = i + 1
            # Find matching closing bracket
            while j < formula_length:
                if formula_cleaned[j] == bracket_type:
                    bracket_level += 1
                elif formula_cleaned[j] == closing_bracket:
                    bracket_level -= 1
                if bracket_level == 0:
                    break
                j += 1

            if bracket_level != 0:
                st.error(f"Mismatched brackets found in '{formula_str}'.")
                return {} # Unbalanced brackets

            sub_formula = formula_cleaned[i + 1:j]
            # Recursively parse the sub-formula
            sub_elements_dict = parse_formula(sub_formula)

            if not sub_elements_dict: # Check if recursive call returned empty dict (error)
                  # Error message should have been displayed by the recursive call
                  return {} # Propagate failure

            # Find the multiplier after the closing bracket
            k = j + 1
            multiplier_str = ''
            while k < formula_length and (formula_cleaned[k].isdigit() or formula_cleaned[k] == '.'):
                multiplier_str += formula_cleaned[k]
                k += 1

            try:
                multiplier = float(multiplier_str) if multiplier_str else 1.0
                if multiplier <= 0:
                      st.error(f"Multiplier after bracket '{closing_bracket}' must be positive, found '{multiplier_str}'.")
                      return {}
            except ValueError:
                  st.error(f"Invalid multiplier format after bracket '{closing_bracket}': '{multiplier_str}'")
                  return {}

            # Add counts from sub-formula multiplied by the multiplier
            for symbol, sub_count in sub_elements_dict.items():
                element_counts[symbol] += sub_count * multiplier
                parsed_elements_in_scope.add(symbol) # Add elements found inside brackets

            i = k # Move index past the multiplier
            continue # Continue parsing after the multiplier

        # --- Handle Elements ---
        if char.isupper():
            symbol = char
            j = i + 1
            # Find full element symbol (e.g., 'He', 'Ba')
            while j < formula_length and formula_cleaned[j].islower():
                symbol += formula_cleaned[j]
                j += 1

            # Validate element symbol and atomic number
            if symbol not in element_cache:
                st.error(f"Unknown element symbol encountered: '{symbol}' in '{formula_str}'.")
                return {}
            atomic_number = element_cache[symbol]
            if atomic_number >= NUM_ELEMENTS:
                  st.error(f"Element '{symbol}' (Z={atomic_number}) exceeds the maximum atomic number ({NUM_ELEMENTS - 1}) supported by the model.")
                  return {}

            # Find coefficient
            count_str = ''
            k = j
            while k < formula_length and (formula_cleaned[k].isdigit() or formula_cleaned[k] == '.'):
                count_str += formula_cleaned[k]
                k += 1

            try:
                count = float(count_str) if count_str else 1.0
                if count <= 0:
                      st.error(f"Coefficient for element '{symbol}' must be positive, found '{count_str}'.")
                      return {}
            except ValueError:
                  st.error(f"Invalid coefficient format for element '{symbol}': '{count_str}'")
                  return {}

            element_counts[symbol] += count
            parsed_elements_in_scope.add(symbol)
            i = k # Move index past the coefficient
            continue # Continue parsing

        # If character is not an uppercase letter or opening bracket, it's invalid here
        st.error(f"Invalid character or format encountered at index {i}: '{char}' in '{formula_str}'.")
        return {}

    # --- Final Validations ---
    # Check if any elements were parsed
    if not element_counts:
          st.error(f"Could not parse any valid elements/structure from '{formula_str}'.")
          return {}

    # Check max unique elements limit
    if len(parsed_elements_in_scope) > MAX_ATOMIC_LEN:
        st.error(f"Formula contains {len(parsed_elements_in_scope)} unique elements, exceeding the maximum ({MAX_ATOMIC_LEN}) supported by the model.")
        return {}

    # Remove elements with zero count (might occur with complex bracket math)
    final_counts = {k: v for k, v in element_counts.items() if v > 1e-9} # Use tolerance

    if not final_counts: # Check if removing zeros made it empty
        st.error(f"Parsed formula resulted in zero counts for all elements: '{formula_str}'.")
        return {}

    return dict(final_counts) # Return as a standard dict


# --- Elemental Data Loading (Cached) ---
@st.cache_data # Cache the loaded elemental data DataFrame
def load_elemental_data(path=ELEMENTAL_DATA_PATH):
    """Loads the elemental properties data from the specified CSV path."""
    if not os.path.exists(path):
        st.error(f"Elemental properties file not found at '{path}'. Cannot generate features.")
        return None
    try:
        element_data = pd.read_csv(path, index_col=0) # Assuming first column is element symbol index
        # Basic check for required 'Z' column
        if 'Z' not in element_data.columns:
              st.error(f"Elemental data file ('{path}') is missing the required 'Z' column for atomic numbers.")
              return None
        print(f"Elemental data '{path}' loaded successfully.")
        # Optional: Further cleaning if needed (e.g., drop specific columns like Group/Period if they exist)
        element_data = element_data.drop(columns=[col for col in element_data.columns if 'Group' in col or 'Period' in col or 'Neutron' in col], errors='ignore')
        return element_data
    except Exception as e:
        st.error(f"Error loading elemental data from '{path}': {e}")
        return None

# --- Feature Calculation Helper ---
def calculate_weighted_stats(elements_dict: dict, element_data_df: pd.DataFrame):
    """
    Calculates weighted mean and std dev for elemental properties based on a dictionary.

    Args:
        elements_dict (dict): Dictionary mapping element symbols (str) to coefficients (float).
        element_data_df (pd.DataFrame): DataFrame of elemental properties, indexed by symbol.

    Returns:
        dict: A dictionary containing the calculated statistics for each property.
              Returns None if errors occur (e.g., unknown element, missing data).
    """
    statistics = {}
    atomic_numbers_list = [] # To store [(symbol, Z), ...] for sorting
    coefficients_list = [] # To store [(symbol, coeff), ...] for sorting

    # Store relevant properties and coefficients for calculation
    properties_data = {} # {property_name: [values_list]}
    coefficients_for_calc = {} # {property_name: [coeffs_list]}

    # Check elements and gather data (element validation already done in parse_formula)
    for element, ratio in elements_dict.items():
        # Assume element exists in element_data_df due to prior validation
        atomic_number = element_data_df.loc[element, 'Z']
        atomic_numbers_list.append((element, int(atomic_number)))
        coefficients_list.append((element, float(ratio)))

        # Gather property data for statistics calculation
        for feature in element_data_df.columns:
            if feature == 'Z': continue # Skip Z itself

            prop_value = element_data_df.loc[element, feature]

            if feature not in properties_data:
                properties_data[feature] = []
                coefficients_for_calc[feature] = []

            properties_data[feature].append(prop_value)
            coefficients_for_calc[feature].append(ratio)

    # Calculate statistics for each property
    for feature, values in properties_data.items():
        coeffs = coefficients_for_calc[feature]
        values_arr = np.array(values, dtype=float) # Ensure float type for NaN checks
        coeffs_arr = np.array(coeffs, dtype=float)

        # Check for NaNs *before* calculation for this specific feature
        if np.isnan(values_arr).any():
            statistics[feature + '_weighted_mean'] = np.nan
            statistics[feature + '_weighted_std'] = np.nan
            continue # Skip calculation for this feature

        if len(values_arr) == 0 or len(coeffs_arr) == 0 or coeffs_arr.sum() < 1e-9: # Use tolerance for sum check
            statistics[feature + '_weighted_mean'] = np.nan
            statistics[feature + '_weighted_std'] = np.nan
            continue # Avoid division by zero or calculation on empty arrays

        try:
            weighted_mean = np.average(values_arr, weights=coeffs_arr)
            # Weighted standard deviation calculation
            variance = np.average((values_arr - weighted_mean) ** 2, weights=coeffs_arr)
            # Ensure variance is non-negative due to potential floating point issues
            weighted_std = np.sqrt(max(0, variance))

            statistics[feature + '_weighted_mean'] = weighted_mean
            statistics[feature + '_weighted_std'] = weighted_std
        except ZeroDivisionError:
              st.warning(f"ZeroDivisionError during stats calculation for '{feature}'. Setting stats to NaN.")
              statistics[feature + '_weighted_mean'] = np.nan
              statistics[feature + '_weighted_std'] = np.nan
        except Exception as e:
              st.error(f"Error calculating stats for '{feature}': {e}")
              statistics[feature + '_weighted_mean'] = np.nan
              statistics[feature + '_weighted_std'] = np.nan


    # Sort atomic numbers and coefficients by element symbol for consistent internal representation if needed
    # Note: These sorted lists are not returned by generate_features anymore
    atomic_numbers_list.sort(key=lambda x: x[0])
    coefficients_list.sort(key=lambda x: x[0])

    # Add the sorted lists to the statistics dictionary (optional, if needed elsewhere)
    # statistics['AtomicNumbers_Sorted'] = [int(an) for _, an in atomic_numbers_list]
    # statistics['Coefficients_Sorted'] = [float(coef) for _, coef in coefficients_list]

    return statistics


# --- Feature Generation Function ---
def generate_features(parsed_formula: dict) -> tuple[pd.DataFrame | None, list[int], list[float]]:
    """
    Generates features for the superconductor model based on the parsed formula dictionary.

    Args:
        parsed_formula (dict): Dictionary mapping element symbols (str) to coefficients (float).
                               (Assumes this dict is already validated by parse_formula).

    Returns:
        tuple[pd.DataFrame | None, list[int], list[float]]:
            - DataFrame with ONE row containing the TOP_FEATURES required by the model, or None if error.
            - List of atomic numbers (original order based on parsed_formula keys).
            - List of coefficients (original order based on parsed_formula values).
            Returns (None, [], []) on failure.
    """
    if not parsed_formula:
        # This case should ideally be handled before calling generate_features
        st.error("Empty formula dictionary provided to generate_features.")
        return None, [], []

    # Load elemental data (cached)
    element_data = load_elemental_data()
    if element_data is None:
        # Error logged in load_elemental_data
        return None, [], []

    try:
        # Calculate weighted statistics
        calculated_stats = calculate_weighted_stats(parsed_formula, element_data)

        if calculated_stats is None:
            # Error likely logged in calculate_weighted_stats
            st.error("Feature calculation failed.")
            return None, [], []

        # --- Prepare Output ---
        # 1. Original Atomic Numbers and Coefficients (order matches input dict)
        original_elements = list(parsed_formula.keys())
        original_coefficients = list(parsed_formula.values())
        # Use the reliable element_cache for atomic numbers
        original_atomic_numbers = [element_cache.get(el, -1) for el in original_elements]
        # Double check if any element wasn't in cache (shouldn't happen if parse_formula worked)
        if -1 in original_atomic_numbers:
              st.error("Internal Error: Element present in parsed formula but not in element_cache.")
              return None, [], []


        # 2. Create the feature vector DataFrame with only the TOP_FEATURES
        feature_vector_data = {}
        missing_model_features = []
        for feature_name in TOP_FEATURES:
            # Check if the feature exists in calculated_stats and is not NaN
            if feature_name in calculated_stats and pd.notna(calculated_stats[feature_name]):
                feature_vector_data[feature_name] = [calculated_stats[feature_name]]
            else:
                # Handle features required by the model but missing/NaN from calculated stats
                feature_vector_data[feature_name] = [0.0] # Fill with 0.0 as placeholder
                missing_model_features.append(feature_name)

        # Issue a single warning if any features were filled with placeholders
        if missing_model_features:
              st.warning(f"Could not calculate values for required features: {', '.join(missing_model_features)}. Filled with 0.0, prediction accuracy may be affected.")

        # Create DataFrame ensuring columns are in the exact order of TOP_FEATURES
        feature_vector_df = pd.DataFrame(feature_vector_data, columns=TOP_FEATURES)

        # Return the DataFrame and the original (unpadded) lists
        return feature_vector_df, original_atomic_numbers, original_coefficients

    except Exception as e:
        st.error(f"Error generating features: {e}")
        st.exception(e)
        return None, [], []


# --- Model Definition (MUST MATCH TRAINING SCRIPT) ---
# Custom activation function
class SuperConActivation(nn.Module):
    """
    Custom activation function based on the formula:
    output = x[..., 0] * exp(-1 / (x[..., 1] + epsilon))
    Ensures the second term (x[..., 1]) is positive before division.
    """
    def forward(self, x):
        denominator = torch.clamp(x[..., 1], min=1e-5) # Clamp to prevent division by zero or negative numbers
        out = x[..., 0] * torch.exp(torch.div(-1., denominator))
        return out.unsqueeze(-1) # Add back the last dimension

# Define the model architecture
class SuperConModel(nn.Module):
    def __init__(self, num_elements, num_features, embedding_size=EMBEDDING_SIZE, hidden_size_features=HIDDEN_SIZE_FEATURES,
                 hidden_size_1=HIDDEN_SIZE_1, hidden_size_2=HIDDEN_SIZE_2, p_drop=P_DROP):
        super(SuperConModel, self).__init__()
        self.embedding = nn.Embedding(num_elements, embedding_size, padding_idx=0)
        self.fc_features = nn.Linear(num_features, hidden_size_features)
        self.fc1 = nn.Linear(embedding_size + hidden_size_features, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, 2)
        self.activation = SuperConActivation()
        self.nonlinear = nn.SiLU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_drop) # Dropout is automatically disabled in model.eval()

    def forward(self, atomic_numbers, coefficients, features):
        embeddings = self.embedding(atomic_numbers)
        weighted_embeddings = torch.sum(embeddings * coefficients.unsqueeze(-1), dim=-2)
        features = self.fc_features(features)
        features = self.nonlinear(features)
        x = torch.cat([weighted_embeddings, features], dim=1)
        # Note: Dropout is automatically skipped when model is in eval mode
        x = self.fc1(x)
        x = self.nonlinear(x)
        x = self.fc2(x)
        x = self.nonlinear(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.activation(x)
        return x

# --- Load Model Function (Cached for Streamlit) ---
@st.cache_resource # Cache the loaded model to avoid reloading on every interaction
def load_supercon_model(model_path=MODEL_PATH):
    """Loads the SuperConModel from the specified path."""
    # Instantiate the model structure
    model_instance = SuperConModel(NUM_ELEMENTS, NUM_TOP_FEATURES)

    # Check if model file exists
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Cannot perform predictions.")
        st.warning("Please ensure the trained model file ('web_model.pth') exists in the application's root directory or the specified path.")
        return None # Return None if file doesn't exist
    else:
        try:
            # Load the saved state dictionary onto CPU
            model_instance.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            # Set the model to evaluation mode
            model_instance.eval()
            print(f"Model '{model_path}' loaded successfully (cached).") # Keep print for server log
            # st.success(f"Model '{model_path}' loaded successfully.") # Optional success message in UI
            return model_instance # Return the loaded model
        except Exception as e:
            st.error(f"Error loading model '{model_path}': {e}")
            return None # Return None if loading fails

# --- Prediction Function ---
def predict_critical_temperature(feature_vector_df: pd.DataFrame, atomic_numbers: list[int], coefficients: list[float]) -> float | None:
    """
    Predicts the critical temperature using the loaded trained model.
    Uses st.cache_resource to load the model efficiently.

    Args:
        feature_vector_df (pd.DataFrame): A DataFrame with ONE row containing the
                                          top features for the material.
                                          Column names must match training features.
        atomic_numbers (list[int]): List of atomic numbers for the material.
        coefficients (list[float]): List of corresponding coefficients (atomic proportions).

    Returns:
        float | None: The predicted critical temperature (Tc) in Kelvin, or None if an error occurs.
    """
    # Load the model using the cached function
    inference_model = load_supercon_model()

    if inference_model is None:
        # Error messages are handled within load_supercon_model
        return None

    # Input validation
    if feature_vector_df is None or feature_vector_df.empty or not atomic_numbers or not coefficients:
        st.error("Invalid input provided to prediction function (features, atomic numbers, or coefficients missing).")
        return None
    if len(atomic_numbers) != len(coefficients):
          st.error("Mismatch between length of atomic numbers and coefficients.")
          return None

    try:
        # --- 1. Preprocessing ---
        # Ensure the feature vector DataFrame has the correct columns in the correct order
        if not all(feature in feature_vector_df.columns for feature in TOP_FEATURES):
              missing = set(TOP_FEATURES) - set(feature_vector_df.columns)
              st.error(f"Input DataFrame is missing required features: {missing}")
              return None
        # Select and reorder features, convert to NumPy array
        features_np = feature_vector_df[TOP_FEATURES].values # Shape: (1, num_features)

        # Convert atomic numbers and coefficients to NumPy arrays
        atomic_numbers_np = np.array(atomic_numbers)
        coefficients_np = np.array(coefficients)

        # Pad sequences
        an_padding_len = MAX_ATOMIC_LEN - len(atomic_numbers_np)
        coeff_padding_len = MAX_ATOMIC_LEN - len(coefficients_np)

        # Note: The check for len > MAX_ATOMIC_LEN should now happen in parse_formula
        # if an_padding_len < 0 or coeff_padding_len < 0:
        #     st.error(f"Input sequence length ({len(atomic_numbers_np)}) exceeds max training length ({MAX_ATOMIC_LEN}).")
        #     return None

        padded_atomic_numbers_np = np.pad(atomic_numbers_np, (0, an_padding_len), 'constant')
        padded_coefficients_np = np.pad(coefficients_np, (0, coeff_padding_len), 'constant')

        # Normalize coefficients
        row_sum = padded_coefficients_np.sum()
        if abs(row_sum) < 1e-8: # Avoid division by zero
              normalized_coefficients_np = padded_coefficients_np
              st.warning("Sum of coefficients is near zero. Using unnormalized coefficients for prediction.")
        else:
              normalized_coefficients_np = padded_coefficients_np / row_sum

        # Convert to PyTorch tensors and add batch dimension (unsqueeze(0))
        # Ensure tensors are on the CPU
        atomic_numbers_tensor = torch.tensor(padded_atomic_numbers_np).long().unsqueeze(0).cpu()
        coefficients_tensor = torch.tensor(normalized_coefficients_np, dtype=torch.float32).unsqueeze(0).cpu()
        features_tensor = torch.tensor(features_np, dtype=torch.float32).cpu() # Already has shape (1, num_features)

        # --- 2. Prediction ---
        with torch.no_grad(): # Disable gradient calculations for inference
            predicted_tc_tensor = inference_model(atomic_numbers_tensor, coefficients_tensor, features_tensor)

        # --- 3. Postprocessing ---
        # Extract the scalar value from the tensor
        predicted_tc = predicted_tc_tensor.item()

        # Optional: Clamp prediction to a reasonable range (e.g., non-negative)
        predicted_tc = max(0.0, predicted_tc)

        return predicted_tc

    except Exception as e:
        # Log the error for debugging using Streamlit
        st.error(f"An error occurred during prediction processing: {e}")
        import traceback
        st.exception(e) # Display the full traceback in the Streamlit app
        return None


# --- Streamlit App Layout and Logic ---

# Set page configuration (title, icon, layout)
# Layout="centered" keeps the main content area constrained to a reasonable width.
# You can experiment with layout="wide".
st.set_page_config(
    page_title="Superconductor Tc Predictor",
    page_icon="✨", # Fun emoji icon
    layout="centered",
    initial_sidebar_state="auto" # Allows users to toggle sidebar if needed later
)

# --- Removed Custom CSS Block ---
# The app will now use Streamlit's default styling, respecting the user's
# light/dark theme preference set in the Streamlit settings menu.

# --- App Title ---
# Use st.title for the main heading. Styling is handled by Streamlit's theme.
st.title("⚛️ Superconductor Tc Predictor") # Added an atom emoji

st.markdown("---") # Visual divider

# --- Input Section ---
# Use st.form to group the input field and the submit button.
# This prevents the app from rerunning on every keystroke in the text input.
with st.form("prediction_form"):
    formula_input = st.text_input(
        label="Enter Chemical Formula:", # Clear label
        placeholder="e.g., MgB2, YBa2Cu3O7, La(Fe0.9Si0.1)13", # Example placeholder
        help="Type the chemical formula and press Enter or click the button below." # Helpful tooltip
    )
    # Use st.form_submit_button.
    # use_container_width=True makes the button span the width of the container.
    submitted = st.form_submit_button("✨ Predict Tc ✨", use_container_width=True)

# --- Processing and Output ---
if submitted and formula_input: # Process only when form is submitted and input is not empty
    # Initialize progress bar for visual feedback during processing
    progress_text = "Starting prediction process..."
    progress_bar = st.progress(0, text=progress_text)

    # 1. Parse the formula
    parsed = parse_formula(formula_input)
    # Update progress (approximate)
    progress_bar.progress(10, text=f"Parsing formula: {formula_input}...")

    if parsed:
        # Update progress after successful parsing
        progress_bar.progress(33, text="Parsed formula. Generating features...")
        time.sleep(0.1) # Small delay for visual flow

        # 2. Generate features
        features, atom_nums, coeffs = generate_features(parsed)

        if isinstance(features, pd.DataFrame) and not features.empty:
            # Update progress after feature generation
            progress_bar.progress(66, text="Generated features. Predicting Tc...")
            time.sleep(0.1) # Small delay

            # 3. Predict Tc
            predicted_tc = predict_critical_temperature(features, atom_nums, coeffs)

            # Update progress after prediction
            progress_bar.progress(100, text="Prediction Complete!")
            time.sleep(0.5) # Keep complete message visible briefly
            progress_bar.empty() # Remove progress bar after completion

            # --- Display Results ---

            st.markdown("---") # Divider before results

            if predicted_tc is not None:
                # Use st.metric for a visually distinct and modern display of the result.
                st.metric(
                    label=f"Predicted Tc for {formula_input}",
                    value=f"{predicted_tc:.2f} K"
                    # delta="Optional: Add change info here", # Example: Can show change from previous prediction
                    # delta_color="normal" # "normal", "inverse", "off"
                )
                st.balloons() # Add a fun element on successful prediction!

            else:
                st.error("Prediction failed after feature generation. Check logs for details.")
                # Optionally keep the progress bar visible on error
                # progress_bar.progress(100, text="Prediction Failed!")

            # Use st.expander to hide detailed information by default.
            with st.expander("View Input Details & Features"):
                st.write("**Input Interpretation (Elements: Count):**")
                # st.json displays dictionaries nicely.
                st.json(parsed)
                # Use st.code for displaying lists or code-like snippets clearly.
                st.write("**Atomic Numbers (Input Order):**")
                st.code(f"{atom_nums}", language=None) # None language for plain text
                st.write("**Coefficients (Input Order):**")
                st.code(f"{[float(f'{c:.4f}') for c in coeffs]}", language=None) # Format for readability
                st.write("**Calculated Feature Vector (Used by Model):**")
                # st.dataframe displays pandas DataFrames.
                st.dataframe(features)

        elif features is None:
            st.error("Feature generation failed. Cannot proceed. Check formula and elemental data.")
            progress_bar.empty() # Hide progress bar on failure
        else: # features is an empty DataFrame
            st.warning("Feature generation resulted in empty data. Cannot predict Tc.")
            progress_bar.empty() # Hide progress bar

    else: # Parsing failed - error message already shown by parse_formula
        st.error("Formula parsing failed. Please check the input format.")
        progress_bar.empty() # Hide progress bar

elif submitted and not formula_input:
      # Handle case where button is clicked with empty input
      st.warning("Please enter a chemical formula before predicting.")


# --- Footer ---
st.markdown("---")
# Use st.caption for less prominent text, like a footer.
st.caption("✨ Built with Streamlit | Model based only on stoichiometric composition ✨")
