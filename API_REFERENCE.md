# API Reference - Advanced Wireline Data Preprocessing System

## Table of Contents
1. [Main Application Class](#main-application-class)
2. [Data Management Classes](#data-management-classes)
3. [Processing Classes](#processing-classes)
4. [Visualization Classes](#visualization-classes)
5. [Utility Classes](#utility-classes)
6. [Configuration Classes](#configuration-classes)
7. [Error Handling Classes](#error-handling-classes)
8. [Extension Classes](#extension-classes)

---

## Main Application Class

### AdvancedPreprocessingApplication

The main application class that orchestrates all system components.

#### Constructor
```python
def __init__(self)
```
**Description**: Initializes the application with all required components and sets up the user interface.

**Parameters**: None

**Returns**: None

**Example**:
```python
app = AdvancedPreprocessingApplication()
```

#### Core Methods

##### setup_ui()
```python
def setup_ui(self) -> None
```
**Description**: Creates and configures the user interface with all tabs and components.

**Parameters**: None

**Returns**: None

**Side Effects**: 
- Creates main window and tabs
- Initializes UI components
- Sets up event handlers

##### load_file()
```python
def load_file(self) -> bool
```
**Description**: Loads and analyzes data from the selected file.

**Parameters**: None

**Returns**: `bool` - True if loading successful, False otherwise

**Raises**: 
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If file format is unsupported
- `Exception`: For other loading errors

**Example**:
```python
if app.load_file():
    print("File loaded successfully")
```

##### start_processing()
```python
def start_processing(self) -> None
```
**Description**: Starts the data processing pipeline in a background thread.

**Parameters**: None

**Returns**: None

**Side Effects**: 
- Updates progress indicators
- Logs processing steps
- Updates UI when complete

##### analyze_curves()
```python
def analyze_curves(self) -> Dict[str, CurveInfo]
```
**Description**: Analyzes loaded curves and identifies petrophysical properties.

**Parameters**: None

**Returns**: `Dict[str, CurveInfo]` - Dictionary mapping curve names to curve information

**Example**:
```python
curve_info = app.analyze_curves()
for curve_name, info in curve_info.items():
    print(f"{curve_name}: {info.description} ({info.unit})")
```

##### update_data_display()
```python
def update_data_display(self) -> None
```
**Description**: Refreshes the data preview and statistics display.

**Parameters**: None

**Returns**: None

##### export_data()
```python
def export_data(self, format: str = "las", filepath: str = None) -> bool
```
**Description**: Exports processed data to specified format.

**Parameters**:
- `format` (str): Export format ("las", "csv", "excel")
- `filepath` (str): Output file path (optional)

**Returns**: `bool` - True if export successful, False otherwise

**Example**:
```python
success = app.export_data(format="las", filepath="processed_well.las")
```

---

## Data Management Classes

### ComprehensiveMnemonicLibrary

Manages petrophysical curve identification and metadata.

#### Constructor
```python
def __init__(self)
```
**Description**: Initializes the mnemonic library with standard curve definitions.

#### Methods

##### identify_curve()
```python
def identify_curve(self, mnemonic: str) -> Optional[CurveInfo]
```
**Description**: Identifies curve type from mnemonic.

**Parameters**:
- `mnemonic` (str): Curve mnemonic to identify

**Returns**: `Optional[CurveInfo]` - Curve information if found, None otherwise

**Example**:
```python
curve_info = library.identify_curve("GR")
if curve_info:
    print(f"Gamma Ray curve: {curve_info.unit}")
```

##### get_curve_family()
```python
def get_curve_family(self, mnemonic: str) -> str
```
**Description**: Returns the curve family for a given mnemonic.

**Parameters**:
- `mnemonic` (str): Curve mnemonic

**Returns**: `str` - Curve family name

**Example**:
```python
family = library.get_curve_family("RHOB")
# Returns: "density"
```

##### get_validation_rules()
```python
def get_validation_rules(self, curve_type: str) -> ValidationRules
```
**Description**: Returns validation rules for a curve type.

**Parameters**:
- `curve_type` (str): Type of curve

**Returns**: `ValidationRules` - Validation rules object

### ComprehensiveCurveManager

Manages curve data lifecycle and statistics.

#### Constructor
```python
def __init__(self)
```
**Description**: Initializes the curve manager.

#### Methods

##### add_curve()
```python
def add_curve(self, name: str, data: pd.Series, metadata: Dict[str, Any]) -> None
```
**Description**: Adds a curve to the manager.

**Parameters**:
- `name` (str): Curve name
- `data` (pd.Series): Curve data
- `metadata` (Dict[str, Any]): Curve metadata

**Returns**: None

##### get_curve_statistics()
```python
def get_curve_statistics(self, curve_name: str) -> CurveStatistics
```
**Description**: Calculates and returns statistics for a curve.

**Parameters**:
- `curve_name` (str): Name of the curve

**Returns**: `CurveStatistics` - Statistics object

**Example**:
```python
stats = curve_manager.get_curve_statistics("GR")
print(f"Mean: {stats.mean}, Std: {stats.std}")
```

##### validate_curve()
```python
def validate_curve(self, curve_name: str) -> ValidationResult
```
**Description**: Validates a curve against quality criteria.

**Parameters**:
- `curve_name` (str): Name of the curve to validate

**Returns**: `ValidationResult` - Validation results

---

## Processing Classes

### AdvancedGapFiller

Handles gap detection and filling operations.

#### Constructor
```python
def __init__(self, parameters: GapFillingParameters)
```
**Description**: Initializes the gap filler with specified parameters.

**Parameters**:
- `parameters` (GapFillingParameters): Gap filling configuration

#### Methods

##### detect_gaps()
```python
def detect_gaps(self, data: pd.Series, gap_threshold: int = 1) -> List[Gap]
```
**Description**: Detects gaps in the data series.

**Parameters**:
- `data` (pd.Series): Data series to analyze
- `gap_threshold` (int): Minimum gap size to detect

**Returns**: `List[Gap]` - List of detected gaps

**Example**:
```python
gaps = gap_filler.detect_gaps(curve_data, gap_threshold=3)
print(f"Found {len(gaps)} gaps")
```

##### fill_gaps()
```python
def fill_gaps(self, data: pd.Series, gaps: List[Gap], method: str = "linear") -> Tuple[pd.Series, GapFillingResults]
```
**Description**: Fills gaps in the data using specified method.

**Parameters**:
- `data` (pd.Series): Data series with gaps
- `gaps` (List[Gap]): List of gaps to fill
- `method` (str): Gap filling method

**Returns**: `Tuple[pd.Series, GapFillingResults]` - Filled data and results

**Example**:
```python
filled_data, results = gap_filler.fill_gaps(curve_data, gaps, method="cubic_spline")
print(f"Filled {results.gaps_filled} gaps")
```

##### get_available_methods()
```python
def get_available_methods(self) -> List[str]
```
**Description**: Returns list of available gap filling methods.

**Returns**: `List[str]` - Available methods

### AdvancedSignalProcessor

Handles denoising and signal processing operations.

#### Constructor
```python
def __init__(self)
```
**Description**: Initializes the signal processor.

#### Methods

##### denoise()
```python
def denoise(self, data: pd.Series, method: str = "bilateral", **kwargs) -> Tuple[pd.Series, DenoisingResults]
```
**Description**: Applies denoising to the data.

**Parameters**:
- `data` (pd.Series): Data to denoise
- `method` (str): Denoising method
- `**kwargs`: Method-specific parameters

**Returns**: `Tuple[pd.Series, DenoisingResults]` - Denoised data and results

**Example**:
```python
denoised_data, results = processor.denoise(noisy_data, method="wavelet", wavelet="db4")
```

##### get_noise_level()
```python
def get_noise_level(self, data: pd.Series) -> float
```
**Description**: Estimates noise level in the data.

**Parameters**:
- `data` (pd.Series): Data to analyze

**Returns**: `float` - Estimated noise level

##### get_available_methods()
```python
def get_available_methods(self) -> List[str]
```
**Description**: Returns list of available denoising methods.

**Returns**: `List[str]` - Available methods

### ScaleAwareProcessor

Processes data with scale awareness for different curve types.

#### Constructor
```python
def __init__(self)
```
**Description**: Initializes the scale-aware processor.

#### Methods

##### process_curve()
```python
def process_curve(self, data: pd.Series, curve_type: str, processing_params: Dict[str, Any]) -> pd.Series
```
**Description**: Processes a curve with scale awareness.

**Parameters**:
- `data` (pd.Series): Curve data
- `curve_type` (str): Type of curve (affects processing)
- `processing_params` (Dict[str, Any]): Processing parameters

**Returns**: `pd.Series` - Processed data

##### get_curve_scale_type()
```python
def get_curve_scale_type(self, curve_name: str) -> str
```
**Description**: Determines the scale type for a curve.

**Parameters**:
- `curve_name` (str): Name of the curve

**Returns**: `str` - Scale type ("log_normal", "bounded", "normal", "discrete")

---

## Visualization Classes

### SecureVisualizationManager

Manages thread-safe visualization operations.

#### Constructor
```python
def __init__(self)
```
**Description**: Initializes the visualization manager.

#### Methods

##### create_plot()
```python
def create_plot(self, plot_type: str, data: pd.DataFrame, parameters: Dict[str, Any]) -> Figure
```
**Description**: Creates a plot in a thread-safe manner.

**Parameters**:
- `plot_type` (str): Type of plot to create
- `data` (pd.DataFrame): Data to plot
- `parameters` (Dict[str, Any]): Plot parameters

**Returns**: `Figure` - Matplotlib figure object

**Example**:
```python
fig = viz_manager.create_plot("comparison", data, {"curves": ["GR", "RHOB"]})
```

##### cleanup_figures()
```python
def cleanup_figures(self) -> None
```
**Description**: Cleans up unused figures to prevent memory leaks.

**Returns**: None

##### get_available_plot_types()
```python
def get_available_plot_types(self) -> List[str]
```
**Description**: Returns list of available plot types.

**Returns**: `List[str]` - Available plot types

### ThreadSafeVisualizationManager

Provides thread-safe visualization operations.

#### Methods

##### plot_comparison()
```python
def plot_comparison(self, original_data: pd.DataFrame, processed_data: pd.DataFrame, curves: List[str]) -> Figure
```
**Description**: Creates a comparison plot of original vs processed data.

**Parameters**:
- `original_data` (pd.DataFrame): Original data
- `processed_data` (pd.DataFrame): Processed data
- `curves` (List[str]): Curves to plot

**Returns**: `Figure` - Comparison plot figure

##### plot_correlation_matrix()
```python
def plot_correlation_matrix(self, data: pd.DataFrame, curves: List[str] = None) -> Figure
```
**Description**: Creates a correlation matrix heatmap.

**Parameters**:
- `data` (pd.DataFrame): Data to analyze
- `curves` (List[str]): Specific curves to include (optional)

**Returns**: `Figure` - Correlation matrix plot

---

## Utility Classes

### DepthValidationManager

Manages depth curve validation and standardization.

#### Constructor
```python
def __init__(self)
```
**Description**: Initializes the depth validation manager.

#### Methods

##### validate_depth_curve()
```python
def validate_depth_curve(self, data: pd.Series, unit: str = None) -> DepthValidationResult
```
**Description**: Validates a depth curve for consistency and range.

**Parameters**:
- `data` (pd.Series): Depth data to validate
- `unit` (str): Depth unit ("m", "ft")

**Returns**: `DepthValidationResult` - Validation results

**Example**:
```python
result = depth_validator.validate_depth_curve(depth_data, unit="m")
if not result.is_valid:
    print(f"Depth validation failed: {result.error_message}")
```

##### standardize_depth_units()
```python
def standardize_depth_units(self, data: pd.Series, from_unit: str, to_unit: str) -> pd.Series
```
**Description**: Converts depth data between units.

**Parameters**:
- `data` (pd.Series): Depth data to convert
- `from_unit` (str): Source unit
- `to_unit` (str): Target unit

**Returns**: `pd.Series` - Converted depth data

### IndustryUnitStandardizer

Handles unit standardization and conversion.

#### Constructor
```python
def __init__(self, app: AdvancedPreprocessingApplication)
```
**Description**: Initializes the unit standardizer.

**Parameters**:
- `app` (AdvancedPreprocessingApplication): Reference to main application

#### Methods

##### analyze_units()
```python
def analyze_units(self, curve_info: Dict[str, CurveInfo]) -> UnitAnalysisResult
```
**Description**: Analyzes units across all curves.

**Parameters**:
- `curve_info` (Dict[str, CurveInfo]): Curve information dictionary

**Returns**: `UnitAnalysisResult` - Unit analysis results

##### standardize_fractional_curves()
```python
def standardize_fractional_curves(self, data: pd.DataFrame, curve_info: Dict[str, CurveInfo]) -> pd.DataFrame
```
**Description**: Converts percentage units to decimal fractions.

**Parameters**:
- `data` (pd.DataFrame): Data to standardize
- `curve_info` (Dict[str, CurveInfo]): Curve information

**Returns**: `pd.DataFrame` - Standardized data

### ProcessingHistoryManager

Manages processing history and undo/redo functionality.

#### Constructor
```python
def __init__(self, max_history: int = 50)
```
**Description**: Initializes the history manager.

**Parameters**:
- `max_history` (int): Maximum number of history entries

#### Methods

##### save_state()
```python
def save_state(self, state: ProcessingState) -> None
```
**Description**: Saves current processing state to history.

**Parameters**:
- `state` (ProcessingState): State to save

**Returns**: None

##### undo()
```python
def undo(self) -> Optional[ProcessingState]
```
**Description**: Restores previous processing state.

**Returns**: `Optional[ProcessingState]` - Previous state if available

##### redo()
```python
def redo(self) -> Optional[ProcessingState]
```
**Description**: Restores next processing state.

**Returns**: `Optional[ProcessingState]` - Next state if available

---

## Configuration Classes

### GapFillingParameters

Configuration for gap filling operations.

#### Constructor
```python
def __init__(self, method: str = "linear", max_gap_size: int = 20, **kwargs)
```
**Description**: Initializes gap filling parameters.

**Parameters**:
- `method` (str): Gap filling method
- `max_gap_size` (int): Maximum gap size to fill
- `**kwargs`: Method-specific parameters

#### Properties

##### method
```python
@property
def method(self) -> str
```
**Description**: Gap filling method to use.

##### max_gap_size
```python
@property
def max_gap_size(self) -> int
```
**Description**: Maximum gap size to fill.

##### parameters
```python
@property
def parameters(self) -> Dict[str, Any]
```
**Description**: Method-specific parameters.

### DenoisingParameters

Configuration for denoising operations.

#### Constructor
```python
def __init__(self, method: str = "bilateral", **kwargs)
```
**Description**: Initializes denoising parameters.

**Parameters**:
- `method` (str): Denoising method
- `**kwargs`: Method-specific parameters

### ProcessingParameters

Overall processing configuration.

#### Constructor
```python
def __init__(self, gap_filling: GapFillingParameters = None, denoising: DenoisingParameters = None, quality_control: QualityControlParameters = None)
```
**Description**: Initializes processing parameters.

**Parameters**:
- `gap_filling` (GapFillingParameters): Gap filling configuration
- `denoising` (DenoisingParameters): Denoising configuration
- `quality_control` (QualityControlParameters): Quality control configuration

---

## Error Handling Classes

### ValidationResult

Results from data validation operations.

#### Properties

##### is_valid
```python
@property
def is_valid(self) -> bool
```
**Description**: Whether validation passed.

##### warnings
```python
@property
def warnings(self) -> List[str]
```
**Description**: List of validation warnings.

##### errors
```python
@property
def errors(self) -> List[str]
```
**Description**: List of validation errors.

#### Methods

##### add_warning()
```python
def add_warning(self, message: str) -> None
```
**Description**: Adds a validation warning.

**Parameters**:
- `message` (str): Warning message

##### add_error()
```python
def add_error(self, message: str) -> None
```
**Description**: Adds a validation error.

**Parameters**:
- `message` (str): Error message

### ProcessingResults

Results from processing operations.

#### Properties

##### success
```python
@property
def success(self) -> bool
```
**Description**: Whether processing was successful.

##### processing_time
```python
@property
def processing_time(self) -> float
```
**Description**: Processing time in seconds.

##### quality_metrics
```python
@property
def quality_metrics(self) -> Dict[str, float]
```
**Description**: Quality metrics from processing.

#### Methods

##### get_summary()
```python
def get_summary(self) -> str
```
**Description**: Returns a summary of processing results.

**Returns**: `str` - Processing summary

---

## Extension Classes

### ProcessingPlugin

Base class for custom processing plugins.

#### Abstract Methods

##### process()
```python
@abstractmethod
def process(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame
```
**Description**: Process data with given parameters.

**Parameters**:
- `data` (pd.DataFrame): Data to process
- `parameters` (Dict[str, Any]): Processing parameters

**Returns**: `pd.DataFrame` - Processed data

##### get_parameters()
```python
@abstractmethod
def get_parameters(self) -> Dict[str, Any]
```
**Description**: Returns parameter schema for the plugin.

**Returns**: `Dict[str, Any]` - Parameter schema

### VisualizationPlugin

Base class for custom visualization plugins.

#### Abstract Methods

##### create_plot()
```python
@abstractmethod
def create_plot(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Figure
```
**Description**: Create plot from data.

**Parameters**:
- `data` (pd.DataFrame): Data to visualize
- `parameters` (Dict[str, Any]): Plot parameters

**Returns**: `Figure` - Matplotlib figure

---

## Data Structures

### CurveInfo

Information about a petrophysical curve.

#### Properties

##### mnemonic
```python
@property
def mnemonic(self) -> str
```
**Description**: Curve mnemonic.

##### description
```python
@property
def description(self) -> str
```
**Description**: Curve description.

##### unit
```python
@property
def unit(self) -> str
```
**Description**: Curve unit.

##### family
```python
@property
def family(self) -> str
```
**Description**: Curve family.

### Gap

Represents a gap in the data.

#### Properties

##### start_index
```python
@property
def start_index(self) -> int
```
**Description**: Starting index of the gap.

##### end_index
```python
@property
def end_index(self) -> int
```
**Description**: Ending index of the gap.

##### size
```python
@property
def size(self) -> int
```
**Description**: Size of the gap.

### ProcessingState

Represents a state in the processing history.

#### Properties

##### data
```python
@property
def data(self) -> pd.DataFrame
```
**Description**: Data at this state.

##### parameters
```python
@property
def parameters(self) -> ProcessingParameters
```
**Description**: Parameters used for this state.

##### timestamp
```python
@property
def timestamp(self) -> datetime
```
**Description**: When this state was created.

---

## Utility Functions

### File Loading Functions

#### load_las_file()
```python
def load_las_file(filepath: str) -> Tuple[pd.DataFrame, Dict[str, Any]]
```
**Description**: Loads a LAS file and returns data with metadata.

**Parameters**:
- `filepath` (str): Path to LAS file

**Returns**: `Tuple[pd.DataFrame, Dict[str, Any]]` - Data and metadata

**Example**:
```python
data, metadata = load_las_file("well_data.las")
```

#### load_csv_file()
```python
def load_csv_file(filepath: str, delimiter: str = None) -> pd.DataFrame
```
**Description**: Loads a CSV file.

**Parameters**:
- `filepath` (str): Path to CSV file
- `delimiter` (str): CSV delimiter (auto-detected if None)

**Returns**: `pd.DataFrame` - Loaded data

### Data Processing Functions

#### detect_outliers()
```python
def detect_outliers(data: pd.Series, method: str = "iqr", threshold: float = 1.5) -> pd.Series
```
**Description**: Detects outliers in data using specified method.

**Parameters**:
- `data` (pd.Series): Data to analyze
- `method` (str): Detection method ("iqr", "zscore", "modified_zscore")
- `threshold` (float): Outlier threshold

**Returns**: `pd.Series` - Boolean series indicating outliers

#### calculate_correlations()
```python
def calculate_correlations(data: pd.DataFrame, method: str = "pearson") -> pd.DataFrame
```
**Description**: Calculates correlation matrix for data.

**Parameters**:
- `data` (pd.DataFrame): Data to analyze
- `method` (str): Correlation method ("pearson", "spearman", "kendall")

**Returns**: `pd.DataFrame` - Correlation matrix

### Utility Functions

#### format_number()
```python
def format_number(value: float, precision: int = 3) -> str
```
**Description**: Formats a number with specified precision.

**Parameters**:
- `value` (float): Number to format
- `precision` (int): Decimal precision

**Returns**: `str` - Formatted number string

#### validate_file_format()
```python
def validate_file_format(filepath: str) -> bool
```
**Description**: Validates that file format is supported.

**Parameters**:
- `filepath` (str): Path to file

**Returns**: `bool` - True if format is supported

---

*This API reference provides comprehensive documentation for all classes, methods, and functions in the Advanced Wireline Data Preprocessing System. For usage examples and detailed explanations, refer to the User Manual and Technical Reference Guide.*
