import hashlib
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
import yaml
from colorama import Fore, Style
from config import PATHS, ProcessingStage, SYSTEM_PARAMS
from validation import get_user_choice
import pandas as pd


@dataclass
class StageInfo:
    """Information about a completed processing stage.
    
    Tracks completion status, input file hashes for change detection,
    and output files produced by the stage.
    
    Attributes:
        completed: Whether the stage has been completed successfully
        input_hashes: Dictionary mapping input file paths to their MD5 hashes
        output_files: List of files produced by this stage
    """

    completed: bool = False
    input_hashes: Optional[Dict[str, str]] = None
    output_files: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Initialize default values for mutable fields."""
        if self.input_hashes is None:
            self.input_hashes = {}
        if self.output_files is None:
            self.output_files = []


class StageManager:
    """Manages processing stages and determines what needs recalculation.
    
    Provides intelligent stage management using file hash comparison to determine
    when stages need to be re-executed. Tracks input file changes and output file
    existence to avoid unnecessary computation while ensuring data consistency.
    
    Attributes:
        status_file: Path to the YAML file storing stage status
        stages: Dictionary mapping stage names to StageInfo objects
    """

    def __init__(self) -> None:
        """Initialize the stage manager and load existing status.
        
        Sets up the stage manager by loading the status file path from configuration
        and attempting to restore previously saved stage completion status from disk.
        """
        self.status_file: str = PATHS["status_file"]
        self.stages: Dict[str, StageInfo] = {}
        self._load_status()

    def _load_status(self) -> None:
        """Load stage status from YAML file.
        
        Reads the status file and reconstructs StageInfo objects from saved data.
        If the file doesn't exist or is corrupted, starts with empty stages.
        """
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, "r") as f:
                    data = yaml.safe_load(f) or {}

                stages_data = data.get("stages", {})
                for stage_name, stage_data in stages_data.items():
                    self.stages[stage_name] = StageInfo(**stage_data)

        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not load stage status: {e}{Style.RESET_ALL}")
            self.stages = {}

    def _save_status(self) -> None:
        """Save stage status to YAML file.
        
        Preserves existing data in the status file and updates only the stages section.
        Creates the file if it doesn't exist.
        """
        try:
            # Load existing data or create new
            data = {}
            if os.path.exists(self.status_file):
                with open(self.status_file, "r") as f:
                    data = yaml.safe_load(f) or {}

            # Update stages section
            data["stages"] = {name: asdict(info) for name, info in self.stages.items()}

            # Save back
            with open(self.status_file, "w") as f:
                yaml.dump(data, f, default_flow_style=False)

        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not save stage status: {e}{Style.RESET_ALL}")

    def _get_file_hash(self, filepath: str) -> Optional[str]:
        """Calculate MD5 hash of a file for change detection.
        
        Computes the MD5 hash of a file to detect changes between processing runs.
        Uses chunked reading to handle large files efficiently.
        
        Args:
            filepath (str): Path to the file to hash
            
        Returns:
            Optional[str]: MD5 hash as hexadecimal string, or None if file doesn't 
                          exist or an error occurs during hashing
        """
        try:
            if not os.path.exists(filepath):
                return None

            hash_md5 = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()

        except Exception:
            return None

    def check_system_specs_stage(self, current_specs) -> ProcessingStage:
        """Check system specifications and return minimum required stage.
        
        Args:
            current_specs: DataFrame with current system specifications
            
        Returns:
            ProcessingStage: Minimum stage that needs recalculation
        """
        saved_file = os.path.join(PATHS["debug_data"], "Saved_System_Specifications.xlsx")
        
        # If no saved file, need full recalculation
        if not os.path.exists(saved_file):
            os.makedirs(PATHS["debug_data"], exist_ok=True)
            current_specs.to_excel(saved_file, index=False)
            return ProcessingStage.CALIBRATION
            
        try:
            saved_specs = pd.read_excel(saved_file, index_col=None)
        except Exception:
            current_specs.to_excel(saved_file, index=False)
            return ProcessingStage.CALIBRATION
            
        # Check for changes and find earliest required stage
        min_stage = ProcessingStage.NO_CHANGES
        changes_detected = False
        
        for param in saved_specs.columns:
            if param not in current_specs.columns:
                continue
                
            try:
                current_val = float(current_specs[param][0])
                saved_val = float(saved_specs[param][0])
                
                # Use small tolerance for float comparison
                if abs(current_val - saved_val) > max(1e-6 * abs(saved_val), 1e-9):
                    print(f"{Fore.YELLOW}System specs changed: {param} ({saved_val} → {current_val}){Style.RESET_ALL}")
                    changes_detected = True
                    
                    # Determine required stage for this parameter
                    required_stage = None
                    if param in SYSTEM_PARAMS["CALIB_PARAMS"]:
                        required_stage = ProcessingStage.CALIBRATION
                    elif param in SYSTEM_PARAMS["LOCATION_PARAMS"] + SYSTEM_PARAMS["TIME_PARAMS"]:
                        required_stage = ProcessingStage.RAW_IRRADIANCE
                    elif param in SYSTEM_PARAMS["IMAGE_PARAMS"]:
                        required_stage = ProcessingStage.SHADING
                    elif param in SYSTEM_PARAMS["SYSTEM_PARAMS"]:
                        required_stage = ProcessingStage.STATE_OF_CHARGE
                    
                    # Update min_stage to earliest required stage
                    if required_stage is not None and required_stage < min_stage:
                        min_stage = required_stage
                            
            except (ValueError, TypeError, IndexError):
                continue
                
        # Save current specs if changes detected
        if changes_detected:
            current_specs.to_excel(saved_file, index=False)
            
        return min_stage

    def _get_files_hash(self, filepaths: List[str]) -> Dict[str, str]:
        """Get MD5 hashes for multiple files.
        
        Args:
            filepaths: List of file paths to hash
            
        Returns:
            Dict mapping file paths to their MD5 hashes (only for existing files)
        """
        return {fp: self._get_file_hash(fp) for fp in filepaths if os.path.exists(fp)}

    def check_stage_needed(self, stage: ProcessingStage, input_files: List[str]) -> bool:
        """Check if a stage needs to be executed based on file changes.
        
        Compares current file hashes with saved hashes and checks if output files exist
        to determine if a stage needs re-execution.
        
        Args:
            stage: Processing stage to check
            input_files: List of input file paths for this stage
            
        Returns:
            bool: True if stage needs to be executed, False otherwise
        """
        stage_name = stage.name.lower()
        stage_info = self.stages.get(stage_name)

        # If stage was never completed, it's needed
        if not stage_info or not stage_info.completed:
            return True

        # Check if any expected output files are missing
        if stage_info.output_files:
            for output_file in stage_info.output_files:
                if not os.path.exists(output_file):
                    print(
                        f"{Fore.YELLOW}Output file missing: {os.path.basename(output_file)} - {stage.name} needed{Style.RESET_ALL}"
                    )
                    return True

        # Check if input files have changed (regular hash checking)
        current_hashes = self._get_files_hash(input_files)
        saved_hashes = stage_info.input_hashes or {}

        for filepath, current_hash in current_hashes.items():
            if current_hash is None:  # File doesn't exist
                continue

            saved_hash = saved_hashes.get(filepath)
            if saved_hash is None or saved_hash != current_hash:
                print(
                    f"{Fore.YELLOW}File changed: {os.path.basename(filepath)} - {stage.name} needed{Style.RESET_ALL}"
                )
                return True

        # Check if saved input files no longer exist in current inputs
        for saved_filepath in saved_hashes.keys():
            if saved_filepath not in current_hashes:
                print(
                    f"{Fore.YELLOW}Input file removed: {os.path.basename(saved_filepath)} - {stage.name} needed{Style.RESET_ALL}"
                )
                return True

        return False

    def mark_stage_completed(
        self, stage: ProcessingStage, input_files: List[str], output_files: List[str] = None
    ) -> None:
        """Mark a processing stage as completed with file tracking.
        
        Records stage completion status and saves file hashes for change detection.
        This enables intelligent re-execution only when input files change.
        
        Args:
            stage (ProcessingStage): The stage to mark as completed
            input_files (List[str]): List of input file paths used by this stage
            output_files (List[str], optional): List of files produced by this stage.
                                              Defaults to empty list if not provided.
        """
        stage_name = stage.name.lower()
        self.stages[stage_name] = StageInfo(
            completed=True,
            input_hashes=self._get_files_hash(input_files),
            output_files=output_files or [],
        )
        self._save_status()

    def reset_stage(self, stage: ProcessingStage) -> None:
        """Reset a stage and all subsequent stages.
        
        When a stage is reset, all stages at or after the specified stage
        in the processing pipeline are marked as not completed, requiring
        them to be re-executed.
        
        Args:
            stage (ProcessingStage): The starting stage to reset. All stages
                                   from this point onwards will be reset.
        """
        stages_to_reset = [s for s in ProcessingStage if s >= stage]

        for reset_stage in stages_to_reset:
            stage_name = reset_stage.name.lower()
            if stage_name in self.stages:
                self.stages[stage_name].completed = False

        self._save_status()
        print(f"{Fore.YELLOW}Reset stages from {stage.name} onwards{Style.RESET_ALL}")

    def reset_all_stages(self) -> None:
        """Reset all processing stages.
        
        Clears all stage completion status, effectively requiring the entire
        processing pipeline to be re-executed from the beginning. The stage
        status is persisted to disk.
        """
        self.stages = {}
        self._save_status()
        print(f"{Fore.YELLOW}All stages reset{Style.RESET_ALL}")

    def get_required_stages(
        self, input_files_per_stage: Dict[ProcessingStage, List[str]], current_specs=None
    ) -> List[ProcessingStage]:
        """Get list of stages that need to be executed based on file changes.

        Analyzes each processing stage in order to determine which stages need
        to be re-executed based on input file changes. Uses specialized handling
        for system specifications.

        Args:
            input_files_per_stage (Dict[ProcessingStage, List[str]]): Dictionary mapping 
                each processing stage to its list of input files
            current_specs: Current system specifications (optional)

        Returns:
            List[ProcessingStage]: List of stages that need execution, in dependency order.
                                  Empty list if no stages need re-execution.
        """
        earliest_required_stage = ProcessingStage.NO_CHANGES
        
        # Check system specifications if provided
        if current_specs is not None:
            spec_required_stage = self.check_system_specs_stage(current_specs)
            if spec_required_stage != ProcessingStage.NO_CHANGES:
                earliest_required_stage = spec_required_stage

        # Check regular file changes for each stage
        for stage in ProcessingStage:
            if stage == ProcessingStage.NO_CHANGES:
                continue

            input_files = input_files_per_stage.get(stage, [])
            if self.check_stage_needed(stage, input_files):
                if stage < earliest_required_stage:
                    earliest_required_stage = stage
                break  # Found earliest stage, no need to check further

        # Return all stages from earliest required stage onwards
        if earliest_required_stage != ProcessingStage.NO_CHANGES:
            return [s for s in ProcessingStage if s >= earliest_required_stage and s != ProcessingStage.NO_CHANGES]
        
        return []

    def show_stage_menu(self) -> ProcessingStage:
        """Display interactive menu for manual stage selection.
        
        Presents a user-friendly menu allowing manual selection of which processing
        stage to start from. Automatically resets the selected stage and all 
        subsequent stages if a specific stage is chosen.
        
        Returns:
            ProcessingStage: The user-selected starting stage for processing
        """
        print(f"\n{Fore.CYAN}=== Processing Stage Selection ==={Style.RESET_ALL}")
        print("Choose which stage to run from:")
        print("1. Camera Calibration (full recalculation)")
        print("2. Solar Position & Irradiance")
        print("3. Shading Factors")
        print("4. State of Charge")
        print("5. No changes (skip to end)")

        choice = get_user_choice("Enter choice (1-5): ", ["1", "2", "3", "4", "5"])

        stage_map = {
            "1": ProcessingStage.CALIBRATION,
            "2": ProcessingStage.RAW_IRRADIANCE,
            "3": ProcessingStage.SHADING,
            "4": ProcessingStage.STATE_OF_CHARGE,
            "5": ProcessingStage.NO_CHANGES,
        }

        selected_stage = stage_map[choice]

        if selected_stage != ProcessingStage.NO_CHANGES:
            self.reset_stage(selected_stage)

        return selected_stage

    def get_stage_status_summary(self) -> str:
        """Generate a formatted summary of current stage completion status.
        
        Creates a color-coded summary showing the completion status of all
        processing stages, useful for user information and debugging.
        
        Returns:
            str: Multi-line formatted string showing each stage's completion status.
                 Completed stages are marked with green checkmarks, needed stages
                 with red X marks.
        """
        lines = [f"{Fore.CYAN}Stage Status Summary:{Style.RESET_ALL}"]

        for stage in ProcessingStage:
            if stage == ProcessingStage.NO_CHANGES:
                continue

            stage_name = stage.name.lower()
            stage_info = self.stages.get(stage_name)

            if stage_info and stage_info.completed:
                status = f"{Fore.GREEN}Completed{Style.RESET_ALL}"
            else:
                status = f"{Fore.RED}Needed{Style.RESET_ALL}"

            lines.append(f"  {stage.name}: {status}")

        return "\n".join(lines)
