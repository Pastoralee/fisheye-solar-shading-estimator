from colorama import Fore, Style, init
from pipeline import SolarEstimationPipeline
from validation import ask_restart
init()


def main() -> None:
    """Main entry point for the solar estimation system.
    
    Initializes the processing pipeline and handles top-level error handling.
    Provides user feedback and graceful shutdown on completion or errors.
    Continuously offers to relaunch the program after completion.
    
    Raises:
        KeyboardInterrupt: If user interrupts the program
        Exception: For any other unexpected errors during execution
    """
    while True:
        try:
            # Create and run pipeline
            pipeline = SolarEstimationPipeline()
            success = pipeline.run_complete_pipeline()

            if success:
                print(f"\n{Fore.GREEN}=== All calculations completed successfully! ==={Style.RESET_ALL}")
                
                # Ask user if they want to restart the program
                if not ask_restart():
                    break

        except KeyboardInterrupt:
            print(f"\n{Fore.RED}Program interrupted by user{Style.RESET_ALL}")
            break

        except Exception as e:
            print(f"\n{Fore.RED}An unexpected error occurred: {e}")
            
            # Ask user if they want to restart after an error
            if not ask_restart():
                break


if __name__ == "__main__":
    main()
