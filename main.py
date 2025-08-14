from colorama import Fore, Style, init
from pipeline import SolarEstimationPipeline
init()


def main() -> None:
    """Main entry point for the solar estimation system.
    
    Initializes the processing pipeline and handles top-level error handling.
    Provides user feedback and graceful shutdown on completion or errors.
    
    Raises:
        KeyboardInterrupt: If user interrupts the program
        Exception: For any other unexpected errors during execution
    """
    try:
        # Create and run pipeline
        pipeline = SolarEstimationPipeline()
        success = pipeline.run_complete_pipeline()

        if success:
            print(f"\n{Fore.GREEN}=== All calculations completed successfully! ==={Style.RESET_ALL}")
            input(f"{Fore.CYAN}Press Enter to exit...{Style.RESET_ALL}")

    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Program interrupted by user{Style.RESET_ALL}")

    except Exception as e:
        print(f"\n{Fore.RED}An unexpected error occurred: {e}")
        input(f"\nPress Enter to exit...{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
