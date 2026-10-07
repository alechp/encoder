#!/usr/bin/env python3
"""
Cross-platform demo runner with interactive menu for Transformer Encoder

This script sets up the environment and runs demos based on user selection.
Works on Windows, macOS, and Linux.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import argparse


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color

    @staticmethod
    def disable():
        """Disable colors (for Windows without ANSI support)"""
        Colors.RED = ''
        Colors.GREEN = ''
        Colors.YELLOW = ''
        Colors.BLUE = ''
        Colors.CYAN = ''
        Colors.BOLD = ''
        Colors.NC = ''


# Disable colors on Windows if not supported
if platform.system() == 'Windows' and not os.environ.get('ANSICON'):
    Colors.disable()


def print_banner():
    """Print the demo banner"""
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                                                            ║")
    print("║          Transformer Encoder from Scratch                 ║")
    print("║          JAX + Haiku + Visualization Layer                ║")
    print("║                                                            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()


def print_step(message):
    """Print a step message"""
    print(f"{Colors.BLUE}▶{Colors.NC} {message}")


def print_success(message):
    """Print a success message"""
    print(f"{Colors.GREEN}✓{Colors.NC} {message}")


def print_error(message):
    """Print an error message"""
    print(f"{Colors.RED}✗{Colors.NC} {message}")


def print_warning(message):
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠{Colors.NC} {message}")


def print_header(message):
    """Print a section header"""
    print(f"{Colors.CYAN}{Colors.BOLD}{message}{Colors.NC}")


def separator():
    """Print a separator line"""
    print()
    print("════════════════════════════════════════════════════════════")
    print()


def run_command(cmd, description=None, quiet=False):
    """
    Run a command and return the result.

    Args:
        cmd: Command to run (list or string)
        description: Optional description of what the command does
        quiet: If True, suppress output

    Returns:
        True if successful, False otherwise
    """
    if description:
        print_step(description)

    try:
        if quiet:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                shell=isinstance(cmd, str)
            )
        else:
            result = subprocess.run(
                cmd,
                check=True,
                shell=isinstance(cmd, str)
            )
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        if quiet and e.stdout:
            print(e.stdout)
        if quiet and e.stderr:
            print(e.stderr)
        return False


def get_venv_python():
    """Get the path to the Python executable in the virtual environment"""
    if platform.system() == 'Windows':
        return Path('venv') / 'Scripts' / 'python.exe'
    else:
        return Path('venv') / 'bin' / 'python'


def get_venv_pip():
    """Get the path to the pip executable in the virtual environment"""
    if platform.system() == 'Windows':
        return Path('venv') / 'Scripts' / 'pip.exe'
    else:
        return Path('venv') / 'bin' / 'pip'


def setup_environment():
    """Setup the Python environment"""
    print_header("Setting Up Environment")
    separator()

    # Check Python version
    print_step("Checking Python installation...")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print_success(f"Found Python {python_version}")

    if sys.version_info < (3, 8):
        print_error("Python 3.8 or higher is required")
        return False

    # Create virtual environment
    venv_path = Path('venv')
    if venv_path.exists():
        print_success("Virtual environment already exists")
    else:
        print_step("Creating virtual environment...")
        if not run_command([sys.executable, '-m', 'venv', 'venv'], quiet=True):
            print_error("Failed to create virtual environment")
            return False
        print_success("Virtual environment created")

    # Upgrade pip
    print_step("Upgrading pip...")
    pip_path = str(get_venv_pip())
    if not run_command([pip_path, 'install', '--upgrade', 'pip'], quiet=True):
        print_warning("Failed to upgrade pip, continuing anyway...")
    else:
        print_success("pip upgraded")

    # Install dependencies
    print_step("Installing dependencies...")
    requirements = Path('requirements.txt')
    if not requirements.exists():
        print_error("requirements.txt not found")
        return False

    if run_command([pip_path, 'install', '-r', 'requirements.txt'], quiet=True):
        print_success("Dependencies installed")
    else:
        print_error("Failed to install dependencies")
        return False

    print()
    return True


def run_encoder_tests():
    """Run encoder tests"""
    print_header("Running Encoder Tests")
    separator()
    print()

    python_path = str(get_venv_python())
    success = run_command([python_path, 'test_encoder.py'])

    if success:
        print()
        print_success("All encoder tests passed!")
    else:
        print()
        print_error("Some encoder tests failed")

    print()
    return success


def run_visualization_tests():
    """Run visualization tests"""
    print_header("Running Visualization Tests")
    separator()
    print()

    python_path = str(get_venv_python())
    success = run_command([python_path, 'test_visualizer.py'])

    if success:
        print()
        print_success("All visualization tests passed!")
    else:
        print()
        print_error("Some visualization tests failed")

    print()
    return success


def run_encoder_examples():
    """Run encoder examples"""
    print_header("Running Encoder Examples")
    separator()
    print()

    python_path = str(get_venv_python())
    success = run_command([python_path, 'example.py'])

    if success:
        print()
        print_success("Encoder examples completed!")
    else:
        print()
        print_error("Encoder examples failed")

    print()
    return success


def run_encoder_demo():
    """Run main encoder demo"""
    print_header("Running Main Encoder Demo")
    separator()
    print()

    python_path = str(get_venv_python())
    run_command([python_path, 'transformer_encoder.py'])

    print()


def run_visualization_demo():
    """Run visualization demo"""
    print_header("Running Visualization Demo")
    separator()
    print()

    print_step("Generating visualizations (non-interactive mode)...")
    print()

    python_path = str(get_venv_python())
    success = run_command([python_path, 'example_with_visualization.py'])

    if success:
        print()
        print_success("Visualization demo completed!")
        print()
        print_step("Generated visualizations saved to: outputs/")
    else:
        print()
        print_error("Visualization demo failed")

    print()
    return success


def run_interactive_viz_menu():
    """Run interactive visualization menu"""
    print_header("Interactive Visualization Menu")
    separator()
    print()

    print_step("Launching interactive visualization menu...")
    print()

    python_path = str(get_venv_python())
    run_command([python_path, 'realtime_demo.py'])

    print()


def start_web_visualizer():
    """Start the web visualizer server"""
    print_header("Web Visualizer")
    separator()
    print()

    # Check if Flask dependencies are installed
    print_step("Checking web dependencies...")
    python_path = str(get_venv_python())
    pip_path = str(get_venv_pip())

    check_cmd = [python_path, '-c', 'import flask, flask_socketio']
    result = subprocess.run(check_cmd, capture_output=True)

    if result.returncode != 0:
        print_warning("Installing web visualizer dependencies...")
        deps = ['flask', 'flask-cors', 'flask-socketio', 'python-socketio', 'simple-websocket']
        if run_command([pip_path, 'install'] + deps, quiet=True):
            print_success("Dependencies installed")
        else:
            print_error("Failed to install dependencies")
            return False
    else:
        print_success("Dependencies already installed")

    print()
    separator()
    print()
    print_success("Starting Web Visualizer Server...")
    print()
    print(f"   {Colors.GREEN}🌐 URL:{Colors.NC} {Colors.CYAN}http://localhost:5000{Colors.NC}")
    print(f"   {Colors.YELLOW}📱 Open this URL in your browser{Colors.NC}")
    print()
    print(f"   {Colors.BOLD}Press Ctrl+C to stop the server{Colors.NC}")
    print()
    separator()
    print()

    # Start the server
    os.chdir('viz_app')
    return run_command([python_path, 'app.py'])


def run_quick_test():
    """Quick installation verification"""
    print_header("Quick Test Mode")
    separator()
    print()

    print_step("Running quick verification...")
    python_path = str(get_venv_python())

    test_code = "import jax; import haiku as hk; import numpy as np; import matplotlib; print('✓ All imports successful')"
    success = run_command([python_path, '-c', test_code])

    if success:
        print()
        print_success("Installation verified! All dependencies are working.")
    else:
        print()
        print_error("Installation verification failed")

    print()
    return success


def show_menu():
    """Display the interactive menu"""
    print()
    print("════════════════════════════════════════════════════════════")
    print_header("Demo Menu")
    print("════════════════════════════════════════════════════════════")
    print()
    print(f"  {Colors.BOLD}[1]{Colors.NC} Run Full Demo (Tests + Encoder + Visualization)")
    print(f"  {Colors.BOLD}[2]{Colors.NC} Run Encoder Demo Only (Tests + Examples)")
    print(f"  {Colors.BOLD}[3]{Colors.NC} Run Visualization Demo Only")
    print(f"  {Colors.BOLD}[4]{Colors.NC} Run Tests Only (Encoder + Visualization)")
    print(f"  {Colors.BOLD}[5]{Colors.NC} Quick Test (Skip demos, just verify installation)")
    print(f"  {Colors.BOLD}[6]{Colors.NC} Interactive Visualization Menu (Python)")
    print(f"  {Colors.BOLD}[7]{Colors.NC} {Colors.GREEN}🌐 Start Web Visualizer{Colors.NC} {Colors.CYAN}(Recommended!){Colors.NC}")
    print()
    print(f"  {Colors.BOLD}[q]{Colors.NC} Quit")
    print()
    print("════════════════════════════════════════════════════════════")
    print()


def run_full_demo():
    """Run the complete demo"""
    print_header("Full Demo Mode")
    separator()
    print()

    if not run_encoder_tests():
        return False
    if not run_visualization_tests():
        return False
    if not run_encoder_examples():
        return False
    run_encoder_demo()
    if not run_visualization_demo():
        return False

    print_success("Full demo completed successfully! 🎉")
    return True


def run_encoder_only():
    """Run encoder demo only"""
    print_header("Encoder Demo Mode")
    separator()
    print()

    if not run_encoder_tests():
        return False
    if not run_encoder_examples():
        return False
    run_encoder_demo()

    print_success("Encoder demo completed successfully! 🎉")
    return True


def run_visualization_only():
    """Run visualization demo only"""
    print_header("Visualization Demo Mode")
    separator()
    print()

    if not run_visualization_tests():
        return False
    if not run_visualization_demo():
        return False

    print_success("Visualization demo completed successfully! 🎉")
    return True


def run_tests_only():
    """Run tests only"""
    print_header("Test Mode")
    separator()
    print()

    if not run_encoder_tests():
        return False
    if not run_visualization_tests():
        return False

    print_success("All tests completed successfully! 🎉")
    return True


def print_next_steps():
    """Print next steps for the user"""
    print()
    print("════════════════════════════════════════════════════════════")
    print_header("Next Steps")
    print("════════════════════════════════════════════════════════════")
    print()
    print("  Manual Commands:")

    if platform.system() == 'Windows':
        activate_cmd = f"{Colors.GREEN}venv\\Scripts\\activate{Colors.NC}"
    else:
        activate_cmd = f"{Colors.GREEN}source venv/bin/activate{Colors.NC}"

    print(f"  - Activate environment:     {activate_cmd}")
    print(f"  - Run encoder tests:        {Colors.GREEN}python test_encoder.py{Colors.NC}")
    print(f"  - Run viz tests:            {Colors.GREEN}python test_visualizer.py{Colors.NC}")
    print(f"  - Run encoder examples:     {Colors.GREEN}python example.py{Colors.NC}")
    print(f"  - Run viz demo:             {Colors.GREEN}python example_with_visualization.py{Colors.NC}")
    print(f"  - Interactive viz menu:     {Colors.GREEN}python realtime_demo.py{Colors.NC}")
    print(f"  - View documentation:       {Colors.GREEN}README.md{Colors.NC}")
    print()
    print(f"  To deactivate: {Colors.YELLOW}deactivate{Colors.NC}")
    print()


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Transformer Encoder Demo Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py              # Interactive menu
  python run.py --full       # Run full demo
  python run.py --encoder    # Run encoder demo only
  python run.py --viz        # Run visualization demo only
  python run.py --tests      # Run tests only
  python run.py --quick      # Quick installation test
  python run.py --interactive # Interactive visualization menu (Python)
  python run.py --web        # Start web visualizer (Recommended!)
        """
    )

    parser.add_argument('--full', '-f', action='store_true',
                       help='Run full demo')
    parser.add_argument('--encoder', '-e', action='store_true',
                       help='Run encoder demo only')
    parser.add_argument('--viz', '-v', action='store_true',
                       help='Run visualization demo only')
    parser.add_argument('--tests', '-t', action='store_true',
                       help='Run tests only')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Quick installation test')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive visualization menu (Python)')
    parser.add_argument('--web', '-w', action='store_true',
                       help='Start web visualizer (Recommended!)')

    args = parser.parse_args()

    print_banner()

    # Setup environment first
    if not setup_environment():
        sys.exit(1)

    # Handle command line arguments
    if args.full:
        success = run_full_demo()
    elif args.encoder:
        success = run_encoder_only()
    elif args.viz:
        success = run_visualization_only()
    elif args.tests:
        success = run_tests_only()
    elif args.quick:
        success = run_quick_test()
    elif args.interactive:
        run_interactive_viz_menu()
        success = True
    elif args.web:
        success = start_web_visualizer()
    else:
        # Interactive mode
        show_menu()
        choice = input("Enter choice: ").strip()

        if choice == '1':
            success = run_full_demo()
        elif choice == '2':
            success = run_encoder_only()
        elif choice == '3':
            success = run_visualization_only()
        elif choice == '4':
            success = run_tests_only()
        elif choice == '5':
            success = run_quick_test()
        elif choice == '6':
            run_interactive_viz_menu()
            success = True
        elif choice == '7':
            success = start_web_visualizer()
        elif choice.lower() == 'q':
            print("Exiting...")
            sys.exit(0)
        else:
            print_error("Invalid choice")
            sys.exit(1)

    print_next_steps()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()
        print_warning("Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print()
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
