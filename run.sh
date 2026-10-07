#!/bin/bash

# Transformer Encoder Demo Runner with Interactive Menu
# This script sets up the environment and runs demos based on user selection

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to print colored messages
print_step() {
    echo -e "${BLUE}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_header() {
    echo -e "${CYAN}${BOLD}$1${NC}"
}

# Print banner
print_banner() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                                                            ║"
    echo "║          Transformer Encoder from Scratch                 ║"
    echo "║          JAX + Haiku + Visualization Layer                ║"
    echo "║                                                            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
}

# Setup environment
setup_environment() {
    print_header "Setting Up Environment"
    echo "════════════════════════════════════════════════════════════"
    echo ""

    # Check if Python 3 is installed
    print_step "Checking Python installation..."
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Found Python $PYTHON_VERSION"

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_step "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_success "Virtual environment already exists"
    fi

    # Activate virtual environment
    print_step "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"

    # Upgrade pip
    print_step "Upgrading pip..."
    pip install --upgrade pip -q
    print_success "pip upgraded"

    # Install dependencies
    print_step "Installing dependencies..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt -q
        print_success "Dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi

    echo ""
}

# Run encoder tests
run_encoder_tests() {
    print_header "Running Encoder Tests"
    echo "════════════════════════════════════════════════════════════"
    echo ""

    python test_encoder.py

    if [ $? -eq 0 ]; then
        echo ""
        print_success "All encoder tests passed!"
    else
        echo ""
        print_error "Some encoder tests failed"
        return 1
    fi
    echo ""
}

# Run visualization tests
run_visualization_tests() {
    print_header "Running Visualization Tests"
    echo "════════════════════════════════════════════════════════════"
    echo ""

    python test_visualizer.py

    if [ $? -eq 0 ]; then
        echo ""
        print_success "All visualization tests passed!"
    else
        echo ""
        print_error "Some visualization tests failed"
        return 1
    fi
    echo ""
}

# Run encoder examples
run_encoder_examples() {
    print_header "Running Encoder Examples"
    echo "════════════════════════════════════════════════════════════"
    echo ""

    python example.py

    if [ $? -eq 0 ]; then
        echo ""
        print_success "Encoder examples completed!"
    else
        echo ""
        print_error "Encoder examples failed"
        return 1
    fi
    echo ""
}

# Run main encoder demo
run_encoder_demo() {
    print_header "Running Main Encoder Demo"
    echo "════════════════════════════════════════════════════════════"
    echo ""

    python transformer_encoder.py

    echo ""
}

# Run visualization demo
run_visualization_demo() {
    print_header "Running Visualization Demo"
    echo "════════════════════════════════════════════════════════════"
    echo ""

    print_step "Generating visualizations (non-interactive mode)..."
    echo ""

    python example_with_visualization.py

    if [ $? -eq 0 ]; then
        echo ""
        print_success "Visualization demo completed!"
        echo ""
        print_step "Generated visualizations saved to: outputs/"
    else
        echo ""
        print_error "Visualization demo failed"
        return 1
    fi
    echo ""
}

# Show menu
show_menu() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    print_header "Demo Menu"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo -e "  ${BOLD}[1]${NC} Run Full Demo (Tests + Encoder + Visualization)"
    echo -e "  ${BOLD}[2]${NC} Run Encoder Demo Only (Tests + Examples)"
    echo -e "  ${BOLD}[3]${NC} Run Visualization Demo Only"
    echo -e "  ${BOLD}[4]${NC} Run Tests Only (Encoder + Visualization)"
    echo -e "  ${BOLD}[5]${NC} Quick Test (Skip demos, just verify installation)"
    echo -e "  ${BOLD}[6]${NC} Interactive Visualization Menu (Python)"
    echo -e "  ${BOLD}[7]${NC} ${GREEN}🌐 Start Web Visualizer${NC} ${CYAN}(Recommended!)${NC}"
    echo ""
    echo -e "  ${BOLD}[q]${NC} Quit"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo ""
}

# Run full demo
run_full_demo() {
    print_header "Full Demo Mode"
    echo "════════════════════════════════════════════════════════════"
    echo ""

    run_encoder_tests || exit 1
    run_visualization_tests || exit 1
    run_encoder_examples || exit 1
    run_encoder_demo || exit 1
    run_visualization_demo || exit 1

    print_success "Full demo completed successfully! 🎉"
}

# Run encoder only
run_encoder_only() {
    print_header "Encoder Demo Mode"
    echo "════════════════════════════════════════════════════════════"
    echo ""

    run_encoder_tests || exit 1
    run_encoder_examples || exit 1
    run_encoder_demo || exit 1

    print_success "Encoder demo completed successfully! 🎉"
}

# Run visualization only
run_visualization_only() {
    print_header "Visualization Demo Mode"
    echo "════════════════════════════════════════════════════════════"
    echo ""

    run_visualization_tests || exit 1
    run_visualization_demo || exit 1

    print_success "Visualization demo completed successfully! 🎉"
}

# Run tests only
run_tests_only() {
    print_header "Test Mode"
    echo "════════════════════════════════════════════════════════════"
    echo ""

    run_encoder_tests || exit 1
    run_visualization_tests || exit 1

    print_success "All tests completed successfully! 🎉"
}

# Quick test
run_quick_test() {
    print_header "Quick Test Mode"
    echo "════════════════════════════════════════════════════════════"
    echo ""

    print_step "Running quick verification..."
    python -c "import jax; import haiku as hk; import numpy as np; import matplotlib; print('✓ All imports successful')"

    if [ $? -eq 0 ]; then
        echo ""
        print_success "Installation verified! All dependencies are working."
    else
        echo ""
        print_error "Installation verification failed"
        return 1
    fi
    echo ""
}

# Interactive visualization menu
run_interactive_viz_menu() {
    print_header "Interactive Visualization Menu"
    echo "════════════════════════════════════════════════════════════"
    echo ""

    print_step "Launching interactive visualization menu..."
    echo ""

    python realtime_demo.py

    echo ""
}

# Start web visualizer
start_web_visualizer() {
    print_header "Web Visualizer"
    echo "════════════════════════════════════════════════════════════"
    echo ""

    # Check if Flask dependencies are installed
    print_step "Checking web dependencies..."
    python -c "import flask, flask_socketio" 2>/dev/null

    if [ $? -ne 0 ]; then
        print_warning "Installing web visualizer dependencies..."
        pip install flask flask-cors flask-socketio python-socketio simple-websocket -q
        if [ $? -eq 0 ]; then
            print_success "Dependencies installed"
        else
            print_error "Failed to install dependencies"
            return 1
        fi
    else
        print_success "Dependencies already installed"
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo ""
    print_success "Starting Web Visualizer Server..."
    echo ""
    echo -e "   ${GREEN}🌐 URL:${NC} ${CYAN}http://localhost:5000${NC}"
    echo -e "   ${YELLOW}📱 Open this URL in your browser${NC}"
    echo ""
    echo -e "   ${BOLD}Press Ctrl+C to stop the server${NC}"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo ""

    # Start the server
    cd viz_app && python app.py
}

# Print next steps
print_next_steps() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    print_header "Next Steps"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "  Manual Commands:"
    echo -e "  - Activate environment:     ${GREEN}source venv/bin/activate${NC}"
    echo -e "  - Run encoder tests:        ${GREEN}python test_encoder.py${NC}"
    echo -e "  - Run viz tests:            ${GREEN}python test_visualizer.py${NC}"
    echo -e "  - Run encoder examples:     ${GREEN}python example.py${NC}"
    echo -e "  - Run viz demo:             ${GREEN}python example_with_visualization.py${NC}"
    echo -e "  - Interactive viz menu:     ${GREEN}python realtime_demo.py${NC}"
    echo -e "  - View documentation:       ${GREEN}cat README.md${NC}"
    echo ""
    echo -e "  To deactivate: ${YELLOW}deactivate${NC}"
    echo ""
}

# Main function
main() {
    print_banner

    # Setup environment first
    setup_environment

    # If arguments provided, run in non-interactive mode
    if [ $# -gt 0 ]; then
        case "$1" in
            --full|-f)
                run_full_demo
                ;;
            --encoder|-e)
                run_encoder_only
                ;;
            --viz|-v)
                run_visualization_only
                ;;
            --tests|-t)
                run_tests_only
                ;;
            --quick|-q)
                run_quick_test
                ;;
            --interactive|-i)
                run_interactive_viz_menu
                ;;
            --web|-w)
                start_web_visualizer
                ;;
            --help|-h)
                echo "Usage: $0 [option]"
                echo ""
                echo "Options:"
                echo "  --full, -f          Run full demo"
                echo "  --encoder, -e       Run encoder demo only"
                echo "  --viz, -v           Run visualization demo only"
                echo "  --tests, -t         Run tests only"
                echo "  --quick, -q         Quick installation test"
                echo "  --interactive, -i   Interactive visualization menu (Python)"
                echo "  --web, -w           Start web visualizer (Recommended!)"
                echo "  --help, -h          Show this help"
                echo ""
                echo "No option: Show interactive menu"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    else
        # Interactive mode
        show_menu
        read -p "Enter choice: " choice

        case "$choice" in
            1)
                run_full_demo
                ;;
            2)
                run_encoder_only
                ;;
            3)
                run_visualization_only
                ;;
            4)
                run_tests_only
                ;;
            5)
                run_quick_test
                ;;
            6)
                run_interactive_viz_menu
                ;;
            7)
                start_web_visualizer
                ;;
            q|Q)
                echo "Exiting..."
                exit 0
                ;;
            *)
                print_error "Invalid choice"
                exit 1
                ;;
        esac
    fi

    print_next_steps
}

# Run main function
main "$@"
