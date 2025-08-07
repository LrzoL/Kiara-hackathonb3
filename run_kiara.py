#!/usr/bin/env python3
"""
🤖 Kiara - AI Documentation Agent
Launcher script for easy execution

Usage:
    python run_kiara.py                    # Interactive enhanced interface
    python run_kiara.py --simple          # Simple CLI interface  
    python run_kiara.py --serve           # API server mode
    python run_kiara.py --test            # Test installation
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_installation():
    """Test if Kiara is properly installed and configured."""
    print("* Testing Kiara installation...")
    print("=" * 50)
    
    try:
        # Test basic imports
        print("* Testing basic imports...")
        from src.config import get_settings
        from src.oci_analyzer import OCIAnalyzer
        from src.simple_commit_tracker import SimpleCommitTracker
        from src.auth_manager import AuthManager
        print("* All imports successful!")
        
        # Test configuration
        print("* Testing configuration...")
        settings = get_settings()
        print(f"* Oracle Cloud AI: {'Enabled' if settings.oci_enabled else 'Disabled'}")
        print(f"* GitHub Token: {'Configured' if settings.github_token else 'Missing'}")
        
        # Test commit tracker
        print("* Testing commit tracker...")
        tracker = SimpleCommitTracker()
        print("* Commit tracker initialized!")
        
        print("=" * 50)
        print("* Kiara installation test completed successfully!")
        print("* You can now use Kiara to generate documentation!")
        return True
        
    except Exception as e:
        print("=" * 50)
        print(f"* Installation test failed: {e}")
        print("* Please check your .env configuration and dependencies")
        return False

async def run_simple_cli():
    """Run the simple CLI interface."""
    print("* Starting Kiara Simple CLI...")
    try:
        import kiara_cli
        await kiara_cli.main()
    except KeyboardInterrupt:
        print("\n* Kiara CLI shutdown requested. Goodbye!")
    except Exception as e:
        print(f"\n* Error running Kiara CLI: {e}")

async def run_interactive():
    """Run the interactive enhanced interface."""
    print("* Starting Kiara Interactive Enhanced...")
    try:
        import kiara_interactive_enhanced
        await kiara_interactive_enhanced.main()
    except KeyboardInterrupt:
        print("\n* Kiara Interactive shutdown requested. Goodbye!")
    except Exception as e:
        print(f"\n* Error running Kiara Interactive: {e}")

def run_server():
    """Run the API server."""
    print("* Starting Kiara API Server...")
    try:
        import uvicorn
        from api.main import app
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\n* Kiara server shutdown requested. Goodbye!")
    except Exception as e:
        print(f"\n* Error running Kiara server: {e}")

def main():
    """Main entry point with argument parsing."""
    # Set UTF-8 encoding for Windows compatibility
    if sys.platform.startswith('win'):
        import os
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        try:
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        except:
            pass
    
    parser = argparse.ArgumentParser(
        description="Kiara - AI Documentation Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_kiara.py                    # Interactive enhanced interface (default)
  python run_kiara.py --simple          # Simple CLI interface
  python run_kiara.py --serve           # Start API server
  python run_kiara.py --test            # Test installation
        """
    )
    
    parser.add_argument(
        "--simple", 
        action="store_true", 
        help="Use simple CLI interface instead of interactive"
    )
    
    parser.add_argument(
        "--serve", 
        action="store_true", 
        help="Start API server mode"
    )
    
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Test installation and configuration"
    )
    
    args = parser.parse_args()
    
    # Show banner (Windows-compatible)
    banner = """
    +==============================================================+
    |   KIARA - AI Documentation Agent                            |
    |                                                              |
    |   * Powered by Oracle Cloud AI (meta.llama-4-maverick)      |
    |   * Automatic README.md generation for any project          |
    |   * Smart commit detection system                           |
    |   * Professional templates + Intelligent AI Analysis        |
    +==============================================================+
    """
    print(banner)
    
    # Route to appropriate function
    if args.test:
        success = test_installation()
        sys.exit(0 if success else 1)
    elif args.serve:
        run_server()
    elif args.simple:
        asyncio.run(run_simple_cli())
    else:
        # Default to interactive enhanced
        asyncio.run(run_interactive())

if __name__ == "__main__":
    main()