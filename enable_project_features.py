#!/usr/bin/env python3
"""
Script to enable/disable project management features
Zero-impact integration control
"""

import sys
import os

def enable_project_features():
    """Enable project management features in app_chromadb.py"""
    try:
        # Read the current app_chromadb.py file
        with open('app_chromadb.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the feature flag
        old_flag = "PROJECT_FEATURES_ENABLED = False  # Default: OFF for safety"
        new_flag = "PROJECT_FEATURES_ENABLED = True"
        
        if old_flag in content:
            content = content.replace(old_flag, new_flag)
            
            # Write back to file
            with open('app_chromadb.py', 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("Project management features ENABLED successfully!")
            print("Please restart your chatbot to activate the features.")
            return True
        else:
            print("Could not find the feature flag to enable.")
            return False
            
    except Exception as e:
        print(f"Error enabling project features: {e}")
        return False

def disable_project_features():
    """Disable project management features in app_chromadb.py"""
    try:
        # Read the current app_chromadb.py file
        with open('app_chromadb.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the feature flag
        old_flag = "PROJECT_FEATURES_ENABLED = True"
        new_flag = "PROJECT_FEATURES_ENABLED = False  # Default: OFF for safety"
        
        if old_flag in content:
            content = content.replace(old_flag, new_flag)
            
            # Write back to file
            with open('app_chromadb.py', 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("Project management features DISABLED successfully!")
            print("Please restart your chatbot to return to zero-impact mode.")
            return True
        else:
            print("Could not find the feature flag to disable.")
            return False
            
    except Exception as e:
        print(f"Error disabling project features: {e}")
        return False

def check_status():
    """Check current status of project features"""
    try:
        with open('app_chromadb.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "PROJECT_FEATURES_ENABLED = True" in content:
            print("Project management features are ENABLED")
            return True
        elif "PROJECT_FEATURES_ENABLED = False" in content:
            print("Project management features are DISABLED (Zero Impact Mode)")
            return False
        else:
            print("Could not determine project features status")
            return None
            
    except Exception as e:
        print(f"Error checking status: {e}")
        return None

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 2:
        print("Project Management Features Control")
        print("=" * 50)
        print("Usage:")
        print("  python enable_project_features.py enable   - Enable project features")
        print("  python enable_project_features.py disable  - Disable project features")
        print("  python enable_project_features.py status   - Check current status")
        print("")
        check_status()
        return
    
    command = sys.argv[1].lower()
    
    if command == "enable":
        enable_project_features()
    elif command == "disable":
        disable_project_features()
    elif command == "status":
        check_status()
    else:
        print(f"Unknown command: {command}")
        print("Use 'enable', 'disable', or 'status'")

if __name__ == "__main__":
    main()
