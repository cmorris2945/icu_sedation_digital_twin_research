#!/bin/bash
#
# initialize_git.sh
# =================
#
# This script helps set up version control for the research repository.
# Running it initializes a Git repository, configures appropriate settings,
# makes the first commit with all the existing files, and prepares the
# repository for being pushed to a remote service like GitHub.
#
# Version control is important for academic research for several reasons.
# It provides a complete history of every change, which helps when writing
# the methods section of a paper because you can see exactly when each
# decision was made. It enables collaboration with advisors and other
# researchers without overwriting each other's work. It serves as a
# backup system because the repository can be pushed to remote servers.
# And it provides the foundation for sharing code publicly when the paper
# is published.
#
# USAGE:
#     bash initialize_git.sh
#
# REQUIREMENTS:
#     Git must be installed on your system. You can check by running
#     git --version in a terminal. If not installed, install from
#     https://git-scm.com/downloads
#
# AFTER RUNNING THIS SCRIPT:
#     The repository will be initialized locally. To push to a remote
#     service like GitHub, you will need to create a repository there
#     first, then add it as a remote and push. Specific commands are
#     printed at the end of this script.

set -e  # Exit immediately if any command fails

echo "====================================================================="
echo "Initializing Git Repository for ICU Sedation Research"
echo "====================================================================="
echo ""

# Verify Git is installed
if ! command -v git &> /dev/null; then
    echo "ERROR: Git is not installed on this system."
    echo "Please install Git from https://git-scm.com/downloads and try again."
    exit 1
fi

# Initialize the repository if not already initialized
if [ ! -d ".git" ]; then
    echo "Step 1: Initializing Git repository..."
    git init
    echo "  Repository initialized"
else
    echo "Step 1: Repository already initialized, skipping init"
fi
echo ""

# Configure user name and email if not already set
# These are required by Git for making commits
echo "Step 2: Checking Git user configuration..."
USER_NAME=$(git config --get user.name || echo "")
USER_EMAIL=$(git config --get user.email || echo "")

if [ -z "$USER_NAME" ]; then
    echo "  No user name configured. Please set it now."
    read -p "  Enter your name: " input_name
    git config user.name "$input_name"
fi

if [ -z "$USER_EMAIL" ]; then
    echo "  No user email configured. Please set it now."
    read -p "  Enter your email: " input_email
    git config user.email "$input_email"
fi

echo "  Configured as: $(git config user.name) <$(git config user.email)>"
echo ""

# Set up the default branch name as main, which is the modern convention
echo "Step 3: Configuring repository settings..."
git config init.defaultBranch main
git branch -M main 2>/dev/null || true
echo "  Default branch set to main"
echo ""

# Verify gitignore is in place
echo "Step 4: Verifying gitignore..."
if [ -f ".gitignore" ]; then
    echo "  gitignore is in place"
    echo "  Sensitive files like patient data are properly excluded"
else
    echo "  WARNING: No gitignore file found. This is dangerous because"
    echo "  patient data could be accidentally committed. Please ensure"
    echo "  .gitignore exists before continuing."
    exit 1
fi
echo ""

# Stage all the files that should be tracked
echo "Step 5: Staging files for first commit..."
git add .
echo "  Staged the following files:"
git status --short | head -30
echo ""

# Make the initial commit
echo "Step 6: Creating initial commit..."
git commit -m "Initial commit of ICU sedation digital twin research repository

This commit contains the complete research repository structure including:
- Source code for feature building, ablation study, and analysis
- SQL query templates for MIMIC-IV data extraction
- Comprehensive documentation including paper skeleton and study guides
- Environment specifications for reproducibility
- Progress reports documenting research history

The repository is structured following conventions for medical informatics
publications to support eventual paper submission and code sharing."

echo "  Initial commit created"
echo ""

# Show the repository status
echo "====================================================================="
echo "Repository Successfully Initialized"
echo "====================================================================="
echo ""
echo "Repository contents tracked by Git:"
git ls-files | head -30
echo ""

echo "Total files tracked: $(git ls-files | wc -l)"
echo ""

echo "====================================================================="
echo "Next Steps"
echo "====================================================================="
echo ""
echo "Your local repository is ready to use. To push it to a remote service"
echo "like GitHub, you have two options."
echo ""
echo "Option 1: Push to your own GitHub account"
echo "  First create a new empty repository on GitHub. Do not initialize it"
echo "  with a README since this repository already has one. Then run these"
echo "  commands, replacing the URL with your repository's URL:"
echo ""
echo "    git remote add origin https://github.com/yourname/icu-sedation-research.git"
echo "    git push -u origin main"
echo ""
echo "Option 2: Push to your advisor's GitHub organization"
echo "  If your advisor has a research group GitHub organization, ask for a"
echo "  new repository to be created there. Then push to that URL instead."
echo ""
echo "For now, you can keep working with the repository locally. Make commits"
echo "as you make changes by running:"
echo ""
echo "    git add ."
echo "    git commit -m \"Description of what you changed\""
echo ""
echo "When you are ready to push your local commits to a remote, you can"
echo "set up the remote and push at that time."
