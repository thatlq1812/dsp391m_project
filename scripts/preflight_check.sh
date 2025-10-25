#!/bin/bash
# Pre-flight check before cloud deployment
# Ensures all prerequisites are met

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASSED=0
FAILED=0
WARNINGS=0

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════${NC}\n"
}

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

print_header "Pre-flight Check for Cloud Deployment"

echo "Checking prerequisites..."
echo ""

# Check 1: gcloud CLI
echo -n "Checking gcloud CLI... "
if command -v gcloud &> /dev/null; then
    VERSION=$(gcloud version --format="value(core)" 2>/dev/null || echo "unknown")
    check_pass "gcloud CLI installed (version: $VERSION)"
else
    check_fail "gcloud CLI not found. Install from: https://cloud.google.com/sdk/docs/install"
fi

# Check 2: Authentication
echo -n "Checking gcloud authentication... "
if gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null)
    check_pass "Authenticated as: $ACCOUNT"
else
    check_fail "Not authenticated. Run: gcloud auth login"
fi

# Check 3: Project ID
echo -n "Checking GCP project... "
if [ -n "$GCP_PROJECT_ID" ]; then
    check_pass "GCP_PROJECT_ID set: $GCP_PROJECT_ID"
elif gcloud config get-value project &> /dev/null; then
    PROJECT=$(gcloud config get-value project 2>/dev/null)
    check_warn "Using default project: $PROJECT (set GCP_PROJECT_ID env var to override)"
    export GCP_PROJECT_ID="$PROJECT"
else
    check_fail "GCP_PROJECT_ID not set. Run: export GCP_PROJECT_ID='your-project-id'"
fi

# Check 4: Git
echo -n "Checking git... "
if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version | awk '{print $3}')
    check_pass "git installed (version: $GIT_VERSION)"
else
    check_warn "git not found (optional, but recommended)"
fi

# Check 5: Repository
echo -n "Checking repository... "
if [ -d ".git" ]; then
    BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
    check_pass "Git repository (branch: $BRANCH)"
else
    check_warn "Not in a git repository (deployment will clone from GitHub)"
fi

# Check 6: Configuration files
echo -n "Checking configuration files... "
if [ -f "configs/project_config.yaml" ]; then
    check_pass "Configuration file exists"
else
    check_fail "configs/project_config.yaml not found"
fi

# Check 7: Deployment script
echo -n "Checking deployment script... "
if [ -f "scripts/deploy_week_collection.sh" ]; then
    if [ -x "scripts/deploy_week_collection.sh" ]; then
        check_pass "Deployment script ready"
    else
        check_warn "Deployment script not executable. Run: chmod +x scripts/deploy_week_collection.sh"
    fi
else
    check_fail "scripts/deploy_week_collection.sh not found"
fi

# Check 8: Compute Engine API
if [ -n "$GCP_PROJECT_ID" ]; then
    echo -n "Checking Compute Engine API... "
    if gcloud services list --enabled --project=$GCP_PROJECT_ID 2>/dev/null | grep -q compute.googleapis.com; then
        check_pass "Compute Engine API enabled"
    else
        check_warn "Compute Engine API not enabled (will be enabled during deployment)"
    fi
fi

# Check 9: Billing
if [ -n "$GCP_PROJECT_ID" ]; then
    echo -n "Checking billing... "
    if gcloud beta billing projects describe $GCP_PROJECT_ID &>/dev/null; then
        BILLING=$(gcloud beta billing projects describe $GCP_PROJECT_ID --format="value(billingEnabled)" 2>/dev/null || echo "unknown")
        if [ "$BILLING" = "True" ]; then
            check_pass "Billing enabled"
        else
            check_fail "Billing not enabled. Enable at: https://console.cloud.google.com/billing"
        fi
    else
        check_warn "Cannot verify billing status (command may not be available)"
    fi
fi

# Check 10: Quotas
echo -n "Checking VM quotas... "
if [ -n "$GCP_PROJECT_ID" ] && [ -n "$GCP_ZONE" ]; then
    # This is a simplified check - full quota check requires more complex logic
    check_warn "Quota check skipped (verify manually if needed)"
else
    check_warn "Quota check skipped (GCP_ZONE not set)"
fi

# Check 11: API Key (optional)
echo -n "Checking Google Maps API key... "
if [ -n "$GOOGLE_MAPS_API_KEY" ]; then
    check_pass "API key set (will use real API)"
    echo -e "${YELLOW}   Expected cost: ~\$168 for 7 days${NC}"
elif [ "$USE_REAL_API" = "true" ]; then
    check_fail "USE_REAL_API=true but GOOGLE_MAPS_API_KEY not set"
else
    check_pass "Using Mock API (FREE)"
    echo -e "${GREEN}   Cost: \$0 for data collection${NC}"
fi

# Check 12: Disk space (local)
echo -n "Checking local disk space... "
if command -v df &> /dev/null; then
    AVAILABLE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$AVAILABLE" -gt 10 ]; then
        check_pass "Sufficient disk space (${AVAILABLE}GB available)"
    else
        check_warn "Low disk space (${AVAILABLE}GB available, recommend 10GB+)"
    fi
else
    check_warn "Cannot check disk space"
fi

# Summary
echo ""
print_header "Pre-flight Check Summary"

echo -e "Checks passed:   ${GREEN}$PASSED${NC}"
echo -e "Checks failed:   ${RED}$FAILED${NC}"
echo -e "Warnings:        ${YELLOW}$WARNINGS${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    echo ""
    echo "Ready to deploy. Run:"
    echo "  ./scripts/deploy_week_collection.sh"
    echo ""
    echo "Estimated costs:"
    if [ "$USE_REAL_API" = "true" ] || [ -n "$GOOGLE_MAPS_API_KEY" ]; then
        echo "  - VM (7 days): ~\$12"
        echo "  - Google API (7 days): ~\$168"
        echo "  - Total: ~\$180"
    else
        echo "  - VM (7 days): ~\$12"
        echo "  - Google API: \$0 (Mock API)"
        echo "  - Total: ~\$12"
    fi
    exit 0
else
    echo -e "${RED}✗ Some critical checks failed.${NC}"
    echo ""
    echo "Please fix the issues above before deploying."
    echo ""
    echo "Quick fixes:"
    echo "  - Install gcloud: https://cloud.google.com/sdk/docs/install"
    echo "  - Authenticate: gcloud auth login"
    echo "  - Set project: export GCP_PROJECT_ID='your-project-id'"
    echo "  - Enable billing: https://console.cloud.google.com/billing"
    exit 1
fi
