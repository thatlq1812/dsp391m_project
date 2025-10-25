#!/bin/bash
# Check available Ubuntu images in GCP
# Useful for troubleshooting deployment issues

echo "Checking available Ubuntu images in GCP..."
echo ""

echo "Ubuntu 22.04 LTS images:"
gcloud compute images list \
  --project=ubuntu-os-cloud \
  --filter="family:ubuntu-2204-lts" \
  --format="table(name,family,status)" \
  --limit=3

echo ""
echo "Ubuntu 24.04 LTS images:"
gcloud compute images list \
  --project=ubuntu-os-cloud \
  --filter="family:ubuntu-2404-lts-amd64" \
  --format="table(name,family,status)" \
  --limit=3

echo ""
echo "All Ubuntu families:"
gcloud compute images list \
  --project=ubuntu-os-cloud \
  --format="value(family)" | sort -u | grep ubuntu
