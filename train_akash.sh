#!/bin/bash
# Build, push, and deploy HandFlow training on Akash Network
#
# Prerequisites:
#   1. Docker installed
#   2. Akash CLI installed: https://docs.akash.network/guides/cli
#   3. Akash wallet funded with AKT
#   4. Docker Hub account

set -e

DOCKERHUB_USER="${DOCKERHUB_USER:-andyhuynh24}"  # Change to your Docker Hub username
IMAGE_NAME="handflow-train"
IMAGE_TAG="latest"
FULL_IMAGE="${DOCKERHUB_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "=== Step 1: Build Docker image ==="
docker build -f Dockerfile.train -t "$FULL_IMAGE" .

echo ""
echo "=== Step 2: Push to Docker Hub ==="
echo "Make sure you're logged in: docker login"
docker push "$FULL_IMAGE"

echo ""
echo "=== Step 3: Update deploy.yaml ==="
# Replace placeholder with actual image name
sed -i.bak "s|<YOUR_DOCKERHUB_USERNAME>/handflow-train:latest|${FULL_IMAGE}|g" deploy.yaml
rm -f deploy.yaml.bak

echo ""
echo "=== Step 4: Deploy to Akash ==="
echo "Run these commands manually:"
echo ""
echo "  # Create deployment"
echo "  akash tx deployment create deploy.yaml --from <your-wallet> --node <akash-node> --chain-id akashnet-2"
echo ""
echo "  # Check bids"
echo "  akash query market bid list --owner <your-address>"
echo ""
echo "  # Accept a bid"
echo "  akash tx market lease create --from <your-wallet> --dseq <deployment-seq> --gseq 1 --oseq 1 --provider <provider-address>"
echo ""
echo "  # Send manifest"
echo "  akash provider send-manifest deploy.yaml --from <your-wallet> --provider <provider-address> --dseq <deployment-seq>"
echo ""
echo "  # Check logs"
echo "  akash provider lease-logs --from <your-wallet> --provider <provider-address> --dseq <deployment-seq>"
echo ""
echo "=== OR use Akash Console (easier) ==="
echo "  1. Go to https://console.akash.network"
echo "  2. Connect wallet"
echo "  3. Click 'Deploy' → 'Upload SDL' → select deploy.yaml"
echo "  4. Choose a provider with GPU"
echo "  5. Deploy!"
echo ""
echo "=== To download trained model after training ==="
echo "  # SSH/exec into the container and copy the model out"
echo "  akash provider lease-shell --from <your-wallet> --provider <provider> --dseq <dseq>"
echo "  # Inside container: the model is at /workspace/project/models/hand_action.keras"
