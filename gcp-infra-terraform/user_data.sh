#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "Starting user data script..."

# Update and install common utilities
sudo yum update -y
sudo yum install -y git docker aws-cli tree

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user

# Install Docker Compose
LATEST_COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d\" -f4)
sudo curl -L "https://github.com/docker/compose/releases/download/${LATEST_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose # Optional: symlink for easier access

echo "Docker and Docker Compose installed."

# Format and mount the data EBS volume (assuming it will be /dev/xvdf or /dev/nvme1n1)
# Find the device name. It can vary. lsblk can help.
# For NVMe instances, it might be /dev/nvme1n1, /dev/nvme2n1 etc.
# For Xen instances (older types), it might be /dev/xvdf, /dev/xvdg etc.
# We'll try to be a bit robust here.
DATA_DEVICE=""
# Wait for devices to be available
sleep 10 

# Check for NVMe devices first (common on newer instance types)
for device in $(ls /dev/nvme*n1 2>/dev/null); do
  if [ -b "$device" ] && ! lsblk -no MOUNTPOINT "$device" | grep -q "."; then # Check if block device and not mounted
    # Further check if it's the correct size (approx 500GB)
    SIZE_IN_BYTES=$(lsblk -b -no SIZE "$device")
    SIZE_IN_GB=$((SIZE_IN_BYTES / 1024 / 1024 / 1024))
    if [ "$SIZE_IN_GB" -gt 450 ] && [ "$SIZE_IN_GB" -lt 550 ]; then # Adjust range if your size differs
        DATA_DEVICE=$device
        break
    fi
  fi
done

# If not found, check for Xen virtual devices
if [ -z "$DATA_DEVICE" ]; then
  for device_letter in {f..p}; do # Check /dev/xvdf through /dev/xvdp
    device="/dev/xvd${device_letter}"
    if [ -b "$device" ] && ! lsblk -no MOUNTPOINT "$device" | grep -q "."; then
      SIZE_IN_BYTES=$(lsblk -b -no SIZE "$device")
      SIZE_IN_GB=$((SIZE_IN_BYTES / 1024 / 1024 / 1024))
      if [ "$SIZE_IN_GB" -gt 450 ] && [ "$SIZE_IN_GB" -lt 550 ]; then
          DATA_DEVICE=$device
          break
      fi
    fi
  done
fi

if [ -n "$DATA_DEVICE" ]; then
    echo "Data device found: $DATA_DEVICE"
    # Check if filesystem already exists (e.g., on reboot if not in fstab yet but formatted)
    FS_TYPE=$(lsblk -no FSTYPE "$DATA_DEVICE")
    if [ -z "$FS_TYPE" ]; then
        echo "Formatting $DATA_DEVICE with xfs..."
        sudo mkfs -t xfs "$DATA_DEVICE"
    else
        echo "Device $DATA_DEVICE already has a filesystem: $FS_TYPE. Not formatting."
    fi
    
    echo "Creating mount point /data..."
    sudo mkdir -p /data
    
    # Add to /etc/fstab to mount on boot
    UUID=$(sudo blkid -s UUID -o value "$DATA_DEVICE")
    if ! grep -q "UUID=$UUID" /etc/fstab; then
        echo "Adding $DATA_DEVICE to /etc/fstab..."
        echo "UUID=$UUID  /data  xfs  defaults,nofail  0  2" | sudo tee -a /etc/fstab
    else
        echo "$DATA_DEVICE already in /etc/fstab."
    fi
    
    echo "Mounting /data..."
    sudo mount -a # Mount all filesystems in fstab, including the new one
    sudo chown ec2-user:ec2-user /data # Give ec2-user ownership
else
    echo "ERROR: Could not identify the 500GB data EBS volume to format and mount."
fi

# (Optional) Clone your application repository for the first time
# cd /home/ec2-user
# git clone YOUR_GIT_REPO_URL gcp_project
# cd gcp_project/services/data_infrastructure
# Create a placeholder .env file if needed, or manage it via GitHub Actions later
# echo "PLACEHOLDER_VAR=yourvalue" > .env

echo "User data script finished."
