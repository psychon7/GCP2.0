terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0" # Use a recent version
    }
    random = {
      source = "hashicorp/random"
      version = "~> 3.1"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

resource "aws_eip" "main_eip" {
  # instance = aws_instance.main.id # Association is now implicitly handled by instance_id in newer AWS provider versions if not using vpc=true
  # For EC2-VPC, it's better to associate it directly to the instance or network interface.
  # However, to avoid circular dependencies if instance depends on EIP for something, or for clarity:
  # We can leave it unassociated here and associate it in the aws_instance, or use depends_on.
  # The simplest for an EIP that should live with an instance is to associate it via the instance argument.
  # Let's try associating it via the instance attribute directly for clarity if the provider version supports it well.
  # If instance is not yet created, this will be pending. Terraform handles this.
  # Update: The `instance` argument in `aws_eip` is for EC2-Classic. For VPC, associate via `aws_eip_association` or let the instance claim it.
  # For simplicity and to ensure it's tied to the lifecycle of this TF config, we'll create it and then it can be associated.
  # The aws_instance below does not directly reference this EIP for its creation, but the EIP can be associated post-creation or by other means.
  # A common pattern is to output the EIP and manually associate or use aws_eip_association.
  # Let's create it and then associate it with aws_eip_association for explicitness.
  tags = {
    Name = "${var.project_name}-eip"
  }
}

resource "aws_instance" "main" {
  ami                         = var.ami_id
  instance_type               = var.instance_type
  subnet_id                   = aws_subnet.public.id
  vpc_security_group_ids      = [aws_security_group.ec2_sg.id]
  key_name                    = var.ssh_key_name
  iam_instance_profile        = aws_iam_instance_profile.ec2_profile.name
  associate_public_ip_address = true # This will assign an ephemeral public IP if no EIP is associated.
                                     # We will explicitly associate our EIP later.

  root_block_device {
    volume_size = 30 # GB
    volume_type = "gp3"
    delete_on_termination = true
  }

  ebs_block_device {
    device_name = "/dev/sdf" # This is a suggestion; Linux may rename it (e.g. /dev/xvdf or /dev/nvme1n1)
                             # The user_data.sh script tries to find it more robustly.
    volume_size = var.ebs_data_volume_size_gb
    volume_type = "gp3" # General Purpose SSD
    # iops        = 3000 # Baseline for gp3, can increase if needed
    # throughput  = 125  # Baseline for gp3, can increase if needed
    delete_on_termination = false # IMPORTANT: Keep data volume if instance is terminated
  }

  user_data = filebase64("${path.module}/user_data.sh")

  tags = {
    Name = "${var.project_name}-ec2"
  }
}

resource "aws_eip_association" "eip_assoc" {
  instance_id   = aws_instance.main.id
  allocation_id = aws_eip.main_eip.id
}
