# Changelog

## [YYYY-MM-DD] - AWS EC2 Infrastructure and CI/CD Setup

- **Terraform Infrastructure (in `gcp-infra-terraform/`):**
  - Created `variables.tf` for configurable parameters (AWS region, instance type, AMI, SSH key, user IP).
  - Created `user_data.sh` for EC2 instance initialization (Docker, Docker Compose, AWS CLI, EBS volume setup).
  - Created `vpc.tf` to define VPC, public subnet, internet gateway, and route table.
  - Created `securitygroup.tf` for EC2 instance security (SSH access, outbound traffic).
  - Created `iam.tf` for EC2 instance IAM role and S3 backup policy.
  - Created `s3.tf` to define S3 bucket for backups with versioning and encryption.
  - Created `main.tf` to define EC2 instance, EBS volumes, Elastic IP, and associate other resources.
  - Created `outputs.tf` to output key resource information (EC2 public IP, S3 bucket name).
- **GitHub Actions CI/CD (in `.github/workflows/`):**
  - Created `deploy.yml` workflow to automate deployment to EC2 on push to `main` branch.
  - Workflow handles: code checkout, SSH connection, `.env` file creation from secrets, git pull, Docker Compose service management (pull, down, up --build), and Docker cleanup.
- **Task Management:**
  - Completed Task 11: Setup Terraform for AWS Infrastructure Provisioning.
  - Completed Task 12: Setup CI/CD Pipeline with GitHub Actions for EC2 Deployment.

**Action Required by User:**
1. Update `gcp-infra-terraform/variables.tf` with actual `ami_id`, `ssh_key_name`, and `your_ip_for_ssh`.
2. Run `terraform init`, `terraform plan`, and `terraform apply` in `gcp-infra-terraform/`.
3. Configure GitHub repository secrets: `EC2_HOST`, `EC2_USERNAME`, `EC2_SSH_KEY`, `DOTENV_CONTENT`.
