variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1" # Choose your preferred region
}

variable "project_name" {
  description = "A name for the project to prefix resources"
  type        = string
  default     = "gcp-data-infra"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "c6a.2xlarge" # Or c6i.2xlarge, etc. Matches your preferred specs
}

variable "ami_id" {
  description = "AMI ID for the EC2 instance (Amazon Linux 2 or Ubuntu). Find the latest for your region."
  type        = string
  # Latest Amazon Linux 2 (Kernel 5.10) for us-east-1 as of 2025-06-11
  default     = "ami-0dc3a08bd93f84a35"
}

variable "ebs_data_volume_size_gb" {
  description = "Size of the EBS data volume in GB"
  type        = number
  default     = 500
}

variable "ssh_key_name" {
  description = "Name of your EC2 Key Pair for SSH access"
  type        = string
  # Ensure this key pair exists in your AWS account in the target region
  default     = "gcp-ec2-key-us-east-1" # Updated via CLI interaction
}

variable "your_ip_for_ssh" {
  description = "Your public IP address for SSH access to the EC2 instance (e.g., 1.2.3.4/32)"
  type        = string
  # You can find this by searching "what is my ip"
  default     = "116.86.8.171/32" # Updated via CLI interaction
}
