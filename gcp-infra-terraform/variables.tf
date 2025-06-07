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
  default     = "c6a.2xlarge" # Or c6i.2xlarge, etc.
}

variable "ami_id" {
  description = "AMI ID for the EC2 instance (Amazon Linux 2 or Ubuntu). Find the latest for your region."
  type        = string
  # Example for Amazon Linux 2 (x86) in us-east-1 - ALWAYS VERIFY LATEST
  # You can use AWS CLI: aws ec2 describe-images --owners amazon --filters "Name=name,Values=amzn2-ami-hvm-*-x86_64-gp2" "Name=state,Values=available" --query "sort_by(Images, &CreationDate)[-1].[ImageId]" --output text
  default     = "ami-0c7217cdde317cfec" # Replace with a current AMI for your region
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
  # e.g., "my-aws-key"
  default     = "" # PLEASE PROVIDE YOUR SSH KEY NAME
}

variable "your_ip_for_ssh" {
  description = "Your public IP address for SSH access to the EC2 instance (e.g., 1.2.3.4/32)"
  type        = string
  # You can find this by searching "what is my ip"
  default     = "0.0.0.0/0" # PLEASE REPLACE WITH YOUR IP ADDRESS (e.g., "your.ip.add.ress/32")
}
