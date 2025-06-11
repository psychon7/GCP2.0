resource "aws_security_group" "ec2_sg" {
  name        = "${var.project_name}-ec2-sg"
  description = "Allow SSH and essential traffic for EC2"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "SSH from your IP"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # WARNING: Allows SSH from anywhere. Required for GitHub Actions.
  }

  # Add other ingress rules if absolutely necessary (e.g., HTTP/S if serving web content)
  # For DBs/Redis/NATS, we'll rely on Docker port mapping to 127.0.0.1 on the host,
  # so no external SG rules needed for them if accessed only from within the EC2.

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1" # Allow all outbound traffic
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-ec2-sg"
  }
}
