resource "aws_iam_role" "ec2_s3_backup_role" {
  name = "${var.project_name}-ec2-s3-backup-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Effect = "Allow",
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
  tags = {
    Name = "${var.project_name}-ec2-s3-backup-role"
  }
}

resource "aws_iam_policy" "s3_backup_policy" {
  name        = "${var.project_name}-s3-backup-policy"
  description = "Allows EC2 to write backups to a specific S3 bucket"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = [
          "s3:PutObject",
          "s3:GetObject", // For restore testing
          "s3:ListBucket" // Useful for scripts
        ],
        Effect   = "Allow",
        Resource = [
          aws_s3_bucket.backups.arn,
          "${aws_s3_bucket.backups.arn}/*" # Bucket and objects within it
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ec2_s3_backup_attach" {
  role       = aws_iam_role.ec2_s3_backup_role.name
  policy_arn = aws_iam_policy.s3_backup_policy.arn
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project_name}-ec2-profile"
  role = aws_iam_role.ec2_s3_backup_role.name
}
