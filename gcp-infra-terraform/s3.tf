resource "aws_s3_bucket" "backups" {
  bucket = "${var.project_name}-backups-${random_id.bucket_suffix.hex}" # Unique bucket name
  # acl    = "private" # Default is private, explicitly setting is good practice

  versioning {
    enabled = true
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  tags = {
    Name        = "${var.project_name}-backup-bucket"
    Environment = "Production" # Or as appropriate
  }
}

resource "random_id" "bucket_suffix" {
 byte_length = 4
}
