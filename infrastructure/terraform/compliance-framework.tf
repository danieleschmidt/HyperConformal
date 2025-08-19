# Comprehensive Compliance Framework
# GDPR, CCPA, PDPA, SOC 2 Type II implementation

# Data Classification and Encryption Keys
resource "aws_kms_key" "data_encryption" {
  for_each = local.regions
  
  provider                = aws.${replace(each.key, "-", "_")}
  description             = "HyperConformal data encryption key for ${each.value}"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow service access"
        Effect = "Allow"
        Principal = {
          Service = [
            "s3.amazonaws.com",
            "rds.amazonaws.com",
            "logs.amazonaws.com",
            "eks.amazonaws.com"
          ]
        }
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        Resource = "*"
      }
    ]
  })
  
  tags = {
    Name        = "hyperconformal-${each.key}-encryption-key"
    Environment = local.environment
    Region      = each.value
    Compliance  = join(",", local.compliance_regions[each.key])
    Purpose     = "data-encryption"
  }
}

resource "aws_kms_alias" "data_encryption" {
  for_each = local.regions
  
  provider      = aws.${replace(each.key, "-", "_")}
  name          = "alias/hyperconformal-${each.key}-data"
  target_key_id = aws_kms_key.data_encryption[each.key].key_id
}

# Data residency S3 buckets with compliance controls
resource "aws_s3_bucket" "data_residency" {
  for_each = local.regions
  
  provider = aws.${replace(each.key, "-", "_")}
  bucket   = "hyperconformal-${each.key}-data-${random_string.bucket_suffix.result}"
  
  tags = {
    Name        = "hyperconformal-${each.key}-data"
    Environment = local.environment
    Region      = each.value
    Compliance  = join(",", local.compliance_regions[each.key])
    DataClass   = "personal-data"
    Purpose     = "data-residency"
  }
}

resource "aws_s3_bucket_encryption" "data_residency" {
  for_each = local.regions
  
  provider = aws.${replace(each.key, "-", "_")}
  bucket   = aws_s3_bucket.data_residency[each.key].id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        kms_master_key_id = aws_kms_key.data_encryption[each.key].arn
        sse_algorithm     = "aws:kms"
      }
      bucket_key_enabled = true
    }
  }
}

resource "aws_s3_bucket_versioning" "data_residency" {
  for_each = local.regions
  
  provider = aws.${replace(each.key, "-", "_")}
  bucket   = aws_s3_bucket.data_residency[each.key].id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "data_residency" {
  for_each = local.regions
  
  provider = aws.${replace(each.key, "-", "_")}
  bucket   = aws_s3_bucket.data_residency[each.key].id

  rule {
    id     = "gdpr_retention_policy"
    status = "Enabled"

    # GDPR: Delete after 2 years
    dynamic "expiration" {
      for_each = contains(local.compliance_regions[each.key], "gdpr") ? [1] : []
      content {
        days = 730
      }
    }
    
    # CCPA: Delete after 3 years
    dynamic "expiration" {
      for_each = contains(local.compliance_regions[each.key], "ccpa") && !contains(local.compliance_regions[each.key], "gdpr") ? [1] : []
      content {
        days = 1095
      }
    }
    
    # PDPA: Delete after 1 year
    dynamic "expiration" {
      for_each = contains(local.compliance_regions[each.key], "pdpa") ? [1] : []
      content {
        days = 365
      }
    }
    
    # SOC 2: Archive to Glacier after 90 days
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    
    # SOC 2: Deep archive after 1 year
    transition {
      days          = 365
      storage_class = "DEEP_ARCHIVE"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data_residency" {
  for_each = local.regions
  
  provider = aws.${replace(each.key, "-", "_")}
  bucket   = aws_s3_bucket.data_residency[each.key].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Audit logging for compliance
resource "aws_cloudtrail" "compliance_audit" {
  for_each = local.regions
  
  provider                      = aws.${replace(each.key, "-", "_")}
  name                          = "hyperconformal-${each.key}-audit-trail"
  s3_bucket_name               = aws_s3_bucket.audit_logs[each.key].bucket
  s3_key_prefix                = "cloudtrail-logs/"
  include_global_service_events = each.key == "us_east_1" ? true : false
  is_multi_region_trail        = false
  enable_logging               = true
  enable_log_file_validation   = true
  kms_key_id                   = aws_kms_key.data_encryption[each.key].arn

  event_selector {
    read_write_type                 = "All"
    include_management_events       = true
    exclude_management_event_sources = []

    data_resource {
      type   = "AWS::S3::Object"
      values = ["${aws_s3_bucket.data_residency[each.key].arn}/*"]
    }
    
    data_resource {
      type   = "AWS::S3::Bucket"
      values = [aws_s3_bucket.data_residency[each.key].arn]
    }
  }
  
  tags = {
    Name        = "hyperconformal-${each.key}-audit-trail"
    Environment = local.environment
    Region      = each.value
    Compliance  = join(",", local.compliance_regions[each.key])
    Purpose     = "compliance-audit"
  }
}

# S3 buckets for audit logs
resource "aws_s3_bucket" "audit_logs" {
  for_each = local.regions
  
  provider = aws.${replace(each.key, "-", "_")}
  bucket   = "hyperconformal-${each.key}-audit-logs-${random_string.bucket_suffix.result}"
  
  tags = {
    Name        = "hyperconformal-${each.key}-audit-logs"
    Environment = local.environment
    Region      = each.value
    Purpose     = "audit-logs"
  }
}

resource "aws_s3_bucket_policy" "audit_logs" {
  for_each = local.regions
  
  provider = aws.${replace(each.key, "-", "_")}
  bucket   = aws_s3_bucket.audit_logs[each.key].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AWSCloudTrailAclCheck"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action   = "s3:GetBucketAcl"
        Resource = aws_s3_bucket.audit_logs[each.key].arn
        Condition = {
          StringEquals = {
            "AWS:SourceArn" = "arn:aws:cloudtrail:${each.value}:${data.aws_caller_identity.current.account_id}:trail/hyperconformal-${each.key}-audit-trail"
          }
        }
      },
      {
        Sid    = "AWSCloudTrailWrite"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action   = "s3:PutObject"
        Resource = "${aws_s3_bucket.audit_logs[each.key].arn}/*"
        Condition = {
          StringEquals = {
            "s3:x-amz-acl" = "bucket-owner-full-control"
            "AWS:SourceArn" = "arn:aws:cloudtrail:${each.value}:${data.aws_caller_identity.current.account_id}:trail/hyperconformal-${each.key}-audit-trail"
          }
        }
      }
    ]
  })
}

# RDS instances for consent management with encryption
resource "aws_db_subnet_group" "consent_db" {
  for_each = local.regions
  
  provider   = aws.${replace(each.key, "-", "_")}
  name       = "hyperconformal-${each.key}-consent-db-subnet-group"
  subnet_ids = module.eks_${replace(each.key, "-", "_")}.private_subnet_ids

  tags = {
    Name        = "hyperconformal-${each.key}-consent-db-subnet-group"
    Environment = local.environment
    Region      = each.value
  }
}

resource "aws_db_instance" "consent_management" {
  for_each = local.regions
  
  provider = aws.${replace(each.key, "-", "_")}
  
  identifier = "hyperconformal-${each.key}-consent-db"
  
  # Database configuration
  engine              = "postgres"
  engine_version      = "15.4"
  instance_class      = "db.t3.micro"
  allocated_storage   = 20
  max_allocated_storage = 100
  storage_type        = "gp3"
  storage_encrypted   = true
  kms_key_id         = aws_kms_key.data_encryption[each.key].arn
  
  # Database credentials
  db_name  = "hyperconformal_consent"
  username = "consent_admin"
  password = random_password.db_password[each.key].result
  
  # Network configuration
  db_subnet_group_name   = aws_db_subnet_group.consent_db[each.key].name
  vpc_security_group_ids = [aws_security_group.consent_db[each.key].id]
  publicly_accessible    = false
  
  # Backup and maintenance
  backup_retention_period = contains(local.compliance_regions[each.key], "gdpr") ? 7 : 1
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  auto_minor_version_upgrade = true
  
  # Compliance settings
  deletion_protection = true
  skip_final_snapshot = false
  final_snapshot_identifier = "hyperconformal-${each.key}-consent-db-final-snapshot"
  
  # Monitoring
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_monitoring[each.key].arn
  enabled_cloudwatch_logs_exports = ["postgresql"]
  
  tags = {
    Name        = "hyperconformal-${each.key}-consent-db"
    Environment = local.environment
    Region      = each.value
    Compliance  = join(",", local.compliance_regions[each.key])
    Purpose     = "consent-management"
    DataClass   = "personal-data"
  }
}

resource "random_password" "db_password" {
  for_each = local.regions
  
  length  = 32
  special = true
}

resource "aws_secretsmanager_secret" "db_password" {
  for_each = local.regions
  
  provider    = aws.${replace(each.key, "-", "_")}
  name        = "hyperconformal-${each.key}-consent-db-password"
  description = "Password for consent management database"
  kms_key_id  = aws_kms_key.data_encryption[each.key].arn
  
  tags = {
    Name        = "hyperconformal-${each.key}-consent-db-password"
    Environment = local.environment
    Region      = each.value
  }
}

resource "aws_secretsmanager_secret_version" "db_password" {
  for_each = local.regions
  
  provider      = aws.${replace(each.key, "-", "_")}
  secret_id     = aws_secretsmanager_secret.db_password[each.key].id
  secret_string = jsonencode({
    username = aws_db_instance.consent_management[each.key].username
    password = random_password.db_password[each.key].result
    engine   = "postgres"
    host     = aws_db_instance.consent_management[each.key].endpoint
    port     = aws_db_instance.consent_management[each.key].port
    dbname   = aws_db_instance.consent_management[each.key].db_name
  })
}

# Security groups for consent database
resource "aws_security_group" "consent_db" {
  for_each = local.regions
  
  provider    = aws.${replace(each.key, "-", "_")}
  name        = "hyperconformal-${each.key}-consent-db-sg"
  description = "Security group for consent management database"
  vpc_id      = module.eks_${replace(each.key, "-", "_")}.vpc_id

  ingress {
    description     = "PostgreSQL from EKS nodes"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks_${replace(each.key, "-", "_")}.node_security_group_id]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "hyperconformal-${each.key}-consent-db-sg"
    Environment = local.environment
    Region      = each.value
  }
}

# IAM role for RDS monitoring
resource "aws_iam_role" "rds_monitoring" {
  for_each = local.regions
  
  provider = aws.${replace(each.key, "-", "_")}
  name     = "hyperconformal-${each.key}-rds-monitoring-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name        = "hyperconformal-${each.key}-rds-monitoring-role"
    Environment = local.environment
    Region      = each.value
  }
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  for_each = local.regions
  
  provider   = aws.${replace(each.key, "-", "_")}
  role       = aws_iam_role.rds_monitoring[each.key].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# Data discovery and classification
resource "aws_macie2_account" "compliance_scanning" {
  for_each = local.regions
  
  provider                     = aws.${replace(each.key, "-", "_")}
  finding_publishing_frequency = "FIFTEEN_MINUTES"
  status                      = "ENABLED"
}

resource "aws_macie2_classification_job" "data_discovery" {
  for_each = local.regions
  
  provider    = aws.${replace(each.key, "-", "_")}
  job_type    = "ONE_TIME"
  name        = "hyperconformal-${each.key}-data-discovery"
  description = "Data discovery and classification for compliance"
  
  s3_job_definition {
    bucket_definitions {
      account_id = data.aws_caller_identity.current.account_id
      buckets    = [aws_s3_bucket.data_residency[each.key].bucket]
    }
  }
  
  tags = {
    Name        = "hyperconformal-${each.key}-data-discovery"
    Environment = local.environment
    Region      = each.value
    Purpose     = "data-classification"
  }
  
  depends_on = [aws_macie2_account.compliance_scanning]
}

# AWS Config for compliance monitoring
resource "aws_config_configuration_recorder" "compliance" {
  for_each = local.regions
  
  provider = aws.${replace(each.key, "-", "_")}
  name     = "hyperconformal-${each.key}-compliance-recorder"
  role_arn = aws_iam_role.config[each.key].arn

  recording_group {
    all_supported                 = true
    include_global_resource_types = each.key == "us_east_1" ? true : false
  }
}

resource "aws_config_delivery_channel" "compliance" {
  for_each = local.regions
  
  provider       = aws.${replace(each.key, "-", "_")}
  name           = "hyperconformal-${each.key}-compliance-delivery"
  s3_bucket_name = aws_s3_bucket.config_compliance[each.key].bucket
  
  depends_on = [aws_config_configuration_recorder.compliance]
}

# S3 bucket for AWS Config
resource "aws_s3_bucket" "config_compliance" {
  for_each = local.regions
  
  provider = aws.${replace(each.key, "-", "_")}
  bucket   = "hyperconformal-${each.key}-config-${random_string.bucket_suffix.result}"
  
  tags = {
    Name        = "hyperconformal-${each.key}-config"
    Environment = local.environment
    Region      = each.value
    Purpose     = "compliance-monitoring"
  }
}

resource "aws_s3_bucket_policy" "config_compliance" {
  for_each = local.regions
  
  provider = aws.${replace(each.key, "-", "_")}
  bucket   = aws_s3_bucket.config_compliance[each.key].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AWSConfigBucketPermissionsCheck"
        Effect = "Allow"
        Principal = {
          Service = "config.amazonaws.com"
        }
        Action   = "s3:GetBucketAcl"
        Resource = aws_s3_bucket.config_compliance[each.key].arn
        Condition = {
          StringEquals = {
            "AWS:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      },
      {
        Sid    = "AWSConfigBucketExistenceCheck"
        Effect = "Allow"
        Principal = {
          Service = "config.amazonaws.com"
        }
        Action   = "s3:ListBucket"
        Resource = aws_s3_bucket.config_compliance[each.key].arn
        Condition = {
          StringEquals = {
            "AWS:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      },
      {
        Sid    = "AWSConfigBucketDelivery"
        Effect = "Allow"
        Principal = {
          Service = "config.amazonaws.com"
        }
        Action   = "s3:PutObject"
        Resource = "${aws_s3_bucket.config_compliance[each.key].arn}/*"
        Condition = {
          StringEquals = {
            "s3:x-amz-acl" = "bucket-owner-full-control"
            "AWS:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      }
    ]
  })
}

# IAM role for AWS Config
resource "aws_iam_role" "config" {
  for_each = local.regions
  
  provider = aws.${replace(each.key, "-", "_")}
  name     = "hyperconformal-${each.key}-config-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "config.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name        = "hyperconformal-${each.key}-config-role"
    Environment = local.environment
    Region      = each.value
  }
}

resource "aws_iam_role_policy_attachment" "config" {
  for_each = local.regions
  
  provider   = aws.${replace(each.key, "-", "_")}
  role       = aws_iam_role.config[each.key].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/ConfigRole"
}

# Data source for current AWS account
data "aws_caller_identity" "current" {}

# Compliance monitoring rules
resource "aws_config_config_rule" "gdpr_encryption" {
  for_each = {
    for region, frameworks in local.compliance_regions : region => frameworks
    if contains(frameworks, "gdpr")
  }
  
  provider = aws.${replace(each.key, "-", "_")}
  name     = "hyperconformal-${each.key}-gdpr-encryption-check"

  source {
    owner             = "AWS"
    source_identifier = "S3_BUCKET_SERVER_SIDE_ENCRYPTION_ENABLED"
  }

  depends_on = [aws_config_configuration_recorder.compliance]
  
  tags = {
    Name        = "hyperconformal-${each.key}-gdpr-encryption-check"
    Environment = local.environment
    Region      = each.key
    Compliance  = "gdpr"
  }
}

resource "aws_config_config_rule" "soc2_logging" {
  for_each = {
    for region, frameworks in local.compliance_regions : region => frameworks
    if contains(frameworks, "soc2")
  }
  
  provider = aws.${replace(each.key, "-", "_")}
  name     = "hyperconformal-${each.key}-soc2-logging-check"

  source {
    owner             = "AWS"
    source_identifier = "CLOUDTRAIL_ENABLED"
  }

  depends_on = [aws_config_configuration_recorder.compliance]
  
  tags = {
    Name        = "hyperconformal-${each.key}-soc2-logging-check"
    Environment = local.environment
    Region      = each.key
    Compliance  = "soc2"
  }
}

# Outputs
output "compliance_framework" {
  description = "Compliance framework information"
  value = {
    encryption_keys = {
      for region, key in aws_kms_key.data_encryption : region => {
        key_id  = key.key_id
        arn     = key.arn
        alias   = aws_kms_alias.data_encryption[region].name
      }
    }
    
    data_residency_buckets = {
      for region, bucket in aws_s3_bucket.data_residency : region => {
        bucket_name = bucket.bucket
        arn         = bucket.arn
        region      = bucket.region
      }
    }
    
    consent_databases = {
      for region, db in aws_db_instance.consent_management : region => {
        endpoint = db.endpoint
        port     = db.port
        database = db.db_name
      }
    }
    
    audit_trails = {
      for region, trail in aws_cloudtrail.compliance_audit : region => {
        name = trail.name
        arn  = trail.arn
      }
    }
    
    compliance_regions = local.compliance_regions
  }
}