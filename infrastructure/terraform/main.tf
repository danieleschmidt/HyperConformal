# HyperConformal Global Production Infrastructure
# Multi-Region AWS Deployment with Compliance Framework

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
  
  backend "s3" {
    bucket         = "hyperconformal-terraform-state"
    key            = "global-production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "hyperconformal-terraform-locks"
  }
}

# Provider configurations for multi-region deployment
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
  
  default_tags {
    tags = {
      Project     = "HyperConformal"
      Environment = "production"
      Region      = "us-east-1"
      Owner       = "terragon-labs"
      CostCenter  = "research-production"
    }
  }
}

provider "aws" {
  alias  = "eu_west_1"
  region = "eu-west-1"
  
  default_tags {
    tags = {
      Project     = "HyperConformal"
      Environment = "production"
      Region      = "eu-west-1"
      Owner       = "terragon-labs"
      CostCenter  = "research-production"
    }
  }
}

provider "aws" {
  alias  = "ap_southeast_1"
  region = "ap-southeast-1"
  
  default_tags {
    tags = {
      Project     = "HyperConformal"
      Environment = "production"
      Region      = "ap-southeast-1"
      Owner       = "terragon-labs"
      CostCenter  = "research-production"
    }
  }
}

provider "aws" {
  alias  = "sa_east_1"
  region = "sa-east-1"
  
  default_tags {
    tags = {
      Project     = "HyperConformal"
      Environment = "production"
      Region      = "sa-east-1"
      Owner       = "terragon-labs"
      CostCenter  = "research-production"
    }
  }
}

# Data sources for availability zones
data "aws_availability_zones" "us_east_1" {
  provider = aws.us_east_1
  state    = "available"
}

data "aws_availability_zones" "eu_west_1" {
  provider = aws.eu_west_1
  state    = "available"
}

data "aws_availability_zones" "ap_southeast_1" {
  provider = aws.ap_southeast_1
  state    = "available"
}

data "aws_availability_zones" "sa_east_1" {
  provider = aws.sa_east_1
  state    = "available"
}

# Global Variables
locals {
  project_name = "hyperconformal"
  environment  = "production"
  
  regions = {
    us_east_1      = "us-east-1"
    eu_west_1      = "eu-west-1"
    ap_southeast_1 = "ap-southeast-1"
    sa_east_1      = "sa-east-1"
  }
  
  # Compliance regions mapping
  compliance_regions = {
    us_east_1      = ["ccpa", "soc2"]
    eu_west_1      = ["gdpr", "soc2"]
    ap_southeast_1 = ["pdpa", "soc2"]
    sa_east_1      = ["lgpd", "soc2"]
  }
  
  # Performance requirements
  performance_targets = {
    response_time_ms    = 100
    availability_sla    = 99.99
    throughput_rps      = 347000
    max_scaling_pods    = 1000
  }
}

# Global Route 53 Hosted Zone
resource "aws_route53_zone" "hyperconformal_global" {
  provider = aws.us_east_1
  name     = "hyperconformal.ai"
  
  tags = {
    Name        = "hyperconformal-global-dns"
    Environment = local.environment
    Purpose     = "global-routing"
  }
}

# S3 Bucket for Terraform State (if not exists)
resource "aws_s3_bucket" "terraform_state" {
  provider = aws.us_east_1
  bucket   = "hyperconformal-terraform-state"
  
  tags = {
    Name        = "hyperconformal-terraform-state"
    Environment = local.environment
    Purpose     = "terraform-state"
  }
}

resource "aws_s3_bucket_versioning" "terraform_state" {
  provider = aws.us_east_1
  bucket   = aws_s3_bucket.terraform_state.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "terraform_state" {
  provider = aws.us_east_1
  bucket   = aws_s3_bucket.terraform_state.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

# DynamoDB table for Terraform locks
resource "aws_dynamodb_table" "terraform_locks" {
  provider   = aws.us_east_1
  name       = "hyperconformal-terraform-locks"
  billing_mode = "PAY_PER_REQUEST"
  hash_key   = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }

  tags = {
    Name        = "hyperconformal-terraform-locks"
    Environment = local.environment
    Purpose     = "terraform-locks"
  }
}

# CloudWatch Log Groups for centralized logging
resource "aws_cloudwatch_log_group" "hyperconformal_us_east_1" {
  provider          = aws.us_east_1
  name              = "/aws/hyperconformal/production/us-east-1"
  retention_in_days = 30
  
  tags = {
    Environment = local.environment
    Region      = "us-east-1"
    Purpose     = "application-logs"
  }
}

resource "aws_cloudwatch_log_group" "hyperconformal_eu_west_1" {
  provider          = aws.eu_west_1
  name              = "/aws/hyperconformal/production/eu-west-1"
  retention_in_days = 30
  
  tags = {
    Environment = local.environment
    Region      = "eu-west-1"
    Purpose     = "application-logs"
  }
}

resource "aws_cloudwatch_log_group" "hyperconformal_ap_southeast_1" {
  provider          = aws.ap_southeast_1
  name              = "/aws/hyperconformal/production/ap-southeast-1"
  retention_in_days = 30
  
  tags = {
    Environment = local.environment
    Region      = "ap-southeast-1"
    Purpose     = "application-logs"
  }
}

resource "aws_cloudwatch_log_group" "hyperconformal_sa_east_1" {
  provider          = aws.sa_east_1
  name              = "/aws/hyperconformal/production/sa-east-1"
  retention_in_days = 30
  
  tags = {
    Environment = local.environment
    Region      = "sa-east-1"
    Purpose     = "application-logs"
  }
}