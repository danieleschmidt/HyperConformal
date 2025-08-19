# Global Load Balancing with CloudFront CDN and Route 53
# Latency-based routing for sub-100ms response times globally

# CloudFront Distribution for Global CDN
resource "aws_cloudfront_distribution" "hyperconformal_global" {
  provider = aws.us_east_1
  
  origin {
    domain_name = "hyperconformal.ai"
    origin_id   = "hyperconformal-global-origin"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }
  
  # Additional origins for each region
  dynamic "origin" {
    for_each = local.regions
    content {
      domain_name = "${origin.key}.hyperconformal.ai"
      origin_id   = "hyperconformal-${origin.key}"
      
      custom_origin_config {
        http_port              = 80
        https_port             = 443
        origin_protocol_policy = "https-only"
        origin_ssl_protocols   = ["TLSv1.2"]
      }
    }
  }
  
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "HyperConformal Global CDN Distribution"
  default_root_object = "index.html"
  
  # Global caching behavior
  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD", "OPTIONS"]
    target_origin_id = "hyperconformal-global-origin"
    
    forwarded_values {
      query_string = true
      headers      = ["Authorization", "Accept-Language", "CloudFront-Viewer-Country"]
      
      cookies {
        forward = "all"
      }
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
    compress               = true
    
    # Lambda@Edge for intelligent routing
    lambda_function_association {
      event_type   = "viewer-request"
      lambda_arn   = aws_lambda_function.edge_router.qualified_arn
      include_body = false
    }
  }
  
  # API caching behavior with optimized settings
  ordered_cache_behavior {
    path_pattern     = "/api/*"
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD", "OPTIONS"]
    target_origin_id = "hyperconformal-global-origin"
    
    forwarded_values {
      query_string = true
      headers      = ["*"]
      
      cookies {
        forward = "all"
      }
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 0
    max_ttl                = 0
    compress               = true
  }
  
  # Static assets caching with long TTL
  ordered_cache_behavior {
    path_pattern     = "/static/*"
    allowed_methods  = ["GET", "HEAD"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "hyperconformal-global-origin"
    
    forwarded_values {
      query_string = false
      
      cookies {
        forward = "none"
      }
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 31536000
    default_ttl            = 31536000
    max_ttl                = 31536000
    compress               = true
  }
  
  # Regional cache behaviors for compliance
  dynamic "ordered_cache_behavior" {
    for_each = local.compliance_regions
    content {
      path_pattern     = "/${ordered_cache_behavior.key}/*"
      allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
      cached_methods   = ["GET", "HEAD", "OPTIONS"]
      target_origin_id = "hyperconformal-${ordered_cache_behavior.key}"
      
      forwarded_values {
        query_string = true
        headers      = ["*"]
        
        cookies {
          forward = "all"
        }
      }
      
      viewer_protocol_policy = "redirect-to-https"
      min_ttl                = 0
      default_ttl            = 300
      max_ttl                = 3600
      compress               = true
    }
  }
  
  price_class = "PriceClass_All"
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate.hyperconformal_global.arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }
  
  web_acl_id = aws_wafv2_web_acl.hyperconformal_global.arn
  
  tags = {
    Name        = "hyperconformal-global-cdn"
    Environment = local.environment
    Purpose     = "global-acceleration"
  }
}

# ACM Certificate for CloudFront
resource "aws_acm_certificate" "hyperconformal_global" {
  provider          = aws.us_east_1
  domain_name       = "hyperconformal.ai"
  validation_method = "DNS"
  
  subject_alternative_names = [
    "*.hyperconformal.ai",
    "api.hyperconformal.ai",
    "us-east-1.hyperconformal.ai",
    "eu-west-1.hyperconformal.ai",
    "ap-southeast-1.hyperconformal.ai",
    "sa-east-1.hyperconformal.ai"
  ]
  
  lifecycle {
    create_before_destroy = true
  }
  
  tags = {
    Name        = "hyperconformal-global-cert"
    Environment = local.environment
  }
}

# ACM Certificate validation
resource "aws_acm_certificate_validation" "hyperconformal_global" {
  provider        = aws.us_east_1
  certificate_arn = aws_acm_certificate.hyperconformal_global.arn
  
  validation_record_fqdns = [
    for record in aws_route53_record.cert_validation : record.fqdn
  ]
}

# Route 53 DNS records for certificate validation
resource "aws_route53_record" "cert_validation" {
  provider = aws.us_east_1
  for_each = {
    for dvo in aws_acm_certificate.hyperconformal_global.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }
  
  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = aws_route53_zone.hyperconformal_global.zone_id
}

# Lambda@Edge function for intelligent routing
resource "aws_lambda_function" "edge_router" {
  provider         = aws.us_east_1
  filename         = "edge-router.zip"
  function_name    = "hyperconformal-edge-router"
  role            = aws_iam_role.edge_lambda.arn
  handler         = "index.handler"
  source_code_hash = data.archive_file.edge_router.output_base64sha256
  runtime         = "nodejs18.x"
  timeout         = 5
  
  publish = true
  
  tags = {
    Name        = "hyperconformal-edge-router"
    Environment = local.environment
    Purpose     = "intelligent-routing"
  }
}

# Edge router source code
data "archive_file" "edge_router" {
  type        = "zip"
  output_path = "edge-router.zip"
  
  source {
    content = templatefile("${path.module}/edge-router.js", {
      regions = jsonencode(local.regions)
    })
    filename = "index.js"
  }
}

# IAM role for Lambda@Edge
resource "aws_iam_role" "edge_lambda" {
  provider = aws.us_east_1
  name     = "hyperconformal-edge-lambda-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = [
            "lambda.amazonaws.com",
            "edgelambda.amazonaws.com"
          ]
        }
      }
    ]
  })
  
  tags = {
    Name        = "hyperconformal-edge-lambda-role"
    Environment = local.environment
  }
}

resource "aws_iam_role_policy_attachment" "edge_lambda_basic" {
  provider   = aws.us_east_1
  role       = aws_iam_role.edge_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Application Load Balancers for each region
resource "aws_lb" "regional_alb" {
  for_each = local.regions
  
  name               = "hyperconformal-${each.key}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb[each.key].id]
  subnets           = module.eks_${replace(each.key, "-", "_")}.public_subnet_ids
  
  enable_deletion_protection = true
  enable_http2              = true
  enable_waf_fail_open      = false
  
  access_logs {
    bucket  = aws_s3_bucket.alb_logs[each.key].bucket
    prefix  = "alb-logs"
    enabled = true
  }
  
  tags = {
    Name        = "hyperconformal-${each.key}-alb"
    Environment = local.environment
    Region      = each.value
    Purpose     = "regional-load-balancing"
  }
}

# ALB Security Groups
resource "aws_security_group" "alb" {
  for_each = local.regions
  
  name        = "hyperconformal-${each.key}-alb-sg"
  description = "Security group for ALB in ${each.value}"
  vpc_id      = module.eks_${replace(each.key, "-", "_")}.vpc_id
  
  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "hyperconformal-${each.key}-alb-sg"
    Environment = local.environment
    Region      = each.value
  }
}

# S3 buckets for ALB access logs
resource "aws_s3_bucket" "alb_logs" {
  for_each = local.regions
  
  bucket = "hyperconformal-${each.key}-alb-logs-${random_string.bucket_suffix.result}"
  
  tags = {
    Name        = "hyperconformal-${each.key}-alb-logs"
    Environment = local.environment
    Region      = each.value
    Purpose     = "alb-access-logs"
  }
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# S3 bucket policies for ALB access logs
resource "aws_s3_bucket_policy" "alb_logs" {
  for_each = local.regions
  
  bucket = aws_s3_bucket.alb_logs[each.key].id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          AWS = data.aws_elb_service_account.main[each.key].arn
        }
        Action   = "s3:PutObject"
        Resource = "${aws_s3_bucket.alb_logs[each.key].arn}/*"
      },
      {
        Effect = "Allow"
        Principal = {
          Service = "delivery.logs.amazonaws.com"
        }
        Action   = "s3:PutObject"
        Resource = "${aws_s3_bucket.alb_logs[each.key].arn}/*"
        Condition = {
          StringEquals = {
            "s3:x-amz-acl" = "bucket-owner-full-control"
          }
        }
      },
      {
        Effect = "Allow"
        Principal = {
          Service = "delivery.logs.amazonaws.com"
        }
        Action   = "s3:GetBucketAcl"
        Resource = aws_s3_bucket.alb_logs[each.key].arn
      }
    ]
  })
}

# ELB service account data sources
data "aws_elb_service_account" "main" {
  for_each = local.regions
}

# Route 53 latency-based routing
resource "aws_route53_record" "api_latency" {
  for_each = local.regions
  
  provider = aws.us_east_1
  zone_id  = aws_route53_zone.hyperconformal_global.zone_id
  name     = "api.hyperconformal.ai"
  type     = "A"
  
  set_identifier = each.key
  
  alias {
    name                   = aws_lb.regional_alb[each.key].dns_name
    zone_id               = aws_lb.regional_alb[each.key].zone_id
    evaluate_target_health = true
  }
  
  latency_routing_policy {
    region = each.value
  }
  
  health_check_id = aws_route53_health_check.regional[each.key].id
}

# Route 53 health checks for each region
resource "aws_route53_health_check" "regional" {
  for_each = local.regions
  
  provider                            = aws.us_east_1
  fqdn                               = aws_lb.regional_alb[each.key].dns_name
  port                               = 443
  type                               = "HTTPS"
  resource_path                      = "/health"
  failure_threshold                  = 3
  request_interval                   = 30
  cloudwatch_alarm_region           = each.value
  cloudwatch_alarm_name             = "hyperconformal-${each.key}-health"
  insufficient_data_health_status   = "Failure"
  
  tags = {
    Name        = "hyperconformal-${each.key}-health-check"
    Environment = local.environment
    Region      = each.value
  }
}

# WAF Web ACL for DDoS protection and security
resource "aws_wafv2_web_acl" "hyperconformal_global" {
  provider = aws.us_east_1
  name     = "hyperconformal-global-waf"
  scope    = "CLOUDFRONT"
  
  default_action {
    allow {}
  }
  
  # Rate limiting rule
  rule {
    name     = "RateLimitRule"
    priority = 1
    
    action {
      block {}
    }
    
    statement {
      rate_based_statement {
        limit              = 10000
        aggregate_key_type = "IP"
        
        scope_down_statement {
          geo_match_statement {
            country_codes = ["CN", "RU", "KP"]
          }
        }
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "RateLimitRule"
      sampled_requests_enabled   = true
    }
  }
  
  # AWS Managed Rules
  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 2
    
    override_action {
      none {}
    }
    
    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "CommonRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }
  
  # Known bad inputs rule
  rule {
    name     = "AWSManagedRulesKnownBadInputsRuleSet"
    priority = 3
    
    override_action {
      none {}
    }
    
    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesKnownBadInputsRuleSet"
        vendor_name = "AWS"
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "KnownBadInputsRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }
  
  tags = {
    Name        = "hyperconformal-global-waf"
    Environment = local.environment
    Purpose     = "ddos-protection"
  }
  
  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "hyperconformalGlobalWAF"
    sampled_requests_enabled   = true
  }
}

# Outputs
output "cloudfront_distribution_id" {
  description = "ID of the CloudFront distribution"
  value       = aws_cloudfront_distribution.hyperconformal_global.id
}

output "cloudfront_distribution_domain_name" {
  description = "Domain name of the CloudFront distribution"
  value       = aws_cloudfront_distribution.hyperconformal_global.domain_name
}

output "regional_load_balancers" {
  description = "Regional load balancer information"
  value = {
    for region, alb in aws_lb.regional_alb : region => {
      dns_name = alb.dns_name
      zone_id  = alb.zone_id
      arn      = alb.arn
    }
  }
}

output "route53_zone_id" {
  description = "Route 53 zone ID for the global domain"
  value       = aws_route53_zone.hyperconformal_global.zone_id
}

output "waf_web_acl_arn" {
  description = "ARN of the WAF Web ACL"
  value       = aws_wafv2_web_acl.hyperconformal_global.arn
}