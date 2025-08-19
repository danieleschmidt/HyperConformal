# EKS Clusters for Multi-Region Deployment
# Production-grade Kubernetes clusters with enterprise security

# US East 1 EKS Cluster
module "eks_us_east_1" {
  source = "./modules/eks-cluster"
  
  providers = {
    aws = aws.us_east_1
  }
  
  cluster_name    = "${local.project_name}-us-east-1"
  cluster_version = "1.28"
  region          = local.regions.us_east_1
  environment     = local.environment
  
  vpc_cidr = "10.0.0.0/16"
  
  # Availability zones
  availability_zones = slice(data.aws_availability_zones.us_east_1.names, 0, 3)
  
  # Node groups configuration
  node_groups = {
    general = {
      instance_types = ["m6i.large", "m6i.xlarge"]
      capacity_type  = "ON_DEMAND"
      min_size       = 3
      max_size       = 50
      desired_size   = 5
      
      labels = {
        role = "general"
        region = "us-east-1"
      }
      
      taints = []
    }
    
    compute_intensive = {
      instance_types = ["c6i.2xlarge", "c6i.4xlarge"]
      capacity_type  = "SPOT"
      min_size       = 0
      max_size       = 100
      desired_size   = 2
      
      labels = {
        role = "compute"
        region = "us-east-1"
        workload = "hyperconformal"
      }
      
      taints = [
        {
          key    = "hyperconformal.io/dedicated"
          value  = "compute"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
  
  # Compliance requirements
  compliance_frameworks = local.compliance_regions.us_east_1
  
  # Performance targets
  performance_targets = local.performance_targets
}

# EU West 1 EKS Cluster (GDPR Compliant)
module "eks_eu_west_1" {
  source = "./modules/eks-cluster"
  
  providers = {
    aws = aws.eu_west_1
  }
  
  cluster_name    = "${local.project_name}-eu-west-1"
  cluster_version = "1.28"
  region          = local.regions.eu_west_1
  environment     = local.environment
  
  vpc_cidr = "10.1.0.0/16"
  
  # Availability zones
  availability_zones = slice(data.aws_availability_zones.eu_west_1.names, 0, 3)
  
  # Node groups configuration
  node_groups = {
    general = {
      instance_types = ["m6i.large", "m6i.xlarge"]
      capacity_type  = "ON_DEMAND"
      min_size       = 3
      max_size       = 50
      desired_size   = 5
      
      labels = {
        role = "general"
        region = "eu-west-1"
        compliance = "gdpr"
      }
      
      taints = []
    }
    
    compute_intensive = {
      instance_types = ["c6i.2xlarge", "c6i.4xlarge"]
      capacity_type  = "SPOT"
      min_size       = 0
      max_size       = 100
      desired_size   = 2
      
      labels = {
        role = "compute"
        region = "eu-west-1"
        workload = "hyperconformal"
        compliance = "gdpr"
      }
      
      taints = [
        {
          key    = "hyperconformal.io/dedicated"
          value  = "compute"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
  
  # GDPR specific configurations
  compliance_frameworks = local.compliance_regions.eu_west_1
  data_residency_required = true
  encryption_at_rest_required = true
  
  # Performance targets
  performance_targets = local.performance_targets
}

# AP Southeast 1 EKS Cluster (PDPA Compliant)
module "eks_ap_southeast_1" {
  source = "./modules/eks-cluster"
  
  providers = {
    aws = aws.ap_southeast_1
  }
  
  cluster_name    = "${local.project_name}-ap-southeast-1"
  cluster_version = "1.28"
  region          = local.regions.ap_southeast_1
  environment     = local.environment
  
  vpc_cidr = "10.2.0.0/16"
  
  # Availability zones
  availability_zones = slice(data.aws_availability_zones.ap_southeast_1.names, 0, 3)
  
  # Node groups configuration
  node_groups = {
    general = {
      instance_types = ["m6i.large", "m6i.xlarge"]
      capacity_type  = "ON_DEMAND"
      min_size       = 3
      max_size       = 50
      desired_size   = 5
      
      labels = {
        role = "general"
        region = "ap-southeast-1"
        compliance = "pdpa"
      }
      
      taints = []
    }
    
    compute_intensive = {
      instance_types = ["c6i.2xlarge", "c6i.4xlarge"]
      capacity_type  = "SPOT"
      min_size       = 0
      max_size       = 100
      desired_size   = 2
      
      labels = {
        role = "compute"
        region = "ap-southeast-1"
        workload = "hyperconformal"
        compliance = "pdpa"
      }
      
      taints = [
        {
          key    = "hyperconformal.io/dedicated"
          value  = "compute"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
  
  # PDPA specific configurations
  compliance_frameworks = local.compliance_regions.ap_southeast_1
  data_residency_required = true
  
  # Performance targets
  performance_targets = local.performance_targets
}

# SA East 1 EKS Cluster (LGPD Compliant)
module "eks_sa_east_1" {
  source = "./modules/eks-cluster"
  
  providers = {
    aws = aws.sa_east_1
  }
  
  cluster_name    = "${local.project_name}-sa-east-1"
  cluster_version = "1.28"
  region          = local.regions.sa_east_1
  environment     = local.environment
  
  vpc_cidr = "10.3.0.0/16"
  
  # Availability zones
  availability_zones = slice(data.aws_availability_zones.sa_east_1.names, 0, 3)
  
  # Node groups configuration
  node_groups = {
    general = {
      instance_types = ["m6i.large", "m6i.xlarge"]
      capacity_type  = "ON_DEMAND"
      min_size       = 3
      max_size       = 50
      desired_size   = 5
      
      labels = {
        role = "general"
        region = "sa-east-1"
        compliance = "lgpd"
      }
      
      taints = []
    }
    
    compute_intensive = {
      instance_types = ["c6i.2xlarge", "c6i.4xlarge"]
      capacity_type  = "SPOT"
      min_size       = 0
      max_size       = 100
      desired_size   = 2
      
      labels = {
        role = "compute"
        region = "sa-east-1"
        workload = "hyperconformal"
        compliance = "lgpd"
      }
      
      taints = [
        {
          key    = "hyperconformal.io/dedicated"
          value  = "compute"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
  
  # LGPD specific configurations
  compliance_frameworks = local.compliance_regions.sa_east_1
  
  # Performance targets
  performance_targets = local.performance_targets
}

# Cross-region VPC peering for disaster recovery
resource "aws_vpc_peering_connection" "us_to_eu" {
  provider    = aws.us_east_1
  vpc_id      = module.eks_us_east_1.vpc_id
  peer_vpc_id = module.eks_eu_west_1.vpc_id
  peer_region = local.regions.eu_west_1
  auto_accept = false
  
  tags = {
    Name = "hyperconformal-us-to-eu-peering"
    Environment = local.environment
    Purpose = "disaster-recovery"
  }
}

resource "aws_vpc_peering_connection_accepter" "us_to_eu" {
  provider                  = aws.eu_west_1
  vpc_peering_connection_id = aws_vpc_peering_connection.us_to_eu.id
  auto_accept               = true
  
  tags = {
    Name = "hyperconformal-us-to-eu-peering-accepter"
    Environment = local.environment
    Purpose = "disaster-recovery"
  }
}

# Outputs
output "eks_clusters" {
  description = "EKS cluster information for all regions"
  value = {
    us_east_1 = {
      cluster_name     = module.eks_us_east_1.cluster_name
      cluster_endpoint = module.eks_us_east_1.cluster_endpoint
      cluster_arn      = module.eks_us_east_1.cluster_arn
      vpc_id          = module.eks_us_east_1.vpc_id
    }
    eu_west_1 = {
      cluster_name     = module.eks_eu_west_1.cluster_name
      cluster_endpoint = module.eks_eu_west_1.cluster_endpoint
      cluster_arn      = module.eks_eu_west_1.cluster_arn
      vpc_id          = module.eks_eu_west_1.vpc_id
    }
    ap_southeast_1 = {
      cluster_name     = module.eks_ap_southeast_1.cluster_name
      cluster_endpoint = module.eks_ap_southeast_1.cluster_endpoint
      cluster_arn      = module.eks_ap_southeast_1.cluster_arn
      vpc_id          = module.eks_ap_southeast_1.vpc_id
    }
    sa_east_1 = {
      cluster_name     = module.eks_sa_east_1.cluster_name
      cluster_endpoint = module.eks_sa_east_1.cluster_endpoint
      cluster_arn      = module.eks_sa_east_1.cluster_arn
      vpc_id          = module.eks_sa_east_1.vpc_id
    }
  }
}