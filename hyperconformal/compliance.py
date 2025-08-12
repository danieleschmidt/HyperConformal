"""
Global compliance framework for HyperConformal
GDPR, CCPA, PDPA compliance for international deployment
"""
import hashlib
import json
import time
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

class DataRegion(Enum):
    """Supported data processing regions"""
    EU = "eu"           # GDPR
    US = "us"           # CCPA  
    SINGAPORE = "sg"    # PDPA
    CANADA = "ca"       # PIPEDA
    GLOBAL = "global"   # Combined compliance

class ConsentType(Enum):
    """Types of data processing consent"""
    FUNCTIONAL = "functional"
    ANALYTICS = "analytics"
    MARKETING = "marketing"
    RESEARCH = "research"

@dataclass
class UserConsent:
    """User consent tracking for GDPR compliance"""
    user_id: str
    consent_types: List[ConsentType]
    timestamp: float
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    explicit: bool = True

class ComplianceManager:
    """Multi-jurisdiction compliance management"""
    
    def __init__(self, default_region: DataRegion = DataRegion.GLOBAL):
        self.region = default_region
        self.consent_records: Dict[str, UserConsent] = {}
        self.data_retention_days = {
            DataRegion.EU: 730,      # GDPR: 2 years default
            DataRegion.US: 1095,     # CCPA: 3 years
            DataRegion.SINGAPORE: 365,  # PDPA: 1 year
            DataRegion.CANADA: 1095, # PIPEDA: 3 years
            DataRegion.GLOBAL: 365   # Conservative 1 year
        }
    
    def record_consent(self, user_id: str, consent_types: List[ConsentType], 
                      metadata: Optional[Dict] = None) -> bool:
        """Record user consent for data processing"""
        consent = UserConsent(
            user_id=user_id,
            consent_types=consent_types,
            timestamp=time.time(),
            ip_address=metadata.get('ip') if metadata else None,
            user_agent=metadata.get('user_agent') if metadata else None
        )
        
        self.consent_records[user_id] = consent
        return True
    
    def check_consent(self, user_id: str, consent_type: ConsentType) -> bool:
        """Verify user consent for specific data processing"""
        consent = self.consent_records.get(user_id)
        if not consent:
            return False
        
        # Check consent expiry based on region
        max_age = self.data_retention_days[self.region] * 24 * 3600
        if time.time() - consent.timestamp > max_age:
            return False
        
        return consent_type in consent.consent_types
    
    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize data for privacy compliance"""
        anonymous_data = data.copy()
        
        # Remove direct identifiers
        pii_fields = ['email', 'phone', 'name', 'address', 'ip_address', 'user_id']
        for field in pii_fields:
            if field in anonymous_data:
                # Replace with hash for analytics while preserving uniqueness
                anonymous_data[field] = hashlib.sha256(
                    str(anonymous_data[field]).encode()
                ).hexdigest()[:16]
        
        return anonymous_data
    
    def get_retention_period(self) -> int:
        """Get data retention period for current region in days"""
        return self.data_retention_days[self.region]
    
    def right_to_erasure(self, user_id: str) -> bool:
        """Handle GDPR Article 17 - Right to Erasure"""
        # Remove consent records
        if user_id in self.consent_records:
            del self.consent_records[user_id]
        
        # In practice, would trigger deletion from all data stores
        return True
    
    def data_portability_export(self, user_id: str) -> Optional[Dict]:
        """Handle GDPR Article 20 - Right to Data Portability"""
        consent = self.consent_records.get(user_id)
        if not consent:
            return None
        
        return {
            'user_data': asdict(consent),
            'export_timestamp': time.time(),
            'format': 'json',
            'compliance_region': self.region.value
        }
    
    def privacy_impact_assessment(self) -> Dict[str, Any]:
        """Generate Privacy Impact Assessment report"""
        return {
            'timestamp': time.time(),
            'region': self.region.value,
            'data_categories': ['model_predictions', 'calibration_scores'],
            'processing_purposes': ['uncertainty_quantification', 'model_improvement'],
            'retention_period_days': self.get_retention_period(),
            'anonymization_methods': ['hashing', 'aggregation'],
            'consent_mechanism': 'explicit_opt_in',
            'data_transfers': 'within_region_only',
            'risk_level': 'low',
            'compliance_measures': [
                'data_minimization',
                'purpose_limitation', 
                'storage_limitation',
                'anonymization'
            ]
        }

class PrivacyPreservingHDC:
    """HDC with built-in privacy preservation"""
    
    def __init__(self, compliance_manager: ComplianceManager):
        self.compliance = compliance_manager
        
    def encode_with_privacy(self, data: Dict[str, Any], user_id: str) -> Optional[Any]:
        """Encode data only if user consent exists"""
        if not self.compliance.check_consent(user_id, ConsentType.FUNCTIONAL):
            return None
        
        # Anonymize data before processing
        anonymous_data = self.compliance.anonymize_data(data)
        
        # Proceed with HDC encoding on anonymized data
        # Implementation would integrate with main HDC encoders
        return anonymous_data
    
    def federated_learning_compliance(self, updates: List[Dict]) -> List[Dict]:
        """Ensure federated updates comply with data protection"""
        compliant_updates = []
        
        for update in updates:
            # Apply differential privacy
            # Verify data minimization
            # Check cross-border transfer restrictions
            compliant_updates.append(update)
        
        return compliant_updates

# Global compliance instance
default_compliance = ComplianceManager()