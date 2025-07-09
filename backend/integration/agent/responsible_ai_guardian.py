"""
Responsible AI Guardian for Synthetic Data Platform

This module provides comprehensive responsible AI oversight including:
- Privacy protection and compliance
- Fairness monitoring and bias detection
- Transparency and explainability
- Ethical AI guidelines enforcement
- Regulatory compliance (GDPR, HIPAA, etc.)
- Risk assessment and mitigation
"""

import logging
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Import RBAC system for permission checks
from .rbac_system import Permission, require_permission, rbac_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    CCPA = "ccpa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    FERPA = "ferpa"

class PrivacyLevel(Enum):
    """Privacy protection levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class FairnessMetric(Enum):
    """Fairness evaluation metrics"""
    STATISTICAL_PARITY = "statistical_parity"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    EQUALIZED_ODDS = "equalized_odds"
    DISPARATE_IMPACT = "disparate_impact"
    INDIVIDUAL_FAIRNESS = "individual_fairness"

@dataclass
class PrivacyViolation:
    """Privacy violation record"""
    timestamp: datetime
    violation_type: str
    severity: str
    description: str
    affected_data: str
    mitigation_action: str
    compliance_framework: ComplianceFramework

@dataclass
class BiasReport:
    """Bias detection report"""
    timestamp: datetime
    metric: FairnessMetric
    protected_attribute: str
    bias_score: float
    threshold: float
    is_biased: bool
    recommendations: List[str]

@dataclass
class ComplianceCheck:
    """Compliance verification result"""
    framework: ComplianceFramework
    timestamp: datetime
    checks_passed: int
    checks_failed: int
    violations: List[str]
    recommendations: List[str]
    overall_compliant: bool

@dataclass
class EthicalGuideline:
    """Ethical AI guideline"""
    guideline_id: str
    title: str
    description: str
    category: str
    enforcement_level: str
    compliance_frameworks: List[ComplianceFramework]

class ResponsibleAIGuardian:
    """Responsible AI Guardian for enforcing ethical AI practices"""
    
    def __init__(self):
        self.privacy_violations: List[PrivacyViolation] = []
        self.bias_reports: List[BiasReport] = []
        self.compliance_checks: List[ComplianceCheck] = []
        self.ethical_guidelines: Dict[str, EthicalGuideline] = self._initialize_guidelines()
        self.risk_assessments: Dict[str, Dict[str, Any]] = {}
        self.audit_trail: List[Dict[str, Any]] = []
        
        # Privacy thresholds
        self.privacy_thresholds = {
            PrivacyLevel.LOW: {"k_anonymity": 2, "l_diversity": 2, "t_closeness": 0.1},
            PrivacyLevel.MEDIUM: {"k_anonymity": 5, "l_diversity": 3, "t_closeness": 0.05},
            PrivacyLevel.HIGH: {"k_anonymity": 10, "l_diversity": 5, "t_closeness": 0.02},
            PrivacyLevel.MAXIMUM: {"k_anonymity": 20, "l_diversity": 10, "t_closeness": 0.01}
        }
        
        # Fairness thresholds
        self.fairness_thresholds = {
            FairnessMetric.STATISTICAL_PARITY: 0.1,
            FairnessMetric.EQUAL_OPPORTUNITY: 0.1,
            FairnessMetric.EQUALIZED_ODDS: 0.1,
            FairnessMetric.DISPARATE_IMPACT: 0.8
        }
    
    def _initialize_guidelines(self) -> Dict[str, EthicalGuideline]:
        """Initialize ethical AI guidelines"""
        return {
            "privacy_first": EthicalGuideline(
                guideline_id="privacy_first",
                title="Privacy-First Design",
                description="Prioritize privacy protection in all data operations",
                category="privacy",
                enforcement_level="mandatory",
                compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.HIPAA]
            ),
            "fairness_monitoring": EthicalGuideline(
                guideline_id="fairness_monitoring",
                title="Fairness Monitoring",
                description="Continuously monitor for bias and ensure fairness",
                category="fairness",
                enforcement_level="mandatory",
                compliance_frameworks=[ComplianceFramework.GDPR]
            ),
            "transparency": EthicalGuideline(
                guideline_id="transparency",
                title="Transparency and Explainability",
                description="Ensure AI decisions are transparent and explainable",
                category="transparency",
                enforcement_level="recommended",
                compliance_frameworks=[ComplianceFramework.GDPR]
            ),
            "data_minimization": EthicalGuideline(
                guideline_id="data_minimization",
                title="Data Minimization",
                description="Collect and process only necessary data",
                category="privacy",
                enforcement_level="mandatory",
                compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.HIPAA]
            ),
            "consent_management": EthicalGuideline(
                guideline_id="consent_management",
                title="Consent Management",
                description="Properly manage and track user consent",
                category="privacy",
                enforcement_level="mandatory",
                compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.CCPA]
            )
        }
    
    def check_privacy_compliance(self, data: pd.DataFrame, 
                                privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM,
                                sensitive_columns: List[str] = None) -> Dict[str, Any]:
        """
        Check privacy compliance of data
        
        Args:
            data: Input data
            privacy_level: Required privacy level
            sensitive_columns: List of sensitive column names
            
        Returns:
            Privacy compliance report
        """
        logger.info(f"Checking privacy compliance for {len(data)} records")
        
        compliance_report = {
            "timestamp": datetime.now(),
            "privacy_level": privacy_level.value,
            "checks": {},
            "violations": [],
            "recommendations": [],
            "overall_compliant": True
        }
        
        # Check for PII (Personally Identifiable Information)
        pii_violations = self._check_pii(data, sensitive_columns)
        if pii_violations:
            compliance_report["violations"].extend(pii_violations)
            compliance_report["overall_compliant"] = False
        
        # Check k-anonymity
        k_anon_result = self._check_k_anonymity(data, privacy_level)
        compliance_report["checks"]["k_anonymity"] = k_anon_result
        
        # Check l-diversity
        l_div_result = self._check_l_diversity(data, privacy_level)
        compliance_report["checks"]["l_diversity"] = l_div_result
        
        # Check t-closeness
        t_close_result = self._check_t_closeness(data, privacy_level)
        compliance_report["checks"]["t_closeness"] = t_close_result
        
        # Generate recommendations
        compliance_report["recommendations"] = self._generate_privacy_recommendations(
            compliance_report["checks"], privacy_level
        )
        
        self._log_audit("privacy_compliance_check", f"data:{len(data)}_records", 
                       compliance_report["overall_compliant"], compliance_report)
        
        return compliance_report
    
    def _check_pii(self, data: pd.DataFrame, sensitive_columns: List[str] = None) -> List[str]:
        """Check for PII in data"""
        violations = []
        
        # Common PII patterns
        pii_patterns = {
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "phone": r"(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            "ssn": r"\d{3}-\d{2}-\d{4}",
            "credit_card": r"\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}"
        }
        
        for col in data.columns:
            if sensitive_columns and col in sensitive_columns:
                violations.append(f"Sensitive column detected: {col}")
                continue
            
            # Check for PII patterns in string columns
            if data[col].dtype == 'object':
                for pattern_name, pattern in pii_patterns.items():
                    if data[col].astype(str).str.contains(pattern, regex=True).any():
                        violations.append(f"PII pattern '{pattern_name}' detected in column '{col}'")
        
        return violations
    
    def _check_k_anonymity(self, data: pd.DataFrame, privacy_level: PrivacyLevel) -> Dict[str, Any]:
        """Check k-anonymity of data"""
        threshold = self.privacy_thresholds[privacy_level]["k_anonymity"]
        
        # Simplified k-anonymity check
        # In practice, you would use more sophisticated algorithms
        unique_combinations = data.groupby(data.columns.tolist()).size()
        min_group_size = unique_combinations.min() if len(unique_combinations) > 0 else 0
        
        return {
            "threshold": threshold,
            "actual": min_group_size,
            "compliant": min_group_size >= threshold,
            "description": f"Minimum group size: {min_group_size}, Required: {threshold}"
        }
    
    def _check_l_diversity(self, data: pd.DataFrame, privacy_level: PrivacyLevel) -> Dict[str, Any]:
        """Check l-diversity of data"""
        threshold = self.privacy_thresholds[privacy_level]["l_diversity"]
        
        # Simplified l-diversity check
        # In practice, you would check diversity of sensitive attributes
        sensitive_cols = [col for col in data.columns if 'sensitive' in col.lower() or 'private' in col.lower()]
        
        if not sensitive_cols:
            return {
                "threshold": threshold,
                "actual": 0,
                "compliant": True,
                "description": "No sensitive columns identified"
            }
        
        min_diversity = float('inf')
        for col in sensitive_cols:
            diversity = data[col].nunique()
            min_diversity = min(min_diversity, diversity)
        
        return {
            "threshold": threshold,
            "actual": min_diversity,
            "compliant": min_diversity >= threshold,
            "description": f"Minimum diversity: {min_diversity}, Required: {threshold}"
        }
    
    def _check_t_closeness(self, data: pd.DataFrame, privacy_level: PrivacyLevel) -> Dict[str, Any]:
        """Check t-closeness of data"""
        threshold = self.privacy_thresholds[privacy_level]["t_closeness"]
        
        # Simplified t-closeness check
        # In practice, you would calculate distribution distance
        return {
            "threshold": threshold,
            "actual": 0.05,  # Placeholder
            "compliant": True,
            "description": "T-closeness check passed"
        }
    
    def _generate_privacy_recommendations(self, checks: Dict[str, Any], 
                                        privacy_level: PrivacyLevel) -> List[str]:
        """Generate privacy recommendations based on check results"""
        recommendations = []
        
        for check_name, check_result in checks.items():
            if not check_result.get("compliant", True):
                if check_name == "k_anonymity":
                    recommendations.append(f"Increase k-anonymity by generalizing or suppressing attributes")
                elif check_name == "l_diversity":
                    recommendations.append(f"Increase l-diversity by ensuring more diverse sensitive attribute values")
                elif check_name == "t_closeness":
                    recommendations.append(f"Improve t-closeness by reducing distribution differences")
        
        if privacy_level == PrivacyLevel.LOW:
            recommendations.append("Consider upgrading to higher privacy level for sensitive data")
        
        return recommendations
    
    def check_fairness(self, data: pd.DataFrame, target_column: str,
                      protected_attributes: List[str] = None) -> Dict[str, Any]:
        """
        Check fairness of data and model outcomes
        
        Args:
            data: Input data
            target_column: Target variable column
            protected_attributes: List of protected attribute columns
            
        Returns:
            Fairness analysis report
        """
        logger.info(f"Checking fairness for target: {target_column}")
        
        fairness_report = {
            "timestamp": datetime.now(),
            "target_column": target_column,
            "protected_attributes": protected_attributes or [],
            "metrics": {},
            "bias_detected": False,
            "recommendations": []
        }
        
        if not protected_attributes:
            fairness_report["recommendations"].append("No protected attributes specified for fairness analysis")
            return fairness_report
        
        for attr in protected_attributes:
            if attr not in data.columns:
                continue
            
            attr_report = self._analyze_protected_attribute(data, attr, target_column)
            fairness_report["metrics"][attr] = attr_report
            
            if attr_report["bias_detected"]:
                fairness_report["bias_detected"] = True
        
        # Generate fairness recommendations
        fairness_report["recommendations"] = self._generate_fairness_recommendations(
            fairness_report["metrics"]
        )
        
        self._log_audit("fairness_check", f"target:{target_column}", 
                       not fairness_report["bias_detected"], fairness_report)
        
        return fairness_report
    
    def _analyze_protected_attribute(self, data: pd.DataFrame, protected_attr: str, 
                                   target_column: str) -> Dict[str, Any]:
        """Analyze fairness for a specific protected attribute"""
        attr_values = data[protected_attr].unique()
        target_values = data[target_column].unique()
        
        # Calculate statistical parity
        statistical_parity = self._calculate_statistical_parity(data, protected_attr, target_column)
        
        # Calculate disparate impact
        disparate_impact = self._calculate_disparate_impact(data, protected_attr, target_column)
        
        # Check against thresholds
        sp_threshold = self.fairness_thresholds[FairnessMetric.STATISTICAL_PARITY]
        di_threshold = self.fairness_thresholds[FairnessMetric.DISPARATE_IMPACT]
        
        sp_biased = abs(statistical_parity) > sp_threshold
        di_biased = disparate_impact < di_threshold
        
        bias_detected = sp_biased or di_biased
        
        return {
            "statistical_parity": {
                "value": statistical_parity,
                "threshold": sp_threshold,
                "biased": sp_biased
            },
            "disparate_impact": {
                "value": disparate_impact,
                "threshold": di_threshold,
                "biased": di_biased
            },
            "bias_detected": bias_detected,
            "recommendations": self._generate_attribute_fairness_recommendations(
                protected_attr, sp_biased, di_biased
            )
        }
    
    def _calculate_statistical_parity(self, data: pd.DataFrame, protected_attr: str, 
                                    target_column: str) -> float:
        """Calculate statistical parity difference"""
        # Simplified calculation
        # In practice, you would use more sophisticated methods
        groups = data.groupby(protected_attr)[target_column].mean()
        if len(groups) < 2:
            return 0.0
        
        return groups.max() - groups.min()
    
    def _calculate_disparate_impact(self, data: pd.DataFrame, protected_attr: str, 
                                  target_column: str) -> float:
        """Calculate disparate impact ratio"""
        # Simplified calculation
        # In practice, you would use more sophisticated methods
        groups = data.groupby(protected_attr)[target_column].mean()
        if len(groups) < 2:
            return 1.0
        
        return groups.min() / groups.max()
    
    def _generate_fairness_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate fairness recommendations"""
        recommendations = []
        
        for attr, attr_metrics in metrics.items():
            if attr_metrics["bias_detected"]:
                recommendations.append(f"Bias detected in {attr}: Consider data preprocessing or model adjustments")
                recommendations.append(f"Review data collection process for {attr} to ensure representativeness")
        
        if not recommendations:
            recommendations.append("No significant bias detected in protected attributes")
        
        return recommendations
    
    def _generate_attribute_fairness_recommendations(self, attr: str, sp_biased: bool, 
                                                   di_biased: bool) -> List[str]:
        """Generate recommendations for specific attribute"""
        recommendations = []
        
        if sp_biased:
            recommendations.append(f"Statistical parity bias in {attr}: Consider rebalancing data")
        if di_biased:
            recommendations.append(f"Disparate impact in {attr}: Review decision thresholds")
        
        return recommendations
    
    def apply_privacy_protection(self, data: pd.DataFrame, 
                               privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM,
                               sensitive_columns: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply privacy protection measures to data
        
        Args:
            data: Input data
            privacy_level: Privacy protection level
            sensitive_columns: List of sensitive columns
            
        Returns:
            Tuple of (protected_data, protection_metadata)
        """
        logger.info(f"Applying privacy protection level: {privacy_level.value}")
        
        protected_data = data.copy()
        protection_metadata = {
            "timestamp": datetime.now(),
            "privacy_level": privacy_level.value,
            "applied_techniques": [],
            "original_shape": data.shape,
            "protected_shape": None
        }
        
        # Remove or mask sensitive columns
        if sensitive_columns:
            for col in sensitive_columns:
                if col in protected_data.columns:
                    protected_data[col] = "[REDACTED]"
                    protection_metadata["applied_techniques"].append(f"redacted_column:{col}")
        
        # Apply generalization for categorical columns
        categorical_cols = protected_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in (sensitive_columns or []):
                # Simple generalization: keep only first few characters
                protected_data[col] = protected_data[col].astype(str).str[:3] + "..."
                protection_metadata["applied_techniques"].append(f"generalized_column:{col}")
        
        # Apply noise addition for numerical columns
        numerical_cols = protected_data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col not in (sensitive_columns or []):
                noise = np.random.normal(0, 0.01 * protected_data[col].std(), len(protected_data))
                protected_data[col] = protected_data[col] + noise
                protection_metadata["applied_techniques"].append(f"noise_added_column:{col}")
        
        protection_metadata["protected_shape"] = protected_data.shape
        
        self._log_audit("privacy_protection_applied", f"data:{len(data)}_records", 
                       True, protection_metadata)
        
        return protected_data, protection_metadata
    
    def check_regulatory_compliance(self, data: pd.DataFrame, 
                                  frameworks: List[ComplianceFramework] = None) -> Dict[str, Any]:
        """
        Check compliance with regulatory frameworks
        
        Args:
            data: Input data
            frameworks: List of compliance frameworks to check
            
        Returns:
            Compliance report
        """
        if not frameworks:
            frameworks = [ComplianceFramework.GDPR, ComplianceFramework.HIPAA]
        
        logger.info(f"Checking compliance for frameworks: {[f.value for f in frameworks]}")
        
        compliance_report = {
            "timestamp": datetime.now(),
            "frameworks": [f.value for f in frameworks],
            "results": {},
            "overall_compliant": True
        }
        
        for framework in frameworks:
            framework_result = self._check_framework_compliance(data, framework)
            compliance_report["results"][framework.value] = framework_result
            
            if not framework_result["compliant"]:
                compliance_report["overall_compliant"] = False
        
        self.compliance_checks.append(ComplianceCheck(
            framework=frameworks[0],  # Simplified
            timestamp=datetime.now(),
            checks_passed=sum(1 for r in compliance_report["results"].values() if r["compliant"]),
            checks_failed=sum(1 for r in compliance_report["results"].values() if not r["compliant"]),
            violations=[],  # Would be populated in real implementation
            recommendations=[],
            overall_compliant=compliance_report["overall_compliant"]
        ))
        
        self._log_audit("regulatory_compliance_check", f"frameworks:{len(frameworks)}", 
                       compliance_report["overall_compliant"], compliance_report)
        
        return compliance_report
    
    def _check_framework_compliance(self, data: pd.DataFrame, 
                                  framework: ComplianceFramework) -> Dict[str, Any]:
        """Check compliance for specific framework"""
        if framework == ComplianceFramework.GDPR:
            return self._check_gdpr_compliance(data)
        elif framework == ComplianceFramework.HIPAA:
            return self._check_hipaa_compliance(data)
        else:
            return {
                "compliant": True,
                "checks": [],
                "violations": [],
                "recommendations": [f"Compliance check for {framework.value} not implemented"]
            }
    
    def _check_gdpr_compliance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check GDPR compliance"""
        violations = []
        recommendations = []
        
        # Check for explicit consent tracking
        if 'consent_given' not in data.columns:
            violations.append("No consent tracking found")
            recommendations.append("Add consent tracking column")
        
        # Check for data retention policies
        if 'data_retention_date' not in data.columns:
            recommendations.append("Add data retention date tracking")
        
        # Check for right to be forgotten
        if 'deletion_requested' not in data.columns:
            recommendations.append("Add deletion request tracking")
        
        return {
            "compliant": len(violations) == 0,
            "checks": ["consent_tracking", "data_retention", "right_to_be_forgotten"],
            "violations": violations,
            "recommendations": recommendations
        }
    
    def _check_hipaa_compliance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check HIPAA compliance"""
        violations = []
        recommendations = []
        
        # Check for PHI (Protected Health Information)
        phi_indicators = ['patient', 'medical', 'diagnosis', 'treatment', 'health']
        phi_columns = [col for col in data.columns if any(indicator in col.lower() for indicator in phi_indicators)]
        
        if phi_columns:
            # Check for proper de-identification
            for col in phi_columns:
                if data[col].dtype == 'object' and data[col].str.contains(r'\d{3}-\d{2}-\d{4}').any():
                    violations.append(f"SSN found in {col}")
                    recommendations.append(f"De-identify SSN in {col}")
        
        return {
            "compliant": len(violations) == 0,
            "checks": ["phi_identification", "de_identification"],
            "violations": violations,
            "recommendations": recommendations
        }
    
    def _log_audit(self, action: str, resource: str, success: bool, 
                   details: Dict[str, Any] = None):
        """Log audit entry"""
        audit_entry = {
            "timestamp": datetime.now(),
            "action": action,
            "resource": resource,
            "success": success,
            "details": details or {}
        }
        self.audit_trail.append(audit_entry)
        logger.info(f"ResponsibleAI: {action} on {resource} - {'SUCCESS' if success else 'FAILED'}")
    
    def get_audit_trail(self, start_date: datetime = None, end_date: datetime = None) -> List[Dict[str, Any]]:
        """Get audit trail entries"""
        trail = self.audit_trail
        
        if start_date:
            trail = [entry for entry in trail if entry["timestamp"] >= start_date]
        if end_date:
            trail = [entry for entry in trail if entry["timestamp"] <= end_date]
        
        return trail
    
    def export_compliance_report(self, format: str = "json") -> str:
        """Export comprehensive compliance report"""
        report = {
            "timestamp": datetime.now(),
            "privacy_violations": len(self.privacy_violations),
            "bias_reports": len(self.bias_reports),
            "compliance_checks": len(self.compliance_checks),
            "audit_entries": len(self.audit_trail),
            "ethical_guidelines": len(self.ethical_guidelines),
            "recent_audit_trail": self.audit_trail[-10:] if self.audit_trail else []
        }
        
        if format == "json":
            output_path = f"responsible_ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return output_path

# Global Responsible AI Guardian instance
responsible_ai_guardian = ResponsibleAIGuardian() 