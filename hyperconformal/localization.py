"""
Enhanced internationalization and localization system for HyperConformal
Comprehensive support for 6 languages with cultural adaptations
"""

import os
import yaml
import json
import locale
import datetime
from typing import Dict, Optional, Any, List
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

class SupportedLanguage(Enum):
    """Supported languages with their codes"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"

class ComplianceRegion(Enum):
    """Compliance regions for localized legal messaging"""
    GDPR = "gdpr"      # European Union
    CCPA = "ccpa"      # California, USA
    PDPA = "pdpa"      # Singapore, Asia-Pacific
    LGPD = "lgpd"      # Brazil, South America

@dataclass
class LanguageConfig:
    """Configuration for a specific language"""
    code: str
    name: str
    locale: str
    direction: str
    currency: str
    date_format: str
    number_format: str

@dataclass
class CulturalSettings:
    """Cultural adaptations for different regions"""
    business_hours: str
    support_phone: str
    privacy_policy_url: str
    terms_url: str

class LocalizationManager:
    """Advanced localization manager with cultural adaptation"""
    
    def __init__(self, default_language: str = "en", translations_file: Optional[str] = None):
        self.current_language = default_language
        self.fallback_language = "en"
        self.translations = {}
        self.language_configs = {}
        self.cultural_settings = {}
        self.compliance_messages = {}
        
        # Load translations
        if translations_file is None:
            translations_file = Path(__file__).parent.parent / "infrastructure" / "localization" / "translations.yaml"
        
        self._load_translations(translations_file)
        self._setup_language_configs()
    
    def _load_translations(self, translations_file: str) -> None:
        """Load translations from YAML file"""
        try:
            with open(translations_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
            self.translations = data.get('messages', {})
            self.cultural_settings = data.get('cultural_settings', {})
            self.compliance_messages = data.get('compliance_messages', {})
            
            # Setup language configs
            language_data = data.get('languages', {})
            for lang_code, config in language_data.items():
                self.language_configs[lang_code] = LanguageConfig(
                    code=lang_code,
                    name=config['name'],
                    locale=config['locale'],
                    direction=config['direction'],
                    currency=config['currency'],
                    date_format=config['date_format'],
                    number_format=config['number_format']
                )
                
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Warning: Could not load translations file: {e}")
            self._setup_fallback_translations()
    
    def _setup_fallback_translations(self) -> None:
        """Setup basic fallback translations if file loading fails"""
        self.translations = {
            "en": {
                "system_ready": "System ready",
                "model_training": "Training model...",
                "prediction_complete": "Prediction complete",
                "error_occurred": "An error occurred"
            }
        }
    
    def _setup_language_configs(self) -> None:
        """Setup language configurations if not loaded from file"""
        if not self.language_configs:
            self.language_configs = {
                "en": LanguageConfig("en", "English", "en-US", "ltr", "USD", "MM/DD/YYYY", "1,234.56"),
                "es": LanguageConfig("es", "Español", "es-ES", "ltr", "EUR", "DD/MM/YYYY", "1.234,56"),
                "fr": LanguageConfig("fr", "Français", "fr-FR", "ltr", "EUR", "DD/MM/YYYY", "1 234,56"),
                "de": LanguageConfig("de", "Deutsch", "de-DE", "ltr", "EUR", "DD.MM.YYYY", "1.234,56"),
                "ja": LanguageConfig("ja", "日本語", "ja-JP", "ltr", "JPY", "YYYY/MM/DD", "1,234"),
                "zh": LanguageConfig("zh", "中文", "zh-CN", "ltr", "CNY", "YYYY-MM-DD", "1,234.56")
            }
    
    def set_language(self, language: str) -> bool:
        """Set the current language"""
        if language in self.translations:
            self.current_language = language
            self._update_system_locale(language)
            return True
        return False
    
    def _update_system_locale(self, language: str) -> None:
        """Update system locale based on language"""
        try:
            if language in self.language_configs:
                locale_str = self.language_configs[language].locale
                locale.setlocale(locale.LC_ALL, locale_str)
        except locale.Error:
            # Fallback to default locale if specific locale not available
            pass
    
    def get_text(self, key: str, **kwargs) -> str:
        """Get localized text with parameter substitution"""
        # Try current language first
        text = None
        if self.current_language in self.translations:
            text = self.translations[self.current_language].get(key)
        
        # Fallback to default language
        if text is None and self.fallback_language in self.translations:
            text = self.translations[self.fallback_language].get(key)
        
        # If still not found, return the key itself
        if text is None:
            text = key
        
        # Perform parameter substitution
        try:
            return text.format(**kwargs)
        except (KeyError, ValueError):
            return text
    
    def get_compliance_message(self, compliance_region: str, message_key: str, language: Optional[str] = None) -> str:
        """Get compliance-specific localized message"""
        lang = language or self.current_language
        
        try:
            return self.compliance_messages[compliance_region][lang][message_key]
        except KeyError:
            # Fallback to English
            try:
                return self.compliance_messages[compliance_region]["en"][message_key]
            except KeyError:
                return message_key
    
    def format_number(self, number: float, language: Optional[str] = None) -> str:
        """Format number according to locale"""
        lang = language or self.current_language
        if lang not in self.language_configs:
            lang = self.fallback_language
        
        config = self.language_configs[lang]
        
        # Simple formatting based on language
        if lang == "fr":
            return f"{number:,.2f}".replace(",", " ").replace(".", ",")
        elif lang in ["de", "es"]:
            return f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        elif lang == "ja" and number == int(number):
            return f"{int(number):,}"
        else:
            return f"{number:,.2f}"
    
    def format_currency(self, amount: float, language: Optional[str] = None) -> str:
        """Format currency according to locale"""
        lang = language or self.current_language
        if lang not in self.language_configs:
            lang = self.fallback_language
        
        config = self.language_configs[lang]
        formatted_number = self.format_number(amount, lang)
        
        currency_symbols = {
            "USD": "$", "EUR": "€", "JPY": "¥", "CNY": "¥"
        }
        
        symbol = currency_symbols.get(config.currency, config.currency)
        
        # Currency placement varies by locale
        if lang == "en":
            return f"{symbol}{formatted_number}"
        elif lang in ["fr", "de"]:
            return f"{formatted_number} {symbol}"
        elif lang == "ja":
            return f"{symbol}{formatted_number}"
        elif lang == "zh":
            return f"{symbol}{formatted_number}"
        else:
            return f"{symbol}{formatted_number}"
    
    def format_date(self, date: datetime.datetime, language: Optional[str] = None) -> str:
        """Format date according to locale"""
        lang = language or self.current_language
        if lang not in self.language_configs:
            lang = self.fallback_language
        
        config = self.language_configs[lang]
        format_str = config.date_format
        
        # Convert format string to strftime format
        format_str = format_str.replace("YYYY", "%Y").replace("MM", "%m").replace("DD", "%d")
        
        return date.strftime(format_str)
    
    def get_cultural_setting(self, setting_key: str, language: Optional[str] = None) -> str:
        """Get cultural setting for specific language"""
        lang = language or self.current_language
        
        try:
            return self.cultural_settings[lang][setting_key]
        except KeyError:
            # Fallback to English
            try:
                return self.cultural_settings["en"][setting_key]
            except KeyError:
                return ""
    
    def get_language_config(self, language: Optional[str] = None) -> Optional[LanguageConfig]:
        """Get language configuration"""
        lang = language or self.current_language
        return self.language_configs.get(lang)
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages"""
        return [
            {"code": config.code, "name": config.name}
            for config in self.language_configs.values()
        ]
    
    def detect_browser_language(self, accept_language_header: str) -> str:
        """Detect preferred language from browser Accept-Language header"""
        if not accept_language_header:
            return self.fallback_language
        
        # Parse Accept-Language header
        languages = []
        for lang in accept_language_header.split(','):
            parts = lang.strip().split(';')
            code = parts[0].strip().lower()
            quality = 1.0
            
            if len(parts) > 1 and parts[1].strip().startswith('q='):
                try:
                    quality = float(parts[1].strip()[2:])
                except ValueError:
                    quality = 1.0
            
            # Extract primary language code
            primary_code = code.split('-')[0]
            languages.append((primary_code, quality))
        
        # Sort by quality and find first supported language
        languages.sort(key=lambda x: x[1], reverse=True)
        
        for lang_code, _ in languages:
            if lang_code in self.translations:
                return lang_code
        
        return self.fallback_language
    
    def get_rtl_languages(self) -> List[str]:
        """Get list of right-to-left languages"""
        return [
            code for code, config in self.language_configs.items()
            if config.direction == "rtl"
        ]
    
    def validate_translation_completeness(self) -> Dict[str, float]:
        """Validate translation completeness for all languages"""
        base_keys = set(self.translations.get(self.fallback_language, {}).keys())
        completeness = {}
        
        for lang_code, translations in self.translations.items():
            if not base_keys:
                completeness[lang_code] = 0.0
            else:
                translated_keys = set(translations.keys())
                completeness[lang_code] = len(translated_keys & base_keys) / len(base_keys) * 100
        
        return completeness

# Global localization manager instance
_localization_manager = LocalizationManager()

def set_language(language: str) -> bool:
    """Set global language"""
    return _localization_manager.set_language(language)

def get_text(key: str, **kwargs) -> str:
    """Get localized text (shorthand function)"""
    return _localization_manager.get_text(key, **kwargs)

def _(key: str, **kwargs) -> str:
    """Even shorter translation function"""
    return _localization_manager.get_text(key, **kwargs)

def format_number(number: float, language: Optional[str] = None) -> str:
    """Format number according to locale"""
    return _localization_manager.format_number(number, language)

def format_currency(amount: float, language: Optional[str] = None) -> str:
    """Format currency according to locale"""
    return _localization_manager.format_currency(amount, language)

def format_date(date: datetime.datetime, language: Optional[str] = None) -> str:
    """Format date according to locale"""
    return _localization_manager.format_date(date, language)

def get_compliance_message(compliance_region: str, message_key: str, language: Optional[str] = None) -> str:
    """Get compliance-specific message"""
    return _localization_manager.get_compliance_message(compliance_region, message_key, language)

def get_cultural_setting(setting_key: str, language: Optional[str] = None) -> str:
    """Get cultural setting"""
    return _localization_manager.get_cultural_setting(setting_key, language)

def detect_browser_language(accept_language_header: str) -> str:
    """Detect browser language"""
    return _localization_manager.detect_browser_language(accept_language_header)

def get_supported_languages() -> List[Dict[str, str]]:
    """Get supported languages"""
    return _localization_manager.get_supported_languages()

# Export the global manager for advanced usage
localization_manager = _localization_manager