"""
Internationalization support for HyperConformal
Multi-language support for global deployment
"""
import json
from typing import Dict, Optional

# Global-first language support
TRANSLATIONS = {
    'en': {
        'model_training': 'Training HDC model...',
        'prediction_complete': 'Prediction complete',
        'coverage_guarantee': 'Coverage guarantee: {coverage}%',
        'invalid_input': 'Invalid input data',
        'model_ready': 'Model ready for inference',
        'calibration_complete': 'Conformal calibration complete',
    },
    'es': {
        'model_training': 'Entrenando modelo HDC...',
        'prediction_complete': 'Predicción completa',
        'coverage_guarantee': 'Garantía de cobertura: {coverage}%',
        'invalid_input': 'Datos de entrada inválidos',
        'model_ready': 'Modelo listo para inferencia',
        'calibration_complete': 'Calibración conformal completa',
    },
    'fr': {
        'model_training': 'Entraînement du modèle HDC...',
        'prediction_complete': 'Prédiction terminée',
        'coverage_guarantee': 'Garantie de couverture: {coverage}%',
        'invalid_input': 'Données d\'entrée invalides',
        'model_ready': 'Modèle prêt pour l\'inférence',
        'calibration_complete': 'Calibration conforme terminée',
    },
    'de': {
        'model_training': 'HDC-Modell wird trainiert...',
        'prediction_complete': 'Vorhersage abgeschlossen',
        'coverage_guarantee': 'Abdeckungsgarantie: {coverage}%',
        'invalid_input': 'Ungültige Eingabedaten',
        'model_ready': 'Modell bereit für Inferenz',
        'calibration_complete': 'Konforme Kalibrierung abgeschlossen',
    },
    'ja': {
        'model_training': 'HDCモデルを訓練中...',
        'prediction_complete': '予測完了',
        'coverage_guarantee': 'カバレッジ保証: {coverage}%',
        'invalid_input': '無効な入力データ',
        'model_ready': '推論準備完了',
        'calibration_complete': '適合的較正完了',
    },
    'zh': {
        'model_training': '正在训练HDC模型...',
        'prediction_complete': '预测完成',
        'coverage_guarantee': '覆盖保证: {coverage}%',
        'invalid_input': '无效输入数据',
        'model_ready': '模型推理就绪',
        'calibration_complete': '保形校准完成',
    }
}

class I18nManager:
    """Global internationalization manager"""
    
    def __init__(self, language: str = 'en'):
        self.language = language
        self.fallback_language = 'en'
    
    def get_text(self, key: str, **kwargs) -> str:
        """Get localized text with parameter substitution"""
        try:
            text = TRANSLATIONS[self.language].get(
                key, 
                TRANSLATIONS[self.fallback_language].get(key, key)
            )
            return text.format(**kwargs)
        except (KeyError, ValueError):
            return key
    
    def set_language(self, language: str) -> None:
        """Set active language"""
        if language in TRANSLATIONS:
            self.language = language

# Global instance
_i18n = I18nManager()

def _(key: str, **kwargs) -> str:
    """Global translation function"""
    return _i18n.get_text(key, **kwargs)

def set_language(language: str) -> None:
    """Set global language"""
    _i18n.set_language(language)