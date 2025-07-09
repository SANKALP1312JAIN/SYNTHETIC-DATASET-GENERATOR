"""
GAN implementations for synthetic data generation
"""

from .base_gan import BaseGAN
from .dp_ctgan import DPCTGAN
from .sdv_gan import SDVGAN
from .smote_ctgan import SMOTECTGAN
from .doppelganger import DoppelGANger
from .med_gan import MedGAN
from .default_gan import DefaultGAN
from .text_gan import TextGAN

# GAN Factory
class GANFactory:
    """Factory class to create appropriate GAN instances"""
    
    @staticmethod
    def create_gan(gan_type: str, config: dict = None) -> BaseGAN:
        """Create a GAN instance based on type"""
        gan_map = {
            'DP-CTGAN': DPCTGAN,
            'SDVGAN': SDVGAN,
            'SMOTE-CTGAN': SMOTECTGAN,
            'DoppelGANger': DoppelGANger,
            'MedGAN': MedGAN,
            'DefaultGAN': DefaultGAN,
            'TextGAN': TextGAN,
        }
        
        gan_class = gan_map.get(gan_type, DefaultGAN)
        return gan_class(config or {})

__all__ = [
    'BaseGAN',
    'DPCTGAN', 
    'SDVGAN',
    'SMOTECTGAN',
    'DoppelGANger',
    'MedGAN',
    'DefaultGAN',
    'GANFactory'
] 