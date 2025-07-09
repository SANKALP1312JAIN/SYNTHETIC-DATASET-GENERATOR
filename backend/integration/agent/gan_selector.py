from .gans import GANFactory
import pandas as pd

class GANSelector:
    def __call__(self, state):
        domain = state.get('domain')
        data_type = state.get('data_type')
        privacy_level = state.get('config', {}).get('privacy_level', 'medium')
        selected_gan = None
        
        # Selection matrix
        if data_type == 'tabular':
            if domain == 'healthcare' and privacy_level == 'high':
                selected_gan = 'DP-CTGAN'
            elif domain == 'finance':
                selected_gan = 'SDVGAN'
            elif state.get('profile', {}).get('class_imbalance', False):
                selected_gan = 'SMOTE-CTGAN'
            else:
                selected_gan = 'SDVGAN'
        elif data_type == 'time_series':
            selected_gan = 'DoppelGANger'
        elif data_type == 'image' and domain == 'healthcare':
            selected_gan = 'MedGAN'
        else:
            selected_gan = 'DefaultGAN'
        
        # Create GAN instance and generate synthetic data
        gan_config = state.get('config', {}).get('gan_config', {})
        gan = GANFactory.create_gan(selected_gan, gan_config)
        
        # Generate synthetic data
        original_data = state.get('data')
        num_samples = state.get('config', {}).get('num_samples', len(original_data))
        
        try:
            synthetic_data = gan.generate(original_data, num_samples)
            state['synthetic_data'] = synthetic_data
            state['selected_gan'] = selected_gan
            state['log'] = state.get('log', []) + [f"Selected GAN: {selected_gan}", f"Generated {len(synthetic_data)} synthetic samples"]
        except Exception as e:
            state['log'] = state.get('log', []) + [f"GAN generation failed: {str(e)}"]
            # Fallback to simple resampling
            if isinstance(original_data, pd.DataFrame):
                synthetic_data = original_data.sample(n=num_samples, replace=True).copy()
                state['synthetic_data'] = synthetic_data
                state['log'] = state.get('log', []) + ["Used fallback resampling method"]
        
        return state
