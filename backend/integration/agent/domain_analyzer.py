class DomainAnalyzer:
    def __call__(self, state):
        # Detect domain from config
        config = state.get('config', {})
        domain = config.get('domain', 'unknown')
        
        # Domain-specific configurations
        domain_configs = {
            'healthcare': {
                'privacy_requirements': 'high',
                'data_sensitivity': 'critical',
                'compliance': ['HIPAA', 'GDPR'],
                'recommended_gan_config': {
                    'epsilon': 0.5,  # Lower epsilon for higher privacy
                    'delta': 1e-6,
                    'latent_dim': 128,
                    'hidden_dim': 512
                }
            },
            'finance': {
                'privacy_requirements': 'medium',
                'data_sensitivity': 'high',
                'compliance': ['SOX', 'GDPR', 'PCI-DSS'],
                'recommended_gan_config': {
                    'epsilon': 1.0,
                    'delta': 1e-5,
                    'latent_dim': 100,
                    'hidden_dim': 256
                }
            },
            'retail': {
                'privacy_requirements': 'medium',
                'data_sensitivity': 'medium',
                'compliance': ['GDPR'],
                'recommended_gan_config': {
                    'epsilon': 2.0,
                    'delta': 1e-5,
                    'latent_dim': 64,
                    'hidden_dim': 128
                }
            },
            'education': {
                'privacy_requirements': 'high',
                'data_sensitivity': 'high',
                'compliance': ['FERPA', 'COPPA'],
                'recommended_gan_config': {
                    'epsilon': 0.8,
                    'delta': 1e-6,
                    'latent_dim': 100,
                    'hidden_dim': 256
                }
            },
            'telecommunications': {
                'privacy_requirements': 'high',
                'data_sensitivity': 'high',
                'compliance': ['GDPR', 'CCPA'],
                'recommended_gan_config': {
                    'epsilon': 0.6,
                    'delta': 1e-6,
                    'latent_dim': 128,
                    'hidden_dim': 512
                }
            }
        }
        
        # Get domain-specific configuration
        domain_config = domain_configs.get(domain, {
            'privacy_requirements': 'medium',
            'data_sensitivity': 'medium',
            'compliance': ['GDPR'],
            'recommended_gan_config': {
                'epsilon': 1.0,
                'delta': 1e-5,
                'latent_dim': 100,
                'hidden_dim': 256
            }
        })
        
        # Update state with domain information
        state['domain'] = domain
        state['domain_config'] = domain_config
        
        # Merge domain-specific GAN config with user config
        user_gan_config = config.get('gan_config', {})
        merged_gan_config = {**domain_config['recommended_gan_config'], **user_gan_config}
        state['config']['gan_config'] = merged_gan_config
        
        # Update privacy level based on domain if not explicitly set
        if 'privacy_level' not in config:
            state['config']['privacy_level'] = domain_config['privacy_requirements']
        
        state['log'] = state.get('log', []) + [
            f"Domain detected: {domain}",
            f"Privacy requirements: {domain_config['privacy_requirements']}",
            f"Data sensitivity: {domain_config['data_sensitivity']}",
            f"Compliance requirements: {', '.join(domain_config['compliance'])}"
        ]
        
        return state
