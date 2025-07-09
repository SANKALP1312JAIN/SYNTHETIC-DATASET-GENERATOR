class FeedbackAgent:
    def __call__(self, state):
        # Dummy feedback and parameter suggestion
        feedback = {
            'quality_score': state.get('quality_metrics', {}),
            'recommendations': ['Increase epochs', 'Adjust learning rate'],
            'suggested_parameters': {'epochs': 300, 'learning_rate': 0.001}
        }
        state['feedback'] = feedback
        state['log'] = state.get('log', []) + ["Feedback processed."]
        return state
