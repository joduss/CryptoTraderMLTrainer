
class Policy:

    def decide(self, observation):
        raise NotImplementedError()

    def next_episode(self):
        """
        Notify the next episode starts
        """
        raise NotImplementedError()
