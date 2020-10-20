from farms_network.network_generator import NetworkGenerator
from scipy.integrate import ode
from .networkx_model import NetworkXModel


class NeuralSystem(NetworkXModel):
    """Neural System.
    """

    def __init__(self, config_path, container):
        """ Initialize neural system. """
        super(NeuralSystem, self).__init__()
        self.container = container
        #: Add name-space for neural system data
        neural_table = self.container.add_namespace('neural')
        self.config_path = config_path
        self.integrator = None
        self.read_graph(config_path)
        #: Create network
        self.network = NetworkGenerator(self.graph, neural_table)

    def setup_integrator(self, x0=None, integrator='dopri5', atol=10,
                         rtol=10, max_step=1, method='adams'):
        """Setup system."""
        self.integrator = ode(self.network.ode).set_integrator(
            integrator,
            method=method)

        if not x0:
            self.integrator.set_initial_value(
                self.container.neural.states.values, 0.0)
        else:
            self.integrator.set_initial_value(x0, 0.0)

    def step(self, dt=1, update=True):
        """Step ode system. """
        self.integrator.set_initial_value(self.integrator.y,
                                          self.integrator.t)
        self.integrator.integrate(self.integrator.t+dt)
        #: Update the logs
        # if update:
        #     #: TO-DO
        #     self.container.neural.update_log()
