"""Factory class for generating the neuron model."""

from farms_network.lif_danner import LIFDanner
from farms_network.lif_danner_nap import LIFDannerNap
from farms_network.lif_daun_interneuron import LIFDaunInterneuron
from farms_network.hh_daun_motorneuron import HHDaunMotorneuron
from farms_network.sensory_neuron import SensoryNeuron
from farms_network.leaky_integrator import LeakyIntegrator
from farms_network.oscillator import Oscillator
from farms_network.morphed_oscillator import MorphedOscillator
from farms_network.fitzhugh_nagumo import FitzhughNagumo
from farms_network.matsuoka_neuron import MatsuokaNeuron
from farms_network.morris_lecar import MorrisLecarNeuron


class NeuronFactory(object):
    """Implementation of Factory Neuron class.
    """
    neurons = {  # 'if': IntegrateAndFire,
            'oscillator': Oscillator,
            'morphed_oscillator': MorphedOscillator,
            'leaky': LeakyIntegrator,
            'sensory': SensoryNeuron,
            'lif_danner_nap': LIFDannerNap,
            'lif_danner': LIFDanner,
            'lif_daun_interneuron': LIFDaunInterneuron,
            'hh_daun_motorneuron': HHDaunMotorneuron,
            'fitzhugh_nagumo': FitzhughNagumo,
            'matsuoka_neuron': MatsuokaNeuron,
            'morris_lecar': MorrisLecarNeuron
        }
    
    def __init__(self):
        """Factory initialization."""
        super(NeuronFactory, self).__init__()        

    @staticmethod
    def register_neuron(neuron_type, neuron_instance):
        """
        Register a new type of neuron that is a child class of Neuron.
        Parameters
        ----------
        self: type
            description
        neuron_type: <str>
            String to identifier for the neuron.
        neuron_instance: <cls>
            Class of the neuron to register.
        """
        NeuronFactory.neurons[neuron_type] = neuron_instance

    @staticmethod
    def gen_neuron(neuron_type):
        """Generate the necessary type of neuron.
        Parameters
        ----------
        self: type
            description
        neuron_type: <str>
            One of the following list of available neurons.
            1. if - Integrate and Fire
            2. lif_danner_nap - LIF Danner Nap
            3. lif_danner - LIF Danner
            4. lif_daun_interneuron - LIF Daun Interneuron
            5. hh_daun_motorneuron - HH_Daun_Motorneuron
        Returns
        -------
        neuron: <cls>
            Appropriate neuron class.
        """
        neuron = NeuronFactory.neurons.get(neuron_type)
        if not neuron:
            raise ValueError(neuron_type)
        return neuron
